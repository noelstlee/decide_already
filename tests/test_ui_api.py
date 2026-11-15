"""The web UI's API, called through FastAPI's test client. Sorting gets a fake backend.

Every client here says it is talking to 127.0.0.1: the server refuses any other host name.
No test can reach the real model. `make_backend` is replaced for the whole file, and the
tests that are about building the real backend put their own stand-in there.

The last part of the file is about the page itself: app.js is run under node with a stand-in
for the browser, and its server is a table inside the script (see PAGE_STAND_IN).
"""

import asyncio
import json
import re
import shutil
import socket
import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decide_already import vault
from decide_already.backends.base import BackendError
from decide_already.cli import app as cli
from decide_already.config import (
    JEV_LOG_FILE,
    LABELS_FILE,
    SUGGESTIONS_FILE,
    Config,
    decide_path,
)
from decide_already.log import append_jsonl, read_jsonl
from decide_already.ui import server
from decide_already.ui.server import STATIC_DIR, create_app
from fakes import FakeBackend

BASE_URL = "http://127.0.0.1:8765"

SOLUTION = "hw3/hw3_solution.py"
CHECKLIST = "misc/apartment_checklist.txt"
EXAMPLES = "project/examples.md"
SHEET = "project/logistics_sheet.md"
SCAN = "scan.bin"
SYLLABUS = "syllabus.md"

TEXTS = {
    SOLUTION: '"""Homework 3 solution."""\n\ndef rbf_kernel(x, y):\n    return 1.0\n',
    CHECKLIST: "Moving out: return the keys, forward the mail, cancel the internet.\n",
    EXAMPLES: "# Project examples\n\nThree past projects with their reports.\n",
    SHEET: "# Project logistics\n\nProposal due in week 4. Late policy: two free days.\n",
    SYLLABUS: "# Syllabus\n\nSchedule, grading, office hours and the late policy.\n",
}
SCAN_BYTES = b"\x00\x01\x02 a scan with no text \x00"

# What the fake model says. The sheet is a close call between Project and Logistics, the
# checklist fits no group, and the scan has no text, so the model is never asked about it.
ANSWERS = {
    SOLUTION: ({"hw": 0.91, "logistics": 0.03, "project": 0.06}, 0.96),
    CHECKLIST: ({"hw": 0.41, "logistics": 0.3, "project": 0.29}, 0.2),
    EXAMPLES: ({"hw": 0.1, "logistics": 0.04, "project": 0.86}, 0.94),
    SHEET: ({"hw": 0.04, "logistics": 0.44, "project": 0.52}, 0.9),
    SYLLABUS: ({"hw": 0.02, "logistics": 0.95, "project": 0.03}, 0.98),
}


class HoldingBackend(FakeBackend):
    """A fake that holds back its answer for one file until the test lets it go."""

    def __init__(self, hold: str, **kwargs) -> None:
        """Set the file to wait at. The other files are answered at once."""
        super().__init__(**kwargs)
        self.hold = hold
        self.reached = threading.Event()  # set when the sort arrives at the held file
        self.release = threading.Event()  # set by the test to let the sort go on

    def sort(self, request):
        """Answer as the fake does, after waiting at the held file."""
        if request.file_id == self.hold:
            self.reached.set()
            if not self.release.wait(timeout=10):
                raise BackendError("the test never let the held file go")
        return super().sort(request)


@pytest.fixture(autouse=True)
def no_real_backend(monkeypatch):
    """Make any attempt to build the real backend fail the test."""

    def make_backend(root, config):
        """Stand in for backends.make_backend."""
        raise AssertionError("a test tried to build the real backend")

    monkeypatch.setattr(server, "make_backend", make_backend)


@pytest.fixture
def root(tmp_path):
    """A vault with three groups (set version 3) and six files in the dump, nothing sorted yet.

    The top cutoff is lowered to 0.4. With the default of 0.6 no answer whose probabilities
    add up to 1 can be a close call, and the board has to show one.
    """
    root = tmp_path / "vault"  # one level down, so a test can check that nothing lands beside it
    root.mkdir()
    config = vault.init_vault(root)
    config.model_copy(update={"unsorted_top_below": 0.4}).save(root)
    vault.add_group(root, "HW", "homework specs, starter code and solutions", "project files")
    vault.add_group(root, "Logistics", "syllabus, schedule and deadlines")
    vault.add_group(root, "Project", "project specs and examples")
    for file_id, text in TEXTS.items():
        drop(root, file_id, text)
    drop(root, SCAN, SCAN_BYTES)
    return root


@pytest.fixture
def backend():
    """The fake model, with an answer for every file that has text."""
    return FakeBackend(answers=dict(ANSWERS))


@pytest.fixture
def client(root, backend):
    """A test client for the vault's app. Sorts and suggests use the fake."""
    return TestClient(create_app(root, lambda: backend), base_url=BASE_URL)


def drop(root, file_id, content):
    """Put a file in the dump folder, as text or as bytes."""
    path = root / "_dump" / file_id
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def wait_for_sort(client, timeout=10.0):
    """Ask for the sort's status until the sort is over, and return the last answer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/api/sort/status").json()
        if not status["running"]:
            return status
        time.sleep(0.01)
    raise AssertionError("the sort did not finish in time")


def sort_now(client):
    """Start a sort through the API, wait for it, and return its report."""
    assert client.post("/api/sort").status_code == 202
    status = wait_for_sort(client)
    assert status["error"] is None
    return status["report"]


def where(root):
    """The folder each file's card note sits in."""
    return {note.placement.file_id: note.folder for note in vault.read_card_notes(root)}


def labels(root):
    """Each line of labels.jsonl as (kind, file id, from group, to group)."""
    lines = read_jsonl(decide_path(root, LABELS_FILE))
    return [(line["kind"], line["file_id"], line["from_group"], line["to_group"]) for line in lines]


def tree(root):
    """Every file under a folder with its bytes, to check that nothing was written."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def column(board, group_id):
    """The file ids in one group's column of a board answer, in the order shown."""
    (found,) = [entry for entry in board["groups"] if entry["group"]["id"] == group_id]
    return [file["file_id"] for file in found["files"]]


def is_error(answer, status_code):
    """True when an answer is an error of this status with one plain line under "detail"."""
    body = answer.json()
    return (
        answer.status_code == status_code
        and set(body) == {"detail"}
        and isinstance(body["detail"], str)
        and body["detail"] != ""
    )


def is_guarded(answer):
    """True when an answer may not be framed, sniffed, or made to load from anywhere else."""
    policy = answer.headers.get("content-security-policy", "")
    return (
        "default-src 'self'" in policy
        and "frame-ancestors 'none'" in policy
        and answer.headers.get("x-frame-options") == "DENY"
        and answer.headers.get("x-content-type-options") == "nosniff"
    )


def hidden_in_dump(root):
    """The names in the dump folder that start with a dot. The sorter never lists those."""
    return sorted(path.name for path in (root / "_dump").iterdir() if path.name.startswith("."))


def put_in_parts(app, name, parts, midway=None, cut_short=False):
    """Send PUT /api/dump/<name> to the app part by part, the way a slow upload arrives.

    The test client only sends whole bodies, so this talks to the app directly. `midway` is
    called once every part has been handed over and the end of the body has not. With
    `cut_short` the browser then goes away and the body never ends. Returns the status and
    the JSON body of the answer.
    """
    messages = [{"type": "http.request", "body": part, "more_body": True} for part in parts]
    messages.append({"type": "http.disconnect" if cut_short else "http.request"})
    answer = {"body": b""}

    async def receive():
        """Hand the app the next piece of the request."""
        if len(messages) == 1 and midway is not None:
            midway()
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        """Keep what the app answers."""
        if message["type"] == "http.response.start":
            answer["status"] = message["status"]
        elif message["type"] == "http.response.body":
            answer["body"] += message.get("body", b"")

    path = f"/api/dump/{name}"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "PUT",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1:8765")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
    }
    asyncio.run(app(scope, receive, send))
    return answer["status"], json.loads(answer["body"])


# The page and its files


def test_page_and_its_files_are_served(client):
    """The page comes with a policy that keeps it to its own files, which are never stale."""
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in page.headers["content-security-policy"]
    for name in ("app.js", "style.css"):
        assert f"/static/{name}" in page.text
        answer = client.get(f"/static/{name}")
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "no-cache"
    # FastAPI's own docs pages would load scripts from a CDN, so they are off.
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_every_answer_carries_the_security_headers(client, root):
    """No answer may be framed by another site: not the page's second address, not an error."""
    served = [
        "/",
        "/static/index.html",  # the same page again, which a route's own headers do not reach
        "/static/app.js",
        "/static/style.css",
        "/api/board",
        "/api/sort/status",
        f"/api/original/{SYLLABUS}",
    ]
    for path in served:
        answer = client.get(path)
        assert answer.status_code == 200, path
        assert is_guarded(answer), path
    refused = [
        client.get("/api/nope"),
        client.get("/static/nope.js"),
        client.get(f"/api/files/{SYLLABUS}"),
        client.post("/api/move", json={}),
        client.put(f"/api/dump/{SYLLABUS}", content=b"again"),
        client.get("/", headers={"host": "evil.example:8765"}),
    ]
    assert [answer.status_code for answer in refused] == [404, 404, 404, 400, 409, 403]
    assert all(is_guarded(answer) for answer in refused)


@pytest.mark.parametrize("name", ["index.html", "app.js", "style.css"])
def test_static_files_load_nothing_from_outside(name):
    """No address of another site, so the page works offline. Also no em dash, no raw HTML."""
    text = (STATIC_DIR / name).read_text(encoding="utf-8")
    assert "://" not in text
    assert not re.search(r"""(src|href)\s*=\s*["'](?!/static/|/api/|data:)""", text)
    assert not re.search(r"url\(|@import", text)
    assert "—" not in text
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write", text)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_script_parses():
    """app.js is valid JavaScript, as far as node can tell without running it."""
    result = subprocess.run(
        ["node", "--check", str(STATIC_DIR / "app.js")], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_page_and_script_agree_on_ids():
    """Every element the script looks up by its id is there in index.html."""
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    in_page = set(re.findall(r'\sid="([^"]+)"', page))
    asked_for = set(re.findall(r'\$\("([^"]+)"\)', script))
    assert asked_for
    assert asked_for <= in_page


def test_every_route_of_the_contract_is_served(client):
    """The routes in docs/implementation.md, by method and path."""
    served = {
        (method, route.path)
        for route in client.app.routes
        for method in getattr(route, "methods", None) or ()
    }
    assert served >= {
        ("GET", "/"),
        ("GET", "/api/board"),
        ("GET", "/api/files/{file_id:path}"),
        ("GET", "/api/original/{file_id:path}"),
        ("POST", "/api/sort"),
        ("GET", "/api/sort/status"),
        ("POST", "/api/move"),
        ("POST", "/api/freeze"),
        ("POST", "/api/unfreeze"),
        ("POST", "/api/groups"),
        ("PATCH", "/api/groups/{group_id}"),
        ("DELETE", "/api/groups/{group_id}"),
        ("PUT", "/api/dump/{name:path}"),
        ("POST", "/api/suggest"),
    }


def test_app_needs_a_vault(tmp_path):
    """A folder that was never set up as a vault is refused when the app is built."""
    with pytest.raises(FileNotFoundError, match="p0 init"):
        create_app(tmp_path)


# GET /api/board


def test_board_of_a_new_vault(tmp_path):
    """No groups and no files yet: an empty board, with what the header shows."""
    vault.init_vault(tmp_path)
    board = TestClient(create_app(tmp_path), base_url=BASE_URL).get("/api/board")
    assert board.status_code == 200
    assert board.json() == {
        "set_version": 0,
        "groups": [],
        "unsorted": [],
        "new_files": [],
        "sorting": False,
        "vault": {
            "name": tmp_path.name,
            "path": str(tmp_path.resolve()),
            "dump_dir": "_dump",
            "model": "jev-1.14.0",
            "spent_usd": 0.0,
            "max_spend_usd": 5.0,
            "cutoffs": {"fits_any": 0.5, "top": 0.6, "close_call": 0.15},
        },
    }


def test_board_before_the_first_sort_lists_the_new_files(client):
    """Groups with empty columns, and every dump file counted as new."""
    board = client.get("/api/board").json()
    assert board["set_version"] == 3
    assert [entry["group"] for entry in board["groups"]] == [
        {
            "id": "hw",
            "name": "HW",
            "covers": "homework specs, starter code and solutions",
            "excludes": "project files",
        },
        {
            "id": "logistics",
            "name": "Logistics",
            "covers": "syllabus, schedule and deadlines",
            "excludes": None,
        },
        {
            "id": "project",
            "name": "Project",
            "covers": "project specs and examples",
            "excludes": None,
        },
    ]
    assert all(entry["files"] == [] for entry in board["groups"])
    assert board["unsorted"] == []
    assert board["new_files"] == [SOLUTION, CHECKLIST, EXAMPLES, SHEET, SCAN, SYLLABUS]
    assert board["vault"]["cutoffs"]["top"] == 0.4


def test_board_after_a_sort(client, root):
    """Each file sits under its group, close calls first, and Unsorted says why."""
    sort_now(client)
    drop(root, "late.txt", "Dropped in after the sort.\n")
    board = client.get("/api/board").json()
    assert column(board, "hw") == [SOLUTION]
    assert column(board, "logistics") == [SYLLABUS]
    assert column(board, "project") == [SHEET, EXAMPLES]  # the close call comes first
    assert board["groups"][2]["files"][0] == {
        "file_id": SHEET,
        "name": "logistics_sheet.md",
        "file_type": "Markdown",
        "group_id": "project",
        "frozen": False,
        "close_call": True,
        "placed_by": "jev",
        "reason": None,
        "top_prob": 0.52,
        "probs": {"hw": 0.04, "logistics": 0.44, "project": 0.52},
        "fits_any": 0.9,
        "stale": False,
    }
    unsorted = {file["file_id"]: file for file in board["unsorted"]}
    assert sorted(unsorted) == [CHECKLIST, SCAN]
    assert (unsorted[CHECKLIST]["reason"], unsorted[CHECKLIST]["top_prob"]) == (
        "no_group_fits",
        0.41,
    )
    assert (unsorted[SCAN]["reason"], unsorted[SCAN]["placed_by"]) == ("no_text", "code")
    assert unsorted[SCAN]["probs"] is None
    assert board["new_files"] == ["late.txt"]


def test_board_picks_up_a_note_dragged_by_hand(client, root):
    """The board runs sync first: a note moved in Obsidian is a move by the user."""
    sort_now(client)
    note = vault.note_name(SHEET)
    shutil.move(root / "Project" / note, root / "Logistics" / note)
    board = client.get("/api/board").json()
    assert column(board, "logistics") == [SHEET, SYLLABUS]
    moved = board["groups"][1]["files"][0]
    assert (moved["placed_by"], moved["frozen"], moved["close_call"]) == ("you", True, False)
    assert labels(root) == [("moved", SHEET, "project", "logistics")]


def test_board_shows_the_spend_so_far(client, root):
    """Spend is the log's input tokens at the config's price, and follows the log as it grows."""
    log = decide_path(root, JEV_LOG_FILE)
    assert client.get("/api/board").json()["vault"]["spent_usd"] == 0.0
    append_jsonl(log, {"file_id": SYLLABUS, "input_tokens": 1_000_000})
    assert client.get("/api/board").json()["vault"]["spent_usd"] == pytest.approx(0.042)
    append_jsonl(log, {"file_id": SOLUTION, "input_tokens": 500_000})
    assert client.get("/api/board").json()["vault"]["spent_usd"] == pytest.approx(0.063)
    log.unlink()
    assert client.get("/api/board").json()["vault"]["spent_usd"] == 0.0


def test_board_reports_a_broken_group_note(client, root):
    """A group note without covers stops the board with the reason, as a plain error."""
    (root / "_groups" / "HW.md").write_text("---\nid: hw\n---\n", encoding="utf-8")
    answer = client.get("/api/board")
    assert is_error(answer, 400)
    assert "no covers description" in answer.json()["detail"]


# GET /api/files and GET /api/original


def test_file_detail(client):
    """One file: where it is, its placement, the probabilities highest first, and its card."""
    sort_now(client)
    answer = client.get(f"/api/files/{SHEET}")
    assert answer.status_code == 200
    detail = answer.json()
    assert sorted(detail) == ["card", "file_id", "group_id", "placement", "probs"]
    assert (detail["file_id"], detail["group_id"]) == (SHEET, "project")
    assert detail["probs"] == [
        {"group_id": "project", "name": "Project", "prob": 0.52},
        {"group_id": "logistics", "name": "Logistics", "prob": 0.44},
        {"group_id": "hw", "name": "HW", "prob": 0.04},
    ]
    placement = detail["placement"]
    assert (placement["group_id"], placement["placed_by"]) == ("project", "jev")
    assert (placement["close_call"], placement["frozen"]) == (True, False)
    assert (placement["fits_any"], placement["set_version"]) == (0.9, 3)
    assert placement["model"] == "fake-0"
    assert "Proposal due in week 4" in detail["card"]


def test_file_detail_of_a_file_the_model_was_not_asked_about(client):
    """A file with no text has a card note too, with no probabilities."""
    sort_now(client)
    detail = client.get(f"/api/files/{SCAN}").json()
    assert (detail["group_id"], detail["probs"]) == (None, [])
    assert (detail["placement"]["reason"], detail["placement"]["placed_by"]) == ("no_text", "code")
    assert "No text could be extracted" in detail["card"]


def test_file_detail_names_the_groups_as_they_are_now(client):
    """After a rename the new name is shown. A deleted group has no name left."""
    sort_now(client)
    assert client.patch("/api/groups/logistics", json={"name": "Admin"}).status_code == 200
    assert client.delete("/api/groups/hw").status_code == 200
    detail = client.get(f"/api/files/{SHEET}").json()
    assert detail["probs"] == [
        {"group_id": "project", "name": "Project", "prob": 0.52},
        {"group_id": "logistics", "name": "Admin", "prob": 0.44},
        {"group_id": "hw", "name": None, "prob": 0.04},
    ]


def test_file_detail_knows_the_group_of_a_folder_spelled_another_way(client, root):
    """A folder named hw is the group HW's folder, as the file system of a Mac has it."""
    sort_now(client)
    (root / "HW").rename(root / "hw")
    assert client.get(f"/api/files/{SOLUTION}").json()["group_id"] == "hw"


def test_file_detail_agrees_with_the_board_for_a_folder_that_is_no_groups(client, root):
    """A group note renamed by hand: until the next sync, panel and board both go by the note."""
    sort_now(client)
    (root / "HW").rename(root / "Old homework")  # the group's folder renamed by hand
    assert client.get(f"/api/files/{SOLUTION}").json()["group_id"] == "hw"
    (root / "_groups" / "HW.md").unlink()  # and now the group itself is gone
    assert client.get(f"/api/files/{SOLUTION}").json()["group_id"] is None


def test_file_detail_of_an_unknown_or_unsorted_file(client):
    """404 for an id that is not in the dump, and for a file with no card note yet."""
    assert is_error(client.get("/api/files/nope.txt"), 404)
    answer = client.get(f"/api/files/{SYLLABUS}")
    assert is_error(answer, 404)
    assert "not been sorted yet" in answer.json()["detail"]


def test_original_is_sent_as_it_is(client, root):
    """The original's bytes come back unchanged, also for a file that was never sorted."""
    answer = client.get(f"/api/original/{SOLUTION}")
    assert answer.status_code == 200
    assert answer.content == (root / "_dump" / SOLUTION).read_bytes()
    assert answer.headers["content-type"].startswith("text/plain")
    assert answer.headers["x-content-type-options"] == "nosniff"
    assert answer.headers["content-disposition"].startswith("inline")


def test_original_is_never_shown_from_the_browsers_store_unasked(client, root):
    """The browser is told to check back each time: a file in the dump may have been replaced."""
    drop(root, "paper.pdf", b"%PDF-1.4\n\x00\x01")
    for file_id in (SYLLABUS, SCAN, "paper.pdf"):
        answer = client.get(f"/api/original/{file_id}")
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "no-cache", file_id


def test_original_is_never_run_as_a_page(client, root):
    """A web page from the dump is shown as its source. A PDF opens, other binaries download."""
    drop(root, "page.html", "<script>fetch('/api/sort', {method: 'POST'})</script>")
    drop(root, "drawing.svg", "<svg xmlns='x' onload='alert(1)'></svg>")
    drop(root, "paper.pdf", b"%PDF-1.4\n\x00\x01")
    for file_id in ("page.html", "drawing.svg"):
        answer = client.get(f"/api/original/{file_id}")
        assert answer.headers["content-type"].startswith("text/plain")
        assert answer.headers["x-content-type-options"] == "nosniff"
    paper = client.get("/api/original/paper.pdf")
    assert paper.headers["content-type"] == "application/pdf"
    assert paper.headers["content-disposition"].startswith("inline")
    scan = client.get(f"/api/original/{SCAN}")
    assert scan.content == SCAN_BYTES
    assert scan.headers["content-type"] == "application/octet-stream"
    assert scan.headers["content-disposition"].startswith("attachment")


@pytest.mark.parametrize(
    "path",
    [
        "/api/original/%2e%2e/secret.txt",
        "/api/original/%2e%2e%2fsecret.txt",
        "/api/original/%2e%2e/.decide/config.yaml",
        "/api/original/%2e%2e/_groups/HW.md",
        "/api/original/_dump/syllabus.md",
        "/api/original/.hidden/secret.txt",
        "/api/original/hw3",
        "/api/original/",
        "/api/files/%2e%2e/secret.txt",
        "/api/files/%2e%2e/.decide/config.yaml",
    ],
)
def test_path_that_leaves_the_dump_is_not_found(client, root, path):
    """Only ids from the dump listing are served: nothing above it, hidden in it, or absolute."""
    (root / "secret.txt").write_text("not for the page", encoding="utf-8")
    drop(root, ".hidden/secret.txt", "hidden files are skipped")
    assert is_error(client.get(path), 404)
    absolute = root / "secret.txt"
    assert is_error(client.get(f"/api/original/{absolute}"), 404)
    assert is_error(client.get(f"/api/original/{absolute.as_posix().lstrip('/')}"), 404)


# GET /api/dump and DELETE /api/dump/{file_id}


def test_dump_listing_gives_every_file_with_its_size(client, root):
    """The Files view lists what is on disk: every file id, its size and when it changed."""
    listing = client.get("/api/dump").json()
    assert listing["dump_dir"] == "_dump"
    sizes = {entry["file_id"]: entry["size"] for entry in listing["files"]}
    assert sizes == {
        **{file_id: len(text.encode("utf-8")) for file_id, text in TEXTS.items()},
        SCAN: len(SCAN_BYTES),
    }
    assert all(isinstance(entry["modified"], float) for entry in listing["files"])


def test_delete_removes_the_original_and_its_card_note(client, root):
    """Deleting a sorted file takes it out of the dump and off the board. Labels stay."""
    sort_now(client)
    moved = client.post("/api/move", json={"file_id": SHEET, "group_id": "logistics"})
    assert moved.status_code == 200
    answer = client.delete(f"/api/dump/{SHEET}")
    assert answer.status_code == 200
    assert answer.json() == {"deleted": SHEET, "card_notes_removed": 1}
    assert not (root / "_dump" / SHEET).exists()
    assert not any(path.name == vault.note_name(SHEET) for path in root.rglob("*.md"))
    board = client.get("/api/board").json()
    on_board = [
        f["file_id"]
        for f in [*board["unsorted"], *(f for g in board["groups"] for f in g["files"])]
    ]
    assert SHEET not in on_board
    assert SHEET not in board["new_files"]
    assert labels(root) == [("moved", SHEET, "project", "logistics")]  # the record stays
    assert (root / "_dump" / EXAMPLES).exists()  # the folder still holds the other file


def test_delete_removes_folders_it_leaves_empty(client, root):
    """A folder of the dump that only held the deleted file goes too; the dump folder stays."""
    assert client.delete(f"/api/dump/{SOLUTION}").status_code == 200
    assert not (root / "_dump" / "hw3").exists()
    for file_id in [CHECKLIST, EXAMPLES, SHEET, SCAN, SYLLABUS]:
        assert client.delete(f"/api/dump/{file_id}").status_code == 200
    assert (root / "_dump").is_dir()
    assert list((root / "_dump").iterdir()) == []


def test_delete_of_a_file_never_sorted(client, root):
    """A file with no card note yet is deleted all the same."""
    answer = client.delete(f"/api/dump/{SYLLABUS}")
    assert answer.json() == {"deleted": SYLLABUS, "card_notes_removed": 0}
    assert not (root / "_dump" / SYLLABUS).exists()


@pytest.mark.parametrize(
    "path",
    [
        "/api/dump/nope.txt",
        "/api/dump/%2e%2e/secret.txt",
        "/api/dump/%2e%2e/.decide/config.yaml",
        "/api/dump/%2e%2e/_groups/HW.md",
        "/api/dump/.hidden/secret.txt",
        "/api/dump/hw3",
    ],
)
def test_delete_reaches_nothing_but_a_file_of_the_dump(client, root, path):
    """Only an id from the dump listing can be deleted: nothing above it, hidden in it, or a folder."""
    (root / "secret.txt").write_text("not for the page", encoding="utf-8")
    drop(root, ".hidden/secret.txt", "hidden files are skipped")
    before = tree(root)
    assert is_error(client.delete(path), 404)
    assert tree(root) == before


def test_delete_of_a_link_removes_the_link_only(client, root, tmp_path):
    """A link in the dump is removed as a link; the file it points to stays."""
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("keep me", encoding="utf-8")
    (root / "_dump" / "link.txt").symlink_to(outside)
    assert client.delete("/api/dump/link.txt").status_code == 200
    assert not (root / "_dump" / "link.txt").is_symlink()
    assert outside.read_text(encoding="utf-8") == "keep me"


def test_delete_is_refused_while_a_sort_runs(root):
    """The sort read the dump at its start, so nothing is deleted under it."""
    backend = HoldingBackend(hold=SHEET, answers=dict(ANSWERS))
    client = TestClient(create_app(root, lambda: backend), base_url=BASE_URL)
    assert client.post("/api/sort").status_code == 202
    assert backend.reached.wait(timeout=10)
    assert is_error(client.delete(f"/api/dump/{SYLLABUS}"), 409)
    assert (root / "_dump" / SYLLABUS).exists()
    backend.release.set()
    wait_for_sort(client)


# POST /api/move, /api/freeze and /api/unfreeze


def test_move_puts_a_file_in_a_group_and_freezes_it(client, root):
    """A move is the user's own placement: the note changes folder and a label is written."""
    sort_now(client)
    answer = client.post("/api/move", json={"file_id": SHEET, "group_id": "logistics"})
    assert answer.status_code == 200
    placement = answer.json()
    assert (placement["file_id"], placement["group_id"]) == (SHEET, "logistics")
    assert (placement["placed_by"], placement["frozen"], placement["close_call"]) == (
        "you",
        True,
        False,
    )
    assert where(root)[SHEET] == "Logistics"

    answer = client.post("/api/move", json={"file_id": SHEET, "group_id": None})
    assert answer.json()["group_id"] is None
    assert where(root)[SHEET] == "Unsorted"
    assert labels(root) == [
        ("moved", SHEET, "project", "logistics"),
        ("moved", SHEET, "logistics", None),
    ]


def test_move_checks_the_file_and_the_group(client, root):
    """Only a file id from the dump and a group id from the current set are taken."""
    sort_now(client)
    before = tree(root)
    assert is_error(client.post("/api/move", json={"file_id": "nope.txt", "group_id": "hw"}), 404)
    assert is_error(client.post("/api/move", json={"file_id": SHEET, "group_id": "nope"}), 404)
    # The commands also take names. The page always sends ids, so the API takes nothing else.
    assert is_error(client.post("/api/move", json={"file_id": SHEET, "group_id": "Logistics"}), 404)
    assert is_error(
        client.post("/api/move", json={"file_id": "logistics_sheet.md", "group_id": "hw"}), 404
    )
    assert is_error(client.post("/api/move", json={"file_id": "../x", "group_id": "hw"}), 404)
    assert tree(root) == before


def test_move_of_a_file_that_was_never_sorted(client, root):
    """A new file has no card note to move: it is not found, and nothing is written."""
    before = tree(root)
    answer = client.post("/api/move", json={"file_id": SYLLABUS, "group_id": "logistics"})
    assert is_error(answer, 404)
    assert "not been sorted yet" in answer.json()["detail"]
    assert tree(root) == before


def test_move_never_takes_one_file_for_another_of_the_same_name(client, root):
    """A new file that is named like a sorted one in a subfolder: the sorted one stays put."""
    sort_now(client)
    drop(root, "logistics_sheet.md", "Another file that happens to have the same name.\n")
    before = tree(root)
    answer = client.post("/api/move", json={"file_id": "logistics_sheet.md", "group_id": "hw"})
    assert is_error(answer, 404)
    assert "not been sorted yet" in answer.json()["detail"]
    assert tree(root) == before
    assert where(root)[SHEET] == "Project"


def test_freeze_and_unfreeze_a_file_or_a_group(client, root):
    """The body names one file, a group, or Unsorted (null). The answer lists what changed."""
    sort_now(client)
    one = {"file_id": SOLUTION}
    assert client.post("/api/freeze", json=one).json() == {"frozen": [SOLUTION]}
    assert client.post("/api/freeze", json=one).json() == {"frozen": []}
    assert client.post("/api/freeze", json={"group_id": "project"}).json() == {
        "frozen": [EXAMPLES, SHEET]
    }
    assert client.post("/api/freeze", json={"group_id": None}).json() == {
        "frozen": [CHECKLIST, SCAN]
    }
    board = client.get("/api/board").json()
    frozen = {
        file["file_id"]: file["frozen"] for entry in board["groups"] for file in entry["files"]
    }
    assert frozen == {SOLUTION: True, SYLLABUS: False, EXAMPLES: True, SHEET: True}

    assert client.post("/api/unfreeze", json={"group_id": "project"}).json() == {
        "unfrozen": [EXAMPLES, SHEET]
    }
    assert client.post("/api/unfreeze", json={"group_id": None}).json() == {
        "unfrozen": [CHECKLIST, SCAN]
    }
    assert client.post("/api/unfreeze", json=one).json() == {"unfrozen": [SOLUTION]}
    assert client.post("/api/unfreeze", json=one).json() == {"unfrozen": []}
    assert [kind for kind, *_ in labels(root)] == ["confirmed"] * 5 + ["unfrozen"] * 5


def test_freeze_of_a_file_named_like_a_group_touches_only_that_file(client, backend, root):
    """A dump file called logistics, HW or Unsorted is a file. The lane is named by group_id."""
    to_logistics = ({"hw": 0.02, "logistics": 0.95, "project": 0.03}, 0.98)
    for name in ("logistics", "HW", "Unsorted"):  # a group's id, a group's name, the lane
        drop(root, name, f"A top-level file that is named {name}.\n")
        backend.answers[name] = to_logistics
    sort_now(client)
    assert column(client.get("/api/board").json(), "logistics") == [
        "HW",
        "logistics",
        SYLLABUS,
        "Unsorted",
    ]

    for name in ("logistics", "HW", "Unsorted"):
        assert client.post("/api/freeze", json={"file_id": name}).json() == {"frozen": [name]}
    assert labels(root) == [
        ("confirmed", name, "logistics", "logistics") for name in ("logistics", "HW", "Unsorted")
    ]
    # The lanes those names could be taken for were not touched, and can still be named.
    assert client.post("/api/freeze", json={"group_id": "logistics"}).json() == {
        "frozen": [SYLLABUS]
    }
    assert client.post("/api/freeze", json={"group_id": "hw"}).json() == {"frozen": [SOLUTION]}
    assert client.post("/api/freeze", json={"group_id": None}).json() == {
        "frozen": [CHECKLIST, SCAN]
    }

    for name in ("logistics", "HW", "Unsorted"):
        assert client.post("/api/unfreeze", json={"file_id": name}).json() == {"unfrozen": [name]}
    board = client.get("/api/board").json()
    still_frozen = [
        file["file_id"]
        for files in (board["unsorted"], *(entry["files"] for entry in board["groups"]))
        for file in files
        if file["frozen"]
    ]
    assert sorted(still_frozen) == [SOLUTION, CHECKLIST, SCAN, SYLLABUS]


def test_freeze_checks_what_it_is_asked_to_freeze(client, root):
    """Only exact ids are taken, and the body has to say whether it names a file or a group."""
    sort_now(client)
    drop(root, "late.txt", "Dropped in after the sort.\n")
    before = tree(root)
    for route in ("/api/freeze", "/api/unfreeze"):
        assert is_error(client.post(route, json={"file_id": "nope"}), 404)
        assert is_error(client.post(route, json={"file_id": "late.txt"}), 404)  # not sorted yet
        assert is_error(client.post(route, json={"file_id": "hw3_solution.py"}), 404)  # a name
        assert is_error(client.post(route, json={"file_id": "project"}), 404)  # a group's id
        assert is_error(client.post(route, json={"file_id": "../.decide/config.yaml"}), 404)
        assert is_error(client.post(route, json={"group_id": "nope"}), 404)
        assert is_error(client.post(route, json={"group_id": "Project"}), 404)  # a group's name
        assert is_error(client.post(route, json={"group_id": "Unsorted"}), 404)  # null means that
        assert is_error(client.post(route, json={"group_id": SOLUTION}), 404)  # a file's id
        assert is_error(client.post(route, json={}), 400)
        assert is_error(client.post(route, json={"file_id": None}), 400)
        assert is_error(client.post(route, json={"file_id": SOLUTION, "group_id": "hw"}), 400)
        assert is_error(client.post(route, json={"file_id": SOLUTION, "group_id": None}), 400)
        assert is_error(client.post(route, json={"ref": SOLUTION}), 400)  # the old body
    assert tree(root) == before


# POST, PATCH and DELETE /api/groups


def test_group_add_edit_rename_and_delete(client, root):
    """Each change answers with the new set version and the group as it is now."""
    answer = client.post("/api/groups", json={"name": "Lecture notes", "covers": "slides, notes"})
    assert answer.status_code == 201
    assert answer.json() == {
        "set_version": 4,
        "group": {
            "id": "lecture-notes",
            "name": "Lecture notes",
            "covers": "slides, notes",
            "excludes": None,
        },
    }
    assert (root / "Lecture notes").is_dir()

    changes = {"covers": "lecture slides and notes", "excludes": "homework"}
    answer = client.patch("/api/groups/lecture-notes", json=changes)
    assert answer.status_code == 200
    assert answer.json()["set_version"] == 5
    assert answer.json()["group"] == {"id": "lecture-notes", "name": "Lecture notes", **changes}

    # An empty excludes text removes it.
    answer = client.patch("/api/groups/lecture-notes", json={"excludes": ""})
    assert (answer.json()["set_version"], answer.json()["group"]["excludes"]) == (6, None)

    # A rename keeps the id, and moves the folder with its card notes.
    answer = client.patch("/api/groups/lecture-notes", json={"name": "Lectures"})
    assert answer.json()["set_version"] == 7
    assert (answer.json()["group"]["id"], answer.json()["group"]["name"]) == (
        "lecture-notes",
        "Lectures",
    )
    assert (root / "Lectures").is_dir()
    assert not (root / "Lecture notes").exists()

    # Name and description in one request: one rename and one edit.
    both = {"name": "Lecture slides", "covers": "slides only"}
    answer = client.patch("/api/groups/lecture-notes", json=both)
    assert answer.json()["group"] == {"id": "lecture-notes", **both, "excludes": None}
    assert answer.json()["set_version"] == 9

    # Nothing to change: the set version stays.
    assert client.patch("/api/groups/lecture-notes", json={}).json()["set_version"] == 9
    assert client.patch("/api/groups/lecture-notes", json=both).json()["set_version"] == 9

    answer = client.delete("/api/groups/lecture-notes")
    assert answer.status_code == 200
    assert answer.json() == {"set_version": 10}
    board = client.get("/api/board").json()
    assert board["set_version"] == 10
    assert [entry["group"]["id"] for entry in board["groups"]] == ["hw", "logistics", "project"]


def test_group_routes_refuse_bad_input(client, root):
    """A taken name is a conflict, a bad name or an empty description a bad request."""
    before = tree(root)
    assert is_error(client.post("/api/groups", json={"name": "HW", "covers": "again"}), 409)
    assert is_error(client.post("/api/groups", json={"name": "HW!", "covers": "same id"}), 400)
    assert is_error(client.post("/api/groups", json={"name": "Exams", "covers": "  "}), 400)
    assert is_error(client.post("/api/groups", json={"name": "Unsorted", "covers": "x"}), 400)
    assert is_error(client.post("/api/groups", json={"name": "a/b", "covers": "x"}), 400)
    assert is_error(client.post("/api/groups", json={"name": "../up", "covers": "x"}), 400)
    assert is_error(client.post("/api/groups", json={"name": "Exams"}), 400)
    assert is_error(client.patch("/api/groups/nope", json={"covers": "x"}), 404)
    assert is_error(client.patch("/api/groups/HW", json={"covers": "x"}), 404)  # a name, not an id
    assert is_error(client.patch("/api/groups/hw", json={"name": "Logistics"}), 409)
    assert is_error(client.patch("/api/groups/hw", json={"name": "_dump"}), 400)
    assert is_error(client.patch("/api/groups/hw", json={"color": "red"}), 400)
    assert is_error(client.delete("/api/groups/nope"), 404)
    assert tree(root) == before


def test_group_edit_with_an_empty_description_changes_nothing(client, root):
    """A good new name with an empty covers text: the rename is not carried out either."""
    before = tree(root)
    answer = client.patch("/api/groups/hw", json={"name": "Homework", "covers": " "})
    assert is_error(answer, 400)
    assert tree(root) == before


def test_deleted_groups_files_go_to_unsorted(client, root):
    """After a delete, the next board shows the group's files in Unsorted, not frozen."""
    sort_now(client)
    assert client.post("/api/freeze", json={"file_id": SOLUTION}).json() == {"frozen": [SOLUTION]}
    assert client.delete("/api/groups/hw").json() == {"set_version": 4}
    board = client.get("/api/board").json()
    assert [entry["group"]["id"] for entry in board["groups"]] == ["logistics", "project"]
    orphan = {file["file_id"]: file for file in board["unsorted"]}[SOLUTION]
    assert (orphan["group_id"], orphan["frozen"], orphan["placed_by"]) == (None, False, "code")
    assert where(root)[SOLUTION] == "Unsorted"


# PUT /api/dump


def test_upload_adds_a_file_to_the_dump(client, root):
    """The raw body becomes a new file in the dump folder, which the board then counts as new."""
    answer = client.put("/api/dump/week 1 notes.txt", content=b"week 1 notes\n")
    assert answer.status_code == 201
    assert answer.json() == {"file_id": "week 1 notes.txt"}
    assert (root / "_dump" / "week 1 notes.txt").read_bytes() == b"week 1 notes\n"
    assert "week 1 notes.txt" in client.get("/api/board").json()["new_files"]


def test_upload_keeps_the_bytes_as_they_are(client, root):
    """A large binary body arrives whole, and an empty one makes an empty file."""
    body = bytes(range(256)) * 8192  # 2 MB
    assert client.put("/api/dump/slides.pptx", content=body).status_code == 201
    assert (root / "_dump" / "slides.pptx").read_bytes() == body
    assert client.put("/api/dump/empty.txt", content=b"").status_code == 201
    assert (root / "_dump" / "empty.txt").read_bytes() == b""


def test_upload_never_overwrites(client, root):
    """A name that is already in the dump is refused, and the file there stays as it was."""
    before = tree(root)
    answer = client.put(f"/api/dump/{SYLLABUS}", content=b"something else")
    assert is_error(answer, 409)
    assert "already in the dump folder" in answer.json()["detail"]
    assert tree(root) == before


def test_upload_of_a_taken_name_is_refused_before_the_body_is_read(client, root):
    """A name that is already there is turned down at once, however long the body would be."""
    asked_for = []
    status, body = put_in_parts(
        client.app, SYLLABUS, [b"part one, ", b"part two"], lambda: asked_for.append("the rest")
    )
    assert status == 409
    assert "already in the dump folder" in body["detail"]
    assert asked_for == []


def test_upload_never_follows_a_link_that_sits_at_its_name(client, root, tmp_path):
    """A link in the dump is a name that is taken, even when it points at nothing."""
    outside = tmp_path / "outside.txt"
    link = root / "_dump" / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("this file system has no symbolic links")
    assert is_error(client.put("/api/dump/linked.txt", content=b"sent through the page"), 409)
    assert link.is_symlink()
    assert not outside.exists()
    assert hidden_in_dump(root) == []


def test_upload_never_writes_over_a_name_that_was_taken_while_it_arrived(client, root):
    """The name is free when the upload starts and taken when it ends: the other file stays."""
    target = root / "_dump" / "notes.txt"

    def copy_in_by_hand():
        """The user puts a file of the same name in the dump while the body is on its way."""
        target.write_bytes(b"copied in by hand")

    status, body = put_in_parts(
        client.app, "notes.txt", [b"sent ", b"through the page"], copy_in_by_hand
    )
    assert status == 409
    assert "already in the dump folder" in body["detail"]
    assert target.read_bytes() == b"copied in by hand"
    assert hidden_in_dump(root) == []


def test_upload_that_is_still_arriving_is_no_file_of_the_dump_yet(client, root):
    """A sort that runs while a body arrives never sees the file. It appears whole, at the end."""
    name = "big_report.md"
    seen = {}

    def look_and_sort():
        """What the dump, the board and a whole sort see while the end of the body is missing."""
        seen["hidden"] = hidden_in_dump(root)
        seen["in_place"] = (root / "_dump" / name).exists()
        seen["listed"] = [file_id for file_id, _ in vault.dump_files(root, Config.load(root))]
        seen["new"] = client.get("/api/board").json()["new_files"]
        sort_now(client)
        seen["sorted"] = sorted(where(root))

    parts = [b"# Project report\n\nThe first half.\n", b"The second half.\n"]
    status, body = put_in_parts(client.app, name, parts, look_and_sort)
    assert (status, body) == (201, {"file_id": name})
    assert len(seen["hidden"]) == 1  # the body is collected under a hidden name
    assert seen["in_place"] is False
    assert name not in seen["listed"]
    assert name not in seen["new"]
    assert seen["sorted"] == sorted([*TEXTS, SCAN])  # the sort placed the others, and only them

    assert (root / "_dump" / name).read_bytes() == b"".join(parts)
    assert hidden_in_dump(root) == []
    assert client.get("/api/board").json()["new_files"] == [name]


def test_upload_that_stops_half_way_leaves_nothing_behind(client, root):
    """The browser goes away before the body ends: no file, not even the hidden one."""
    before = tree(root)
    seen = {}

    def look():
        """What is in the dump just before the connection is lost."""
        seen["hidden"] = hidden_in_dump(root)

    status, body = put_in_parts(client.app, "half.bin", [b"the first half"], look, cut_short=True)
    assert len(seen["hidden"]) == 1
    assert status == 400
    assert "did not arrive in full" in body["detail"]
    assert tree(root) == before


@pytest.mark.parametrize(
    "name",
    [
        "",
        "%20",
        ".hidden",
        ".env",
        "%2e%2e",
        "hw3/new.py",
        "a%2Fb.txt",
        "%2e%2e%2fescape.txt",
        "%2e%2e/escape.txt",
        "a%5Cb.txt",
        "/etc/passwd",
    ],
)
def test_upload_takes_a_plain_file_name_only(client, root, name):
    """Empty, hidden, or a path of any kind: refused, and nothing is written anywhere."""
    before = tree(root.parent)
    assert is_error(client.put(f"/api/dump/{name}", content=b"x"), 400)
    assert tree(root.parent) == before


# POST /api/sort and GET /api/sort/status


def test_sort_status_before_any_sort(client):
    """Nothing has run yet: not running, nothing done, no report and no error."""
    assert client.get("/api/sort/status").json() == {
        "running": False,
        "done": 0,
        "total": 0,
        "report": None,
        "error": None,
    }


def test_sort_runs_in_the_background_and_reports(client, backend, root):
    """The sort is started, followed through its status, and leaves its report there."""
    answer = client.post("/api/sort")
    assert answer.status_code == 202
    assert answer.json()["running"] is True
    status = wait_for_sort(client)
    assert status == {
        "running": False,
        "done": 6,
        "total": 6,
        "report": {
            "set_version": 3,
            "asked": 5,
            "placed": 4,
            "unsorted": 2,
            "close_calls": 1,
            "frozen_skipped": 0,
            "errors": {},
            "spent_usd": None,
        },
        "error": None,
    }
    assert [request.file_id for request in backend.calls] == sorted(ANSWERS)
    assert where(root) == {
        SOLUTION: "HW",
        CHECKLIST: "Unsorted",
        EXAMPLES: "Project",
        SHEET: "Project",
        SCAN: "Unsorted",
        SYLLABUS: "Logistics",
    }


def test_only_one_sort_runs_and_nothing_changes_the_vault_meanwhile(root):
    """While a sort runs: 409 for a second one and for every change, but the board still reads."""
    backend = HoldingBackend(hold=SHEET, answers=dict(ANSWERS))
    client = TestClient(create_app(root, lambda: backend), base_url=BASE_URL)
    assert client.post("/api/sort").status_code == 202
    assert backend.reached.wait(timeout=10)

    # Files are sorted in id order: three are done, and the model is being asked about the fourth.
    status = client.get("/api/sort/status").json()
    assert (status["running"], status["report"], status["error"]) == (True, None, None)
    assert (status["done"], status["total"]) == (3, 6)

    again = client.post("/api/sort")
    assert is_error(again, 409)
    assert "already running" in again.json()["detail"]
    changes = [
        client.post("/api/move", json={"file_id": SOLUTION, "group_id": "project"}),
        client.post("/api/freeze", json={"file_id": SOLUTION}),
        client.post("/api/unfreeze", json={"group_id": "hw"}),
        client.post("/api/groups", json={"name": "Exams", "covers": "exams"}),
        client.patch("/api/groups/hw", json={"covers": "homework"}),
        client.delete("/api/groups/hw"),
        client.post("/api/suggest"),
    ]
    for answer in changes:
        assert is_error(answer, 409)
        assert "a sort is running" in answer.json()["detail"]

    board = client.get("/api/board")
    assert board.status_code == 200
    assert board.json()["sorting"] is True
    assert column(board.json(), "hw") == [SOLUTION]  # placed before the sort stopped to wait
    assert client.get(f"/api/files/{SOLUTION}").status_code == 200
    assert (
        client.put("/api/dump/late.txt", content=b"dropped in during the sort").status_code == 201
    )

    backend.release.set()
    status = wait_for_sort(client)
    assert (status["running"], status["error"]) == (False, None)
    assert (status["done"], status["total"]) == (6, 6)
    assert status["report"]["placed"] == 4
    assert labels(root) == []  # none of the refused changes left a trace
    assert client.get("/api/board").json()["new_files"] == ["late.txt"]

    # The next sort may start, and its status starts over.
    assert client.post("/api/sort").status_code == 202
    assert wait_for_sort(client)["total"] == 7


def test_board_leaves_the_vault_alone_while_a_sort_runs(root):
    """During a sort the board only reads: no hand edit is picked up, and the log is left alone."""
    backend = HoldingBackend(hold=SHEET, answers=dict(ANSWERS))
    client = TestClient(create_app(root, lambda: backend), base_url=BASE_URL)
    backend.release.set()  # the first sort runs through
    sort_now(client)
    assert client.get("/api/board").json()["vault"]["spent_usd"] == 0.0

    backend.release.clear()
    backend.reached.clear()
    assert client.post("/api/sort").status_code == 202
    assert backend.reached.wait(timeout=10)
    # While the sort waits at the sheet, a card note it is done with is dragged by hand to
    # another group's folder, and the Jev log grows as it does when calls are logged.
    note = vault.note_name(SOLUTION)
    dragged = root / "Project" / note
    shutil.move(root / "HW" / note, dragged)
    as_dragged = dragged.read_bytes()
    append_jsonl(decide_path(root, JEV_LOG_FILE), {"file_id": SOLUTION, "input_tokens": 1_000_000})

    board = client.get("/api/board").json()
    assert board["sorting"] is True
    assert SOLUTION in column(board, "project")  # shown where the note sits
    assert labels(root) == []  # but the drag is not recorded as a move yet
    assert dragged.read_bytes() == as_dragged  # and the note is not rewritten
    assert board["vault"]["spent_usd"] == 0.0  # the spend is the one from before the sort

    backend.release.set()
    assert wait_for_sort(client)["error"] is None
    board = client.get("/api/board").json()
    assert labels(root) == [("moved", SOLUTION, "hw", "project")]
    assert board["vault"]["spent_usd"] == pytest.approx(0.042)


def test_sort_lists_the_files_the_backend_failed_on(root):
    """A failure on one file is in the report, and the other files are still sorted."""
    backend = FakeBackend(answers=dict(ANSWERS), fail={SYLLABUS})
    client = TestClient(create_app(root, lambda: backend), base_url=BASE_URL)
    client.post("/api/sort")
    status = wait_for_sort(client)
    assert status["error"] is None
    assert list(status["report"]["errors"]) == [SYLLABUS]
    assert "told to fail" in status["report"]["errors"][SYLLABUS]
    assert SYLLABUS not in where(root)
    assert status["report"]["placed"] == 3


def test_sort_without_groups_ends_with_the_error(tmp_path):
    """A sort that cannot run leaves its reason in the status, and the next one may start."""
    vault.init_vault(tmp_path)
    drop(tmp_path, SYLLABUS, TEXTS[SYLLABUS])
    client = TestClient(create_app(tmp_path, FakeBackend), base_url=BASE_URL)
    assert client.post("/api/sort").status_code == 202
    status = wait_for_sort(client)
    assert (status["running"], status["report"]) == (False, None)
    assert "no groups yet" in status["error"]
    assert client.post("/api/sort").status_code == 202
    wait_for_sort(client)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_sort_that_hits_a_bug_still_ends(root):
    """An error nobody planned for ends the sort as well: the page is told where to look."""

    def broken_factory():
        """Stand in for a backend that cannot even be built, for a reason the tool does not know."""
        raise RuntimeError("a bug")

    client = TestClient(create_app(root, broken_factory), base_url=BASE_URL)
    assert client.post("/api/sort").status_code == 202
    status = wait_for_sort(client)
    assert (status["running"], status["report"]) == (False, None)
    assert "unexpected error" in status["error"]
    assert "RuntimeError: a bug" in status["error"]  # the reason itself, not only where to look
    assert where(root) == {}
    assert client.post("/api/sort").status_code == 202
    wait_for_sort(client)


def test_real_backend_is_built_only_when_a_sort_or_suggest_starts(root, monkeypatch):
    """Without a factory the page opens with no API key. The key is asked for by the sort."""
    asked = []

    def make_backend(vault_root, config):
        """Stand in for backends.make_backend on a machine with no key."""
        asked.append((vault_root, config.model))
        raise BackendError("no TypeSafe API key found: set TYPESAFE_API_KEY")

    monkeypatch.setattr(server, "make_backend", make_backend)
    client = TestClient(create_app(root), base_url=BASE_URL)
    assert client.get("/").status_code == 200
    assert client.get("/api/board").status_code == 200
    assert client.get("/api/sort/status").status_code == 200
    assert asked == []

    assert client.post("/api/sort").status_code == 202
    status = wait_for_sort(client)
    assert (status["running"], status["report"]) == (False, None)
    assert "TYPESAFE_API_KEY" in status["error"]
    assert asked == [(root.resolve(), "jev-1.14.0")]
    assert where(root) == {}

    answer = client.post("/api/suggest")
    assert is_error(answer, 409)
    assert "TYPESAFE_API_KEY" in answer.json()["detail"]
    assert len(asked) == 2


def test_real_backend_is_closed_after_each_use(root, monkeypatch):
    """The backend the app builds itself is closed when the sort or the suggest is over."""

    class ClosableBackend(FakeBackend):
        """A fake that counts how often it is closed, as the Jev client must be."""

        closed = 0

        def close(self):
            """Count one close."""
            type(self).closed += 1

    monkeypatch.setattr(
        server, "make_backend", lambda vault_root, config: ClosableBackend(answers=dict(ANSWERS))
    )
    client = TestClient(create_app(root), base_url=BASE_URL)
    sort_now(client)
    assert ClosableBackend.closed == 1
    assert client.post("/api/suggest").status_code == 200
    assert ClosableBackend.closed == 2


# POST /api/suggest


def test_suggest_lists_frozen_files_the_model_would_now_place_elsewhere(client, backend, root):
    """Only frozen files are asked about. A differing answer is listed, and nothing moves."""
    sort_now(client)
    client.post("/api/freeze", json={"file_id": SOLUTION})
    client.post("/api/freeze", json={"file_id": SYLLABUS})
    backend.answers[SOLUTION] = ({"hw": 0.1, "logistics": 0.1, "project": 0.8}, 0.9)
    asked_before = len(backend.calls)

    answer = client.post("/api/suggest")
    assert answer.status_code == 200
    body = answer.json()
    assert sorted(body) == ["failed", "suggestions"]
    assert body["failed"] == {}
    (found,) = body["suggestions"]
    assert (found["file_id"], found["current_group"], found["suggested_group"]) == (
        SOLUTION,
        "hw",
        "project",
    )
    assert (found["probs"], found["fits_any"]) == (
        {"hw": 0.1, "logistics": 0.1, "project": 0.8},
        0.9,
    )
    assert (found["set_version"], found["model"]) == (3, "fake-0")
    assert [request.file_id for request in backend.calls[asked_before:]] == [SOLUTION, SYLLABUS]
    assert where(root)[SOLUTION] == "HW"
    assert [line["file_id"] for line in read_jsonl(decide_path(root, SUGGESTIONS_FILE))] == [
        SOLUTION
    ]


def test_suggest_says_which_files_could_not_be_asked_about(client, backend, root):
    """A failure is not agreement: the file is listed under `failed` with the reason."""
    sort_now(client)
    client.post("/api/freeze", json={"file_id": SOLUTION})
    backend.fail.add(SOLUTION)
    body = client.post("/api/suggest").json()
    assert body["suggestions"] == []
    assert list(body["failed"]) == [SOLUTION]
    assert "told to fail" in body["failed"][SOLUTION]


def test_suggest_with_nothing_frozen(client, backend):
    """No frozen files: nothing to ask, nothing to suggest."""
    sort_now(client)
    asked_before = len(backend.calls)
    assert client.post("/api/suggest").json() == {"suggestions": [], "failed": {}}
    assert len(backend.calls) == asked_before


# Errors and where requests may come from


def test_errors_come_back_as_one_plain_line(client):
    """Every failure is {"detail": "..."} with a 4xx status, so the page can always show it."""
    assert is_error(client.get("/api/files/nope.txt"), 404)
    assert is_error(client.get("/api/nope"), 404)
    assert is_error(client.put("/api/board"), 405)
    assert is_error(client.post("/api/groups", json={"name": "HW", "covers": "again"}), 409)

    missing = client.post("/api/move", json={"file_id": SYLLABUS})
    assert is_error(missing, 400)
    assert missing.json()["detail"] == "group_id: Field required"
    extra = client.post("/api/freeze", json={"file_id": SYLLABUS, "force": True})
    assert is_error(extra, 400)
    assert extra.json()["detail"] == "force: Extra inputs are not permitted"
    several = client.post("/api/groups", json={"covers": 3})
    assert is_error(several, 400)
    assert several.json()["detail"].count(";") == 1  # two problems on one line
    not_json = client.post(
        "/api/move", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert is_error(not_json, 400)
    as_text = client.post(
        "/api/move",
        content=b'{"file_id": "syllabus.md", "group_id": null}',
        headers={"content-type": "text/plain"},
    )
    assert is_error(as_text, 400)


def test_error_nobody_planned_for_comes_back_with_its_reason(root, backend, monkeypatch):
    """Any other error is one plain line too, with the error's type and message, as a 500."""

    def broken_board(vault_root):
        """Stand in for a bug in the code that builds the board."""
        raise RuntimeError("a bug\nin the board")

    monkeypatch.setattr(server, "build_board", broken_board)
    app = create_app(root, lambda: backend)
    # The error is raised again after the answer is sent, which a test client stops on.
    quiet = TestClient(app, base_url=BASE_URL, raise_server_exceptions=False)
    answer = quiet.get("/api/board")
    assert is_error(answer, 500)
    assert "RuntimeError: a bug in the board" in answer.json()["detail"]
    assert is_guarded(answer)
    # Raised again means that the server's terminal gets the traceback.
    with pytest.raises(RuntimeError, match="a bug"):
        TestClient(app, base_url=BASE_URL).get("/api/board")


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_group_note_that_is_not_valid_yaml_is_reported_with_a_reason(root, backend):
    """A slip of the hand in a group note never takes the page down with a bare status."""
    app = create_app(root, lambda: backend)
    client = TestClient(app, base_url=BASE_URL, raise_server_exceptions=False)
    sort_now(client)
    note = root / "_groups" / "HW.md"
    note.write_text("---\nid: hw\ncovers: homework: specs and solutions\n---\n", encoding="utf-8")
    answers = [
        client.get("/api/board"),
        client.get(f"/api/files/{SYLLABUS}"),
        client.post("/api/move", json={"file_id": SYLLABUS, "group_id": "project"}),
        client.post("/api/freeze", json={"file_id": SYLLABUS}),
        client.post("/api/groups", json={"name": "Exams", "covers": "exams"}),
    ]
    for answer in answers:
        # 400 when the vault itself names the broken note, 500 when the last handler answers.
        assert is_error(answer, 400) or is_error(answer, 500)
        assert "mapping values are not allowed" in answer.json()["detail"]
    assert client.post("/api/sort").status_code == 202
    status = wait_for_sort(client)
    assert status["report"] is None
    assert "mapping values are not allowed" in status["error"]


def test_requests_from_other_sites_and_other_hosts_are_refused(root, backend):
    """Only this app's own page on this machine may call the API or load an original."""
    app = create_app(root, lambda: backend)
    client = TestClient(app, base_url=BASE_URL)

    # A page on another site, open in the same browser, posts to this server.
    for origin in ("http://evil.example", "http://127.0.0.1:9999", "http://localhost:8765", "null"):
        answer = client.post("/api/sort", headers={"origin": origin})
        assert is_error(answer, 403)
    assert client.get("/api/sort/status").json()["running"] is False
    assert backend.calls == []

    # The page's own requests carry the server's address as their origin.
    own = client.post("/api/sort", headers={"origin": BASE_URL})
    assert own.status_code == 202
    wait_for_sort(client)

    # A site that points its own domain at 127.0.0.1 would otherwise pass as this app.
    elsewhere = TestClient(app, base_url="http://evil.example:8765")
    for path in ("/", "/api/board", f"/api/original/{SYLLABUS}", f"/api/files/{SYLLABUS}"):
        assert is_error(elsewhere.get(path), 403)
    # The page's own files and an address that nothing serves are behind the same check.
    for path in ("/static/index.html", "/static/app.js", "/api/nope"):
        assert is_error(elsewhere.get(path), 403)
        assert is_error(client.get(path, headers={"origin": "http://evil.example"}), 403)
    assert is_error(elsewhere.put("/api/dump/x.txt", content=b"x"), 403)
    assert not (root / "_dump" / "x.txt").exists()

    for base_url in ("http://localhost:8765", "http://127.0.0.1", "http://localhost"):
        assert TestClient(app, base_url=base_url).get("/api/board").status_code == 200


# p0 ui


def free_port():
    """A port on this machine that nothing is listening on right now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def served(monkeypatch):
    """Stand in for uvicorn's server and the browser. Returns what `p0 ui` did with them."""
    record = {"apps": [], "addresses": [], "opened": []}

    def run(self, sockets=None):
        """Note the app and the address it would be served on, and return at once."""
        record["apps"].append(self.config.app)
        record["addresses"].extend(listener.getsockname() for listener in sockets)

    monkeypatch.setattr("uvicorn.Server.run", run)
    monkeypatch.setattr("webbrowser.open", record["opened"].append)
    return record


def test_ui_command_serves_the_vault_and_opens_the_browser(root, served):
    """`p0 ui` prints the address, opens it, and serves the vault's app on 127.0.0.1 only."""
    port = free_port()
    result = CliRunner().invoke(cli, ["ui", "--vault", str(root), "--port", str(port)])
    assert result.exit_code == 0, result.output
    url = f"http://127.0.0.1:{port}"
    assert url in result.output
    assert served["opened"] == [url]
    assert served["addresses"] == [("127.0.0.1", port)]
    (app,) = served["apps"]
    board = TestClient(app, base_url=url).get("/api/board")
    assert board.json()["vault"]["path"] == str(root.resolve())


def test_ui_command_with_no_open_leaves_the_browser_alone(root, served):
    """--no-open serves the same way and opens nothing."""
    port = free_port()
    result = CliRunner().invoke(cli, ["ui", "--vault", str(root), "--port", str(port), "--no-open"])
    assert result.exit_code == 0, result.output
    assert f"http://127.0.0.1:{port}" in result.output
    assert served["opened"] == []
    assert served["addresses"] == [("127.0.0.1", port)]


def test_ui_command_fails_plainly(root, tmp_path_factory, served):
    """A port that is taken, or a folder that is no vault: one error line, and nothing served."""
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = str(taken.getsockname()[1])
        result = CliRunner().invoke(cli, ["ui", "--vault", str(root), "--port", port])
    assert result.exit_code == 1
    assert f"error: port {port} cannot be used" in result.output
    assert "--port" in result.output

    empty = tmp_path_factory.mktemp("not_a_vault")
    result = CliRunner().invoke(cli, ["ui", "--vault", str(empty)])
    assert result.exit_code == 1
    assert "error: no vault" in result.output
    assert served == {"apps": [], "addresses": [], "opened": []}


# The page itself: app.js, run under node with a stand-in for the browser

# Just enough of a browser for app.js to run: elements that keep their attributes, children
# and listeners, a clock the test controls, and a server that answers from a table. run_page
# puts FIXED_ELEMENTS (the elements of index.html that have an id) in front of this text,
# and the page's script and the steps of one test behind it.
PAGE_STAND_IN = r"""
const browser = (() => {
  // One element. Its methods are the browser's own, each doing the least the page needs.
  class FakeElement {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.attrs = new Map();
      this.nodes = []; // elements and strings of text, in order
      this.parentElement = null;
      this.listeners = {};
      this.style = {};
    }
    classes() {
      return (this.attrs.get("class") || "").split(" ");
    }
    setAttribute(name, value) {
      this.attrs.set(name, String(value));
    }
    removeAttribute(name) {
      this.attrs.delete(name);
    }
    get dataset() {
      const data = {};
      for (const [name, value] of this.attrs) {
        if (name.startsWith("data-")) data[name.slice(5)] = value;
      }
      return data;
    }
    get children() {
      return this.nodes.filter((node) => node instanceof FakeElement);
    }
    get firstElementChild() {
      return this.children[0] || null;
    }
    get textContent() {
      const text = (node) => (node instanceof FakeElement ? node.textContent : node);
      return this.nodes.map(text).join("");
    }
    set textContent(text) {
      this.replaceChildren(String(text));
    }
    append(...nodes) {
      for (const node of nodes) {
        if (node instanceof FakeElement) {
          node.remove();
          node.parentElement = this;
        }
        this.nodes.push(node instanceof FakeElement ? node : String(node));
      }
    }
    replaceChildren(...nodes) {
      for (const child of this.children) child.parentElement = null;
      this.nodes = [];
      this.append(...nodes);
    }
    remove() {
      if (!this.parentElement) return;
      this.parentElement.nodes = this.parentElement.nodes.filter((node) => node !== this);
      this.parentElement = null;
    }
    addEventListener(type, listener) {
      (this.listeners[type] ||= []).push(listener);
    }
    focus() {
      if (!this.disabled) document.activeElement = this;
    }
    setSelectionRange() {}
    scrollIntoView() {}
    // A click from code, as the page gives the hidden file picker. The test counts them.
    click() {
      this.clicks = (this.clicks || 0) + 1;
      for (const listener of this.listeners.click || []) listener({ target: this });
    }
    // Only the selectors the page uses: .class, [attribute] and [attribute="value"].
    matches(selector) {
      if (selector.startsWith(".")) return this.classes().includes(selector.slice(1));
      const [, name, value] = /^\[([\w-]+)(?:="(.*)")?\]$/s.exec(selector);
      return this.attrs.has(name) && (value === undefined || this.attrs.get(name) === value);
    }
    closest(selector) {
      for (let node = this; node; node = node.parentElement) {
        if (node.matches(selector)) return node;
      }
      return null;
    }
    querySelectorAll(selector) {
      const found = [];
      for (const child of this.children) {
        if (child.matches(selector)) found.push(child);
        found.push(...child.querySelectorAll(selector));
      }
      return found;
    }
    querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    }
  }
  // Two attributes that the browser also offers as properties of the element.
  for (const name of ["hidden", "disabled"]) {
    Object.defineProperty(FakeElement.prototype, name, {
      get() {
        return this.attrs.has(name);
      },
      set(on) {
        if (on) this.attrs.set(name, "");
        else this.attrs.delete(name);
      },
    });
  }

  const body = new FakeElement("body");
  const byId = new Map();
  for (const [tag, id, key] of FIXED_ELEMENTS) {
    const node = new FakeElement(tag);
    if (key) node.setAttribute("data-key", key);
    byId.set(id, node);
    body.append(node);
  }
  const onWindow = {};
  globalThis.Element = FakeElement;
  globalThis.CSS = { escape: (text) => text };
  globalThis.document = {
    title: "",
    activeElement: body,
    getElementById: (id) => byId.get(id) || null,
    createElement: (tag) => new FakeElement(tag),
    querySelector: (selector) => body.querySelector(selector),
    querySelectorAll: (selector) => body.querySelectorAll(selector),
    addEventListener() {}, // presses and drags, which no test here goes through
  };
  globalThis.window = {
    addEventListener: (type, listener) => (onWindow[type] ||= []).push(listener),
  };

  // The clock. A message that would go away after seconds stays. The wait before the page's
  // next look at a running sort ends when the test says so (see look). Any shorter wait is
  // over at once.
  const realTimeout = setTimeout;
  let waits = 0; // short waits that are not over yet
  let nextLook = null;
  globalThis.setTimeout = (work, wait = 0) => {
    if (wait >= 1000) return;
    if (wait >= 100) {
      nextLook = work;
      return;
    }
    waits += 1;
    realTimeout(() => {
      waits -= 1;
      work();
    }, 0);
  };

  // The server. `routes` maps "METHOD /path" to a function of the request's JSON body that
  // gives the answer as {status, body}. A request that is held waits until it is let go.
  const calls = [];
  const unanswered = [];
  const routes = {};
  const held = {};
  globalThis.fetch = async (path, options = {}) => {
    const name = `${options.method || "GET"} ${path}`;
    const sent = typeof options.body === "string" ? JSON.parse(options.body) : null;
    calls.push([name, sent]);
    if (held[name]) await held[name];
    if (!routes[name]) {
      unanswered.push(name);
      throw new Error(`the test has no answer for ${name}`);
    }
    const answer = routes[name](sent);
    return { ok: answer.status < 400, status: answer.status, json: async () => answer.body };
  };

  const happening = (target) => ({ target, preventDefault() {}, stopPropagation() {} });
  const find = (key) => body.querySelector(`[data-key="${key}"]`);
  return {
    calls,
    routes,
    find,
    // How often the page has sent this request.
    count: (name) => calls.filter(([each]) => each === name).length,
    // Hold back the answer to one request. Gives the function that lets it go.
    hold(name) {
      let letGo;
      held[name] = new Promise((resolve) => {
        letGo = resolve;
      });
      return () => {
        delete held[name];
        letGo();
      };
    },
    // Let the page finish what it has started. It is done when all that is left to wait for
    // is a held answer or the next look at a running sort.
    async settle() {
      for (let quiet = 0; quiet < 2; quiet = waits === 0 ? quiet + 1 : 0) {
        await new Promise((resolve) => realTimeout(resolve, 0));
      }
    },
    // End the wait before the page's next look at a running sort.
    look() {
      const work = nextLook;
      nextLook = null;
      if (work) work();
    },
    // Press the element with this data-key, as a click would. False when it cannot be
    // pressed. A press on a form's save button reaches the page as the form's submit.
    press(key) {
      const node = find(key);
      if (!node || node.disabled) return false;
      const form = node.attrs.get("type") === "submit" ? node.closest("[novalidate]") : null;
      const [target, type] = form ? [form, "submit"] : [node, "click"];
      for (const listener of target.listeners[type] || []) listener(happening(target));
      return true;
    },
    // Type a text into the field with this data-key.
    type(key, text) {
      const node = find(key);
      node.value = text;
      for (const listener of node.listeners.input) listener(happening(node));
    },
    // The window gets the focus back, as when the user returns from another app.
    comeBack() {
      for (const listener of onWindow.focus) listener(happening(null));
    },
    text: (id) => byId.get(id).textContent,
    toasts: () => byId.get("toasts").children.map((node) => node.textContent),
    // The lists of problems above the board, each as {title, rows}.
    problems: () =>
      body.querySelectorAll(".problems").map((list) => ({
        title: list.querySelector(".problems-head").firstElementChild.textContent,
        rows: list.querySelector(".problems-list").children.map((row) => row.textContent),
      })),
    // Hand the test what it wants to check, and stop.
    done(result) {
      if (unanswered.length > 0) throw new Error(`no answer was set up for: ${unanswered}`);
      console.log(JSON.stringify(result));
      process.exit(0);
    },
  };
})();

// What the server holds when a test starts: two groups and two sorted files, one of them in
// Unsorted. The steps of a test change `server` and `browser.routes` to stage their case.
const boardFile = (fileId, groupId) => ({
  file_id: fileId,
  name: fileId.split("/").pop(),
  file_type: "Text",
  group_id: groupId,
  frozen: false,
  close_call: false,
  placed_by: "jev",
  reason: null,
  top_prob: 0.9,
  probs: { hw: 0.9, notes: 0.1 },
  fits_any: 0.9,
  stale: false,
});
const sortReport = (errors) => ({
  set_version: 2,
  asked: 2,
  placed: 1,
  unsorted: 1,
  close_calls: 0,
  frozen_skipped: 0,
  errors,
  spent_usd: null,
});
const server = {
  board: {
    set_version: 2,
    groups: [
      {
        group: { id: "hw", name: "HW", covers: "homework", excludes: null },
        files: [boardFile("hw3/solution.py", "hw")],
      },
      { group: { id: "notes", name: "Notes", covers: "lecture notes", excludes: null }, files: [] },
    ],
    unsorted: [boardFile("scan.txt", null)],
    new_files: [],
    sorting: false,
    vault: {
      name: "vault",
      path: "/vault",
      dump_dir: "_dump",
      model: "fake-0",
      spent_usd: 0,
      max_spend_usd: 5,
      cutoffs: { fits_any: 0.5, top: 0.6, close_call: 0.15 },
    },
  },
  status: { running: false, done: 0, total: 0, report: null, error: null },
  detail: {
    file_id: "hw3/solution.py",
    group_id: "hw",
    placement: { reason: null, fits_any: null, model: null, set_version: 2 },
    probs: [],
    card: "",
  },
};
const ok = (body) => ({ status: 200, body });
const refused = (status, detail) => ({ status, body: { detail } });
const running = (done) => ({ running: true, done, total: 2, report: null, error: null });
const over = (report, error = null) => ({ running: false, done: 2, total: 2, report, error });
browser.routes["GET /api/board"] = () => ok(server.board);
browser.routes["GET /api/sort/status"] = () => ok(server.status);
browser.routes["GET /api/files/hw3/solution.py"] = () => ok(server.detail);
"""

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def run_page(steps):
    """Run the page's script under node, then the steps of one test, and return their result.

    The steps are JavaScript. They start when the page has loaded its first board, drive
    the page through `browser` (see PAGE_STAND_IN) and end with `browser.done(result)`.
    Nothing here touches a vault or the network: the server is a table inside the script.
    """
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    fixed = []
    for tag in re.findall(r'<\w+\s[^>]*\bid="[^"]+"[^>]*>', page):
        key = re.search(r'\bdata-key="([^"]+)"', tag)
        fixed.append([tag[1:].split()[0], re.search(r'\bid="([^"]+)"', tag)[1], key and key[1]])
    script = "\n".join(
        [
            '"use strict";',
            f"const FIXED_ELEMENTS = {json.dumps(fixed)};",
            PAGE_STAND_IN,
            (STATIC_DIR / "app.js").read_text(encoding="utf-8"),
            "(async () => {",
            "  await browser.settle();",
            steps,
            "})().catch((error) => {",
            "  console.error(error);",
            "  process.exit(1);",
            "});",
        ]
    )
    result = subprocess.run(
        ["node", "-"], input=script, capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_node
def test_page_says_whether_a_freeze_names_a_file_or_a_group():
    """A lane's button sends the group's id (null for Unsorted), a file's button the file's id."""
    result = run_page(
        """
        browser.routes["POST /api/freeze"] = () => ok({ frozen: ["one"] });
        browser.press("freeze:hw");
        await browser.settle();
        browser.press("freeze:Unsorted");
        await browser.settle();
        browser.press("card:hw3/solution.py");
        await browser.settle();
        browser.press("panel:freeze");
        await browser.settle();
        browser.done(browser.calls.filter(([name]) => name.startsWith("POST")));
        """
    )
    assert result == [
        ["POST /api/freeze", {"group_id": "hw"}],
        ["POST /api/freeze", {"group_id": None}],
        ["POST /api/freeze", {"file_id": "hw3/solution.py"}],
    ]


@needs_node
def test_page_ignores_a_second_press_on_sort_while_the_first_is_not_answered():
    """Two quick presses start one sort, and no false error is shown while it runs."""
    result = run_page(
        """
        browser.routes["POST /api/sort"] = () => {
          if (server.status.running) return refused(409, "a sort is already running");
          server.status = running(0);
          return { status: 202, body: server.status };
        };
        const letGo = browser.hold("POST /api/sort");
        const pressed = [browser.press("sort"), browser.press("sort")];
        await browser.settle();
        const notAnswered = browser.count("POST /api/sort");
        letGo();
        await browser.settle();
        const during = [browser.text("sort-label"), browser.press("sort")];
        server.status = over(sortReport({}));
        browser.look();
        await browser.settle();
        browser.done({
          pressed,
          notAnswered,
          during,
          sent: browser.count("POST /api/sort"),
          label: browser.text("sort-label"),
          toasts: browser.toasts(),
        });
        """
    )
    assert result == {
        "pressed": [True, True],  # the button still looked pressable both times
        "notAnswered": 1,
        "during": ["Sorting 0 of 2", False],
        "sent": 1,
        "label": "Sort",
        "toasts": ["Sorted: 1 placed, 1 in Unsorted."],
    }


@needs_node
def test_page_ignores_a_second_press_on_save_and_on_delete():
    """Two quick presses in the group form add one group, and delete one group once."""
    result = run_page(
        """
        const exams = { id: "exams", name: "Exams", covers: "midterms", excludes: null };
        browser.routes["POST /api/groups"] = () => ({ status: 201, body: { group: exams } });
        browser.routes["DELETE /api/groups/hw"] = () => ok({ set_version: 4 });
        browser.press("add-group");
        browser.type("form:name", "Exams");
        browser.type("form:covers", "midterms");
        let letGo = browser.hold("POST /api/groups");
        browser.press("form:save");
        browser.press("form:save");
        await browser.settle();
        letGo();
        await browser.settle();
        const saved = [browser.count("POST /api/groups"), Boolean(browser.find("form:save"))];

        browser.press("edit:hw");
        browser.press("form:delete");
        letGo = browser.hold("DELETE /api/groups/hw");
        browser.press("form:confirm");
        browser.press("form:confirm");
        await browser.settle();
        letGo();
        await browser.settle();
        browser.done({
          saved,
          deleted: browser.count("DELETE /api/groups/hw"),
          toasts: browser.toasts(),
        });
        """
    )
    assert result == {
        "saved": [1, False],  # one request, and the form is closed
        "deleted": 1,
        "toasts": [
            "Added Exams. Sort to place files in it.",
            "Deleted HW. Any files it held are in Unsorted now.",
        ],
    }


@needs_node
def test_page_follows_a_sort_it_did_not_start_once_the_server_refuses():
    """Sort or save is refused because a sort runs (another tab started it): the page shows it."""
    result = run_page(
        """
        const busy = "a sort is running; try again when it is done";
        browser.routes["POST /api/sort"] = () => refused(409, "a sort is already running");
        browser.routes["PATCH /api/groups/notes"] = () => refused(409, busy);
        const look = () => [browser.text("sort-label"), state.watching, browser.press("sort")];
        const elsewhere = (done) => {
          server.board.sorting = true;
          server.status = running(done);
        };
        const ends = async () => {
          server.board.sorting = false;
          server.status = over(sortReport({}));
          browser.look();
          await browser.settle();
        };

        elsewhere(1);
        const before = look();
        await browser.settle();
        const afterSort = look();
        await ends();

        browser.press("edit:notes");
        browser.type("form:covers", "slides and notes");
        elsewhere(0);
        browser.press("form:save");
        await browser.settle();
        const afterSave = [...look(), state.form.covers];
        await ends();
        browser.done({ before, afterSort, afterSave, toasts: browser.toasts() });
        """
    )
    assert result == {
        "before": ["Sort", False, True],  # that press is the one the server refuses
        "afterSort": ["Sorting 1 of 2", True, False],
        "afterSave": ["Sorting 0 of 2", True, False, "slides and notes"],  # what was typed stays
        "toasts": [
            "Sort not started: a sort is already running",
            "Sorted: 1 placed, 1 in Unsorted.",
            "Not saved: a sort is running; try again when it is done",
            "Sorted: 1 placed, 1 in Unsorted.",
        ],
    }


@needs_node
def test_page_keeps_the_list_of_what_a_sort_could_not_do():
    """Every failed file stays listed with its reason until it is dismissed or the next sort."""
    result = run_page(
        """
        let outcome;
        browser.routes["POST /api/sort"] = () => {
          server.status = outcome;
          return { status: 202, body: running(0) };
        };
        const sort = async (how) => {
          outcome = how;
          browser.press("sort");
          await browser.settle();
          return browser.problems();
        };
        const failed = { "a.txt": "HTTP 429 from the model", "b/c.pdf": "the model timed out" };
        const afterSort = await sort(over(sortReport(failed)));
        const toasts = browser.toasts();
        browser.comeBack();
        await browser.settle();
        const afterRefresh = browser.problems();
        browser.press("dismiss:sort");
        const afterDismiss = browser.problems();

        // What stops a sort often stops the board too: a group note that cannot be read.
        const broken = "the frontmatter of HW.md is not valid YAML";
        browser.routes["GET /api/board"] = () => refused(400, broken);
        const afterStop = await sort(over(null, broken));
        const stopToasts = browser.toasts().slice(toasts.length);
        browser.routes["GET /api/board"] = () => ok(server.board);
        const afterCleanSort = await sort(over(sortReport({})));
        browser.done({
          afterSort,
          toasts,
          afterRefresh,
          afterDismiss,
          afterStop,
          stopToasts,
          afterCleanSort,
        });
        """
    )
    listed = [
        {
            "title": "2 files were not sorted",
            "rows": ["a.txt: HTTP 429 from the model", "b/c.pdf: the model timed out"],
        }
    ]
    broken = "the frontmatter of HW.md is not valid YAML"
    assert result == {
        "afterSort": listed,
        "toasts": [
            "Sorted: 1 placed, 1 in Unsorted.",
            "2 files were not sorted. The list is above the board.",
        ],
        "afterRefresh": listed,
        "afterDismiss": [],
        "afterStop": [{"title": "The sort stopped", "rows": [broken]}],
        "stopToasts": [f"Board not refreshed: {broken}", f"The sort stopped: {broken}"],
        "afterCleanSort": [],
    }


@needs_node
def test_page_holds_sort_back_while_files_arrive_and_lists_the_ones_not_added():
    """While dropped files are on their way Sort cannot start. Each refused file stays listed."""
    result = run_page(
        """
        const dropped = (name) => Object.assign(new Blob(["x"]), { name });
        const taken = "scan.txt is already in the dump folder";
        browser.routes["PUT /api/dump/new.txt"] = () => ({ status: 201, body: {} });
        browser.routes["PUT /api/dump/.hidden"] = () => refused(400, "'.hidden' starts with a dot");
        browser.routes["PUT /api/dump/scan.txt"] = () => refused(409, taken);
        const letGo = browser.hold("PUT /api/dump/new.txt");
        addFiles({ files: ["new.txt", ".hidden", "scan.txt"].map(dropped), folders: 0 });
        await browser.settle();
        const sort = browser.find("sort");
        const during = [sort.disabled, sort.title, browser.text("pending")];
        startSort(); // a press that got through anyway, before the button was redrawn
        letGo();
        await browser.settle();
        browser.done({
          during,
          after: [sort.disabled, document.getElementById("pending").hidden],
          sorts: browser.count("POST /api/sort"),
          problems: browser.problems(),
          toasts: browser.toasts(),
        });
        """
    )
    assert result == {
        "during": [True, "Files are still being added.", "Adding 3 files"],
        "after": [False, True],
        "sorts": 0,
        "problems": [
            {
                "title": "2 files were not added",
                "rows": [
                    ".hidden: '.hidden' starts with a dot",
                    "scan.txt: scan.txt is already in the dump folder",
                ],
            }
        ],
        "toasts": [
            "Added 1 file to the dump folder. Press Sort to place it.",
            "2 files were not added. The list is above the board.",
        ],
    }


@needs_node
def test_page_adds_the_files_picked_with_the_add_files_button():
    """Add files opens the picker; the files picked go to the dump like dropped ones."""
    result = run_page(
        """
        const picked = (name) => Object.assign(new Blob(["x"]), { name });
        browser.routes["PUT /api/dump/a.txt"] = () => ({ status: 201, body: {} });
        browser.routes["PUT /api/dump/b.pdf"] = () => ({ status: 201, body: {} });
        const picker = document.getElementById("file-picker");
        const enabled = !browser.find("add-files").disabled;
        browser.press("add-files");
        picker.files = ["a.txt", "b.pdf"].map(picked);
        picker.value = "C:\\fakepath\\a.txt";
        for (const listener of picker.listeners.change) listener({ target: picker });
        await browser.settle();
        browser.done({
          enabled,
          opened: picker.clicks,
          sent: browser.calls.filter(([name]) => name.startsWith("PUT")).map(([name]) => name),
          cleared: picker.value,
          toasts: browser.toasts(),
        });
        """
    )
    assert result == {
        "enabled": True,
        "opened": 1,
        "sent": ["PUT /api/dump/a.txt", "PUT /api/dump/b.pdf"],
        "cleared": "",
        "toasts": ["Added 2 files to the dump folder. Press Sort to place them."],
    }


@needs_node
def test_page_shows_the_dump_folder_as_a_tree_and_deletes_after_a_yes():
    """Files lists every file by folder with where it is sorted; Delete asks before it acts."""
    result = run_page(
        """
        server.board.new_files = ["notes/week 1/new.md"];
        server.dump = {
          dump_dir: "_dump",
          files: [
            { file_id: "hw3/solution.py", size: 2048, modified: 1 },
            { file_id: "notes/week 1/new.md", size: 10, modified: 1 },
            { file_id: "scan.txt", size: 5, modified: 1 },
          ],
        };
        browser.routes["GET /api/dump"] = () => ok(server.dump);
        browser.routes["DELETE /api/dump/hw3/solution.py"] = () => {
          server.dump.files = server.dump.files.filter((f) => f.file_id !== "hw3/solution.py");
          server.board.groups[0].files = [];
          return ok({ deleted: "hw3/solution.py", card_notes_removed: 1 });
        };
        browser.press("view:files");
        await browser.settle();
        const rows = () =>
          document.querySelectorAll(".tree-row").map((row) =>
            [".folder-name", ".file-name", ".file-where"]
              .flatMap((name) => row.querySelectorAll(name))
              .map((n) => n.textContent),
          );
        const shown = rows();
        browser.press("folder:notes/");
        const folded = rows();
        browser.press("remove:hw3/solution.py");
        const asked = [browser.count("DELETE /api/dump/hw3/solution.py"),
          document.querySelector(".confirm-text").textContent];
        browser.press("remove-yes:hw3/solution.py");
        await browser.settle();
        browser.done({
          shown,
          folded,
          asked,
          deletes: browser.count("DELETE /api/dump/hw3/solution.py"),
          after: rows(),
          toasts: browser.toasts(),
          pressed: document.getElementById("view-files").attrs.get("aria-pressed"),
        });
        """
    )
    assert result == {
        "shown": [
            ["hw3/"],
            ["solution.py", "HW"],
            ["notes/"],
            ["week 1/"],
            ["new.md", "Not sorted yet"],
            ["scan.txt", "Unsorted"],
        ],
        "folded": [["hw3/"], ["solution.py", "HW"], ["notes/"], ["scan.txt", "Unsorted"]],
        "asked": [0, "Delete for good? This cannot be undone."],
        "deletes": 1,
        "after": [["notes/"], ["scan.txt", "Unsorted"]],
        "toasts": ["Deleted hw3/solution.py."],
        "pressed": "true",
    }


@needs_node
def test_page_says_why_the_open_file_was_closed():
    """The file in the side panel can no longer be read: the panel closes with a message."""
    result = run_page(
        """
        const gone = "no file with the id 'hw3/solution.py' in the dump folder";
        browser.press("card:hw3/solution.py");
        await browser.settle();
        const panel = document.getElementById("panel");
        const open = !panel.hidden;
        // The original leaves the dump folder. Its card stays on the board.
        browser.routes["GET /api/files/hw3/solution.py"] = () => refused(404, gone);
        browser.comeBack();
        await browser.settle();
        browser.done({ open, closed: panel.hidden, toasts: browser.toasts() });
        """
    )
    assert result == {
        "open": True,
        "closed": True,
        "toasts": [
            (
                "The panel of hw3/solution.py was closed: "
                "no file with the id 'hw3/solution.py' in the dump folder"
            )
        ],
    }
