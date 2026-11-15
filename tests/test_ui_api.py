"""The web UI's API, called through FastAPI's test client. Sorting gets a fake backend.

Every client here says it is talking to 127.0.0.1: the server refuses any other host name.
No test can reach the real model. `make_backend` is replaced for the whole file, and the
tests that are about building the real backend put their own stand-in there.
"""

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
from decide_already.config import JEV_LOG_FILE, LABELS_FILE, SUGGESTIONS_FILE, decide_path
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
    """A new file has no card note to move: a plain error, and nothing is written."""
    before = tree(root)
    answer = client.post("/api/move", json={"file_id": SYLLABUS, "group_id": "logistics"})
    assert is_error(answer, 400)
    assert tree(root) == before


def test_freeze_and_unfreeze_a_file_or_a_group(client, root):
    """`ref` is a file id, a group id, or Unsorted. The answer lists the files it changed."""
    sort_now(client)
    assert client.post("/api/freeze", json={"ref": SOLUTION}).json() == {"frozen": [SOLUTION]}
    assert client.post("/api/freeze", json={"ref": SOLUTION}).json() == {"frozen": []}
    assert client.post("/api/freeze", json={"ref": "project"}).json() == {
        "frozen": [EXAMPLES, SHEET]
    }
    assert client.post("/api/freeze", json={"ref": "Unsorted"}).json() == {
        "frozen": [CHECKLIST, SCAN]
    }
    board = client.get("/api/board").json()
    frozen = {
        file["file_id"]: file["frozen"] for entry in board["groups"] for file in entry["files"]
    }
    assert frozen == {SOLUTION: True, SYLLABUS: False, EXAMPLES: True, SHEET: True}

    assert client.post("/api/unfreeze", json={"ref": "project"}).json() == {
        "unfrozen": [EXAMPLES, SHEET]
    }
    assert client.post("/api/unfreeze", json={"ref": SOLUTION}).json() == {"unfrozen": [SOLUTION]}
    assert client.post("/api/unfreeze", json={"ref": SOLUTION}).json() == {"unfrozen": []}
    assert [kind for kind, *_ in labels(root)] == ["confirmed"] * 5 + ["unfrozen"] * 3


def test_freeze_checks_what_it_is_asked_to_freeze(client, root):
    """A ref that is no group id and no dump file is not found. Group names are not taken."""
    sort_now(client)
    before = tree(root)
    for route in ("/api/freeze", "/api/unfreeze"):
        assert is_error(client.post(route, json={"ref": "nope"}), 404)
        assert is_error(client.post(route, json={"ref": "Project"}), 404)
        assert is_error(client.post(route, json={"ref": "../.decide/config.yaml"}), 404)
        assert is_error(client.post(route, json={}), 400)
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
    client.post("/api/freeze", json={"ref": SOLUTION})
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
        client.post("/api/freeze", json={"ref": SOLUTION}),
        client.post("/api/unfreeze", json={"ref": "hw"}),
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
    client.post("/api/freeze", json={"ref": SOLUTION})
    client.post("/api/freeze", json={"ref": SYLLABUS})
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
    client.post("/api/freeze", json={"ref": SOLUTION})
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
    extra = client.post("/api/freeze", json={"ref": SYLLABUS, "force": True})
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
