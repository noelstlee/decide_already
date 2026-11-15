"""The local web UI: a small FastAPI app over the functions the p0 commands call.

Each route checks the ids in the request against the vault, calls vault.py, sort.py, board.py
or corrections.py, and answers with JSON. No sorting or correction logic lives here. The page
is plain HTML, CSS and JS in static/. It is served from this machine only and loads nothing
from anywhere else.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import corrections, vault
from ..backends import make_backend
from ..backends.base import Backend, BackendError
from ..board import build_board
from ..config import JEV_LOG_FILE, UNSORTED_DIR, Config, decide_path
from ..extract.base import BINARY_PROBE_BYTES
from ..log import total_input_tokens
from ..sort import sort_vault
from ..types import Group, GroupSet, StrictModel

STATIC_DIR = Path(__file__).parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
# The page may load its own files and nothing else: no CDN, no web fonts, no trackers.
PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "Cache-Control": "no-cache",
}
# Originals the browser may show as they are. Any other file is sent as plain text or as a
# download (see _original_response).
SHOWN_AS_IS = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class MoveRequest(StrictModel):
    """Body of POST /api/move."""

    file_id: str
    group_id: str | None  # None means Unsorted; the field has to be sent either way


class RefRequest(StrictModel):
    """Body of POST /api/freeze and /api/unfreeze: a file id, a group id, or "Unsorted"."""

    ref: str


class NewGroup(StrictModel):
    """Body of POST /api/groups."""

    name: str
    covers: str
    excludes: str | None = None


class GroupChanges(StrictModel):
    """Body of PATCH /api/groups/{group_id}. A field that is left out is not touched."""

    name: str | None = None
    covers: str | None = None
    excludes: str | None = None  # an empty string removes the excludes text


class SortJob:
    """The one sort that may run at a time, and what the page needs to follow it."""

    def __init__(self) -> None:
        """Start with no sort running and nothing to report."""
        self._lock = threading.Lock()  # the sort's thread writes while request threads read
        self._status = {"running": False, "done": 0, "total": 0, "report": None, "error": None}

    @property
    def running(self) -> bool:
        """True while a sort is under way."""
        with self._lock:
            return self._status["running"]

    def claim(self) -> bool:
        """Mark a sort as started and clear the last outcome. False when one is running."""
        with self._lock:
            if self._status["running"]:
                return False
            self._status = {"running": True, "done": 0, "total": 0, "report": None, "error": None}
            return True

    def progress(self, done: int, total: int) -> None:
        """Note how far the sort has come. sort_vault calls this after each file."""
        with self._lock:
            self._status.update(done=done, total=total)

    def finish(self, report: dict | None = None, error: str | None = None) -> None:
        """Mark the sort as over, with its report or with the reason it failed."""
        with self._lock:
            self._status.update(running=False, report=report, error=error)

    def status(self) -> dict:
        """The state right now: running, done, total, report and error."""
        with self._lock:
            return dict(self._status)


class SpendCounter:
    """What the vault has spent so far, worked out from its Jev log.

    The board asks after every action, and a log can be many megabytes, so the log is read
    again only when its size or its time of change differs from the last read.
    """

    def __init__(self, log_path: Path) -> None:
        """Remember where the log is. Nothing is read yet."""
        self._path = log_path
        self._seen: tuple[int, int] | None = None  # the log's size and time at the last read
        self._tokens = 0

    def usd(self, config: Config, reread: bool = True) -> float:
        """The spend in dollars: the log's input tokens at the config's price. 0 without a log.

        With `reread` false the log is left alone and the last total is given. That is for
        the time a sort runs, when a line may be half written.
        """
        if reread:
            stat = self._path.stat() if self._path.exists() else None
            seen = (stat.st_size, stat.st_mtime_ns) if stat else None
            if seen != self._seen:
                self._tokens = total_input_tokens(self._path)
                self._seen = seen
        return self._tokens / 1e6 * config.usd_per_million_tokens


class FreshStaticFiles(StaticFiles):
    """The page's files, sent so that the browser asks for changes on every load.

    Without this a browser keeps using a script it has stored for a while after the file on
    disk has changed.
    """

    def file_response(self, *args, **kwargs) -> Response:
        """Answer as StaticFiles does, and tell the browser to check back each time."""
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _local_only(request: Request) -> None:
    """Refuse a request that does not come from this app's own page on this machine.

    The server holds the user's files and can start a paid sort, and any website open in the
    same browser can send requests to 127.0.0.1. Two checks keep those out. The Host header
    must name this machine, which stops a site that points its own domain at 127.0.0.1. And a
    request that says which page sent it (the Origin header) must have been sent by this page.
    """
    host = request.headers.get("host", "")
    if host.rsplit(":", 1)[0] not in LOCAL_HOSTS:
        raise HTTPException(403, "this server only answers as 127.0.0.1 or localhost")
    origin = request.headers.get("origin")
    if origin is not None and origin.split("://", 1)[-1] != host:
        raise HTTPException(403, "requests from other sites are refused")


def _as_detail(status_code: int) -> Callable[[Request, Exception], JSONResponse]:
    """Make a handler that answers an error of one kind with its text as {"detail": ...}."""

    def handler(request: Request, error: Exception) -> JSONResponse:
        """Answer with the error's own message, which the page shows."""
        return JSONResponse({"detail": str(error)}, status_code=status_code)

    return handler


def _invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    """Answer a malformed request with one plain line in place of FastAPI's list of errors."""
    problems = []
    for problem in error.errors():
        field = ".".join(str(step) for step in problem["loc"] if step != "body")
        problems.append(f"{field}: {problem['msg']}" if field else problem["msg"])
    return JSONResponse({"detail": "; ".join(problems)}, status_code=400)


def _check_upload_name(name: str) -> None:
    """Accept only a plain file name for an upload: no path, no hidden file, not empty."""
    if not name.strip():
        raise HTTPException(400, "the file needs a name")
    if "/" in name or "\\" in name:
        raise HTTPException(400, f"{name!r} is a path; send a plain file name")
    if name.startswith("."):
        raise HTTPException(400, f"{name!r} starts with a dot; hidden files are never sorted")


def _original_response(path: Path) -> FileResponse:
    """Send an original so that the browser shows it and never runs it.

    A PDF or a picture opens in the tab. Any other file that reads as text is sent as plain
    text, so a web page or a script from the dump is shown as its source: opened as a page it
    would run inside this app, where it could call the API. The rest (slides, archives)
    download.
    """
    media_type = SHOWN_AS_IS.get(path.suffix.lower())
    if media_type is None:
        with path.open("rb") as handle:
            binary = b"\x00" in handle.read(BINARY_PROBE_BYTES)
        media_type = "application/octet-stream" if binary else "text/plain"
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        content_disposition_type="attachment" if media_type.endswith("octet-stream") else "inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )


def _group_answer(group_set: GroupSet, group_id: str) -> dict:
    """What a group route answers: the set version it left behind, and the group as it is now."""
    group = next(group for group in group_set.groups if group.id == group_id)
    return {"set_version": group_set.version, "group": group.model_dump()}


def create_app(root: Path, backend_factory: Callable[[], Backend] | None = None) -> FastAPI:
    """Build the web app for one vault.

    `backend_factory` gives the backend for a sort or a suggest; tests pass one that returns
    a fake. Without it the real backend is built, and only when a sort or a suggest starts,
    so the page opens without an API key.
    """
    root = root.resolve()
    if not Config.path(root).exists():
        raise FileNotFoundError(f"no vault at {root}; run `p0 init` first")
    lock = threading.Lock()  # one request at a time reads or changes the vault
    job = SortJob()
    spend = SpendCounter(decide_path(root, JEV_LOG_FILE))
    # FastAPI's own docs pages load their scripts from a CDN, so they are switched off.
    app = FastAPI(
        title="decide_already",
        dependencies=[Depends(_local_only)],
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_exception_handler(RequestValidationError, _invalid_request)
    app.add_exception_handler(ValueError, _as_detail(400))
    app.add_exception_handler(FileNotFoundError, _as_detail(404))
    app.add_exception_handler(FileExistsError, _as_detail(409))
    app.add_exception_handler(BackendError, _as_detail(409))
    app.add_exception_handler(OSError, _as_detail(500))

    @contextmanager
    def changing() -> Iterator[None]:
        """Hold the vault for one change. Refused with 409 while a sort runs.

        The sort read the card notes when it started, and would write over a change made
        after that.
        """
        with lock:
            if job.running:
                raise HTTPException(409, "a sort is running; try again when it is done")
            yield

    @contextmanager
    def backend() -> Iterator[Backend]:
        """The backend for one sort or suggest. The real one is built here and closed after."""
        if backend_factory is not None:
            yield backend_factory()
            return
        built = make_backend(root, Config.load(root))
        try:
            yield built
        finally:
            if hasattr(built, "close"):
                built.close()

    def dump_path(file_id: str) -> Path:
        """The original a file id names. 404 for anything that is not a file in the dump folder.

        The id is looked up in the dump listing and never joined to a path, so no request
        can reach a file outside the vault.
        """
        paths = dict(vault.dump_files(root, Config.load(root)))
        if file_id not in paths:
            raise HTTPException(404, f"no file with the id {file_id!r} in the dump folder")
        return paths[file_id]

    def current_group(group_id: str) -> Group:
        """The group with this id in the current set. 404 when there is none."""
        for group in vault.sync_group_set(root).groups:
            if group.id == group_id:
                return group
        raise HTTPException(404, f"no group with the id {group_id!r}")

    def check_ref(ref: str) -> None:
        """Check what a freeze or unfreeze names: a current group, Unsorted, or a dump file.

        As in corrections, a reference that names a group wins over a file of the same name.
        """
        groups = vault.sync_group_set(root).groups
        if ref != UNSORTED_DIR and all(group.id != ref for group in groups):
            dump_path(ref)

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        """The page itself."""
        return FileResponse(STATIC_DIR / "index.html", headers=PAGE_HEADERS)

    @app.get("/api/board")
    def board() -> dict:
        """The board, plus what the header shows about the vault.

        Edits made by hand are picked up first (corrections.sync). While a sort runs that
        step is left out: the sort did it when it started, and the two must not write card
        notes at the same time. The spend is then the one from before the sort.
        """
        with lock:
            sorting = job.running
            if not sorting:
                corrections.sync(root)
            config = Config.load(root)
            return {
                **build_board(root).model_dump(mode="json"),
                "sorting": sorting,
                "vault": {
                    "name": root.name,
                    "path": str(root),
                    "dump_dir": config.dump_dir,
                    "model": config.model,
                    "spent_usd": spend.usd(config, reread=not sorting),
                    "max_spend_usd": config.max_spend_usd,
                    "cutoffs": {
                        "fits_any": config.unsorted_fits_any_below,
                        "top": config.unsorted_top_below,
                        "close_call": config.close_call_margin,
                    },
                },
            }

    @app.get("/api/files/{file_id:path}")
    def file_detail(file_id: str) -> dict:
        """One sorted file: its placement, the model's probabilities by group, its card text.

        `group_id` is the group of the folder the card note sits in, as on the board. The
        probabilities come highest first, each with the group's current name (null for a
        group that no longer exists).
        """
        with lock:
            dump_path(file_id)
            notes = [
                note for note in vault.read_card_notes(root) if note.placement.file_id == file_id
            ]
            if not notes:
                raise HTTPException(404, f"{file_id} has not been sorted yet")
            note = notes[-1]  # a file with several notes: the last one counts, as in the sorter
            groups = vault.sync_group_set(root).groups
            names = {group.id: group.name for group in groups}
            probs = sorted(
                (note.placement.probs or {}).items(), key=lambda item: (-item[1], item[0])
            )
            return {
                "file_id": file_id,
                "group_id": next((group.id for group in groups if group.name == note.folder), None),
                "placement": note.placement.model_dump(mode="json"),
                "probs": [
                    {"group_id": group_id, "name": names.get(group_id), "prob": prob}
                    for group_id, prob in probs
                ],
                "card": note.body,
            }

    @app.get("/api/original/{file_id:path}")
    def original(file_id: str) -> FileResponse:
        """The original file, for reading only."""
        return _original_response(dump_path(file_id))

    def run_sort() -> None:
        """The sort itself, on its own thread. However it ends, the job is told.

        A failure the tool knows (no API key, no groups, a file in the way of a card note)
        ends the job with its message. Anything else is a bug: its traceback goes to the
        terminal, and the page is pointed there.
        """
        report = None
        error: str | None = "the sort stopped on an unexpected error; see the terminal of p0 ui"
        try:
            with backend() as model:
                report = sort_vault(root, model, job.progress).model_dump(mode="json")
            error = None
        except (BackendError, ValueError, OSError) as known:
            error = str(known)
        finally:
            job.finish(report=report, error=error)

    @app.post("/api/sort", status_code=202)
    def start_sort() -> dict:
        """Start a sort in the background. 409 when one is already running."""
        with lock:  # a change that is under way finishes before the sort reads the vault
            if not job.claim():
                raise HTTPException(409, "a sort is already running")
        threading.Thread(target=run_sort, name="p0-sort", daemon=True).start()
        return job.status()

    @app.get("/api/sort/status")
    def sort_status() -> dict:
        """How the sort is doing: running, done, total, the last report, and the error text."""
        return job.status()

    @app.post("/api/move")
    def move(body: MoveRequest) -> dict:
        """Put a file in a group, or in Unsorted for null, and freeze it there."""
        with changing():
            dump_path(body.file_id)
            if body.group_id is not None:
                current_group(body.group_id)
            placement = corrections.move_file(root, body.file_id, body.group_id)
        return placement.model_dump(mode="json")

    @app.post("/api/freeze")
    def freeze(body: RefRequest) -> dict:
        """Freeze one file, or every file in a group. Answers with the ids it froze."""
        with changing():
            check_ref(body.ref)
            return {"frozen": corrections.freeze(root, body.ref)}

    @app.post("/api/unfreeze")
    def unfreeze(body: RefRequest) -> dict:
        """Unfreeze one file, or every file in a group. Answers with the ids it unfroze."""
        with changing():
            check_ref(body.ref)
            return {"unfrozen": corrections.unfreeze(root, body.ref)}

    @app.post("/api/groups", status_code=201)
    def add_group(body: NewGroup) -> dict:
        """Add a group. Nothing moves until the next sort."""
        with changing():
            group_set = vault.add_group(root, body.name, body.covers, body.excludes)
        return _group_answer(group_set, vault.slugify(body.name))

    @app.patch("/api/groups/{group_id}")
    def change_group(group_id: str, body: GroupChanges) -> dict:
        """Rename a group, change its covers or excludes text, or several of these at once.

        The covers text is checked first, so that a request with a good name and an empty
        description changes nothing.
        """
        with changing():
            name = current_group(group_id).name
            if body.covers is not None and not body.covers.strip():
                raise HTTPException(400, "every group needs a covers description")
            if body.name is not None and body.name != name:
                vault.rename_group(root, name, body.name)
                name = body.name
            if body.covers is not None or body.excludes is not None:
                vault.edit_group(root, name, body.covers, body.excludes)
            return _group_answer(vault.sync_group_set(root), group_id)

    @app.delete("/api/groups/{group_id}")
    def delete_group(group_id: str) -> dict:
        """Delete a group. The next board sends its files to Unsorted, unfrozen (sync does)."""
        with changing():
            group_set = vault.delete_group(root, current_group(group_id).name)
        return {"set_version": group_set.version}

    @app.put("/api/dump/{name:path}", status_code=201)
    async def upload(name: str, request: Request) -> dict:
        """Add one file to the dump folder from the raw request body.

        Only a plain file name is taken, and a name that is already there is refused (409):
        nothing in the dump is ever written over. The file is created new and filled as the
        body arrives, and it is removed again if the body does not arrive in full.
        """
        _check_upload_name(name)
        target = root / Config.load(root).dump_dir / name
        try:
            handle = target.open("xb")  # x: fails when the name exists, and never follows a link
        except FileExistsError:
            raise HTTPException(409, f"{name} is already in the dump folder") from None
        try:
            with handle:
                async for chunk in request.stream():
                    handle.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return {"file_id": name}

    @app.post("/api/suggest")
    def suggest() -> dict:
        """Ask the model about every frozen file, and list the ones it would now place elsewhere.

        Nothing moves. `failed` holds the files the model could not be asked about, with the
        reason: a failure is not an answer, so it is not agreement either.
        """
        failed: dict[str, str] = {}
        with changing(), backend() as model:
            found = corrections.suggest(root, model, failed)
        return {"suggestions": [item.model_dump(mode="json") for item in found], "failed": failed}

    app.mount("/static", FreshStaticFiles(directory=STATIC_DIR), name="static")
    return app
