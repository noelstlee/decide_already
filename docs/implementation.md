# Phase 0 implementation notes

How the spec in [phase0.md](phase0.md) maps to code. This is the working contract between modules: if code and this file disagree, fix one of them in the same change. Records live in `src/decide_already/types.py`, settings in `config.py`, the backend interface in `backends/base.py`. Read those three files first; they are short.

## Module map

| Module | Does | Step |
| --- | --- | --- |
| `types.py` | Every on-disk record (`Group`, `GroupSet`, `File`, `Placement`, `Label`, `Suggestion`) | 1 |
| `config.py` | `Config` (`.decide/config.yaml`), folder and file names, `env_value` for secrets, `folder_key`, `clean_dump_dir`, the YAML loader | 1 |
| `vault.py` | Vault layout, group notes, set versions, dump listing, card notes | 1, 4 |
| `extract/` | Text out of one file: `extract(path) -> Extracted` | 2 |
| `card.py` | The card and its experiment variants: `build_card(...) -> Card` | 2 |
| `questions.py` | The two questions, in wire format, with shuffled options | 3 |
| `log.py` | JSONL append and read, the append check, token totals | 3 |
| `backends/base.py` | `Backend`, `SortRequest`, `SortResult`, `BackendError`, `RunStopped` (with `BudgetExceeded` and `LogFailed`), `state_text` | 3 |
| `backends/jev.py` | The only code that talks to TypeSafe | 3 |
| `backends/__init__.py` | `make_backend(root, config)` | 3 |
| `sort.py` | `decide` (the placement rules), `judge`, `sort_vault` | 4 |
| `board.py` | `build_board(root) -> Board` for the CLI and the UI | 4 |
| `corrections.py` | `sync`, `move_file`, `freeze`, `unfreeze`, `suggest` | 5 |
| `cli.py` | The `p0` commands, thin wrappers only | 1, 4, 5 |
| `ui/server.py`, `ui/static/` | Local web UI over the same functions | UI |
| `eval/` | Dataset, gold labels, baselines, variants, metrics, experiments, report | 6, 7, 8 |
| `scripts/make_smoke_vault.py` | Writes the ten synthetic smoke files into `sandbox/smoke/_dump/` | 0 |

## Rules for all code

- Every function gets a short docstring saying what it does. Stan reads the code file by file.
- No em dashes anywhere. Plain wording.
- Records are `StrictModel` subclasses. Anything written to disk is validated on the way in and out.
- Tests never call a real API. The fake backend lives in `tests/fakes.py`, never in `src/`. HTTP is mocked only inside tests (`httpx2.MockTransport` for the Jev client).
- Tests use `tmp_path` vaults built with `vault.init_vault`. The page's script is run under node with a stand-in for the browser, in `tests/test_ui_api.py` (skipped when node is missing).
- Originals in the dump folder are opened read-only. Nothing in `src/` writes, moves or deletes there, with two exceptions: the UI upload, which only adds new files (it keeps its own hidden `.p0-upload-*` file there while a body arrives and removes it again), and `vault.delete_original`, which deletes one file the user named by its exact id (the page's Files view, `p0 remove`). No function renames or removes the dump folder, the folder it sits in, or a tool folder, whatever a group is called.
- A failure that is about the run and not about one file is a `RunStopped`: the spend cap (`BudgetExceeded`) or a call log that cannot be written (`LogFailed`). The sorter, `suggest` and the evaluation runner end the run on either.
- Only `backends/jev.py` imports `typesafe_sdk`. Only `eval/baselines/` imports `fastembed` or `anthropic`.
- The model version always comes from `Config.model` and is passed on every call.

## Config (`config.py`)

- `model` must be an exact version: a lower-case name, then a full dotted version, as in `jev-1.14.0`. An alias (`jev-latest`), a partial version (`jev-1.14`) or any other spelling is refused when the config is built or loaded, before a backend exists.
- `dump_dir` is kept in one plain form: spaces around it, trailing slashes and a leading `./` are dropped, so an old `course/` reads as `course`. It is refused when it is empty, `.`, absolute, holds a backslash or a `..` part, or starts in `_groups`, `Unsorted` or `.decide`. `p0 init --dump .` is refused.
- `folder_key(name)` is a folder name as the file system may compare it: one Unicode form, capitals ignored. Folders are matched to groups, and names are checked against the protected folders, by key. So the tool behaves the same on a Mac (where `HW` and `hw` are one folder) and on Linux.
- `Config.load` raises `ValueError` that names the file for bad YAML and for a config that does not validate. It lists setting names and messages only, never the values typed in, so a key pasted into the config by mistake is not echoed.

## Extraction (`extract/`)

```python
class Extracted(StrictModel):
    file_type: str            # "Python code", "PDF, 12 pages", "Slides, 24", "Jupyter notebook, 14 cells", "Markdown", "Text", "HTML page"
    title: str | None = None
    outline: list[str] = []   # headings, slide titles, or function and class names; at most 40, each at most 120 chars
    text: str = ""            # all extracted text; "" when nothing could be extracted

def extract(path: Path) -> Extracted   # dispatches on the file extension
```

| Kind | Extensions | Title | Outline | Text |
| --- | --- | --- | --- | --- |
| Markdown | `.md`, `.markdown` | first heading, else the first line that holds a word | headings | the file |
| Text | `.txt`, `.rst`, any other readable text | the first line that holds a word | none | the file |
| Code | `.py`, `.m`, `.r`, `.jl`, `.js`, `.ts`, `.java`, `.c`, `.h`, `.cpp`, `.sh`, `.tex`, `.sql` | module docstring's first line, or the first comment within the first 10 lines | function and class names (`ast` for Python, simple patterns elsewhere) | the file |
| Notebook | `.ipynb` | first markdown heading | markdown headings, then function and class names | cell sources joined; outputs dropped (nbformat 4 only) |
| PDF | `.pdf` | document title metadata if sensible, else the first line that holds a word | the PDF's table of contents, if it has one | text of every page (PyMuPDF) |
| Slides | `.pptx` | first slide's title | slide titles | text of every slide (python-pptx) |
| HTML | `.html`, `.htm` | `<title>` or first `<h1>` | `h1` to `h3` | visible text; only `<main>` or `<article>` when present; scripts, styles and nav dropped (stdlib `html.parser`) |
| Anything else or binary (type `Binary file`) | | none | none | `""` |

Read text as UTF-8 with `errors="replace"`. A file with NUL bytes in its first 8 KB is binary. Only the first 8 MB of a text file is read (`MAX_TEXT_BYTES`), so its card, excerpts included, comes from that part; a notebook is read whole, since it is JSON. Images are dropped. Text is cleaned: control characters and base64 data URIs are removed. For Markdown and notebooks, leading headings that start with "Copyright" or "License" are passed over for the title (they stay in the outline); when there is no other heading, the first one is kept. A file with no text gets `text=""` and later goes to Unsorted with reason `no_text`. Extractors never raise on a damaged file: they return `text=""`.

## Card (`card.py`)

```python
VARIANTS = ("card", "card_no_name", "full_text", "window_vote", "names_only")

class Card(StrictModel):
    file: File
    title: str | None
    states: list[dict]   # one state per request; more than one only for window_vote; [] when the file has no text

def estimate_tokens(text: str) -> int    # len(text) / 4, rounded up
def hash_file(path: Path) -> str         # sha256 hex of the bytes
def build_card(path: Path, file_id: str, config: Config, variant: str | None = None) -> Card   # variant defaults to config.variant
def render_card(card: Card) -> str       # markdown body of the card note: name, type, title, outline, then the text the model saw
```

The token count is an estimate (4 characters per token). The API's own `input_tokens` is logged per call as the real number. `build_card` raises `ValueError` for an unknown variant and `OSError` for a file that cannot be opened; `extract` itself never raises.

The default `card` state is a JSON object with named fields. Empty fields are left out, because an empty list or string can read as a signal:

```json
{
  "file": {"name": "hw3_solution.py", "path": "hw3/hw3_solution.py", "type": "Python code"},
  "title": "Homework 3: Kernel Methods",
  "outline": ["rbf_kernel()", "train_svm()", "predict()"],
  "opening": "the first 1,500 tokens of the text",
  "excerpts": {"at_40_percent": "...", "at_70_percent": "...", "at_95_percent": "..."}
}
```

- `opening`: the first 6,000 characters, cut at a whitespace boundary.
- `excerpts`: three passages of up to 1,200 characters starting at 40%, 70% and 95% of the text. Only for text longer than the opening, and never overlapping the opening or each other.
- The whole state must fit `config.card_token_budget`. If it does not, shorten the outline, then the opening.
- `file.card_tokens` is the estimate for all states together.

Variants:

| Variant | States | Note |
| --- | --- | --- |
| `card` | the object above | default |
| `card_no_name` | the same without `file.name` and `file.path` (`file.type` stays) | |
| `full_text` | `{"file": {...}, "text": ...}` with the text cut at 30,000 tokens | one request |
| `window_vote` | `{"file": {...}, "text": window}` in equal windows of at most 2,000 tokens; when there are more than 8 windows, 8 evenly spaced ones including the first and the last | one request per window; probabilities are averaged by `sort.judge` |
| `names_only` | same as `card` | the request sets `with_descriptions=False` |

## Questions (`questions.py`)

All question wording lives here and nowhere else.

```python
GROUP_QUESTION = "group"
FITS_ANY_QUESTION = "fits_any"

def build_questions(groups: list[Group], rng: random.Random, with_descriptions: bool = True) -> tuple[dict, list[str]]
# returns (questions in wire format, option order as group ids)
```

- `group`: a Choice. Instructions: "Which group does this document belong in?". Criteria: one entry per group, keyed by the group name, in shuffled order. The value is `{"covers": ..., "excludes": ...}` (no `excludes` key when the group has none), or `None` when `with_descriptions` is false.
- `fits_any`: a Noul answered on its own, so it must carry the group descriptions itself. Instructions are a JSON object with the statement "This document clearly belongs in one of these groups." and the same groups in the same shuffled order. Criteria: `true` is "The document's content matches what one of the groups covers.", `false` is "The document is about something that none of the groups cover.". With `with_descriptions` false, only the names are listed.
- There is no "none of these" option. Unsorted is decided in code.
- `build_questions` raises `ValueError` when there are no groups or two groups share a name.
- The shapes were checked against the live docs (https://docs.typesafe.ai/primitives/choice.md, noul.md, advanced.md) and against the live API on 2026-11-15. A probe on five hand-written cards showed why `fits_any` carries the descriptions: with group names only it scored an unrelated checklist 0.67 and a syllabus 0.57; with descriptions, 0.11 and 0.97.

## Jev backend (`backends/jev.py`, `log.py`)

```python
class JevBackend:
    name = "jev"
    def __init__(self, config: Config, log_path: Path, *, api_key: str | None = None,
                 rng: random.Random | None = None, transport=None): ...
    def sort(self, request: SortRequest) -> SortResult: ...
    def close(self) -> None: ...

def make_backend(root: Path, config: Config) -> Backend   # in backends/__init__.py; key from env_value("TYPESAFE_API_KEY", root)
```

- One `TypeSafeClient` per backend, built with the key, a `RetryPolicy` (up to 4 retries on 408, 429 and 5xx with backoff, honors Retry-After, 60 seconds in all) and a 30 second timeout per HTTP operation. `transport` is for tests.
- The key is passed in as `api_key`. Only `make_backend` looks it up.
- Every call passes `model=config.model`. If the response's `model` differs from the pin, log the call and raise `BackendError`.
- A missing key raises `BackendError` with a message that names `TYPESAFE_API_KEY` and `.env`. The key is never logged: it is blanked (`***`) in error text and in every string of a logged response body.
- Before each request, compare the spend so far with `config.max_spend_usd`. Spend is `input tokens / 1e6 * config.usd_per_million_tokens`, counted from the log at start and updated per call. Every call that reports input tokens counts, usable or not, because it was billed. At or over the cap, raise `BudgetExceeded` without sending (and without a log line).
- Next, make sure the log can be added to (`log.check_appendable`: create `.decide/` and open the file for append; a missing log is created empty). When it cannot, raise `LogFailed` without sending. Both checks run before every request, under the lock.
- After the call its line is appended under the lock. When that fails (a full disk, a record that is not JSON), raise `LogFailed` naming the log file, the reason and the file id, whatever the call's own outcome was. The call's input tokens still count toward the cap in that backend object. From then on that backend refuses every call with `LogFailed` before sending, because a full disk passes the open check.
- A 200 the SDK cannot validate whose body is a JSON object is an answered call: the body is logged as `response`, and its `usage.input_tokens` is logged and counted.
- A request cut off by Ctrl+C, or by any exception that is not the SDK's own, is logged with `status: "error"` and `error: "interrupted: ..."`, then passed on unchanged. Not covered: a process that is killed outright, and the UI's sort thread when the server exits in the middle of a request.
- Probabilities come back keyed by group name. Convert to group ids. `option_order` is the shuffled order as ids.
- Safe to call from several threads: one lock is held for the shuffle, both checks, every change of the spend counter and every log append.
- The SDK's own logger stays at its default. Our log is the record.

One line per call in `.decide/jev.jsonl`:

```json
{"at": "2026-11-15T10:00:00+00:00", "file_id": "hw3/hw3_solution.py", "set_version": 2, "variant": "card",
 "backend": "jev", "model_requested": "jev-1.14.0", "model_answered": "jev-1.14.0",
 "option_order": ["logistics", "hw", "project"],
 "request": {"model": "jev-1.14.0", "state": {}, "questions": {}},
 "response": {},
 "input_tokens": 2710, "request_id": "req_...", "latency_ms": 412.3, "status": "ok", "error": null}
```

`request` is exactly what was sent. `response` is the raw JSON body (with the API key blanked out, should the server quote it). A call with no usable body (an HTTP error, a timeout, an interrupted request) is logged with `status: "error"`, `response: null` and the error text. A call that was answered but cannot be used (another model version answered, an answer is missing, the SDK could not validate the body) is also `status: "error"`, but keeps its response, tokens and request id.

`log.py`: `append_jsonl(path, record)` (a failed append leaves the file as it was: a line that only partly fit is cut off again), `check_appendable(path)` (creates the folder and the file, writes nothing, raises `OSError`), `read_jsonl(path) -> list[dict]`, `total_input_tokens(path) -> int` (every line that has `input_tokens`). A line that is not a JSON object raises `ValueError` naming the file and line.

The fake for tests, in `tests/fakes.py` (import it with `from fakes import FakeBackend`):

```python
class FakeBackend:
    name = "fake"
    def __init__(self, answers: dict[str, tuple[dict[str, float], float]] | None = None,
                 default: tuple[dict[str, float], float] | None = None, fail: set[str] | None = None): ...
    calls: list[SortRequest]
```

`answers` maps a file id to `(probs by group id, fits_any)`. A file id in `fail` raises `BackendError`. With no answer and no default, it spreads probability evenly and returns `fits_any=0.9`. `model` in results is `"fake-0"`.

## Placement rules (`sort.py`)

```python
class Decision(StrictModel):
    group_id: str | None
    close_call: bool
    reason: UnsortedReason | None

def decide(probs: dict[str, float], fits_any: float, config: Config) -> Decision
def judge(card: Card, groups: list[Group], set_version: int, backend: Backend, variant: str,
          per_state: list[SortResult] | None = None) -> SortResult
```

| `fits_any` | top probability | top minus second | Outcome |
| --- | --- | --- | --- |
| under `unsorted_fits_any_below` | any | any | Unsorted, `no_group_fits` |
| at or over | under `unsorted_top_below` | any | Unsorted, `weak_top` |
| at or over | at or over | under `close_call_margin` | placed in the top group, `close_call` true |
| at or over | at or over | at or over | placed in the top group |

With one group, the second probability counts as 0. Ties go to the lower group id. A value exactly at a cutoff counts as at or over. `judge` sends one request per state and averages `probs` and `fits_any` when there are several (window vote); tokens and latency are summed, and stay `None` when the backend reports none (the fake does not). When `per_state` is a list, each state's own answer is appended to it (the evaluation reads the window votes from it). A group missing from one state's answer counts as 0 for that state, and a card with no states raises `ValueError`.

With the default cutoffs the third row can never apply when the probabilities sum to 1: a top probability of 0.6 or more leaves at most 0.4 for the second, a gap of at least 0.2, and the margin is 0.15. The close call flag only becomes live when `unsorted_top_below` is under 0.575 or the margin is raised. The defaults are Stan's call (see docs/decisions.md, still open).

Code alone handles two cases: a card with no states goes to Unsorted with reason `no_text`, `placed_by: code`, no model asked. A `BackendError` is never an answer: the file keeps its previous note, if any, and is reported.

## Card notes (`vault.py`)

```python
def dump_files(root: Path, config: Config) -> list[tuple[str, Path]]   # (file id, path), sorted by id
def note_name(file_id: str) -> str                                     # "/" becomes "__", plus ".md"
def source_link(file_id: str, config: Config) -> str                   # "[[<dump_dir>/<file_id>]]"

class CardNote(StrictModel):
    path: Path
    folder: str          # name of the top-level folder the note is under
    placement: Placement # the placement fields of the frontmatter
    body: str
    extras: dict         # any other frontmatter keys (properties added in Obsidian), kept on every rewrite
    stamp: tuple[int, int] | None   # the file's size and time of change when it was read

def read_card_notes(root: Path) -> list[CardNote]
def write_card_note(root: Path, placement: Placement, body: str, notes: list[CardNote] | None = None) -> Path
def delete_original(root: Path, file_id: str) -> int                 # deletes one original and its card notes; returns how many notes
def folder_for(root: Path, group_id: str | None) -> Path               # group name folder, or Unsorted
def group_id_of_folder(root: Path, folder: str) -> str | None          # raises KeyError for a folder that is no current group
```

- `delete_original` is the only code that removes an original. It takes the file by its exact id in the dump listing (never a joined path), removes a link as a link, removes the file's card notes, and removes folders inside the dump that are left empty (never the dump folder itself). Labels stay: they record what the user decided. The commands and the page call it through `corrections.delete_file`, which then withdraws the file's active label with an "unfrozen" line, so a file added later under the same id is sorted as a new file.
- `dump_files` walks the dump folder recursively and skips any path with a component that starts with a dot. It does not follow symlinked folders. The file id is the path relative to the dump folder, with forward slashes.
- A card note is a markdown file whose frontmatter holds a valid `Placement` once other keys are set aside. Other keys (a tag or any property added in Obsidian) are kept in `extras` and written back after the placement fields by `write_card_note` and by every rewrite in `corrections`. The body is `render_card(card)`.
- A card note sits in its group's folder (the folder is named after the group) or in `Unsorted/`. The folder a note is under is the file's current group. `placement.group_id` is the group as of the last write or sync.
- `read_card_notes` reads the markdown files in every visible top-level folder and in the folders inside it, except `_groups`, `.decide` and the top-level folder that holds the dump. A note's folder is the top-level folder it is under, so a note filed into `HW/week 3` is in HW. A markdown file whose frontmatter holds no `Placement` is one of the user's own notes and is ignored. A card note whose YAML is broken counts as not a card note.
- `write_card_note` writes at the top of the folder of `placement.group_id`, carries over the extras of the note it replaces, and removes any other card note for the same file id. It never overwrites a file that is not that file's card note: that raises `FileExistsError`. Paths are compared with `samefile`. `notes` is what `read_card_notes` just returned; a caller that writes many notes in a row passes it so the vault is read once, not once per write.
- Two file ids that flatten to the same note name (`a/b.txt` and `a__b.txt`) cannot both have a note in one group. The second is reported as an error.
- Folders are matched to groups by `folder_key` (`folder_ids`, `group_id_of_folder`), so a folder is its group's under any capitals or Unicode form.
- A group's name cannot be a tool folder (`_groups`, `Unsorted`, `.decide`), the first part of the dump folder, or a name that ends in ` (old)` or ` (old) N`, all compared by key. This holds for notes made by hand too, and two notes whose names have the same key are refused. A group id cannot be `Unsorted`.
- `rename_group` (the command) moves the folder directly. It records pending hand edits first, checks everything before the note is renamed (a file where the folder has to go, a target folder that holds card notes, files of the same name; a hidden file such as `.DS_Store` is never a clash), and renames the note back if the folder cannot follow.
- `sync_group_set` never moves a folder to a new name. Before a new set version is written, the folder of each group that is gone or renamed since the last version is taken out of the way (`_free_old_folders`): respelled when only the spelling changed, removed when empty, and otherwise set aside as `<old name> (old)` (then ` (old) 2`, and so on). Nothing is merged, so a name that is used again never brings another group's card notes with it. `corrections.sync` then puts each set-aside note where it belongs (see Corrections) and removes the set-aside folder once it is empty.
- `read_note` raises `ValueError` that names the file for text that is not UTF-8 and for frontmatter that is not valid YAML or not `key: value` lines. `read_groups` raises `ValueError` that names the note for a covers or excludes that is not text. `slugify` keeps letters and digits of any script; ASCII names give the id they always gave.
- Notes are read with PyYAML's C loader when it is available: every command reads every card note, and the C loader is about four times faster.

## Sort (`sort.py`)

```python
class SortReport(StrictModel):
    set_version: int
    asked: int
    placed: int
    unsorted: int
    close_calls: int
    frozen_skipped: int
    errors: dict[str, str]     # file id to message
    spent_usd: float | None    # this vault's total so far, from log.total_input_tokens on .decide/jev.jsonl

def sort_vault(root: Path, backend: Backend, on_progress: Callable[[int, int], None] | None = None) -> SortReport
```

1. Run `corrections.sync(root)` first, so hand edits made in Obsidian are never overwritten (added in step 5).
2. Read the current set (`vault.sync_group_set`). No groups is an error with a clear message.
3. For each dump file: hash it. A frozen note with the same hash is skipped. A frozen note whose hash changed since the freeze was given is unfrozen first (label `unfrozen`) and then sorted like any other. A dump file that has an active label but no card note is not sent to the model: it is listed in `errors` (`sort.NO_NOTE`), and `p0 unfreeze <file id>` is the way to start it over.
4. Every other file is carded with `config.variant`, judged, decided, and written as a card note with `placed_by: jev`.
5. Any `RunStopped` (the spend cap, a call log that cannot be written) stops the run. The file that hit it is listed in `errors` with the backend's message, and every later file with a fixed one (`NOT_REACHED` for the cap, `STOPPED_BEFORE` otherwise).
6. A card note changed by hand while the sort runs is never written over. Before a file is carded and again before its note is written, the note read at the start must still be at the same path with the same size and time of change. Otherwise it is left alone and the file is listed in `errors` (`sort.CHANGED_BY_HAND`), so the next sync applies the usual rules.

`asked` counts the files sent to the model, answered or failed with a `BackendError`. It leaves out files with no text and files the spend cap stopped. An `OSError` for one file (the original cannot be read, or its note cannot be written) is listed in `errors` and the run goes on. A sort reads the card notes twice: once in `sync`, once for itself. `p0 sort` exits 1 when `errors` is not empty, after printing the summary line.

`sort.py` imports `corrections` inside `sort_vault`, because `corrections` imports `judge` and `decide` from `sort`.

## Board (`board.py`)

```python
class BoardFile(StrictModel):
    file_id: str
    name: str                 # last path component
    file_type: str
    group_id: str | None
    frozen: bool
    close_call: bool
    placed_by: str
    reason: UnsortedReason | None
    top_prob: float | None
    probs: dict[str, float] | None
    fits_any: float | None
    stale: bool               # sorted under an older set version than the current one

class BoardGroup(StrictModel):
    group: Group
    files: list[BoardFile]

class Board(StrictModel):
    set_version: int
    groups: list[BoardGroup]
    unsorted: list[BoardFile]
    new_files: list[str]      # dump files that have no card note yet

def build_board(root: Path) -> Board
```

Files are grouped by the folder they sit in, so `BoardFile.group_id` is the folder's group and not `placement.group_id` (the two differ for a note that was dragged and not synced yet). Folders are matched by key. A note in a folder that is no group's is listed where sync will put it: under its own group when that group exists, else under Unsorted. Within a group: close calls first, then by name (case ignored, ties by file id). `build_board` calls `vault.sync_group_set`, so group notes edited by hand become a new set version, as with `p0 group list`.

## Corrections (`corrections.py`)

```python
def sync(root: Path) -> SyncReport            # moved, confirmed, unfrozen, orphaned, rehomed, missing_sources: lists of file ids; set_version
def move_file(root: Path, file_ref: str, group_ref: str | None, *, exact: bool = False) -> Placement
def freeze(root: Path, ref: str, *, kind: Literal["file", "group"] | None = None) -> list[str]   # returns the file ids it froze
def unfreeze(root: Path, ref: str, *, kind: Literal["file", "group"] | None = None) -> list[str]
def suggest(root: Path, backend: Backend, failed: dict[str, str] | None = None) -> list[Suggestion]
def read_labels(root: Path) -> list[Label]
def active_labels(root: Path) -> dict[str, Label]     # by file id
def unfreeze_note(root: Path, note: CardNote, set_version: int) -> Placement   # used by sort_vault
```

A file reference is a file id, or a file name that is unique in the vault. A group reference is a group id or name; `None` or `Unsorted` means Unsorted.

| Action | Effect on the note | Label written |
| --- | --- | --- |
| `move_file` | moved to the group's folder, `group_id` set, `placed_by: you`, `frozen: true`, `close_call: false`, `reason: null` | `moved` (none if it is already there; then it is a freeze) |
| `freeze` | `frozen: true` | `confirmed`, `from_group` and `to_group` both the current group |
| `unfreeze` | `frozen: false` | `unfrozen` |
| `suggest` | nothing moves | none; one line per differing file in `.decide/suggestions.jsonl` |

Labels record `jev_probs` and `jev_set_version` from the note as it was, and `set_version` as the current set version. `jev_set_version` is null when the note has no `probs` (no model was asked). A file's active label is its last line in `labels.jsonl` when that line is `moved` or `confirmed`.

`move_file`, `freeze`, `unfreeze` and `suggest` each call `sync` first, so an action always sees hand edits, and a drag by hand plus a command gives the same labels as two commands. Inside an action the card note is written first and the label second: a note that cannot be written leaves no label, and a lost label is repaired by the next sync. In `freeze` and `unfreeze` a reference that names a group wins over a file of the same name. That is the commands' reading. A caller that knows what it names passes `kind` ("file": an exact file id, never a group; "group": never a file), and `move_file(..., exact=True)` takes a file id and a group id only. The page always does. When a file has several card notes (one was copied by hand), the last one by path counts.

A note that becomes frozen (`move_file`, `freeze`, sync rules 2 and 3) takes the hash, file type and card of its original as it is at that moment, when the original can be read. So a freeze covers the content the file has when the freeze is given, and only a later change takes it off. `unfreeze` on a file id that has an active label and no card note writes `unfrozen` and returns the id.

`sync` picks up edits made by hand, in this order per card note:

1. The note sits in a folder that is neither a current group's nor `Unsorted` (a set-aside folder, a folder renamed by hand, any other folder): its own `group_id` decides. If that group is gone, move the note to `Unsorted/`, `group_id: null`, `placed_by: code`, `frozen: false`, `close_call: false`, and if it had an active label write `unfrozen` (from the old group to null). Otherwise (Unsorted included) put it back in its group's folder unchanged, with no label, and list it in `rehomed`; rules 3 and 4 still apply to it.
2. The folder's group differs from `placement.group_id` (a drag): update the note as `move_file` does, then write `moved`.
3. `frozen` is true and the file has no active label (set by hand): write `confirmed`.
4. `frozen` is false and the file has an active label (cleared by hand): write `unfrozen`.

Card notes whose source file is gone from the dump are listed in `missing_sources` and left alone: no rule is applied to them, even when they were dragged or frozen by hand. `sync` first calls `vault.sync_group_set`, so edits to group notes become a new set version. A group note renamed by hand keeps its id: its old folder is set aside, and rule 1 puts its notes in the folder of the new name one by one, with every label kept. This holds for a chain of renames and for a swap. A deleted group's notes go to Unsorted, also when its name was taken by another group before any sync. A file is in at most one of `moved`, `confirmed`, `unfrozen`, `orphaned` and `missing_sources`; a rehomed file can also be in `confirmed` or `unfrozen`. After the rules, set-aside folders that are empty are removed. `sync` raises `FileExistsError` when a file of the user's sits where a card note has to go, which stops the whole sync (and the sort or board that called it) with a message that names the file.

`suggest` asks the backend about every frozen file, runs `decide`, and reports the files whose decision differs from their current group. A failure is not an answer, so it is not agreement either: when `failed` is a dict, each file the model could not be asked about is added to it with the reason.

## CLI (`cli.py`)

`p0 init`, `p0 group add|edit|rename|delete|list` (step 1), `p0 sort`, `p0 board` (step 4), `p0 move <file> <group>`, `p0 freeze <file or group>`, `p0 unfreeze <file or group>`, `p0 suggest`, `p0 sync` (step 5), `p0 ui [--port 8765] [--no-open]` (UI). Every command takes `--vault`. Commands print short plain lines and exit 1 with `error: ...` on a known failure (every `ValueError` and `OSError`, a broken group note or config included). The group commands need a vault. `p0 sync` also prints the files it put back in their group's folder. `p0 sort` and `p0 suggest` also exit 1 when a file could not be sorted or asked about, after the summary line and one `error: <file id>: <message>` line per file.

## UI (`ui/`)

`create_app(root: Path, backend_factory: Callable[[], Backend] | None = None) -> FastAPI` in `ui/server.py`. Tests pass a factory that returns the fake backend. `p0 ui` serves it on `127.0.0.1` only. The page is plain HTML, CSS and JS in `ui/static/`, with no build step and no external requests.

| Method and path | Does |
| --- | --- |
| `GET /` | the page |
| `GET /api/board` | runs `sync` (skipped while a sort runs), then returns the board plus `sorting` and vault info: name, path, dump folder, model, spend and cap, and the three cutoffs |
| `GET /api/files/{file_id:path}` | one file: `{file_id, group_id (by folder), placement, probs: [{group_id, name, prob}] highest first, card}`; 404 for a file with no card note yet |
| `GET /api/original/{file_id:path}` | the original file, read-only. PDFs and pictures open inline, any other text file is sent as `text/plain` with `nosniff` (a web page from the dump must never run inside the app), other binaries download. Sent with `Cache-Control: no-cache` |
| `POST /api/sort` | starts a sort in the background; 409 if one is running |
| `GET /api/sort/status` | `{running, done, total, report, error}` for the progress bar |
| `POST /api/move` | `{file_id, group_id}`; `group_id: null` is Unsorted; both matched by exact id; 404 for a file with no card note |
| `POST /api/freeze`, `POST /api/unfreeze` | `{file_id}` for one file (exact id, 404 without a card note) or `{group_id}` for every file of a group (its id; null is Unsorted). Exactly one of the two, else 400 |
| `POST /api/groups` | `{name, covers, excludes}` |
| `PATCH /api/groups/{group_id}` | any of `name`, `covers`, `excludes` |
| `DELETE /api/groups/{group_id}` | deletes the group |
| `GET /api/dump` | every file in the dump folder: `{dump_dir, files: [{file_id, size, modified}]}`, for the Files view |
| `DELETE /api/dump/{file_id:path}` | deletes one original for good, with its card note, and withdraws its label (`corrections.delete_file`); 404 for an id that is not a file of the dump, 409 while a sort runs |
| `PUT /api/dump/{name}` | adds one file to the dump folder from the raw request body. The body is collected under a hidden name (`.p0-upload-<random>`), which `dump_files` skips, and linked to its own name with `os.link` once it is whole. Never overwrites (409) and never accepts a path; a body cut short answers 400 and leaves nothing |

Errors come back as `{"detail": "..."}`: 400 bad input, 403 a request from another site, 404 unknown id, 409 conflict (a sort is running, a name is taken, a file exists, the backend cannot be built), 500 for other OS errors and for any error no handler knows (the detail then names the error's type and message, and a sort that hits one ends with the same text in its status). File ids are checked against `dump_files` and never joined to a path, so no request can reach outside the vault.

Safety of the server:

- Every request passes `GuardMiddleware`, the files under `/static` and unknown addresses included. It refuses a request that fails `_local_only` (the Host header must be `127.0.0.1` or `localhost`, and an Origin header must match the Host), and it adds to every answer the Content-Security-Policy (`default-src 'self'`, `frame-ancestors 'none'`), `X-Frame-Options: DENY` and `X-Content-Type-Options: nosniff`. Without the check any website open in the same browser could read originals or start a paid sort; without the headers it could show the page inside a frame and catch a click. Tests therefore build the client as `TestClient(app, base_url="http://127.0.0.1:8765")`.
- One lock serializes vault access. The sort runs on a background thread. While it runs, every change (move, freeze, unfreeze, group add, edit and delete, suggest) answers 409, because the sort read the card notes at its start and would write over the change. Uploads, file detail and originals still work.
- The real backend is built only when a sort or a suggest starts, and closed after, so the page opens without an API key.
- The page loads nothing from anywhere else. FastAPI's docs routes are off (they load from a CDN). The page, the static files and originals are sent with `Cache-Control: no-cache`.
- `p0 ui` binds the socket itself before it opens the browser, so the first load is never refused and a taken port is a plain `error:` line.

The page: the Unsorted column first, then one column per group, a card per file. Drag a card to another column to move it (which freezes it). A card shows the file name, type, the top probability, and marks for close call, frozen, and moved by you. Clicking a card opens a side panel with every group's probability, `fits_any`, the reason when Unsorted, the card text, and buttons to freeze, unfreeze or open the original. The header has the vault name, set version, model, spend, a Sort button with progress, and Add group. Dropping files from the desktop onto the page adds them to the dump. Group descriptions are edited in place. A Board and Files switch in the header shows either the lanes or the Files view: the dump folder as a tree of folders (each can be folded), every file with its size and where it is sorted (a group, Unsorted, or not sorted yet), and Show card, Open and Delete. Delete asks first, with Keep focused, and then removes the file for good. Add files opens the system's file picker; dropping files on the page does the same. What a sort could not do (the reason it stopped, or every file that was not sorted with its reason) and dropped files that were not added stay in a list above the lanes until dismissed. Sort is disabled while dropped files are on their way. A second press on Sort, on the form's save or on delete is ignored while the first is not answered, and when the server refuses one of them the board is loaded again. When the open file can no longer be read, the panel closes with a message. Minimal look: system fonts, plenty of white space, one accent color, light and dark from the system setting, keyboard reachable.

## Evaluation (`eval/`)

Everything the evaluation downloads or writes goes under the gitignored `sandbox/`. The evaluation vault is `sandbox/eval/`, a normal vault with the groups from `eval/groups.py`.

Dataset (`eval/dataset.py`, run as `python -m decide_already.eval.dataset`):

- Downloads `6.867-fall-2006.zip` from ocw.mit.edu (CC BY-NC-SA 4.0) and the lab notebooks, solutions and helper modules of the MIT 6.S191 labs from GitHub `MITDeepLearning/introtodeeplearning` (MIT license), caches them in `sandbox/datasets/`, and copies the files to sort into `sandbox/eval/_dump/`.
- Gold labels come from the sources' own structure and never from a model: OCW's resource metadata for 6.867, the repo's folders for the labs. The OCW files sit flat in `6.867/`, so no folder there gives a group away. The labs keep their paths under `6.S191/` and every lab file is Homework, so that folder name does (22 of the 73 test files with text). They are written before any sort, to `sandbox/eval/gold.jsonl`, one line per file: `{file_id, gold_group, source_url, source_type, license, sha256, split}`. `gold_group` is a group id, or `null` for a file that fits no group.
- `split` is `dev` for 20% of the files and `test` for the rest, stratified by gold group, with a fixed seed.
- `sandbox/eval/DATASET.md` lists sources, licenses, counts per group and the mapping used.
- For other modules: `read_gold(eval_root) -> list[GoldLabel]` and `build_dataset(root, fetch=download) -> list[GoldLabel]`. The network step is the one `download(url, dest)` function.
- The labs are pinned to a commit. Course pages are rendered from their metadata without the site's navigation. Pages that only list resources are left out. The build fails when the dump holds a file with no gold label.
- As built on 2026-11-15: 95 files. Homework 52, Lecture notes 24, Exams 17, Projects 1, Logistics 1. Dev 18, test 77. Four zip archives (gold: Homework, all in test) have no extractable text.

Baselines (`eval/baselines/`), each a `Backend`:

| Baseline | How it sorts |
| --- | --- |
| `rules.py` | file name keywords and extensions per group, first match wins, written blind; one-hot `probs`; when no rule matches, `probs` is empty and `fits_any` is 0 (no pick) |
| `embed.py` | cosine similarity between the card text (`state_text`) and each group's name and covers text, softmax into `probs`; a local model through `fastembed` (`BAAI/bge-small-en-v1.5`, which reads the first 512 tokens); `fits_any` 1.0. Excludes text is not embedded, since an embedding has no notion of "not". `chunk_chars` embeds the whole card in pieces and averages them |
| `llm.py` | one Claude model (`claude-fable-5-2`) through the Anthropic API, zero-shot, same card and groups, a structured single choice, no fallback model; one-hot `probs`; reports input and output tokens; key from `env_value("ANTHROPIC_API_KEY")`; unavailable without a key |

Metrics (`eval/metrics.py`): `Prediction`, `Cutoffs`, `summarize`, `outcome`, `tune_cutoffs`, and for the report `paired` and `sign_test` (two configurations on the files both answered), `close_call_can_fire` and `tuning_ties`. Reported: top-1 accuracy overall and per group, macro-F1, accuracy among placed files, share Unsorted and how many of those had the right top group, expected calibration error of the top probability, cost (input and, where billed, output tokens) and latency per file. Top-1 ignores the cutoffs. `outcome` is the same truth table as `sort.decide`, written separately, and a test checks that the two agree. `tune_cutoffs` grid-searches the two Unsorted cutoffs to maximize right placements minus wrong placements; the defaults win ties.

Runner (`eval/run.py`, run as `python -m decide_already.eval.run [--root sandbox/eval] [--configs a,b] [--workers 6] [--limit N] [--dry-run] [--no-experiments]`): runs every configuration over the dataset, caches every prediction so nothing is paid for twice, tunes the two Unsorted cutoffs on the dev split, reports on the test split, and writes `results/metrics.json` and `results/report.md`. All Jev calls go through one `JevBackend` logging to `sandbox/eval/.decide/jev.jsonl`, so the spend cap covers the whole evaluation.

- Configurations, written `backend:variant`: `jev:card`, `jev:card_no_name`, `jev:full_text`, `jev:window_vote`, `jev:names_only`, `rules:card`, `embed:card`, `embed-chunks:card` (the whole card in pieces of 1,600 characters), `llm:card`. A configuration whose backend cannot be built (no key) is pending and the report says why. With the default `--configs`, `llm:card` runs whenever `ANTHROPIC_API_KEY` is found, and it has no spend cap of its own: leave it out of `--configs` to be sure.
- Cache: `results/predictions/<name>.jsonl` with `:` written as `_`, one line per file (`file_id, gold, split, no_text, probs, fits_any, input_tokens, output_tokens, latency_ms, model, set_version, groups_hash, error, window_tops`). A line is reused only under the same set version and the same `groups_hash` (a fingerprint of the group list), and never when it holds an error. The cache does not key on the model pin, the card format or the file content: delete the folder when one of those changes.
- Cards are built in the main thread (PyMuPDF is not safe across threads). Jev requests run in a thread pool; the other backends run one at a time. Files with no text are recorded without asking; the file-name rules get a state with the name and path only.
- Exit status: 0 done, 1 a problem with the arguments or the vault, 2 a backend stopped the run (`RunStopped`: the spend cap, or a call log that cannot be written). The report is still written from what is cached, and a request that hit a `RunStopped` is not cached as a failed file.
- `--dry-run` asks and writes nothing. It reads the last recorded set (`vault.latest_set`) because syncing the set can write.
- A configuration is scored on the files it answered among the eligible ones. A file with an error is left out of that row, never counted as wrong, and named in the caveats; `n` is shown per row.
- Layering: `experiments.py` (pure) <- `report.py` <- `run.py`. `run.py` hands the report an `Inputs` record made from the cache.

Scoring rules:

- The headline number is top-1 accuracy on the test split over files that have extractable text, with the default cutoffs. Every configuration is scored on the same files.
- Files with no extractable text never reach a model. They are counted and reported on their own line, and a second accuracy over all files (where they count as wrong for every content-based configuration) is given next to the headline.
- A majority-class row is reported for context: the largest gold group among the files the row is scored on.
- Tuned cutoffs are reported next to the defaults, never instead of them. When several pairs of cutoffs tie for best on the dev split and the defaults are among them, the row reads "not tunable on this dev split" with the count of tied pairs.
- Go/no-go: each comparison is paired on the files both configurations answered and printed with its discordant files (the files only one of the two got right). A condition that needs one configuration above another is met only when the counts meet it and an exact two-sided sign test on the discordant files gives p under 0.05, which takes at least 6 files, all one way. The verdicts are yes, no, not separable, "cannot be told: the card is at the ceiling" (the card leaves less than the 5 points a condition needs) and n/a. The test counts every file as independent, so near copies make it easier to pass than it should be.
- What is fixed by construction is not printed as a measurement: close calls under cutoffs that cannot flag one (n/a), and the placement and calibration cells of every backend whose probabilities are one-hot or set by a temperature (the rules, the embeddings, the LLM, the majority class). `metrics.json` keeps the raw numbers and gains flags.
- The report also prints the card's accuracy under each wording of the groups on the same test files (the vault's descriptions, names only, each description edit), and a caveat computed from the data when every file under a top-level folder has the same gold group, because that folder name gives the group away.

Experiments (`eval/experiments.py`), each an extra pass of a Jev configuration under changed groups, logged under set version 0 (which no vault set ever has):

1. Group added mid-run (`jev:card@without-exams`): every file is sorted under the reduced set, which is `HELD_OUT_GROUP_ID` removed and the excludes text of `eval/groups.py` `HELD_OUT_EXCLUDES` applied (Lecture notes excludes "homework", Homework excludes "course projects"), so that no remaining description names the held-out group. The user then adds the group, freezes each file that sits in its gold group, and reruns the rest against the full set. Reported: the AUROC of `fits_any` between the held-out group's files and the others, then, at the default cutoffs, how many held-out files go to Unsorted, how many others wrongly do, and the accuracy of the rerun. No file in the data set fits no group, so the `fits_any` cutoff cannot be tuned on it, and the report says so.
2. Wording of `fits_any` (`jev:names_only@without-exams`): the same pass with group names only. The AUROC is the comparison between the two columns.
3. Description edits (`jev:card@edit-<name>`): each entry of `<root>/description_edits.yaml` (`name`, `group_id`, `covers`, optional `excludes`, `target_file_ids`, optional `note`: who wrote the edit, when and why) gives one group a new description. Every file is sorted again and compared with its answer before: targets fixed, and every other file whose answer changed. Notes are printed under the table. When no edit names a target, the report says that the spec's experiment (fix each wrong placement) was not run and why.

Experiments 1 and 2 are computed over every file with text, dev and test together, under the default cutoffs. A pass runs only when the configuration it copies is in `--configs`.
