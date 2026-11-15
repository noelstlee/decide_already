# Phase 0 implementation notes

How the spec in [phase0.md](phase0.md) maps to code. This is the working contract between modules: if code and this file disagree, fix one of them in the same change. Records live in `src/decide_already/types.py`, settings in `config.py`, the backend interface in `backends/base.py`. Read those three files first; they are short.

## Module map

| Module | Does | Step |
| --- | --- | --- |
| `types.py` | Every on-disk record (`Group`, `GroupSet`, `File`, `Placement`, `Label`, `Suggestion`) | 1 |
| `config.py` | `Config` (`.decide/config.yaml`), folder and file names, `env_value` for secrets | 1 |
| `vault.py` | Vault layout, group notes, set versions, dump listing, card notes | 1, 4 |
| `extract/` | Text out of one file: `extract(path) -> Extracted` | 2 |
| `card.py` | The card and its experiment variants: `build_card(...) -> Card` | 2 |
| `questions.py` | The two questions, in wire format, with shuffled options | 3 |
| `log.py` | JSONL append and read, token totals | 3 |
| `backends/base.py` | `Backend`, `SortRequest`, `SortResult`, `BackendError`, `BudgetExceeded`, `state_text` | 3 |
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
- Tests use `tmp_path` vaults built with `vault.init_vault`.
- Originals in the dump folder are opened read-only. Nothing in `src/` writes, moves or deletes there, except the UI upload, which only adds new files.
- Only `backends/jev.py` imports `typesafe_sdk`. Only `eval/baselines/` imports `fastembed` or `anthropic`.
- The model version always comes from `Config.model` and is passed on every call.

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
| Markdown | `.md`, `.markdown` | first `#` heading, else first non-empty line | headings | the file |
| Text | `.txt`, `.rst`, any other readable text | first non-empty line | none | the file |
| Code | `.py`, `.m`, `.r`, `.jl`, `.js`, `.ts`, `.java`, `.c`, `.h`, `.cpp`, `.sh`, `.tex`, `.sql` | module docstring's first line or first comment line | function and class names (`ast` for Python, a simple regex elsewhere) | the file |
| Notebook | `.ipynb` | first markdown heading | markdown headings, then function and class names | cell sources joined; outputs dropped |
| PDF | `.pdf` | document title metadata if sensible, else first non-empty line | the PDF's table of contents, if it has one | text of every page (PyMuPDF) |
| Slides | `.pptx` | first slide's title | slide titles | text of every slide (python-pptx) |
| HTML | `.html`, `.htm` | `<title>` or first `<h1>` | `h1` to `h3` | visible text; only `<main>` or `<article>` when present; scripts, styles and nav dropped (stdlib `html.parser`) |
| Anything else or binary | | none | none | `""` |

Read text as UTF-8 with `errors="replace"`. A file with NUL bytes in its first 8 KB is binary. Images are dropped. A file with no text gets `text=""` and later goes to Unsorted with reason `no_text`. Extractors never raise on a damaged file: they return `text=""`.

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

The token count is an estimate (4 characters per token). The API's own `input_tokens` is logged per call as the real number.

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
| `window_vote` | `{"file": {...}, "text": window}` per 2,000-token window; when there are more than 8 windows, 8 evenly spaced ones including the first and the last | one request per window; probabilities are averaged by `sort.judge` |
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
- A missing key raises `BackendError` with a message that names `TYPESAFE_API_KEY` and `.env`. The key is never logged.
- Before each request, compare the spend so far with `config.max_spend_usd`. Spend is `input tokens / 1e6 * config.usd_per_million_tokens`, counted from the log at start and updated per call. Every call that reports input tokens counts, usable or not, because it was billed. At or over the cap, raise `BudgetExceeded` without sending (and without a log line).
- Probabilities come back keyed by group name. Convert to group ids. `option_order` is the shuffled order as ids.
- Safe to call from several threads: guard the log append and the spend counter with a lock.
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

`request` is exactly what was sent. `response` is the raw JSON body. A call with no usable body (an HTTP error, a timeout) is logged with `status: "error"`, `response: null` and the error text. A call that was answered but cannot be used (another model version answered, an answer is missing) is also `status: "error"`, but keeps its response, tokens and request id.

`log.py`: `append_jsonl(path, record)`, `read_jsonl(path) -> list[dict]`, `total_input_tokens(path) -> int` (every line that has `input_tokens`). A line that is not a JSON object raises `ValueError` naming the file and line.

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
def judge(card: Card, groups: list[Group], set_version: int, backend: Backend, variant: str) -> SortResult
```

| `fits_any` | top probability | top minus second | Outcome |
| --- | --- | --- | --- |
| under `unsorted_fits_any_below` | any | any | Unsorted, `no_group_fits` |
| at or over | under `unsorted_top_below` | any | Unsorted, `weak_top` |
| at or over | at or over | under `close_call_margin` | placed in the top group, `close_call` true |
| at or over | at or over | at or over | placed in the top group |

With one group, the second probability counts as 0. Ties go to the lower group id. A value exactly at a cutoff counts as at or over. `judge` sends one request per state and averages `probs` and `fits_any` when there are several (window vote); tokens and latency are summed, and stay `None` when the backend reports none (the fake does not).

Code alone handles two cases: a card with no states goes to Unsorted with reason `no_text`, `placed_by: code`, no model asked. A `BackendError` is never an answer: the file keeps its previous note, if any, and is reported.

## Card notes (`vault.py`)

```python
def dump_files(root: Path, config: Config) -> list[tuple[str, Path]]   # (file id, path), sorted by id
def note_name(file_id: str) -> str                                     # "/" becomes "__", plus ".md"
def source_link(file_id: str, config: Config) -> str                   # "[[<dump_dir>/<file_id>]]"

class CardNote(StrictModel):
    path: Path
    folder: str          # name of the folder the note sits in
    placement: Placement
    body: str

def read_card_notes(root: Path) -> list[CardNote]
def write_card_note(root: Path, placement: Placement, body: str) -> Path
def folder_for(root: Path, group_id: str | None) -> Path               # group name folder, or Unsorted
def group_id_of_folder(root: Path, folder: str) -> str | None          # raises KeyError for a folder that is no current group
```

- `dump_files` walks the dump folder recursively and skips any path with a component that starts with a dot. The file id is the path relative to the dump folder, with forward slashes.
- A card note is a markdown file whose frontmatter is exactly a `Placement`, in the field order of the model. The body is `render_card(card)`.
- A card note sits in its group's folder (the folder is named after the group) or in `Unsorted/`. The folder a note sits in is the file's current group. `placement.group_id` is the group as of the last write or sync.
- `read_card_notes` looks in every top-level folder except the dump folder, `_groups`, `.decide` and hidden folders. Markdown files whose frontmatter is not a `Placement` are the user's own notes and are ignored.
- `write_card_note` writes into the folder of `placement.group_id` and removes any other card note for the same file id.

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
3. For each dump file: hash it. A frozen note with the same hash is skipped. A frozen note whose hash changed is unfrozen first (label `unfrozen`) and then sorted like any other.
4. Every other file is carded with `config.variant`, judged, decided, and written as a card note with `placed_by: jev`.
5. `BudgetExceeded` stops the run. Files not reached are listed in `errors`.

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

Files are grouped by the folder they sit in. Within a group: close calls first, then by name.

## Corrections (`corrections.py`)

```python
def sync(root: Path) -> SyncReport            # moved, confirmed, unfrozen, orphaned, missing_sources: lists of file ids
def move_file(root: Path, file_ref: str, group_ref: str | None) -> Placement
def freeze(root: Path, ref: str) -> list[str]     # ref is a file or a group; returns the file ids it froze
def unfreeze(root: Path, ref: str) -> list[str]
def suggest(root: Path, backend: Backend) -> list[Suggestion]
```

A file reference is a file id, or a file name that is unique in the vault. A group reference is a group id or name; `None` or `Unsorted` means Unsorted.

| Action | Effect on the note | Label written |
| --- | --- | --- |
| `move_file` | moved to the group's folder, `group_id` set, `placed_by: you`, `frozen: true`, `close_call: false`, `reason: null` | `moved` (none if it is already there; then it is a freeze) |
| `freeze` | `frozen: true` | `confirmed`, `from_group` and `to_group` both the current group |
| `unfreeze` | `frozen: false` | `unfrozen` |
| `suggest` | nothing moves | none; one line per differing file in `.decide/suggestions.jsonl` |

Labels record `jev_probs` and `jev_set_version` from the note as it was, and `set_version` as the current set version. A file's active label is its last line in `labels.jsonl` when that line is `moved` or `confirmed`.

`sync` picks up edits made by hand, in this order per card note:

1. The note sits in a folder that is neither a current group nor `Unsorted` (its group was deleted): move it to `Unsorted/`, `group_id: null`, `placed_by: code`, `frozen: false`. If it had an active label, write `unfrozen`.
2. The folder's group differs from `placement.group_id` (a drag): write `moved`, then update the note as `move_file` does.
3. `frozen` is true and the file has no active label (set by hand): write `confirmed`.
4. `frozen` is false and the file has an active label (cleared by hand): write `unfrozen`.

Card notes whose source file is gone from the dump are listed in `missing_sources` and left alone. `sync` also calls `vault.sync_group_set`, so edits to group notes become a new set version.

`suggest` asks the backend about every frozen file, runs `decide`, and reports the files whose decision differs from their current group.

## CLI (`cli.py`)

`p0 init`, `p0 group add|edit|rename|delete|list` (step 1), `p0 sort`, `p0 board` (step 4), `p0 move <file> <group>`, `p0 freeze <file or group>`, `p0 unfreeze <file or group>`, `p0 suggest`, `p0 sync` (step 5), `p0 ui [--port 8765] [--no-open]` (UI). Every command takes `--vault`. Commands print short plain lines and exit 1 with `error: ...` on a known failure.

## UI (`ui/`)

`create_app(root: Path, backend_factory: Callable[[], Backend] | None = None) -> FastAPI` in `ui/server.py`. Tests pass a factory that returns the fake backend. `p0 ui` serves it on `127.0.0.1` only. The page is plain HTML, CSS and JS in `ui/static/`, with no build step and no external requests.

| Method and path | Does |
| --- | --- |
| `GET /` | the page |
| `GET /api/board` | runs `sync`, then returns the board plus vault info: path, dump folder, model, set version, spend and cap |
| `GET /api/files/{file_id:path}` | one file: its placement, probabilities by group, and the card text |
| `GET /api/original/{file_id:path}` | the original file, read-only |
| `POST /api/sort` | starts a sort in the background; 409 if one is running |
| `GET /api/sort/status` | `{running, done, total, report, error}` for the progress bar |
| `POST /api/move` | `{file_id, group_id}`; `group_id: null` is Unsorted |
| `POST /api/freeze`, `POST /api/unfreeze` | `{ref}`: a file id or a group id |
| `POST /api/groups` | `{name, covers, excludes}` |
| `PATCH /api/groups/{group_id}` | any of `name`, `covers`, `excludes` |
| `DELETE /api/groups/{group_id}` | deletes the group |
| `PUT /api/dump/{name}` | adds one file to the dump folder from the raw request body; never overwrites (409) and never accepts a path |
| `POST /api/suggest` | runs `suggest` and returns the list |

Errors come back as `{"detail": "..."}` with a 4xx status. File ids are checked against `dump_files`, so no request can reach outside the vault.

The page: one column per group plus Unsorted, a card per file. Drag a card to another column to move it (which freezes it). A card shows the file name, type, the top probability, and marks for close call, frozen, and moved by you. Clicking a card opens a side panel with every group's probability, `fits_any`, the reason when Unsorted, the card text, and buttons to freeze, unfreeze or open the original. The header has the vault name, set version, model, spend, a Sort button with progress, and Add group. Dropping files from the desktop onto the page adds them to the dump. Group descriptions are edited in place. Minimal look: system fonts, plenty of white space, one accent color, light and dark from the system setting, keyboard reachable.

## Evaluation (`eval/`)

Everything the evaluation downloads or writes goes under the gitignored `sandbox/`. The evaluation vault is `sandbox/eval/`, a normal vault with the groups from `eval/groups.py`.

Dataset (`eval/dataset.py`, run as `python -m decide_already.eval.dataset`):

- Downloads `6.867-fall-2006.zip` from ocw.mit.edu (CC BY-NC-SA 4.0) and the lab notebooks, solutions and helper modules of the MIT 6.S191 labs from GitHub `MITDeepLearning/introtodeeplearning` (MIT license), caches them in `sandbox/datasets/`, and copies the files to sort into `sandbox/eval/_dump/`.
- Gold labels come from the sources' own structure and never from a model: OCW's resource metadata for 6.867, the repo's folders for the labs. They are written before any sort, to `sandbox/eval/gold.jsonl`, one line per file: `{file_id, gold_group, source_url, source_type, license, sha256, split}`. `gold_group` is a group id, or `null` for a file that fits no group.
- `split` is `dev` for 20% of the files and `test` for the rest, stratified by gold group, with a fixed seed.
- `sandbox/eval/DATASET.md` lists sources, licenses, counts per group and the mapping used.
- For other modules: `read_gold(eval_root) -> list[GoldLabel]` and `build_dataset(root, fetch=download) -> list[GoldLabel]`. The network step is the one `download(url, dest)` function.
- The labs are pinned to a commit. Course pages are rendered from their metadata without the site's navigation. Pages that only list resources are left out. The build fails when the dump holds a file with no gold label.
- As built on 2026-11-15: 95 files. Homework 52, Lecture notes 24, Exams 17, Projects 1, Logistics 1. Dev 18, test 77. Four zip archives (gold: Homework, all in test) have no extractable text.

Baselines (`eval/baselines/`), each a `Backend`:

| Baseline | How it sorts |
| --- | --- |
| `rules.py` | file name keywords and extensions per group; one-hot `probs`; `fits_any` 0 when no rule matches |
| `embed.py` | cosine similarity between the card text (`state_text`) and each group's description, softmax into `probs`; a local model through `fastembed`; `fits_any` 1.0 |
| `llm.py` | one Claude model through the Anthropic API, zero-shot, same card and groups; key from `env_value("ANTHROPIC_API_KEY")`; unavailable without a key |

Metrics (`eval/metrics.py`): top-1 accuracy overall and per group, macro-F1, accuracy among placed files, share Unsorted and how many of those had the right top group, expected calibration error of the top probability, cost and latency per file.

Runner (`eval/run.py`, run as `python -m decide_already.eval.run`): runs every configuration (Jev with each of the five variants, plus each baseline on the card) over the dataset, caches every prediction in `sandbox/eval/results/` so nothing is paid for twice, tunes the two Unsorted cutoffs on the dev split, reports on the test split, and writes `metrics.json` and `report.md`. All Jev calls go through one `JevBackend` logging to `sandbox/eval/.decide/jev.jsonl`, so the spend cap covers the whole evaluation.

Scoring rules:

- The headline number is top-1 accuracy on the test split over files that have extractable text, with the default cutoffs. Every configuration is scored on the same files.
- Files with no extractable text never reach a model. They are counted and reported on their own line, and a second accuracy over all files (where they count as wrong for every content-based configuration) is given next to the headline.
- A majority-class row (always the largest gold group) is reported for context.
- Tuned cutoffs are reported next to the defaults, never instead of them. The dev split is small.

Experiments (`eval/experiments.py`): group added mid-run (sort with `HELD_OUT_GROUP_ID` removed, then add it back and rerun the unfrozen files), and move versus description edit.
