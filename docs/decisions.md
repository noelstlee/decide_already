# Decisions so far

Newest first. "Decided" means Stan made the call; "Proposed" means it is in a blueprint but Stan has not confirmed it. Change a row only when Stan decides, and add the date.

| Date | Decision | Status | Why |
| --- | --- | --- | --- |
| 2026-11-15 | Phase 0 gets a minimal local web UI (`p0 ui`): FastAPI and uvicorn serving one plain HTML, CSS and JS page over the same functions as the CLI | Decided | Stan wants to see the loop in a UI and judge the UX; the CLI and Obsidian stay |
| 2026-11-15 | The smoke test runs on 10 synthetic course files of mixed types, not personal files. `smoke_test/mock_data` (real CS 4641 material) is not sent anywhere | Decided | Small, safe to send, exercises every extractor; course material still needs the instructor's OK |
| 2026-11-15 | Public evaluation set: MIT OCW 6.867 (CC BY-NC-SA) plus the MIT 6.S191 lab files (MIT license). Five groups: Lecture notes, Homework, Exams, Projects, Logistics | Decided | Stan delegated the dataset and the groups; 6.867 has no code, the labs add notebooks and Python |
| 2026-11-15 | Gold labels come from the sources' own metadata (OCW resource types, repo folders), written by code before any sort; Stan spot-checks a sample | Decided | Blind by construction, and the run can finish without hand labeling. Hand labels can replace them later |
| 2026-11-15 | Baselines: file-name rules; embeddings from a local model (fastembed, bge-small); an LLM classifier through the Anthropic API, pending a key | Decided | The two non-LLM baselines the go/no-go depends on need no key |
| 2026-11-15 | Pin `jev-1.14.0` for Phase 0 | Decided | Newest; both exact versions were verified live; the companion yes/no is the documented pattern for 1.14 |
| 2026-11-15 | Spend cap per vault (`max_spend_usd`, default 5.00), enforced by the Jev backend from its own log | Decided | Bring-your-own-key tool; Stan set a $5 cap for the Phase 0 run |
| 2026-11-15 | Fakes live in `tests/` only; the config no longer offers a fake backend | Decided | Stan's rule: no mocking outside the test suite |
| 2026-11-15 | Record changes for the correction loop: `Placement` gains `source` and `file_type`; `Label` gains the kind `unfrozen`, a nullable `to_group` and `jev_set_version`; `Suggestion` is new; card note frontmatter uses group ids, not names | Proposed (implemented, confirm) | Needed to implement the spec's own action table; ids survive renames |
| 2026-11-15 | A file whose content changed loses its freeze and is re-sorted | Proposed (implemented, confirm) | The spec's "How it works" says so; it was also listed as an open question |
| 2026-11-15 | The `fits_any` question carries the group descriptions itself | Proposed (implemented, confirm) | Each Jev question is answered on its own and cannot see the Choice's criteria |
| 2026-11-15 | A group note renamed by hand (in Obsidian) is a rename, not a delete: the group keeps its id and its folder follows the new name | Proposed (implemented, confirm) | Without it the next sync sent every file of the group to Unsorted and withdrew the user's labels |
| 2026-11-15 | Every correction (`move`, `freeze`, `unfreeze`, `suggest`) runs `sync` first; the card note is written before the label | Proposed (implemented, confirm) | A drag by hand plus a command gives the same labels as two commands; a note that cannot be written leaves no label |
| 2026-11-15 | A card note whose original is gone from the dump is listed and otherwise left alone, by every sync rule | Proposed (implemented, confirm) | The tool has nothing to card or check it against; explicit move and freeze still work on it |
| 2026-11-15 | In `freeze` and `unfreeze`, a name that is both a group and a file means the group | Proposed (implemented, confirm) | One rule for the CLI and the UI; `p0 move` still reaches the file |
| 2026-11-15 | The UI server refuses requests whose Host is not 127.0.0.1 or localhost or whose Origin is another site, and refuses changes (409) while a sort runs | Proposed (implemented, confirm) | Any website open in the same browser could otherwise read originals or start a paid sort; a running sort would write over the change |
| 2026-11-15 | A file a configuration could not answer (an API error) is left out of that row of the evaluation and named in the caveats, not counted as wrong | Proposed (implemented, confirm) | A failure is not an answer; `n` is shown per row so uneven rows are visible |
| 2026-10-25 | Jev client: the official `typesafe-sdk`, pinned at 0.7.5, imported only inside `backends/jev.py`; raw httpx not used | Decided | Official and MIT; Pydantic v2; retries honor Retry-After; raw response exposed for the log; no telemetry; option order preserved. Early (two breaking changes in Sept 2026), so pin exactly |
| 2026-10-25 | Stack confirmed as in `CLAUDE.md`: Python 3.12 via uv, Pydantic v2, Typer, PyYAML, pytest, ruff; PDF and slide extractors join with the public dataset | Decided | Approved with the Phase 0 build plan |
| 2026-10-24 | Repo `docs/` is the working copy of the design; the Claude Docs blueprints are snapshots | Proposed | Claude Code reads the repo; one place to edit avoids drift |
| 2026-10-24 | CLM is not a replacement for Jev in Phase 0 or 1; it is a candidate second backend and Phase 2 local model | Proposed | Open (Apache 2.0) and local, but evidence is thin and off-task; weak on subtle judgments in the one independent test; 2048-token default context |
| 2026-10-24 | Smoke-test Phase 0 on personal files before any course files | Decided | Checks the concept fast, with no instructor OK needed |
| 2026-10-24 | Local-first, Obsidian-compatible vault; no hosted database; users bring their own Jev key; backend swappable; LLM plug-in later through the vault or MCP | Decided (direction); layout proposed | Privacy, zero setup, git-friendly, matches how people already use Obsidian |
| 2026-10-24 | Unsorted is decided in code from a fits_any yes/no and thresholds; no "none" option | Proposed | Jev rarely picks a rejection option |
| 2026-10-24 | Jev sees a ~3k-token card per file, not the whole file; full text and window voting are tested as variants | Proposed | 32k limit per question; accuracy drops with unrelated state |
| 2026-10-24 | Add Phase 0: Jev sorts a dump into groups the user defines, with freeze, move and rerun | Decided | Cheapest test of whether Jev classifies real files well before building Phase 1 |
| 2026-10-14 | Phase 1 scope: all four Jev role groups (write path, router, guardrails, rerank and sufficiency); CS 4641/7641 Fall 2025 and Fall 2026; stop at ranked evidence and decisions, no writer LLM; hosted Jev only, zero-shot, no fine-tuning, no local hosting | Decided | Proof of concept first; deployment questions wait for results |
| 2026-10-14 | Student data (Ed posts) stays on Georgia Tech hardware and needs IRB and Registrar approval; not used in Phase 1 | Decided | FERPA |
| 2026-10-14 | Code handles facts, Jev handles judgments; Jev only picks among options fixed in code; a person approves anything that retires or changes a node | Decided | Jev is weak at dates and numbers and can be flipped by context |
| 2026-10-14 | Ontology-lite: about 11 classes and 9 relations defined in code, validity windows, retire instead of delete; no OWL stack | Leaning, adopted in the Phase 1 design | Course structure is known in advance; owning the schema keeps comparisons controlled |
| 2026-10-14 | One markdown file per node with Pydantic-checked frontmatter, in git; indexes are build outputs | Adopted in the Phase 1 design | Readable in Obsidian; git history gives the change log |
| 2026-10-14 | Visibility tiers: public, staff, hidden; solutions and tests hidden by rule; Jev may only move a node toward hidden | Leaning | A leaked solution costs far more than a false alarm |
| 2026-10-14 | Text first, images later | Decided | Jev is text only |
| 2026-10-14 | Open-source a general tool; the AI TA for CS 4641/7641 is the benchmark and evaluation use case; research-focused, not HCI; first users are TAs | Decided | Stan's goals for the MS project and the repo |
| 2026-10-14 | Problem: a knowledge base agents can query that stays correct as documents change, so agents don't answer from stale or wrong material | Decided | Restated from the original proposal (a Socratic AI TA with ontology-typed memory) |

## Still open

- The close call cutoffs. With `unsorted_top_below` 0.6 and `close_call_margin` 0.15 a close call can never happen when the probabilities sum to 1 (a top of 0.6 leaves at most 0.4 for the second, a gap of 0.2). Either lower the top cutoff under 0.575, raise the margin, or define a close call another way. Found 2026-11-15; the defaults are unchanged.
- License for the repo. Note: PyMuPDF is AGPL-3.0 (or commercial), which constrains the choice; pypdf (BSD) is a possible swap, isolated to `extract/`.
- CLI and tool-folder names (`p0`, `.decide/`).
- Whether design docs stay public in this repo.
- New dates for the Phase 0 go/no-go (the proposed 2026-11-01 has passed) and their effect on Phase 1.
- Phase 2 choices: see the last section of [phase1.md](phase1.md).
