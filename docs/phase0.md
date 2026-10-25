# Phase 0 blueprint: Jev sorts the dump

> Status: ready to build. This is the working copy Claude Code builds from.
> Source: the Claude Docs doc "Phase 0 blueprint: Jev sorts the dump" (https://claude.ai/artifact/J6MiX9btbtqJYyAGu76RZN), exported 2026-10-24, plus two changes Stan made in chat the same night that the Claude Docs version does not have yet:
> 1. **Smoke test on personal files first** (new step 0), before any course files.
> 2. **Local-first, Obsidian-compatible vault** as the storage design (replaces the `phase0/` state-file layout). Direction accepted; layout details are proposed and listed in Open questions.
>
> "You" means the person using the tool (Stan, for now).

Phase 0 tests one thing before any knowledge base exists: can Jev, with no reasoning LLM, sort a dump of files into groups you name? It also measures how often you have to correct it, and hands Phase 1 a working Jev client, a correction loop and the first labeled data.

## Scope

Phase 0 is one Jev call per file, a loop for correcting it, and the numbers that say whether it worked. Everything else waits for Phase 1.

| In Phase 0 | Left for Phase 1 or later |
| --- | --- |
| A dump folder you drop whole files into | Splitting files into sections |
| Categories you name and describe, one level deep | The fixed classes, relations and tiers in code |
| Text extraction and a short card per file, built by code | Search, indexes, chunking |
| One Jev Choice plus one yes/no per file | Update relations, routing, reranking, leak checks |
| Freeze, move, add or edit a category, rerun | Images, multi-label tags, subcategories |
| Personal files for the smoke test, then public course files | CS 4641/7641 files (need the instructor's OK) |
| Commands plus an Obsidian vault as the board | A custom drag-and-drop UI |

No reasoning LLM sits in the sorting path; one appears only as a baseline. Phase 1 inherits the Jev client and its log, the correction loop, the card format for D1, and every correction you make as a labeled example.

## How it works

The loop: **dump folder** (whole files, never moved or edited) → **card builder** (code: extracts text, about 3k tokens per file, no LLM) → **Jev**, one request per file (a Choice over your groups plus a yes/no "does any group fit?", every call logged) → **board** (your groups with probabilities, Unsorted when no group fits well, close calls flagged) → **you** (add or edit a group, move a file, which freezes it, freeze or unfreeze) → **rerun** (unfrozen files only, against the full current group set). Every move and freeze is written to the labels file, which scores the sort and seeds D1 in Phase 1.

A frozen file never goes back to Jev unless you unfreeze it or its content changes. A rerun sends only unfrozen files, scored against the full current set of groups.

## Walkthrough

Two corrections take six files from three groups to four, and every file Jev placed right stays put.

1. **Name the groups.** You create HW, Logistics and Project, each with a one-line description. This is category set v1.
2. **Dump the files.** HW skeleton code, the code solution, project examples, the syllabus, lecture slides and the project logistics sheet go into the dump folder.
3. **Code builds a card per file.** It extracts the text and writes a short card: file name, type, outline and opening text.
4. **Jev sorts all six against v1.** HW gets the skeleton and the solution. Logistics gets the syllabus and the lecture slides. Project gets the examples and the logistics sheet.
5. **You add a group.** Lecture slides don't belong in Logistics, so you add "Lecture slides and notes" (set v2) and freeze the other five files.
6. **Rerun.** Only the lecture slides go back to Jev, choosing among all four groups, and they land in the new one.
7. **You fix the project logistics sheet, one of two ways.**
    1. Move it to Logistics. That fixes one file, freezes it, and logs a correction.
    2. Or edit Logistics to cover "deadlines and policies for anything, including projects", unfreeze, and rerun. That fixes the rule for every future file like it.
8. **Freeze everything.** The final groups are now your labels, and the log shows what Jev got right on its own.

## The Jev call

Each file gets one request with two questions: a Choice over your groups, and a yes/no on whether any group clearly fits. Both read the same card, so the second question costs almost nothing.

```json
{
  "model": "jev-1.14.0",
  "state": "File: hw3/hw3_solution.py\nType: Python code\nOutline: rbf_kernel(), train_svm(), predict()\nStart:\n...",
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "Which group of course materials does this document belong in?",
      "criteria": {
        "HW": {"covers": "homework specs, starter code, solutions, tests", "excludes": "project files"},
        "Logistics": {"covers": "syllabus, schedule, deadlines, grading and submission rules, including for projects"},
        "Project": {"covers": "project specs, examples, starter material", "excludes": "deadlines and policies"},
        "Lecture slides and notes": {"covers": "lecture slides, lecture notes"}
      }
    },
    "fits_any": {
      "type": "noul",
      "instructions": "Does this document clearly belong to one of these groups: HW, Logistics, Project, Lecture slides and notes?"
    }
  }
}
```

The `instructions` wording is generic in the tool ("Which group does this document belong in?"); the course wording above is the test-run example. Response fields: a Choice answer has `choice`, `confidence` and `probabilities` (keyed by option name); a Noul answer has `noul`, the probability of yes. See [jev-notes.md](jev-notes.md).

Rules the code enforces:

- **Every group has a description.** Its name becomes the option and its description the criteria, as covers and excludes. Option names can outweigh descriptions, so names should say what they hold.
- **Unsorted is decided in code, not by Jev.** A file goes to Unsorted when fits_any is under 0.5 or the top group is under 0.6. Jev rarely picks a "none of these" option, so Phase 0 doesn't offer one. Both cutoffs are starting values, tuned on the dev split.
- **Close calls are flagged.** When the top two groups are within 0.15, the file is placed but marked so you check it first.
- **Reruns send only unfrozen files, always against the full current set.**
- **The category set is versioned.** Any add, rename, description edit or delete bumps the version, and every result records which version it ran under. Probabilities from different versions aren't compared.
- **Option order is shuffled per call** and the order is logged; Jev leans slightly toward the first option.
- **One model version is pinned** for the whole phase.

At about 3k tokens per card, sorting 300 files is roughly 0.9M tokens, about $0.04.

## What Jev sees

Jev gets a card of about 3k tokens per file, not the whole file. One question can read at most 32k tokens, and accuracy drops as the state fills with text that doesn't bear on the decision. A top-level group needs the gist, which a card carries.

| Card field | Built from | Example |
| --- | --- | --- |
| File name and path | The dump folder | hw3/hw3_solution.py |
| Type | Extension, page or slide count | Python code; PDF, 12 pages; slides, 24 |
| Title | First heading, first slide title, or module docstring | Homework 3: Kernel Methods |
| Outline | Headings, slide titles, or function and class names, up to 40 lines | rbf_kernel(), train_svm(), predict() |
| Opening text | The first 1,500 tokens | The first page or first cells |
| Excerpts | Three 300-token passages at 40%, 70% and 95% of the text | Catches files whose start is boilerplate |

Text comes from plain extractors: PyMuPDF for PDFs, python-pptx for slides, cell text for notebooks, and the raw file for code and markdown. Images are dropped. A file with no extractable text, such as a scanned PDF, goes straight to Unsorted with that reason.

The card is the only part of a file that leaves the machine.

### Variants to compare

| Variant | What Jev sees | What it tells you |
| --- | --- | --- |
| Card (default) | The card above | The baseline input |
| Card without file name | The card with name and path removed | How much the file name alone decides |
| Full text | All extracted text, cut at 30k tokens | Whether more text helps or hurts |
| Window vote | 2k-token windows, up to 8, one request each, probabilities averaged | Whether long files need every part read; windows that disagree mark the file as mixed |
| Names only | Groups without descriptions | What the descriptions add |

The winner becomes the input format for D1 in Phase 1.

## Your actions

Commands come first; Obsidian (or Finder) is the board, and a custom drag-and-drop UI can come later by calling the same functions. Only moving and freezing turn a file's group into a label.

| Action | Command | What happens | What gets logged |
| --- | --- | --- | --- |
| Add a group | `p0 group add "Lecture slides and notes" --covers ...` | New set version; nothing moves until you rerun | The new set |
| Edit a group | `p0 group edit Logistics --covers ...` (or edit `_groups/Logistics.md`) | New set version | Old and new description |
| Rename or delete a group | `p0 group rename` / `p0 group delete` | New set version; a deleted group's files become unfrozen and Unsorted | The new set |
| Sort or rerun | `p0 sort` | Every unfrozen file goes to Jev against the current set | Each request, answer and probability |
| Move a file | `p0 move <file> <group>`, or drag its card note to another group folder, then `p0 sync` | Places the file and freezes it | A correction: from, to, Jev's probabilities, set version |
| Freeze | `p0 freeze <file or group>` (or set `frozen: true` in the card note) | Jev won't re-sort it | A confirmation: Jev's pick accepted |
| Unfreeze | `p0 unfreeze <file or group>` | Back in the next rerun | The unfreeze |
| Check frozen files | `p0 suggest` | Lists frozen files Jev would now place elsewhere; nothing moves | Suggestions, kept apart from labels |
| Pick up edits made in Obsidian | `p0 sync` | Reads moved notes, frontmatter edits and group-note edits | Moves and freezes as labels; group edits as a new set |
| See the board | `p0 board` | Groups, Unsorted and close calls, with probabilities | Nothing |

`p0` is a working name for the CLI.

## Data model: a local-first vault

Everything lives in one folder on your machine that Obsidian can open as a vault. No database, no server. Your original files are never moved or edited; each one gets a small **card note** (markdown with YAML frontmatter, like an ADR note), and the folder a card note sits in is the file's current group.

```text
my-vault/
  _dump/                     drop files here; originals are never moved or edited
  _groups/
    HW.md                    one note per group: frontmatter covers, excludes
    Logistics.md
  HW/
    hw3_solution.py.md       one card note per file, in its current group's folder
  Logistics/
  Unsorted/
  .decide/                   tool state, not meant for hand editing
    config.yaml              model pin, cutoffs, backend (jev now; openjev or clm later)
    sets/v1.yaml, v2.yaml    every category set version, kept
    labels.jsonl             every move and confirmation: the labeled data
    jev.jsonl                every request, answer, probability and option order
    cache.sqlite             optional later; rebuildable from the files
```

A card note:

```markdown
---
source: "[[_dump/hw3_solution.py]]"
content_hash: 9f2c1e...
jev_group: HW
probs: {HW: 0.91, Project: 0.06, Logistics: 0.03}
fits_any: 0.94
close_call: false
frozen: false
placed_by: jev
set_version: 2
variant: card
model: jev-1.14.0
---
# hw3_solution.py
Type: Python code
Outline: rbf_kernel(), train_svm(), predict()
...
```

Sync rules:

- **The folder is the current group.** If a card note sits in a folder other than its `jev_group` and the move is not yet logged, `p0 sync` logs a "moved" label and sets `frozen: true`.
- **`frozen` in frontmatter is honored.** Setting it to true on a note Jev placed logs a "confirmed" label.
- **Group notes are the category set.** Any change to `_groups/` bumps the set version on the next run.
- **A changed source file** (new `content_hash`) gets a new card; whether it stays frozen is an open question.
- **Secrets never go in the vault.** The API key comes from the environment or a gitignored `.env`.

The records, as Pydantic models:

```python
# decide_already/types.py (Phase 0)
from datetime import datetime
from typing import Literal
from pydantic import BaseModel

class Group(BaseModel):
    id: str                       # slug, stable across renames
    name: str                     # the option name Jev sees
    covers: str
    excludes: str | None = None

class GroupSet(BaseModel):
    version: int                  # bumps on any add, rename, edit or delete
    groups: list[Group]

class File(BaseModel):
    id: str                       # path in _dump/
    content_hash: str
    file_type: str
    card_tokens: int

class Placement(BaseModel):       # stored in the card note's frontmatter
    file_id: str
    group_id: str | None          # None means Unsorted
    placed_by: Literal["jev", "you"]
    frozen: bool
    close_call: bool
    set_version: int
    probs: dict[str, float] | None    # Jev's distribution over group ids
    fits_any: float | None
    variant: str                  # "card", "full_text", "window_vote", ...
    model: str                    # e.g. "jev-1.14.0"

class Label(BaseModel):           # one line in .decide/labels.jsonl
    file_id: str
    kind: Literal["moved", "confirmed"]
    from_group: str | None
    to_group: str
    jev_probs: dict[str, float] | None
    set_version: int
    at: datetime
```

## Evaluation

Phase 0 passes if the card-based sort beats file-name rules and embedding similarity on labels you made before seeing Jev's picks. The correction rate then says whether sorting by hand would have been faster.

### Dataset

- **Smoke test (step 0):** a few dozen of your own files with 3 to 5 groups you pick. No metrics, just "does the loop work and do the picks look sane". Only use files you are fine sending to TypeSafe as cards.
- **Public course:** [MIT OCW 6.867 Machine Learning, Fall 2006](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/resources/hw2_soln/), under CC BY-NC-SA 4.0. It has a syllabus, lecture notes, problem sets and solutions, exams and projects, all as PDFs, with a whole-course download.
- **Code files:** 6.867 has none, so add starter code and solutions from a public course repo to test HW code.
- **CS 4641/7641:** joins once the instructor OKs sending it to TypeSafe.
- **Blind labels:** before the first sort, you place every file (or 100, if there are more) into the starting groups yourself. These are the gold labels. The final groups after corrections are reported separately, since seeing Jev's pick first pulls your label toward it.
- **Split:** 20% dev for the cutoffs, 80% test.

### Metrics

- Accuracy against the blind labels, overall and per group
- Correction rate: the share of files you moved
- Unsorted: the share sent there, and how many of those Jev had right anyway
- Calibration: expected calibration error of the top group's probability
- Rerun accuracy: after a group is added, the share of unfrozen files placed right
- Cost and latency per file

### Baselines

| Baseline | How it sorts |
| --- | --- |
| File-name rules | Keywords and extensions: "soln" or .py to HW, "syllabus" to Logistics |
| Embedding similarity | The card's embedding against each group's description; nearest wins |
| LLM classifier | One frontier API model, zero-shot, same card and groups |

### Experiments

- **Input variants:** the five rows in What Jev sees, on the same files.
- **Move vs description edit:** for each wrong placement, fix it with a description edit instead, then count which other files changed and whether they changed for the better.
- **Group added mid-run:** replay the walkthrough on the full dataset. Start with three groups, add the fourth, and measure rerun accuracy.

### Go/no-go (proposed defaults)

| Result | Then |
| --- | --- |
| Card accuracy of 90% or more, above both non-LLM baselines | Start Phase 1 as designed |
| Full text or window vote beats the card by 5 points or more | Use that input for D1 |
| Jev doesn't beat embedding similarity | Rethink Jev's role on the write path before Phase 1 |

## Build order

Steps 0 to 4 fit in one night. The go/no-go call is proposed for 2026-11-01, the day before Phase 1's ingest calls start.

0. **Smoke test setup.** Pick a few dozen personal files and 3 to 5 groups. Done when they sit in a test vault's `_dump/` outside the repo (or under the gitignored `sandbox/`).
1. **Scaffold.** Package layout, `types.py`, config, and group notes with three groups. Done when `p0 group add` writes a new set version.
2. **Extractors and cards.** PDFs, slides, notebooks, code and markdown. Done when every file in the dump has a card under 3k tokens.
3. **Jev client.** One sort call with the model pinned, shuffled options, retries on rate limits, and every request logged. Done when one made-up file comes back with probabilities in the log.
4. **Sort and board.** `p0 sort` with the Unsorted and close-call rules, card notes written into group folders, plus `p0 board`. Done when the smoke-test files sort and print.
5. **Corrections.** Move, freeze, unfreeze, suggest, sync, and reruns of unfrozen files only. Done when the walkthrough runs end to end (including a drag in Obsidian) and every move and confirmation is in the labels file.
6. **Dataset and blind labels.** Download 6.867, add code files, and label everything before any sort. Done when every file has a gold label.
7. **Baselines and variants.** The three baselines and five input variants. Done when one script writes every metric for every configuration.
8. **Decision.** Fill in the go/no-go table and carry the winning card format into Phase 1's D1.

## Open questions

- [ ] Model version: pin `jev-1.14.0` (newest, out Oct 14) or `jev-1.13.0` to match the Phase 1 blueprint? Use one for both phases.
- [ ] Starting groups and their descriptions, for the smoke test and for the course test.
- [ ] Which public repo supplies starter code and solution files.
- [ ] Keep the starting cutoffs: Unsorted under 0.5 fits_any or 0.6 top group, close call within 0.15?
- [ ] One group per file only, or also a yes/no tag per group for files that span two, like the project logistics sheet?
- [ ] Should an edited file lose its freeze and be re-sorted?
- [ ] Vault layout details: folder names (`_dump/`, `_groups/`, `.decide/`), card note naming when two files share a name, and the CLI name.
- [ ] Run CLM as a second backend in the same harness?
- [ ] Go/no-go numbers: 90% accuracy and a 5-point margin.

## Sources

- [TypeSafe docs: Models](https://docs.typesafe.ai/models): versions, price, the 32k and 64k limits, the long-context header
- [TypeSafe docs: Choice](https://docs.typesafe.ai/primitives/choice): 255 options, covers and excludes, the companion yes/no for "none fits"
- [TypeSafe docs: API](https://docs.typesafe.ai/api): endpoint, auth header, response fields
- [TypeSafe docs: Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13): accuracy loss from unrelated state, first-option lean
- [arXiv 2609.39496](https://arxiv.org/abs/2609.39496): Jev rarely picks an explicit rejection option
- [arXiv 2609.26758](https://arxiv.org/abs/2609.26758): option names can outweigh their definitions
- [arXiv 2610.00346](https://arxiv.org/abs/2610.00346): calibration shifts as the number of options changes
- [MIT OCW 6.867 Machine Learning, Fall 2006](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/resources/hw2_soln/): the public test course
