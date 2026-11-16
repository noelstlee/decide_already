# Phase 0 results

Run on 2026-11-15 with `jev-1.14.0`, default cutoffs, group set version 5 of the evaluation vault. Everything here can be rebuilt from `sandbox/eval/results/` (the cache, `metrics.json` and the full `report.md`). `sandbox/` is gitignored, so the numbers that matter are copied into this file.

The go/no-go decision is Stan's. This file gives the numbers, what they can support, and a recommendation.

## Short version

- The sorter works end to end, from the command line and from the web page, against the real API. The whole run cost $0.14 of the $5 cap.
- On the public course set Jev placed every test file that has text in its gold group (73 of 73). So did every other way of feeding it the file, and a table of file-name keywords got 72 of 73. **The data set is too easy to carry the go/no-go.** Jev against the rules differs on one file, and the input variants cannot be told apart at all.
- What the data does show: Jev is far ahead of embedding similarity (30 files, all one way), group descriptions matter (6 files, all one way), and a call costs about $0.00012 and 90 ms per file.
- The Unsorted rule needs work before Phase 1 leans on it. With a group held out, the fits-any score ranked its files below the others almost perfectly (AUROC 0.996), but the default cutoff of 0.5 held back only 8 of the 17. A cutoff of 0.7 would have held back all 17 at the cost of 1 of 74 other files, and this data set has nothing to tune that on.
- A review by independent agents recomputed all 628 numbers of the report with their own code and found no mismatch. It also found three serious bugs and a confound in the first version of the held-out experiment. All are fixed, and the numbers below are from after the fixes.

## What ran

| Part | What | Jev requests | Cost |
| --- | --- | --- | --- |
| Smoke test | 10 synthetic files, the full walkthrough by command line and in the page | 22 | $0.0017 |
| Evaluation | 95 public course files, 5 input variants, 3 experiments with 4 extra passes (2 of them run twice) | 1,119 | $0.1374 |

No request failed. Every request is in the vaults' `jev.jsonl` with its option order, the answer, the model that answered and the set version. The LLM baseline was not run: it needs an Anthropic key.

## Smoke test

Ten made-up course files of seven types (`scripts/make_smoke_vault.py`), three groups to start with: HW, Logistics, Project.

| Step | Result |
| --- | --- |
| First sort, three groups | 7 placed, 3 in Unsorted: both lecture files and the apartment checklist, all for "no group fits" |
| Add the group "Lecture slides and notes", freeze HW, sort again | The 3 frozen files were skipped. Both lecture files went to the new group at 1.00. The checklist stayed in Unsorted |
| `p0 move` the project logistics sheet to Project | Moved, frozen, one `moved` label |
| Drag a card note to another folder by hand, then `p0 sync` | Picked up as a move, one `moved` label |
| The page (`p0 ui`) | Board, side panel, a card dragged between lanes, Sort with progress, the group form, freeze and unfreeze: all worked against the live vault, in light and dark |

Two things to know from the answers:

- The off-topic file was a close thing. The apartment checklist scored 0.45, 0.40 and 0.45 on fits-any in three runs, against a cutoff of 0.50.
- The same file can score differently from run to run. `project_guidelines.pdf` got 0.75 for Project in one run and 0.90 in the next, under the same groups. The option order is shuffled per call, so this is most likely order sensitivity.

## The evaluation data

MIT OCW 6.867 (Fall 2006) plus the lab files of MIT 6.S191: 95 files. Gold labels come from the sources' own structure (OCW resource types, repository folders), written by code before any sort.

| Group | Files | In the test split, with text |
| --- | --- | --- |
| Homework | 52 | 38 |
| Lecture notes | 24 | 19 |
| Exams | 17 | 14 |
| Projects | 1 | 1 |
| Logistics | 1 | 1 |

Dev 18 files, test 77. Four zip archives (gold: Homework, all in test) have no text a card can be built from, so no content-based method is asked about them.

## Headline

Test split, the 73 files with text, default cutoffs. Top-1 is the share of files whose top group is the gold group; it ignores the cutoffs.

| Configuration | Top-1 | Placed | Placed right | Unsorted | Input tokens | USD per file | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jev, card | 100.0% (73) | 100.0% | 100.0% | 0.0% | 2,910 | $0.000122 | 90 |
| Jev, card without file name | 100.0% (73) | 100.0% | 100.0% | 0.0% | 2,877 | $0.000121 | 94 |
| Jev, full text | 100.0% (73) | 100.0% | 100.0% | 0.0% | 4,713 | $0.000198 | 92 |
| Jev, window vote | 100.0% (73) | 100.0% | 100.0% | 0.0% | 5,763 | $0.000242 | 169 |
| Jev, card, group names only | 91.8% (67) | 95.9% | 95.7% | 4.1% | 2,572 | $0.000108 | 85 |
| File-name rules | 98.6% (72) | | | | none | free | 0 |
| Embedding, first 512 tokens | 58.9% (43) | | | | none | free | 123 |
| Embedding, whole card in pieces | 54.8% (40) | | | | none | free | 624 |
| Majority class (Homework) | 52.1% (38) | | | | | | |
| LLM classifier (Claude) | not run | | | | | | |

The placement cells are empty for the baselines because their probabilities are one-hot or set by a temperature chosen by hand, so where the cutoffs put a file is fixed by construction. Close calls are left out: with the default cutoffs they cannot happen (see "Found along the way").

Over all 77 test files, counting the four with no text as wrong for every content-based method: Jev 94.8%, the rules 93.5%.

### Which differences the data can carry

Two methods are compared file by file on the 73 test files. What counts is the files only one of them got right, and whether an exact sign test on those separates the two.

| Comparison | Files only one got right | Sign test p | Reading |
| --- | --- | --- | --- |
| Jev card against embedding (512 tokens) | 30, all for Jev | about 2e-9 | Jev is clearly ahead |
| Jev card against embedding (whole card) | 33, all for Jev | about 2e-10 | Jev is clearly ahead |
| Descriptions against group names only | 6, all for descriptions | 0.031 | Descriptions help, on one kind of file |
| Jev card against file-name rules | 1, for Jev | 1.0 | Not separable |
| Card against card without name, full text, window vote | 0 | 1.0 | Cannot be told: the card is at the ceiling |

The test treats every file as independent. This set has many near copies (lectures from one template, exams with their solutions), so even these p-values are on the confident side.

## What the numbers say

1. **Jev is far ahead of embedding similarity.** That is the one comparison between methods this data settles.
2. **Jev and a keyword table are indistinguishable here.** The rules' one miss is `lagrange.pdf`, a lecture note with no telling word in its name, where the rules had no pick at all. One well-run course names its files very regularly.
3. **The card could not be beaten, because it left no room.** Full text and the window vote cost 1.6 to 2 times as much and changed no answer. Across the 51 files with more than one window, the windows never disagreed. This is a reason to start with the card on cost, not evidence that more text never helps.
4. **Descriptions matter.** With group names only, six test files go wrong, all of them small helper code files (`plot.m`, `matrix.m`, `util.py` and the like) that the Homework description names ("starter and helper code for assignments").
5. **Cost and speed are not a concern.** About $0.12 per thousand files, and about a tenth of a second each.
6. **Confidence is high.** 82 of the 91 answered files have a top probability of 1.00, and 87 are at 0.97 or more. The four least sure ones (0.63 to 0.93) are the same helper code files. Since every answer is right, this does not test calibration.

## Experiments

### A group added mid-run

Every file with text (dev and test, 91 files) is sorted with the Exams group left out. Its 17 files then fit no group, so Unsorted is the right outcome for them. Then the group is added, each file that sits in its gold group is frozen, and the rest are sorted again.

| Measure | With descriptions | Group names only |
| --- | --- | --- |
| AUROC of fits-any (other files against exam files) | 0.996 | 0.988 |
| Mean fits-any of the 17 exam files | 0.478 | 0.578 |
| Top-1 accuracy of the other 74 files | 100.0% | 93.2% |
| At the default cutoffs: exam files sent to Unsorted | 8 of 17 | 2 of 17 |
| At the default cutoffs: other files wrongly sent to Unsorted | 0 of 74 | 3 of 74 |
| Files rerun after adding the group | 17 | 23 |
| Rerun accuracy | 100.0% | 73.9% |

The fits-any score does its job as a ranking: exam files score 0.30 to 0.67, the other files 0.60 and up, with one file in the overlap. The default cutoff does not. At 0.5 it lets 9 of the 17 exam files through, and all 9 land in Homework with high confidence (an exam looks like a problem set). The same scores under other cutoffs:

| fits-any cutoff | With descriptions: exams held back, others lost | Names only: exams held back, others lost |
| --- | --- | --- |
| 0.5 (default) | 8 of 17, 0 of 74 | 2 of 17, 0 of 74 |
| 0.6 | 12 of 17, 0 of 74 | 10 of 17, 0 of 74 |
| 0.7 | 17 of 17, 1 of 74 | 16 of 17, 3 of 74 |
| 0.8 | 17 of 17, 3 of 74 | 17 of 17, 5 of 74 |

So a cutoff near 0.7 would have worked on this set, and descriptions give the cleaner split there. That number is read off the same files it would be applied to, so it is a hint and not a tuned value. No file in the public set has "fits no group" as its gold label, which is what tuning needs.

Once the group was added, every rerun file was placed right. That assumes the user did not freeze the nine exam files that sat in Homework.

**What the excludes text does.** The first version of this experiment left the other groups' descriptions as they were, and two of them said "excludes: exams". With that text all 17 exam files were held back (fits-any 0.07 to 0.16). A reviewer pointed out that a user who has not thought of an Exams group would not have written it. Without it the result is the table above. Two lessons: the first result was too good, and an excludes line is a strong lever. Telling a group what it does not hold moved those files more than any other wording change in this run.

**Run-to-run variation.** The names-only pass was asked twice with identical questions; only the shuffled option order differed. The first time 6 of the 17 exam files scored under 0.5, the second time 2. Scores near a cutoff move with the order of the options.

### Description edits

The spec's experiment is: for each wrong placement, fix it with a description edit and count what else changes. The base run had no wrong placement, so there was nothing to fix and that experiment was not run. I ran the question the other way instead, with two edits I wrote myself after seeing the first results (they are in `sandbox/eval/description_edits.yaml`, each with a note that says so):

| Edit | Files whose top group changed | All for the worse? |
| --- | --- | --- |
| Homework covers cut down to "problem sets and their solutions" | 4 of 91 (`files.m`, `matrix.m`, `plot.m`, `util.py`) | yes, all to Lecture notes |
| Lecture notes covers widened to "anything that explains or teaches the course material" | 1 of 91 (`matrix.m`) | yes |

Read forward, the first row says: adding "starter and helper code for assignments" to the Homework description fixes four files and breaks none. So a description edit reaches several files at once, where a move reaches one. On this data the reach was small and stayed inside the one kind of file the wording was about.

## Go/no-go table

The conditions are the proposed defaults in [phase0.md](phase0.md). "Above" and "beats" are judged by the paired test described earlier.

| Result | Then | Measured (n = 73) | Met |
| --- | --- | --- | --- |
| Card accuracy of 90% or more, above both non-LLM baselines | Start Phase 1 as designed | Jev card 100.0%, rules 98.6%, embedding 58.9% | **Not separable.** 90% holds and Jev is clearly above the embedding, but it leads the rules by one file |
| Full text or window vote beats the card by 5 points or more | Use that input for D1 | all three at 100.0% | **Cannot be told.** The card is at the ceiling |
| Jev doesn't beat embedding similarity | Rethink Jev's role on the write path | 100.0% against 58.9% | **No.** Jev wins clearly |

## What these numbers can and cannot support

- **The data set is at the ceiling.** It cannot rank the input variants and cannot show where Jev starts to fail.
- **The lead over the file-name rules is one file out of 73.**
- **Three real classes.** Projects and Logistics have one test file each, and those two files carry 40% of the macro-F1.
- **The label is visible in other ways than the content.** All 26 lab files are Homework by their source and keep their path under `6.S191/`, so the folder name gives the group away for 22 of the 73 test files. All 31 test files that are not PDF or HTML are Homework, so the file type alone does too. Some lab notebooks carry their own repository path in their text, which the "card without file name" variant still shows.
- **The group descriptions and the label mapping share an author.** The gold mapping sends OCW "Tools" files and the lab helper package to Homework, and the Homework description says "starter and helper code for assignments". The labels are blind to Jev's answers, but not to the wording Jev was given. The names-only row (91.8%) and the cut-down Homework description (94.5%) show what the result is without that wording.
- **No file in the set fits no group.** Unsorted was tested only by holding a group out, and by the one off-topic file in the smoke test. The cutoffs could not be tuned: the dev split has 18 files, Jev was right on all of them, and several hundred pairs of cutoffs tie for best.
- **One model, one run, unseeded shuffle.** Answers near a cutoff vary with option order.
- **The LLM baseline is missing.**

## Recommendation

1. **Treat the go/no-go as open.** The loop works and costs almost nothing, and nothing here argues against Phase 1. But the condition "above both non-LLM baselines" is not shown, and this data cannot show it either way. I would not cite "Jev beats rules" from this run.
2. **Get harder data before deciding what Jev is for.** The real CS 4641/7641 material (needs the instructor's OK), where names are messier and groups overlap, or a second public course with real project and logistics files. Have the descriptions written, or at least approved, by someone who has not seen the label mapping, and include files that fit no group.
3. **Use the card as D1's input for now, on cost.** Revisit when a data set leaves room for the long inputs to win.
4. **Fix the two Unsorted rules before Phase 1 depends on them.** The fits-any cutoff of 0.5 is too low on the one test it got, and the close call flag cannot fire (next section). Both are default thresholds, so they are Stan's to change.
5. **Measure order sensitivity.** Ask the same file several times under different option orders and report the spread. It is cheap, and it says how far a single answer near a cutoff can be trusted.
6. **Add the Anthropic key** to get the LLM row. About $2.60 for one pass, at list prices I have not checked myself.

## The review, and what it changed

After the build, five reviewer agents went over the code and the evaluation. They reported 38 findings. The 30 about code and method went to a second agent whose job was to refute them, and all 30 held; the 8 about test quality I checked myself. The ones that mattered:

- **A group named like the dump folder could move the originals.** `Lectures` over a dump called `lectures` on a Mac, or `lectures` over `lectures/`, and a group rename moved the dump folder itself. That broke a non-negotiable. Fixed: names are compared the way the file system compares them, and no function renames those folders.
- **A card note touched in Obsidian could lose its correction.** A tag added to the note, or the note filed into a subfolder, and the file was sorted again and its label withdrawn. Fixed: extra properties are kept, and subfolders are read.
- **The folder move I had added for groups renamed by hand was wrong for chains and swaps of names.** It wrote "moved by you" labels for whole groups. Replaced by a simpler rule: every note goes back to its own group.
- **A call log that could not be written did not stop the sort.** Cards kept going out unlogged. Fixed: the run stops.
- **The page could be framed by another site** through a second address without security headers, and a file named like a group froze the whole group. Fixed.
- **The generated report said "yes" to "above both baselines" on a lead of one file**, and the held-out experiment carried the hint described above. Fixed, and that experiment was rerun.

The fixes are in five commits after the build commits, each with tests. The suite went from 789 to 1,093 tests.

## Found along the way

- **A close call cannot happen with the default cutoffs.** A file is placed only when its top probability is 0.6 or more. That leaves at most 0.4 for the second group, a gap of at least 0.2, and the close call margin is 0.15. Options: lower the top cutoff under 0.575, raise the margin, or flag a placement as a close call when the top probability is under some value such as 0.8.
- **`p0 init --dump .` is now refused**, and so is a dump folder outside the vault. The originals have to sit in a folder of their own inside the vault.
- **PyMuPDF is AGPL-3.0.** It constrains the repo's license. It is used only in `extract/pdf.py`; pypdf (BSD) is a possible swap.
- **`.gitignore` ignores `.claude/`**, so the project `CLAUDE.md` is not in a clone. `.env.example` was ignored too; it is tracked since 2026-11-16.
- **Not seen in a real browser:** PDF originals opened from the page under the new security headers. The app's own browser pane downloads PDFs instead of showing them.
- Smaller open items are listed under "Still open" in [decisions.md](decisions.md).

## How to rerun

```bash
uv run python -m decide_already.eval.run --root sandbox/eval --dry-run
```

```bash
uv run python -m decide_already.eval.run --root sandbox/eval --configs jev:card,jev:card_no_name,jev:full_text,jev:window_vote,jev:names_only,rules:card,embed:card,embed-chunks:card
```

Everything is cached, so a rerun asks nothing and costs nothing. Delete `sandbox/eval/results/predictions/` to start over. Leave `--configs` out to include the LLM baseline once `ANTHROPIC_API_KEY` is in `.env`.
