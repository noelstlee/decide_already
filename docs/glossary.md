# Glossary

## Decision models

- **Jev**: TypeSafe AI's hosted decision model. Picks among options and returns probabilities; does not write text. See [jev-notes.md](jev-notes.md).
- **Decision model / System One model**: the class of models like Jev, OpenJev, Kev and CLM.
- **State**: the text Jev reads for a request (in Phase 0, a file's card).
- **Choice**: a question that picks one option from a set; probabilities sum to 1.
- **Noul**: a yes/no question; returns the probability of yes.
- **Score**: a question that places the input on an ordered scale.
- **Criteria**: the option descriptions sent with a Choice (in Phase 0, a group's covers and excludes).
- **Backend**: which decision model the code calls (Jev now; OpenJev or CLM later).

## Phase 0

- **Dump**: the `_dump/` folder where you drop whole files. Originals are never moved or edited.
- **Card**: the ~3k-token text code builds from a file (name, type, title, outline, opening text, excerpts). The only part of a file sent to the API.
- **Card note**: a markdown file with YAML frontmatter, one per file, sitting in its current group's folder in the vault.
- **Group**: a category you define, with a name and a description (covers, excludes). Becomes one Choice option.
- **Category set / set version**: all groups at a point in time. Any add, rename, edit or delete bumps the version.
- **Placement**: where a file currently sits, with Jev's probabilities and the set version it ran under.
- **fits_any**: the yes/no probability that any group clearly fits the file.
- **Unsorted**: where code puts a file when fits_any is under 0.5 or the top group is under 0.6.
- **Close call**: the top two groups are within 0.15; placed but flagged.
- **Freeze**: mark a file so reruns skip it. Moving a file freezes it.
- **Label**: a record in `labels.jsonl`. "moved" when you change Jev's pick, "confirmed" when you freeze Jev's pick as is.
- **Rerun**: sending only unfrozen files back to Jev, against the full current set.
- **Blind labels**: the gold groups you assign before seeing any Jev pick.
- **Correction rate**: the share of files you had to move.
- **Vault**: the folder the tool works in; Obsidian can open it.

## Phase 1

- **Node**: one section of a course file (a heading, a slide, a HW question), stored as a markdown file.
- **Class**: a node's type (course, offering, assignment, announcement, spec section, rubric item, syllabus section, lecture slide, other, solution, test file).
- **Relation / edge**: a typed link between nodes (part_of, instance_of, references, grades, solves, tests, supersedes, amends, clarifies).
- **Tier**: who may see a node (public, staff, hidden). Solutions and tests are hidden by rule.
- **Authority**: a fixed rank per class (announcement 4, spec 3, rubric 3, syllabus 2, lecture slide 1) passed to D3.
- **Validity window**: `valid_from` to `valid_to`. An empty `valid_to` means current.
- **Retire, don't delete**: an outdated node keeps its file, gets `valid_to` and `superseded_by`, and drops out of search.
- **Supersedes / amends / clarifies**: replaces an older node / partly changes it / explains it without changing it.
- **Shortlist**: the 5 to 30 existing nodes code gathers for D3 to compare against a new node.
- **Review queue**: calls waiting for a person (retirements, low-confidence calls).
- **D1 to D7**: the seven Jev calls in Phase 1: class, tier, update relation, intent and unit, rerank, enough evidence, leak check.
- **Semester replay**: ingesting Fall 2025 then Fall 2026 in date order and asking questions at checkpoints with known answers.
- **Stale-answer rate**: the share of questions where a retired node reaches the final evidence.
- **ECE**: expected calibration error; how far probabilities are from observed accuracy.
- **Risk-coverage**: how many questions the system answers at a given error rate, versus escalating.
