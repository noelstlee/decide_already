# Phase 1 blueprint: Jev in a course knowledge base

> Status: design, not yet started. Phase 1 starts after the Phase 0 decision (proposed for 2026-11-01), which may change D1's input format and the pinned model version.
> Source: the "Build blueprint" tab of the Claude Docs doc "AI TA knowledge base blueprint" (https://claude.ai/artifact/UnfNiQUuQCX8nL8oAbUnsP), exported 2026-10-24. Diagrams there are written out as text here. "You" means Stan.

Phase 1 tests whether Jev, called through TypeSafe's API with no fine-tuning, earns its place in four parts of a course knowledge base: the write path, the query router, guardrails, and reranking with an answer-or-escalate check. The corpus is CS 4641/7641's Fall 2025 and Fall 2026 materials. It stops at ranked evidence and decisions; no answers are generated. The go/no-go gate is mid-December.

## Architecture

The diagram in the Claude Docs version shows two paths over one knowledge base. Jev makes the judgments; code handles every fact.

**Ingest path** (runs once per semester replay and on every changed file)

1. Course files: Fall 2025 and Fall 2026 CS 4641/7641.
2. Parser: sections by heading, slide or HW question.
3. Class and tier (D1, D2): one Jev call per node.
4. Shortlist builder: 5 to 30 existing nodes per new node.
5. Update relation (D3): one Jev call per candidate pair.
6. Review queue (you): retirements and low-confidence calls.
7. Knowledge base: markdown nodes (one file per node, in git), then the index builder rebuilds BM25 and dense indexes at two sizes plus the typed edge list.

**Question path**

1. Question: synthetic, hand-written or unanswerable.
2. Intent and unit (D4): picks which index to search.
3. Hybrid search: top 30 from the chosen index.
4. Filters and expansion: drop superseded and hidden nodes; add 1-hop edges.
5. Rerank (D5): one yes/no per candidate, in one call.
6. Enough evidence? (D6): return ranked evidence, or escalate.
7. Leak check (D7): tested offline only, on labeled drafts.

The shortlist builder and hybrid search both read the indexes.

Ingest runs once per semester replay and on every changed file; only retirements and low-confidence calls stop at you. The question path ends at ranked evidence or an escalation, since Phase 1 writes no answers.

## Worked example

The diagram in the Claude Docs version walks one correction and one question through the pipeline. Probabilities are illustrative.

**Ingest path: the Oct 3 correction**

1. New file: `announcements/2026-10-03.md`, "Q2 kernel width should be 0.5, not 0.1".
2. Class and tier (D1, D2): announcement 0.97, public 0.93. Both above 0.8, so applied automatically.
3. Shortlist builder: reference to HW3 Q2, same assignment, search. 5 candidates; the retired 2025 spec is skipped.
4. Update relation (D3): vs Q2 spec 2026, amends 0.86. vs Lecture 9 slide, unrelated 0.71.
5. Review queue (you): approve "amends Q2 spec 2026". Lecture 9 is under 0.8, so you confirm unrelated.
6. Knowledge base updated: edge written, amends cs4641-f26-hw3-q2. Indexes and edge list rebuilt.

**Question path: a student asks about Q2**

7. Question: "HW3 Q2 says gamma = 0.1 but my RBF results look terrible. Is that right?"
8. Intent and unit (D4): assignment clarification 0.81; section 0.74, so search the section index.
9. Hybrid search: top hit is the Q2 spec 2025 at 0.84, now stale. Missed: the Oct 3 fix, which never says "gamma".
10. Filters and expansion: dropped the Q2 spec 2025 (retired). Added the Oct 3 fix via its amends edge (written at step 6).
11. Rerank (D5): Oct 3 fix 0.95, Q2 spec 2026 0.90, kept. Lecture 9 0.44, rubric 0.12, dropped.
12. Enough evidence? (D6): yes 0.92, above the 0.80 threshold. Output: Oct 3 fix, then Q2 spec 2026.

**Scored against the gold nodes:** this pipeline gets Recall@5 = 1.0 with no stale node in the evidence; plain hybrid search gets Recall@5 = 0.5, missing the fix and serving the 2025 spec.

Search alone misses the correction, because it says "kernel width" and the student says "gamma". The amends edge written at step 6 is what brings it back at step 10, and the date rule is what keeps last year's spec out.

## Jev decisions

Seven Jev calls cover the four roles. Code handles everything that is a fact rather than a judgment: dates, authority, hard tiers, file metadata. Every Choice question includes an "other" or "none" option, and the model is pinned to `jev-1.13.0`.

| # | Decision | Path | Jev question | Options | Input | Policy |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | Node class | Ingest | Choice | Syllabus section, lecture slide, spec section, announcement, rubric, solution, other | Section text plus file path and title | Below 0.8 confidence goes to the review queue |
| D2 | Visibility tier | Ingest | Choice | Public, staff, hidden | Section text plus its class | Solutions and tests are hidden by rule. Jev can only move a node down. Doubtful calls default to staff |
| D3 | Update relation | Ingest | Choice per pair | Replaces, amends, clarifies, unrelated | New node, one shortlisted node, which is older and each one's authority (all set by code) | Every replaces or amends goes to review before an old node is retired |
| D4 | Intent and unit | Question | Two Choices in one call | Intent: logistics, conceptual, assignment clarification, code debugging, other. Unit: section, small chunk, both | The question | Unit picks which index to search |
| D5 | Rerank | Question | One yes/no per candidate, all in one call | Helps answer, or not | The question plus up to 30 candidate cards (text, class, relations) | Keep cards above a threshold tuned on the dev split |
| D6 | Enough evidence | Question | Yes/no | Enough to answer, or not | The question plus the kept cards | Below threshold, escalate to a TA |
| D7 | Leak check | Answer | Yes/no | Reveals solution code or values, or not | A draft reply plus the hidden solution | Above threshold, block. Tested offline on labeled drafts, since Phase 1 writes no answers |

## Components

No model is served locally in Phase 1: Jev and embeddings come through APIs, and everything else is plain Python over files in git.

| Component | What it does | Built with |
| --- | --- | --- |
| Parser | Splits PDFs, slides and markdown into sections by heading, slide or HW question, keeping page references | Python text extraction; images dropped |
| Node writer | Writes one markdown file per section, with frontmatter checked against the schema | Python, Pydantic |
| Shortlist builder | For each new node, gathers 5 to 30 existing nodes: same assignment or question, explicit references, top search hits | Code plus the search index |
| Review queue | Holds retirements and low-confidence calls until you approve them; every decision is saved as a label | A command-line tool first |
| Index builder | Builds BM25 and embedding indexes at two unit sizes, plus the edge list, from the files | BM25 in Python, an embeddings API; rebuilt on every commit |
| Search | Hybrid BM25 and dense search over the index the router picked, merged with reciprocal rank fusion | Python |
| Filters | Drop superseded nodes and anything above the request's tier | Code, no model |
| Edge expansion | Adds 1-hop neighbors through amends, clarifies and references edges, plus grades for staff requests | Code |
| Jev client | One wrapper for D1 to D7 that logs every request, answer and probability | TypeSafe API |
| Eval harness | Runs every configuration on every dataset and writes the metrics | Python |

## Knowledge base

The Phase 1 knowledge base is a git folder of markdown files, one per section, typed by 11 classes and linked by 9 relations. All of them are fixed in code: Jev only picks among these values, and you approve anything that retires or changes a node.

The schema diagram in the Claude Docs version shows Course, Offering and Assignment at the top; the content classes (Announcement, Spec section, Lecture slide, Syllabus section, Rubric item, Other) in one zone; Solution and Test file in a hidden zone read only by the leak check. Code sets part_of, instance_of, references, grades, solves and tests. Jev (D3) plus the review queue sets supersedes, amends and clarifies, between any two content nodes. The tables below are the full definition.

Code sets six of the nine relations from folders, file names and text. Only the three update relations come from Jev, and they pass through the review queue.

The second diagram shows a slice of the knowledge base: HW3 Q2 across two terms (ids and dates illustrative).

- `cs4641` (course) and `cs4641-hw3` (assignment, part_of the course).
- Fall 2025: `cs4641-f25-hw3-q2` (spec section), retired 2026-09-20, kept on disk, out of search.
- Fall 2026: `cs4641-f26-hw3-q2` (spec section, public, live) supersedes the 2025 spec. Both specs are part_of `cs4641-hw3`.
- `cs4641-f26-ann-1003` (announcement, public) amends the 2026 spec.
- The 2026 spec references `cs4641-f26-lec09-s12` (lecture slide, public).
- `cs4641-f26-hw3-q2-rubric` (rubric item, staff) grades the 2026 spec.
- `cs4641-f26-hw3-q2-sol` (solution, hidden) solves the 2026 spec.

Search only sees live nodes within the request's tier. The retired 2025 spec keeps its file and its valid_to, so the semester replay can rebuild what the knowledge base looked like on any date.

### On disk

```text
kb/
  schema/types.py      every class, relation and enum below (Pydantic)
  course.yaml          the course, its 2 offerings and its assignments (the structural nodes)
  sources/f25/ f26/    original files, never edited
  nodes/f25/ f26/      one markdown file per content node: frontmatter plus the section text
  hidden/f25/ f26/     solution and test nodes and their files, never indexed
  review/queue.jsonl   calls waiting for you
  review/labels.jsonl  every approve and reject, kept as labels
  logs/jev.jsonl       every Jev request, answer and probability
  build/               BM25 and dense indexes at section and chunk size, plus edges.jsonl;
                       rebuilt on each commit, not in git
```

### Classes

| Class | One node is | Set by | Default tier | Authority |
| --- | --- | --- | --- | --- |
| Course | CS 4641/7641 | Code, course.yaml | Public |  |
| Offering | One term: Fall 2025 or Fall 2026 | Code, course.yaml | Public |  |
| Assignment | One homework, shared across terms | Code, course.yaml | Public |  |
| Announcement | One instructor post | D1 | Public | 4 |
| Spec section | One homework question or part | D1 | Public | 3 |
| Rubric item | One grading criterion | D1 | Staff | 3 |
| Syllabus section | One syllabus heading | D1 | Public | 2 |
| Lecture slide | One slide | D1 | Public | 1 |
| Other | A section D1 can't place; always reviewed | D1 | Staff |  |
| Solution | One solution file or question | Code, hidden/ folder | Hidden |  |
| Test file | One autograder test file | Code, hidden/ folder | Hidden |  |

Structural nodes have no text; they give edges something to point at. D1 also offers solution as an option, so a solution filed outside hidden/ is caught and sent to review. D2 can move a node toward hidden but never back. Authority goes into D3's input.

### Relations

| Relation | From → to | Set by | What retrieval does with it |
| --- | --- | --- | --- |
| part_of | Any term node → its offering; spec section → its assignment; assignment → course | Code, paths and course.yaml | Term filters, semester replay, shortlist |
| instance_of | Offering → course | Code, course.yaml | Structure only |
| references | A node → a node its text names ("see Lecture 9") | Code, text patterns | 1-hop expansion, shortlist |
| grades | Rubric item → spec section | Code, file names | 1-hop expansion, staff requests only |
| solves | Solution → spec section | Code, file names | Leak check (D7) only |
| tests | Test file → spec section | Code, file names | Leak check (D7) only |
| supersedes | Newer node → the version it replaces | D3 "replaces", always reviewed | Older node gets valid_to and superseded_by, and leaves search |
| amends | Newer node → a node it partly changes | D3, always reviewed | Both stay live; either pulls in the other |
| clarifies | Newer node → a node it explains without changing | D3, reviewed below 0.8 | Both stay live; either pulls in the other |

Each edge is stored on the node it starts from. edges.jsonl indexes both directions, so expansion works from either end.

### Types in code

Everything above lives in one file. Every Jev answer and every node file is checked against it before anything is written.

```python
# kb/schema/types.py, Phase 1. Changing any value here is a schema version bump.
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

NodeClass = Literal[
    "course", "offering", "assignment",              # structural, from course.yaml
    "announcement", "spec_section", "rubric_item",   # content, picked by D1
    "syllabus_section", "lecture_slide", "other",
    "solution", "test_file",                         # hidden, set by folder
]
Relation = Literal[
    "part_of", "instance_of", "references",          # set by code
    "grades", "solves", "tests",                     # set by code
    "supersedes", "amends", "clarifies",             # D3, then review
]
Tier = Literal["public", "staff", "hidden"]           # D2 may move a node toward hidden, never back
Intent = Literal["logistics", "conceptual", "assignment_clarification",
                 "code_debugging", "other"]           # D4
Unit = Literal["section", "chunk", "both"]            # D4: which index to search
UpdateRelation = Literal["replaces", "amends", "clarifies", "unrelated"]   # D3
DecidedBy = Literal["rule", "jev", "human"]
ReviewStatus = Literal["auto_applied", "pending", "approved", "rejected"]

D3_TO_EDGE = {"replaces": "supersedes", "amends": "amends",
              "clarifies": "clarifies", "unrelated": None}
EXPAND = {"public": {"amends", "clarifies", "references"},     # 1 hop, by request tier
          "staff": {"amends", "clarifies", "references", "grades"}}
REVIEW_BELOW = 0.8    # any call under this waits for you; D3 replaces and amends always do
CHUNK_TOKENS = 200

class Edge(BaseModel):
    type: Relation
    target: str                      # node id
    decided_by: DecidedBy
    p: float | None = None           # Jev's probability, when Jev proposed it

class Decision(BaseModel):
    value: str
    decided_by: DecidedBy
    p: float | None = None
    model: str | None = None         # "jev-1.13.0"
    status: ReviewStatus = "auto_applied"

class Node(BaseModel):
    id: str                          # term + file + heading, stable across edits
    cls: NodeClass = Field(alias="class")
    title: str
    offering: str | None = None      # its part_of edge to a term
    assignment: str | None = None    # spec sections only: part_of edge to the homework
    source: str | None = None        # file path and page
    tier: Tier
    authority: int | None = None     # from the class (Classes table)
    valid_from: date
    valid_to: date | None = None     # set when retired; the file stays
    superseded_by: str | None = None
    relations: list[Edge] = []
    decisions: dict[str, Decision] = {}   # "class" (D1) and "tier" (D2)
    content_hash: str                # D1 to D3 rerun only when this changes
    body: str = ""                   # the text under the frontmatter

# Chunks are not files: the index builder cuts each body into ~CHUNK_TOKENS pieces
# with ids like cs4641-f26-hw3-q2#c2 that inherit every field above.
```

### A node file

The Oct 3 correction from the worked example, after you approved D3's call:

```yaml
---
id: cs4641-f26-ann-1003
class: announcement
title: "HW3 Q2 correction"
offering: cs4641-f26
source: sources/f26/announcements/2026-10-03.md
tier: public
authority: 4
valid_from: 2026-10-03
valid_to: null
superseded_by: null
relations:
  - {type: references, target: cs4641-f26-hw3-q2, decided_by: rule}
  - {type: amends, target: cs4641-f26-hw3-q2, decided_by: human, p: 0.86}
decisions:
  class: {value: announcement, decided_by: jev, p: 0.97, model: jev-1.13.0, status: auto_applied}
  tier: {value: public, decided_by: jev, p: 0.93, model: jev-1.13.0, status: auto_applied}
content_hash: 9f2c1e...
---
HW3 Q2: the RBF kernel width should be 0.5, not 0.1. The spec has a typo.
```

Code wrote the references edge when it saw "HW3 Q2" in the text. D3 judged the pair amends at 0.86 and you approved it, so that edge is decided_by human.

## Evaluation

Each decision is scored on accuracy and calibration against the cheapest strong baseline, and retrieval is scored before and after Jev so recall and ranking stay separate. Every dataset gets a 20% dev split for thresholds, frozen before testing.

### Datasets

| Dataset | Size | How it's built | Tests |
| --- | --- | --- | --- |
| Node labels | 300 nodes | You label class and tier on a sample of both offerings | D1, D2 |
| Update pairs | 150 pairs | Fall 2025 vs Fall 2026 sections matched by assignment and question, plus announcement vs spec pairs; you label the relation | D3 |
| Synthetic questions | 300 | Ground truth first: pick a node or fact, generate a question about it, filter out questions that copy the node's words; gold node IDs known by construction | D4, D5, D6 |
| Hand-written questions | 50 | You write them in the style of real Ed questions and mark gold nodes | Checks that synthetic rankings hold on real-style questions |
| Unanswerable questions | 60 | Questions whose evidence was removed or never existed | D6 |
| Draft replies | 200 | An LLM writes drafts with and without solution leaks; you spot-check labels | D7 |
| Semester replay | Both offerings | Ingest Fall 2025, then Fall 2026 in date order; ask questions at checkpoints whose current answer is known | Stale-answer rate |

### Metrics

- **Decisions:** accuracy, macro-F1, calibration (ECE, Brier, reliability plots).
- **Retrieval:** candidate Recall@20 before Jev; nDCG@10 and Recall@5 after.
- **Answer or escalate:** risk-coverage curve, coverage at 5% and 10% risk.
- **Staleness:** share of questions where a retired node reaches the final evidence.
- **Cost:** latency p50 and p95, dollars per question.

### Baselines (API only)

| Decision | Baselines |
| --- | --- |
| D1, D2 | Rules on file paths and class; an LLM classifier via API |
| D3 | A fixed same-key rule (MemStrata style); an LLM judge |
| D4 | One fixed unit size; an LLM router |
| D5 | No reranking; Cohere Rerank API; an LLM judge |
| D6 | The LLM's own yes probability |
| D7 | Code and n-gram overlap with the solution; an LLM judge |

Trained classifiers wait for Phase 2, since they'd need local training. Two ablations ride along: typed-edge expansion on vs off, and the unit router vs a fixed unit.

## Timeline

Planned dates (a starting draft; the gate date is the one to protect):

| Work | Dates (2026) |
| --- | --- |
| Files and instructor OK | Oct 19 to Oct 25 |
| Parser, schema, node files | Oct 26 to Nov 8 |
| Datasets and labels | Oct 26 to Nov 22 |
| Ingest calls, D1 to D3 | Nov 2 to Nov 15 |
| Question path, D4 to D6 | Nov 9 to Nov 29 |
| Baselines and eval harness | Nov 16 to Dec 6 |
| Replay, D7 and ablations | Nov 30 to Dec 12 |
| Go/no-go gate | Dec 15 |

Datasets get built alongside the code from late October, so evaluation can start as soon as the question path works. The dates are a draft; the gate date is the one to protect.

## Cost

Jev itself should cost under $10 for all of Phase 1; the baselines cost more. A $150 budget leaves plenty of room. These are estimates assuming about 1,000 nodes across both offerings, at Jev's $0.042 per million input tokens.

| Item | Volume | Estimate |
| --- | --- | --- |
| Ingest, D1 and D2 (one call per node) | 1,000 calls at about 2k tokens | About $0.10 |
| Ingest, D3 (about 10 pairs per node) | 10,000 calls at about 3k tokens | About $1.30 |
| Questions, D4 to D6, per full run | 410 questions at about 17k tokens | About $0.30 |
| Ten full runs, three ingest replays, D7 | Repeats of the above | About $8 total |
| Cohere Rerank baseline | About 4,000 searches at $2 to $2.50 per 1,000 | About $10 |
| Embeddings for two unit sizes | A few million tokens | Under $1 |
| LLM judge and LLM router baselines | Depends on the model picked | $20 to $100 |

Rate limits aren't a concern: Jev allows 80 requests per second.

## Defaults I picked

None of these change the research question, so I set them rather than ask. Overrule any of them in a comment.

| Decision | Default | Why |
| --- | --- | --- |
| Node unit | Sections (heading, slide or HW question) plus small chunks of about 200 tokens, both indexed | The router needs two sizes to choose between |
| Authority order | Announcement, spec, syllabus, lecture slide | Matches how corrections flow in the course |
| Embeddings | One API embedding model, fixed for all runs | No local hosting in Phase 1 |
| LLM baselines and question generator | One frontier API model, fixed for all runs | Comparisons stay fair |
| Thresholds | Tuned on the 20% dev split, then frozen | Avoids tuning on the test set |
| Review queue | Command-line tool | Fastest to build; a page can come in Phase 2 |
| Images | Dropped; slides use extracted text only | Text first, as you decided |
| Jev version | `jev-1.13.0`, pinned | `jev-latest` moves and would shift thresholds |

Two actions block real work: getting the instructor's OK to send CS 4641/7641 materials through TypeSafe's and the embedding provider's APIs, and collecting both offerings' files.

## Phase 2: deployment version (open)

Phase 2 turns whatever survives the gate into a system TAs can use. Its scope depends on Phase 1 results, so it's written as open decisions, each with the result that settles it.

| Open decision | Options | Settled by |
| --- | --- | --- |
| Which Jev roles continue | Keep roles where Jev beat its baseline; drop or replace the rest | Per-role results at the gate |
| Model backend | Fine-tune Kev or OpenJev-4B; train your own decision head; stay hosted for non-student data | Size of the gap between zero-shot Jev and the baselines, and your label budget |
| Ed history | Ingest past semesters, add similar-post retrieval, add TA and student authority levels | IRB and Registrar approval |
| Answer generation | Add a writer LLM, end-to-end answer grading, and privileged-reader mistake diagnosis | Whether answer-or-escalate is calibrated well enough to trust |
| Typed edges | Keep, expand or drop | The typed-vs-untyped ablation |
| Unit router | Keep the router or fix one unit size | The router ablation |
| Agent interface | MCP server, Ed bot, TA dashboard, or several | Who uses it first |
| Images | OCR and captions, or a multimodal decision model | Share of real questions that include images |

Fixed regardless of results: every model touching student data runs on Georgia Tech hardware, the review queue stays, model versions stay pinned, and the Phase 1 eval harness keeps running so results stay comparable.
