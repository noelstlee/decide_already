# Decisions so far

Newest first. "Decided" means Stan made the call; "Proposed" means it is in a blueprint but Stan has not confirmed it. Change a row only when Stan decides, and add the date.

| Date | Decision | Status | Why |
| --- | --- | --- | --- |
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

- Jev version to pin: `jev-1.14.0` or `jev-1.13.0` (one for both phases).
- Code stack: the defaults in `CLAUDE.md` are proposals.
- License for the repo.
- CLI and tool-folder names (`p0`, `.decide/`).
- Whether design docs stay public in this repo.
- Phase 2 choices: see the last section of [phase1.md](phase1.md).
