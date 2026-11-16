# decide_already

Drop files into a folder. A decision model sorts them into groups you name. You move what it got wrong, and every correction becomes data. No chatbot in the loop, and your files stay on your machine: only a short text card per file goes to the decision model's API.

> **Status: pre-alpha.** Phase 0 (the sorter) is built and evaluated: a command line (`p0`) and a local web page. There is no package to install yet: you run it from a clone with uv. Phase 1 is designed in [`docs/`](docs/).

## The idea

Most "AI knowledge base" tools hand your documents to a generative LLM and hope it organizes them well. This project tries a different split:

- **Code handles facts**: file types, paths, dates, versions, hashes.
- **A decision model handles judgments**: which group a file belongs in, whether a new document replaces an old one. It picks among options you define and returns probabilities, so low-confidence calls can go to you instead of being guessed.
- **You approve what matters**, and each move or confirmation is logged as a labeled example.

The decision model today is [Jev](https://docs.typesafe.ai/models) by TypeSafe AI. Open models (OpenJev, CLM) are planned as swappable backends.

## How it works (Phase 0)

1. Name a few groups and describe each in a line.
2. Drop files into the vault's `_dump/` folder.
3. The tool builds a short text card per file and asks the decision model which group fits, and whether any group fits at all.
4. Each file gets a small markdown note in its group's folder. Open the vault in Obsidian to see the result.
5. Drag a note to the right folder, add a group, or freeze what's right, then rerun. Only unfrozen files are re-sorted.

Details: [docs/phase0.md](docs/phase0.md).

## Quick start

You need [uv](https://docs.astral.sh/uv/) (it brings Python 3.12) and your own TypeSafe API key.

```bash
git clone <this repo> decide_already
```

```bash
cd decide_already
```

```bash
uv sync
```

```bash
cp .env.example .env
```

Then put your key after `TYPESAFE_API_KEY=` in `.env`. The full walkthrough (a vault with ten synthetic course files, the page, the command line, the tests) is in [Run it](docs/phase0.md#run-it).

## Roadmap

| Phase | What | Status |
| --- | --- | --- |
| 0 | The sorter: user-defined groups, decision-model sorting, freeze, move and rerun | Built and evaluated: [docs/phase0.md](docs/phase0.md), [results](docs/phase0-results.md) |
| 1 | A knowledge base that stays correct as documents change: typed nodes, update detection, query routing, reranking, answer-or-escalate, evaluated on two semesters of a real course | Designed: [docs/phase1.md](docs/phase1.md) |
| 2 | Local decision models, plugging in your own LLM (Claude, ChatGPT) through the vault or MCP, deployment | Open |

## Requirements

- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 for the project
- Your own TypeSafe API key for Jev (`TYPESAFE_API_KEY`, in `.env` or the environment). Only the short cards are sent to the API, and sorting stops at a spend cap per vault (`max_spend_usd`, $5 by default).
- Optional: [Obsidian](https://obsidian.md) to browse and drag notes. Any file manager works.

## Docs

- [Phase 0 blueprint](docs/phase0.md)
- [Phase 1 blueprint](docs/phase1.md)
- [Decisions so far](docs/decisions.md)
- [Jev notes: API facts and known weak spots](docs/jev-notes.md)
- [Glossary](docs/glossary.md)

## Background

Built as a Georgia Tech MS project (CS 6999). The evaluation use case is a teaching-assistant knowledge base for a large machine learning course, where specs, announcements and answers change every semester.

## License

Not chosen yet.
