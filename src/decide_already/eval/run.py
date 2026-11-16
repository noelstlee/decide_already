"""The evaluation runner (build step 7): every configuration over the dataset, each file once.

Run as:

    python -m decide_already.eval.run [--root sandbox/eval] [--configs a,b,...] [--workers 6]
        [--limit N] [--dry-run] [--no-experiments]

A configuration is a backend and an input variant, written "backend:variant": Jev with each
of the five variants, and each baseline on the card. Every answer is cached as one line of
<root>/results/predictions/<configuration>.jsonl. A rerun asks only about the files that have
no answer under the current groups, or whose last try failed, so nothing is paid for twice.
At the end, report.py turns what is cached into results/metrics.json and results/report.md.

The experiments (experiments.py) add passes: a Jev configuration run again under changed
groups. A pass runs after the configurations, when the one it copies is in the run.

Every Jev request, those of the passes included, goes through one JevBackend that logs to the
vault's own Jev log, so the vault's spend cap covers the whole evaluation. When a backend says
the run must stop (RunStopped: the cap is reached, the call log cannot be written, or the key is refused), the run
stops asking, still writes the report from what is cached, says why, and exits with status 2.
A configuration whose backend cannot be built (Jev or the LLM without a key) is pending: it is
skipped, and the report says why.

Cards are built in the main thread, once per file and variant, because PyMuPDF is not safe
across threads. Jev requests then run in a thread pool; the other backends run one at a time.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, NoReturn

from pydantic import Field

from .. import vault
from ..backends import make_backend
from ..backends.base import Backend, BackendError, RunStopped, SortResult
from ..card import Card, build_card
from ..config import JEV_LOG_FILE, Config, decide_path
from ..log import read_jsonl, total_input_tokens
from ..sort import judge
from ..types import Group, GroupSet, StrictModel
from .baselines.embed import EmbeddingBackend
from .baselines.llm import LLMBackend
from .baselines.rules import RulesBackend
from .dataset import GOLD_FILE, GoldLabel, read_gold
from .experiments import SET_VERSION, Edit, Pass, passes, read_edits
from .metrics import Prediction, top_group
from .report import (
    REPORT_FILE,
    RESULTS_DIR,
    Answers,
    Inputs,
    prices,
    reads_content,
    write_report,
)

CONFIG_NAMES = (
    "jev:card",
    "jev:card_no_name",
    "jev:full_text",
    "jev:window_vote",
    "jev:names_only",
    "rules:card",
    "embed:card",
    "embed-chunks:card",
    "llm:card",
)
PREDICTIONS_DIR = "predictions"  # inside the results folder
EMBED_CHUNK_CHARS = 1600  # embed-chunks reads the card in pieces of this many characters
QUESTION_TOKENS = 700  # about what the two questions and the group descriptions add per request
DEFAULT_ROOT = "sandbox/eval"
DEFAULT_WORKERS = 6
EXIT_ERROR = 1
EXIT_STOPPED = 2  # a backend stopped the run: the spend cap, or a call log that cannot be written

# One step of a run: a configuration name, the groups to sort into, and the set version to log.
Job = tuple[str, list[Group], int]


class Row(StrictModel):
    """One line of a configuration's prediction cache: one file's answer, or why it has none."""

    file_id: str
    gold: str | None
    split: Literal["dev", "test"]
    no_text: bool  # the file has no extractable text; only the file-name rules still answer
    probs: dict[str, float] = Field(default_factory=dict)  # by group id; empty with no answer
    fits_any: float | None = None  # None with no answer
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    model: str | None = None
    set_version: int  # as sent with the request: the vault's set version, or 0 for an experiment
    groups_hash: str  # fingerprint of the groups the answer was computed under (see groups_hash)
    error: str | None = None  # the backend's message, when the request failed
    window_tops: list[str | None] | None = None  # window vote only: the top group of each window


def split_name(name: str) -> tuple[str, str]:
    """The backend and the variant in a configuration name: "jev:card@x" gives ("jev", "card")."""
    backend, _, variant = name.partition("@")[0].partition(":")
    return backend, variant


def parse_configs(text: str) -> list[str]:
    """Split the --configs value into configuration names, in the order given.

    Raises ValueError for a name that is not a configuration, or when no name is given.
    """
    names = list(dict.fromkeys(name.strip() for name in text.split(",") if name.strip()))
    unknown = [name for name in names if name not in CONFIG_NAMES]
    if unknown:
        raise ValueError(
            f"unknown configuration {', '.join(unknown)}; pick from: {', '.join(CONFIG_NAMES)}"
        )
    if not names:
        raise ValueError(f"no configuration named; pick from: {', '.join(CONFIG_NAMES)}")
    return names


def build_backend(key: str, root: Path, config: Config) -> Backend:
    """Build the backend a configuration name starts with.

    Jev logs to the vault's own Jev log and takes its key from TYPESAFE_API_KEY. The embedding
    model is kept in <root>/../models. Raises BackendError when a backend cannot be built, such
    as Jev or the LLM without a key.
    """
    if key == "jev":
        return make_backend(root, config)
    if key == "rules":
        return RulesBackend()
    if key == "embed":
        return EmbeddingBackend(cache_dir=root.parent / "models")
    if key == "embed-chunks":
        return EmbeddingBackend(cache_dir=root.parent / "models", chunk_chars=EMBED_CHUNK_CHARS)
    if key == "llm":
        return LLMBackend()
    raise ValueError(f"no backend is called {key!r}")


def groups_hash(groups: list[Group]) -> str:
    """A short fingerprint of a list of groups: their ids, names and descriptions.

    A cached answer is reused only under the same fingerprint. The set version alone cannot
    tell, because every experiment pass is logged under set version 0.
    """
    text = json.dumps([group.model_dump() for group in groups], sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def cache_path(root: Path, name: str) -> Path:
    """The cache file of a configuration: results/predictions/<name>.jsonl, ":" written "_"."""
    return root / RESULTS_DIR / PREDICTIONS_DIR / (name.replace(":", "_") + ".jsonl")


def read_rows(root: Path, name: str) -> dict[str, Row]:
    """A configuration's cached rows, by file id. Empty when it has no cache file yet."""
    records = read_jsonl(cache_path(root, name))
    return {row.file_id: row for row in map(Row.model_validate, records)}


def write_rows(path: Path, rows: dict[str, Row]) -> None:
    """Write a configuration's cache: one line per file, sorted by file id.

    The lines go to a temporary file that then takes the cache's place, so a run that is
    killed in the middle of a write still has every answer it paid for.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(rows[file_id].model_dump()) + "\n" for file_id in sorted(rows))
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(lines, encoding="utf-8")
    temporary.replace(path)


def select(labels: list[GoldLabel], limit: int | None) -> list[GoldLabel]:
    """The files of a run, sorted by file id: all of them, or the first `limit` of each split."""
    ordered = sorted(labels, key=lambda label: label.file_id)
    if limit is None:
        return ordered
    dev = [label for label in ordered if label.split == "dev"][:limit]
    test = [label for label in ordered if label.split == "test"][:limit]
    return sorted([*dev, *test], key=lambda label: label.file_id)


def _current(row: Row | None, set_version: int, fingerprint: str) -> bool:
    """Whether a cached row was written under this set version and these groups."""
    return row is not None and (row.set_version, row.groups_hash) == (set_version, fingerprint)


def _ask(
    backend: Backend,
    card: Card,
    groups: list[Group],
    set_version: int,
    variant: str,
    fixed: dict,
) -> Row:
    """Ask a backend about one file and make its cache row: the answer, or the failure.

    This runs in a worker thread. `fixed` holds the row's fields that do not depend on the
    answer. A RunStopped (the spend cap, a call log that cannot be written, a refused key) says nothing about
    the file, so it is passed on for the run to stop on.
    """
    per_state: list[SortResult] = []
    try:
        result = judge(card, groups, set_version, backend, variant, per_state)
    except RunStopped:
        raise
    except BackendError as error:
        return Row(**fixed, error=str(error))
    tops = [top_group(state.probs) for state in per_state] if variant == "window_vote" else None
    return Row(
        **fixed,
        probs=result.probs,
        fits_any=result.fits_any,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        model=result.model,
        window_tops=tops,
    )


class Evaluation:
    """One evaluation of a vault: its files, groups and settings, and what is built once.

    Cards are kept per file and variant, and backends per name, so the five Jev configurations
    and the experiments share one Jev backend and with it one spend counter.
    """

    def __init__(
        self,
        root: Path,
        group_set: GroupSet,
        limit: int | None = None,
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        """Read the vault's config, gold labels and dump listing. Nothing is asked or written.

        Raises ValueError when there are no groups or no gold labels, or when a gold label
        names a file that is not in the dump folder.
        """
        if not group_set.groups:
            raise ValueError(f"{root} has no groups to sort into")
        if not (root / GOLD_FILE).exists():
            raise ValueError(
                f"{root} has no {GOLD_FILE}; build the dataset first with "
                "python -m decide_already.eval.dataset"
            )
        self.root = root
        self.config = Config.load(root)
        self.group_set = group_set
        self.limit = limit
        self.workers = workers
        self.labels = select(read_gold(root), limit)
        self.paths = dict(vault.dump_files(root, self.config))
        missing = [label.file_id for label in self.labels if label.file_id not in self.paths]
        if missing:
            raise ValueError(
                f"{GOLD_FILE} names files that are not in the dump folder "
                f"({len(missing)} in all, the first is {missing[0]})"
            )
        self.pending: dict[str, str] = {}  # configuration name to why it has no backend
        self.stopped: str | None = None  # why a backend stopped the run, once one has
        self._cards: dict[tuple[str, str], Card] = {}
        self._backends: dict[str, Backend] = {}

    def card(self, file_id: str, variant: str) -> Card:
        """One file's card under one variant, built on first use. Main thread only."""
        key = (file_id, variant)
        if key not in self._cards:
            self._cards[key] = build_card(self.paths[file_id], file_id, self.config, variant)
        return self._cards[key]

    def asked_card(self, name: str, file_id: str) -> Card:
        """The card a configuration is asked about for one file. No states: it is not asked.

        A file with no text has a card with no states, and no content-based backend is asked
        about it. The file-name rules never read the text, so for such a file they get one
        state that holds just the file's name, path and type.
        """
        backend, variant = split_name(name)
        card = self.card(file_id, variant)
        if not card.states and not reads_content(backend):
            info = {
                "name": file_id.rsplit("/", 1)[-1],
                "path": file_id,
                "type": card.file.file_type,
            }
            return card.model_copy(update={"states": [{"file": info}]})
        return card

    def no_text(self) -> list[str]:
        """The ids of this run's files that have no extractable text."""
        return [
            label.file_id for label in self.labels if not self.card(label.file_id, "card").states
        ]

    def unanswered(
        self, rows: dict[str, Row], groups: list[Group], set_version: int
    ) -> list[GoldLabel]:
        """The files that have no usable row among a configuration's cached rows.

        A row is usable when it holds no error and was written under the same set version and
        the same groups. Everything else is asked again.
        """
        fingerprint = groups_hash(groups)
        return [
            label
            for label in self.labels
            if not _current(rows.get(label.file_id), set_version, fingerprint)
            or rows[label.file_id].error is not None
        ]

    def backend(self, key: str) -> Backend:
        """The backend of the given name, built on first use and shared from then on."""
        if key not in self._backends:
            self._backends[key] = build_backend(key, self.root, self.config)
        return self._backends[key]

    def run(self, name: str, groups: list[Group], set_version: int) -> str:
        """Ask one configuration about each file it has no usable answer for, and cache them.

        `groups` and `set_version` are the vault's own for a configuration, and the changed
        groups and 0 for an experiment pass. The cache file is rewritten after every answer.
        A request that fails is cached with its error and tried again on the next run. When
        the backend cannot be built, the configuration is noted as pending and nothing is
        written. When a backend stops the run (RunStopped), `stopped` is set to its message
        and the files not yet asked are left for a later run.

        Returns one line for the terminal that says what was done.
        """
        backend_key, variant = split_name(name)
        fingerprint = groups_hash(groups)
        path = cache_path(self.root, name)
        rows = read_rows(self.root, name)
        todo = self.unanswered(rows, groups, set_version)
        jobs: list[tuple[Card, dict]] = []  # the card to ask about, and the row's fixed fields
        for label in todo:
            card = self.asked_card(name, label.file_id)
            fixed = {
                "file_id": label.file_id,
                "gold": label.gold_group,
                "split": label.split,
                "no_text": not self.card(label.file_id, variant).states,
                "set_version": set_version,
                "groups_hash": fingerprint,
            }
            if card.states:
                jobs.append((card, fixed))
            else:
                rows[label.file_id] = Row(**fixed)  # no text, so there is nothing to ask
        asked = failed = 0
        if jobs:
            try:
                backend = self.backend(backend_key)
            except RunStopped as error:
                self.stopped = str(error)
                return f"{name}: stopped ({error})"
            except BackendError as error:
                self.pending[name] = str(error)
                return f"{name}: pending ({error})"
        if len(jobs) < len(todo):
            write_rows(path, rows)
        if jobs:
            pool = ThreadPoolExecutor(max_workers=self.workers if backend_key == "jev" else 1)
            try:
                futures = [
                    pool.submit(_ask, backend, card, groups, set_version, variant, fixed)
                    for card, fixed in jobs
                ]
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    try:
                        row = future.result()
                    except RunStopped as error:
                        self.stopped = str(error)
                        for other in futures:
                            other.cancel()
                        continue
                    rows[row.file_id] = row
                    write_rows(path, rows)
                    asked += 1
                    failed += row.error is not None
            finally:
                pool.shutdown(cancel_futures=True)
        line = (
            f"{name}: asked {asked}, failed {failed}, no text {len(todo) - len(jobs)}, "
            f"already cached {len(self.labels) - len(todo)}"
        )
        return f"{line}, stopped ({self.stopped})" if self.stopped else line

    def answers(self, name: str, groups: list[Group], set_version: int) -> Answers:
        """What a configuration has cached for this run's files under these groups.

        This is the hand-over to the report. Gold labels come from the current gold.jsonl,
        not from the cached rows. A file with no text has no prediction, unless the file-name
        rules answered it.
        """
        backend_key, _ = split_name(name)
        fingerprint = groups_hash(groups)
        rows = read_rows(self.root, name)
        answers = Answers(name=name, backend=backend_key, pending=self.pending.get(name))
        for label in self.labels:
            row = rows.get(label.file_id)
            if not _current(row, set_version, fingerprint):
                continue
            if row.error is not None:
                answers.errors[label.file_id] = row.error
            elif row.fits_any is not None:
                answers.predictions.append(
                    Prediction(
                        file_id=label.file_id,
                        gold=label.gold_group,
                        probs=row.probs,
                        fits_any=row.fits_any,
                        input_tokens=row.input_tokens,
                        output_tokens=row.output_tokens,
                        latency_ms=row.latency_ms,
                    )
                )
                if row.window_tops is not None:
                    answers.window_tops[label.file_id] = row.window_tops
        return answers

    def inputs(self, extra: list[Pass], edits: list[Edit] | None) -> Inputs:
        """Everything the report needs: what every configuration and every pass has cached."""
        groups, version = self.group_set.groups, self.group_set.version
        return Inputs(
            config=self.config,
            set_version=version,
            groups=groups,
            labels=self.labels,
            no_text=self.no_text(),
            answers=[self.answers(name, groups, version) for name in CONFIG_NAMES],
            passes=[self.answers(item.name, item.groups, SET_VERSION) for item in extra],
            edits=edits,
            limit=self.limit,
            stopped=self.stopped,
        )

    def close(self) -> None:
        """Close the backends that hold a connection open (the Jev client)."""
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if close is not None:
                close()


def dry_run(evaluation: Evaluation, plan: list[Job]) -> str:
    """Say what a run would ask and about what it would cost. Nothing is built, sent or written.

    Per step of the plan: the files it would ask about, the requests that takes (a window vote
    sends one per window), and for a billed backend the estimated input tokens (each card's
    size plus QUESTION_TOKENS per request) and their cost. Then the vault's Jev spend so far
    against its cap.
    """
    config = evaluation.config
    labels = evaluation.labels
    dev = sum(label.split == "dev" for label in labels)
    files = (
        f"{len(labels)} files (dev {dev}, test {len(labels) - dev}), "
        f"{len(evaluation.no_text())} with no text"
    )
    lines = [
        f"Dry run for {evaluation.root}",
        f"Set version {evaluation.group_set.version}, model {config.model}. {files}.",
        "",
        f"{'configuration':<32}{'to ask':>8}{'requests':>10}{'est. tokens':>14}{'est. USD':>11}",
    ]
    jev_tokens = 0
    for name, groups, set_version in plan:
        backend_key, _ = split_name(name)
        todo = evaluation.unanswered(read_rows(evaluation.root, name), groups, set_version)
        cards = [evaluation.asked_card(name, label.file_id) for label in todo]
        cards = [card for card in cards if card.states]
        requests = sum(len(card.states) for card in cards)
        usd_input, usd_output = prices(backend_key, config)
        line = f"{name:<32}{len(cards):>8}{requests:>10}"
        if usd_input is None:
            lines.append(f"{line}{'n/a':>14}{'free':>11}")
            continue
        tokens = sum(card.file.card_tokens for card in cards) + QUESTION_TOKENS * requests
        line += f"{tokens:>14,}{tokens / 1e6 * usd_input:>11.4f}"
        if usd_output is not None:
            line += "  input only; output tokens are billed too"
        lines.append(line)
        if backend_key == "jev":
            jev_tokens += tokens
    spent = total_input_tokens(decide_path(evaluation.root, JEV_LOG_FILE))
    usd = config.usd_per_million_tokens / 1e6
    lines.append("")
    lines.append(
        f"Jev: about {jev_tokens:,} tokens, ${jev_tokens * usd:.4f}. "
        f"Spent so far: ${spent * usd:.4f} of the ${config.max_spend_usd:.2f} cap."
    )
    if (spent + jev_tokens) * usd >= config.max_spend_usd:
        lines.append("That reaches the cap: the run would stop part way.")
    lines.append("Nothing was built, sent or written.")
    return "\n".join(lines)


class _Parser(argparse.ArgumentParser):
    """An argument parser whose usage errors exit with status 1: 2 means a stopped run here."""

    def error(self, message: str) -> NoReturn:
        """Print the problem as one line and exit with status 1."""
        self.exit(EXIT_ERROR, f"error: {message}\n")


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line."""
    parser = _Parser(
        prog="python -m decide_already.eval.run",
        description="Run every configuration over the evaluation vault and write the report.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(DEFAULT_ROOT),
        help=f"the evaluation vault (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--configs",
        default=",".join(CONFIG_NAMES),
        help="configurations to run, comma separated (default: all of them)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"threads for Jev requests (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--limit", type=int, help="only the first N files of each split, by file id"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be asked and the estimated cost; ask and write nothing",
    )
    parser.add_argument(
        "--no-experiments", action="store_true", help="skip the experiments' extra passes"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation, or its dry run, from the command line. Returns the exit status.

    0 when the run finished, 1 for a problem with the arguments or the vault, and 2 when a
    backend stopped the run: the spend cap was reached, or the call log could not be written.
    The report is still written from what is cached, and the reason is printed.
    """
    args = _arguments(argv)
    try:
        names = parse_configs(args.configs)
        if args.workers < 1 or (args.limit is not None and args.limit < 1):
            raise ValueError("--workers and --limit must be at least 1")
        root = args.root.resolve()
        if not Config.path(root).exists():
            raise ValueError(f"{root} is not a vault: it has no .decide/config.yaml")
        # Syncing the set can write (a new version file, an id into a group note). A dry run
        # must write nothing, so it reads the last version on record.
        group_set = vault.latest_set(root) if args.dry_run else vault.sync_group_set(root)
        if group_set is None:
            raise ValueError(f"{root} has no group set on record yet")
        evaluation = Evaluation(root, group_set, args.limit, args.workers)
        edits = read_edits(root)
        extra = passes(group_set.groups, edits)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    plan: list[Job] = [(name, group_set.groups, group_set.version) for name in names]
    if not args.no_experiments:
        # A pass copies one configuration, so it runs only when that one is in the run.
        plan += [(item.name, item.groups, SET_VERSION) for item in extra if item.base in names]
    if args.dry_run:
        print(dry_run(evaluation, plan))
        return 0
    try:
        for name, groups, set_version in plan:
            if evaluation.stopped:
                break
            print(evaluation.run(name, groups, set_version), flush=True)
    finally:
        evaluation.close()
    write_report(root, evaluation.inputs(extra, edits))
    print(f"Report: {root / RESULTS_DIR / REPORT_FILE}")
    if evaluation.stopped:
        print(f"stopped: {evaluation.stopped}", file=sys.stderr)
        print(
            "The report covers what is cached. Fix the cause (for the spend cap, raise "
            "max_spend_usd) and run again to finish.",
            file=sys.stderr,
        )
        return EXIT_STOPPED
    return 0


if __name__ == "__main__":
    sys.exit(main())
