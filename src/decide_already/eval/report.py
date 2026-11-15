"""The evaluation report (build steps 7 and 8): results/metrics.json and results/report.md.

Nothing here asks a model or reads the prediction cache. The runner (run.py) hands over what
each configuration answered (Inputs), build_metrics turns that into every number, and
render_report lays the numbers out as markdown tables.

How files are scored:

- The headline is top-1 accuracy on the test split, over the files that have extractable
  text, under the vault's default cutoffs.
- A file with no extractable text is never shown to a content-based configuration. A second
  table gives the accuracy over all test files, where such a file counts as wrong.
- A configuration is scored on the files it answered. When that is fewer files than the
  others (an error, a run cut short), the caveats say so.

The report states results. It does not make the go/no-go decision.
"""

import json
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path

from pydantic import Field

from ..config import Config
from ..types import Group, StrictModel
from . import experiments
from .baselines.llm import USD_PER_MILLION_INPUT_TOKENS, USD_PER_MILLION_OUTPUT_TOKENS
from .dataset import GoldLabel
from .experiments import Edit
from .groups import HELD_OUT_GROUP_ID
from .metrics import Cutoffs, Metrics, Prediction, summarize, tune_cutoffs

RESULTS_DIR = "results"
METRICS_FILE = "metrics.json"
REPORT_FILE = "report.md"

# The configurations the go/no-go table reads.
CARD = "jev:card"
RULES = "rules:card"
EMBEDDINGS = ("embed:card", "embed-chunks:card")
LONG_INPUTS = ("jev:full_text", "jev:window_vote")
WINDOW_VOTE = "jev:window_vote"

# The go/no-go numbers proposed in docs/phase0.md.
GO_ACCURACY = 0.90
GO_MARGIN_POINTS = 5.0

SMALL_GROUP = 5  # a group with fewer gold files than this among the scored files gets a caveat
LISTED = 10  # how many file ids a caveat names before it says "and N more"


class Answers(StrictModel):
    """What one configuration, or one experiment pass, answered: the runner's hand-over."""

    name: str  # the configuration name, such as "jev:card"
    backend: str  # the part before the colon: jev, rules, embed, embed-chunks or llm
    # One per file it answered, with the gold label attached.
    predictions: list[Prediction] = Field(default_factory=list)
    # File id to message, for each file it failed on.
    errors: dict[str, str] = Field(default_factory=dict)
    # Window vote only: file id to the top group of each of its windows.
    window_tops: dict[str, list[str | None]] = Field(default_factory=dict)
    pending: str | None = None  # why its backend could not be built, when that is so


class Inputs(StrictModel):
    """Everything the report is computed from."""

    config: Config
    set_version: int
    groups: list[Group]
    labels: list[GoldLabel]  # the files of this run, dev and test
    no_text: list[str]  # ids of the files with no extractable text
    answers: list[Answers]  # the configurations, in the order the tables show them
    # The experiments' extra passes (see experiments.py).
    passes: list[Answers] = Field(default_factory=list)
    edits: list[Edit] | None = None  # None when the vault has no description_edits.yaml
    limit: int | None = None  # set when the run took only the first files of each split
    stopped: str | None = None  # the spend cap's message, when it cut the run short


def prices(backend: str, config: Config) -> tuple[float | None, float | None]:
    """USD per million input and output tokens for a backend. None means nothing is billed.

    Jev bills input tokens only, at the vault config's price. The LLM baseline bills both, at
    the prices in baselines/llm.py. The rules and the embeddings run on this machine for free.
    """
    if backend == "jev":
        return config.usd_per_million_tokens, None
    if backend == "llm":
        return USD_PER_MILLION_INPUT_TOKENS, USD_PER_MILLION_OUTPUT_TOKENS
    return None, None


def reads_content(backend: str) -> bool:
    """Whether a backend needs a file's text. The file-name rules do not."""
    return backend != "rules"


def build_metrics(inputs: Inputs) -> dict:
    """Compute every number of the report. The result is what metrics.json holds."""
    config = inputs.config
    cutoffs = Cutoffs.from_config(config)
    no_text = set(inputs.no_text)
    test = [label for label in inputs.labels if label.split == "test"]
    dev = [label for label in inputs.labels if label.split == "dev"]
    test_ids = {label.file_id for label in test}
    dev_ids = {label.file_id for label in dev}
    test_text = test_ids - no_text
    dev_text = dev_ids - no_text

    headline: dict[str, dict | None] = {}
    all_files: dict[str, dict | None] = {}
    tuned: dict[str, dict] = {}
    for answers in inputs.answers:
        scored = _pick(answers.predictions, test_text)
        if not scored:
            headline[answers.name] = all_files[answers.name] = None
            continue
        usd = prices(answers.backend, config)
        row = _score(scored, cutoffs, usd)
        headline[answers.name] = row.model_dump()
        if reads_content(answers.backend):
            # The files with no text were never asked about. Here each counts as wrong.
            every = row.n + len(test_ids - test_text)
            all_files[answers.name] = {
                "n": every,
                "top1_accuracy": row.top1_accuracy * row.n / every,
            }
        else:
            whole = summarize(_pick(answers.predictions, test_ids), cutoffs, None)
            all_files[answers.name] = {"n": whole.n, "top1_accuracy": whole.top1_accuracy}
        if answers.backend == "jev":
            dev_scored = _pick(answers.predictions, dev_text)
            best = tune_cutoffs(dev_scored, cutoffs)
            tuned[answers.name] = {
                "dev_n": len(dev_scored),
                "cutoffs": best.model_dump(),
                "metrics": _score(scored, best, usd).model_dump(),
            }

    card = headline.get(CARD)
    variants = [a.name for a in inputs.answers if a.backend == "jev" and a.name != CARD]
    everything = test_ids | dev_ids
    text = test_text | dev_text
    metrics = {
        "model": config.model,
        "set_version": inputs.set_version,
        "limit": inputs.limit,
        "stopped": inputs.stopped,
        "cutoffs": cutoffs.model_dump(),
        "groups": {group.id: group.name for group in inputs.groups},
        "files": {
            "dev": len(dev),
            "test": len(test),
            "dev_with_text": len(dev_text),
            "test_with_text": len(test_text),
            "dev_no_text": len(dev_ids - dev_text),
            "test_no_text": len(test_ids - test_text),
            "test_gold": dict(
                Counter(label.gold_group or "none" for label in test if label.file_id in test_text)
            ),
        },
        "configurations": {a.name: _status(a, text, everything) for a in inputs.answers},
        "passes": {a.name: _status(a, text, everything) for a in inputs.passes},
        "headline": headline,
        "majority": _majority(test, test_text, cutoffs),
        "all_files": all_files,
        "tuned": tuned,
        "per_group": card["per_group"] if card else None,
        "variant_differences": _differences(headline, variants),
        "window_vote": _window_vote(inputs.answers, test_text),
        "go_no_go": _go_no_go(headline),
        "experiments": _experiments(inputs, text, cutoffs),
        "errors": {a.name: a.errors for a in [*inputs.answers, *inputs.passes] if a.errors},
    }
    metrics["caveats"] = _caveats(metrics)
    return metrics


def write_report(root: Path, inputs: Inputs) -> dict:
    """Compute the metrics and write results/metrics.json and results/report.md in the vault.

    Returns the metrics.
    """
    metrics = build_metrics(inputs)
    folder = root / RESULTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / METRICS_FILE).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (folder / REPORT_FILE).write_text(render_report(metrics), encoding="utf-8")
    return metrics


# The numbers


def _pick(predictions: Sequence[Prediction], ids: Collection[str]) -> list[Prediction]:
    """The predictions for the given files."""
    return [prediction for prediction in predictions if prediction.file_id in ids]


def _score(
    predictions: Sequence[Prediction], cutoffs: Cutoffs, usd: tuple[float | None, float | None]
) -> Metrics:
    """Every metric for some predictions, with the prices (input, output) of their backend."""
    row = summarize(predictions, cutoffs, *usd)
    if usd[0] is None:
        row.usd_per_file = 0.0  # a free backend reports no tokens, and costs nothing per file
    return row


def _status(answers: Answers, text: set[str], everything: set[str]) -> dict:
    """How far one configuration got over the files it should answer.

    complete: it answered them all. partial: it answered some, or failed on some. pending: its
    backend could not be built. not run: there are no answers and nothing went wrong.
    """
    expected = text if reads_content(answers.backend) else everything
    answered = len(_pick(answers.predictions, expected))
    failed = len(answers.errors)
    if answered == len(expected):
        state = "complete"
    elif answers.pending and not answered:
        state = "pending"
    elif not answered and not failed:
        state = "not run"
    else:
        state = "partial"
    return {
        "status": state,
        "pending": answers.pending,
        "expected": len(expected),
        "answered": answered,
        "errors": failed,
        "not_asked": max(len(expected) - answered - failed, 0),
    }


def _majority(test: Sequence[GoldLabel], test_text: set[str], cutoffs: Cutoffs) -> dict | None:
    """The majority-class row: every file is given the largest gold group of the test split.

    A tie goes to the lower group id. None when no test file has a gold group.
    """
    counts = Counter(label.gold_group for label in test if label.gold_group is not None)
    if not counts:
        return None
    group = min(counts, key=lambda group_id: (-counts[group_id], group_id))
    scored = [
        Prediction(file_id=label.file_id, gold=label.gold_group, probs={group: 1.0}, fits_any=1.0)
        for label in test
        if label.file_id in test_text
    ]
    return {
        "group": group,
        "headline": _score(scored, cutoffs, (None, None)).model_dump() if scored else None,
        "all_files": {"n": len(test), "top1_accuracy": counts[group] / len(test)},
    }


def _gap(value: float | None, reference: float | None) -> float | None:
    """value minus reference in percentage points. None when either is missing."""
    if value is None or reference is None:
        return None
    return (value - reference) * 100


def _differences(headline: dict[str, dict | None], variants: Sequence[str]) -> dict[str, dict]:
    """How each other Jev variant differs from the card, in percentage points.

    For each variant that has a headline row: the gap in top-1 accuracy, and the gap in
    recall (the top-1 accuracy within a group) for every group both rows hold.
    """
    card = headline.get(CARD)
    if card is None:
        return {}
    differences = {}
    for name in variants:
        row = headline.get(name)
        if row is None:
            continue
        differences[name] = {
            "top1_accuracy": _gap(row["top1_accuracy"], card["top1_accuracy"]),
            "recall": {
                group_id: _gap(row["per_group"][group_id]["recall"], group["recall"])
                for group_id, group in card["per_group"].items()
                if group_id in row["per_group"]
            },
        }
    return differences


def _window_vote(answers: Sequence[Answers], ids: set[str]) -> dict | None:
    """How often a file's windows disagree on the top group, over the given files.

    None when the window vote has no answers for them.
    """
    tops = [
        file_tops
        for configuration in answers
        if configuration.name == WINDOW_VOTE
        for file_id, file_tops in configuration.window_tops.items()
        if file_id in ids
    ]
    if not tops:
        return None
    disagree = sum(len(set(file_tops)) > 1 for file_tops in tops)
    return {
        "n": len(tops),
        "several_windows": sum(len(file_tops) > 1 for file_tops in tops),
        "disagree": disagree,
        "disagree_share": disagree / len(tops),
    }


def _all(checks: Sequence[bool | None]) -> bool | None:
    """True when every check passed, False when one failed, else None: it cannot be told yet."""
    if any(check is False for check in checks):
        return False
    return None if any(check is None for check in checks) else True


def _any(checks: Sequence[bool | None]) -> bool | None:
    """True when one check passed, False when every check failed, else None."""
    if any(check is True for check in checks):
        return True
    return None if any(check is None for check in checks) else False


def _go_no_go(headline: dict[str, dict | None]) -> list[dict]:
    """The three rows of the go/no-go table in docs/phase0.md, with the measured numbers.

    Every number is a headline top-1 accuracy. `met` is None when a number the condition
    needs is missing and the numbers at hand do not already settle it.
    """

    def accuracy(name: str) -> float | None:
        """A configuration's headline top-1 accuracy, when it has one."""
        row = headline.get(name)
        return row["top1_accuracy"] if row else None

    card = accuracy(CARD)

    def card_above(name: str) -> bool | None:
        """Whether the card's accuracy is above another configuration's."""
        other = accuracy(name)
        return None if card is None or other is None else card > other

    def beats_card(name: str) -> bool | None:
        """Whether a configuration beats the card by the go/no-go margin."""
        gap = _gap(accuracy(name), card)
        return None if gap is None else gap >= GO_MARGIN_POINTS - 1e-9

    above_embeddings = [card_above(name) for name in EMBEDDINGS]
    beats_embeddings = _all(above_embeddings)
    return [
        {
            "result": f"Card accuracy of {GO_ACCURACY:.0%} or more, above both non-LLM baselines",
            "then": "Start Phase 1 as designed",
            "measured": {name: accuracy(name) for name in (CARD, RULES, *EMBEDDINGS)},
            "met": _all(
                [
                    None if card is None else card >= GO_ACCURACY,
                    card_above(RULES),
                    *above_embeddings,
                ]
            ),
        },
        {
            "result": (
                f"Full text or window vote beats the card by {GO_MARGIN_POINTS:g} points or more"
            ),
            "then": "Use that input for D1",
            "measured": {name: accuracy(name) for name in (*LONG_INPUTS, CARD)},
            "met": _any([beats_card(name) for name in LONG_INPUTS]),
        },
        {
            "result": "Jev doesn't beat embedding similarity",
            "then": "Rethink Jev's role on the write path before Phase 1",
            "measured": {name: accuracy(name) for name in (CARD, *EMBEDDINGS)},
            "met": None if beats_embeddings is None else not beats_embeddings,
        },
    ]


def _experiments(inputs: Inputs, text: set[str], cutoffs: Cutoffs) -> dict:
    """The experiments' numbers, over every file with text (dev and test).

    A pass with no answers gives None. description_edits is None when the vault has no edits
    file, else one entry per edit.
    """
    by_name = {answers.name: answers for answers in [*inputs.answers, *inputs.passes]}

    def predictions(name: str) -> list[Prediction]:
        """What a configuration or a pass answered for the files with text."""
        return _pick(by_name[name].predictions, text) if name in by_name else []

    group_added = {}
    for base in experiments.HELD_OUT_BASES:
        reduced = predictions(experiments.held_out_name(base))
        group_added[base] = (
            experiments.group_added(reduced, predictions(base), cutoffs) if reduced else None
        )
    edits = None
    if inputs.edits is not None:
        before = predictions(experiments.EDIT_BASE)
        edits = []
        for edit in inputs.edits:
            after = predictions(experiments.edit_name(edit))
            effect = None
            if before and after:
                effect = experiments.edit_effect(before, after, edit.target_file_ids)
            edits.append({**edit.model_dump(), "effect": effect})
    return {
        "held_out_group": HELD_OUT_GROUP_ID,
        "has_held_out_group": any(group.id == HELD_OUT_GROUP_ID for group in inputs.groups),
        "group_added": group_added,
        "description_edits": edits,
    }


def _listed(file_ids: Sequence[str]) -> str:
    """A few file ids, comma separated, with a count of the rest."""
    shown = ", ".join(file_ids[:LISTED])
    more = len(file_ids) - LISTED
    return f"{shown} and {more} more" if more > 0 else shown


def _caveats(metrics: dict) -> list[str]:
    """What a reader should know before trusting the numbers. Every line comes from the data."""
    files = metrics["files"]
    notes = []
    if metrics["limit"] is not None:
        notes.append(
            f"This run took only the first files of each split (--limit {metrics['limit']}): "
            f"every number rests on {files['dev'] + files['test']} files in all."
        )
    if metrics["stopped"]:
        notes.append(
            f"The run stopped at the spend cap ({metrics['stopped']}). "
            "Whatever had not been asked by then is missing."
        )
    small = [
        f"{name} {files['test_gold'].get(group_id, 0)}"
        for group_id, name in metrics["groups"].items()
        if files["test_gold"].get(group_id, 0) < SMALL_GROUP
    ]
    if small:
        notes.append(
            f"Groups with fewer than {SMALL_GROUP} gold files among the test files with text: "
            f"{', '.join(small)}. Their per-group numbers, and their share of the macro-F1, "
            "rest on very few files."
        )
    notes.append(
        f"Files in the dev split: {files['dev']} ({files['dev_with_text']} with text). "
        "The tuned cutoffs rest on them."
    )
    if files["dev_no_text"] or files["test_no_text"]:
        notes.append(
            f"Files with no extractable text: {files['dev_no_text'] + files['test_no_text']} "
            f"({files['test_no_text']} in the test split). No content-based configuration is "
            "asked about them. The headline leaves them out; the all-files table counts them "
            "as wrong."
        )
    statuses = [*metrics["configurations"].items(), *metrics["passes"].items()]
    for name, status in statuses:
        if status["pending"]:
            notes.append(f"{name} is pending: {status['pending']}")
        elif status["status"] == "not run" and name in metrics["configurations"]:
            notes.append(f"{name} was not run: it has no cached answers for these files.")
        if status["status"] == "partial":
            note = (
                f"{name} answered {status['answered']} of {status['expected']} files, so its "
                "numbers cover only those."
            )
            failed = sorted(metrics["errors"].get(name, {}))
            if failed:
                note += (
                    f" Failed: {len(failed)} ({_listed(failed)}); the messages are in "
                    f"{METRICS_FILE} under errors."
                )
            if status["not_asked"]:
                note += f" Not asked yet: {status['not_asked']}."
            notes.append(note)
    for edit in metrics["experiments"]["description_edits"] or []:
        missing = edit["effect"]["targets_missing"] if edit["effect"] else []
        if missing:
            notes.append(
                f"Edit {edit['name']} lists target files that could not be compared: "
                f"{_listed(missing)}."
            )
    return notes


# The markdown


def _pct(value: float | None) -> str:
    """A share as a percentage with one decimal."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 0) -> str:
    """A number with a fixed count of decimals."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None) -> str:
    """A difference in points, with its sign. Adding 0.0 turns a rounded "-0.0" into "+0.0"."""
    return "n/a" if value is None else f"{round(value, 1) + 0.0:+.1f}"


def _usd(value: float | None) -> str:
    """A cost per file in USD. Six decimals, because one Jev call costs about $0.0001."""
    return "n/a" if value is None else f"${value:.6f}"


def _yes_no(value: bool | None) -> str:
    """A three-way answer."""
    return "n/a" if value is None else "yes" if value else "no"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A markdown table. A pipe inside a cell is escaped so that it cannot split the cell."""
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join(
        "| " + " | ".join(str(cell).replace("|", "\\|") for cell in line) + " |" for line in lines
    )


def render_report(metrics: dict) -> str:
    """Lay the metrics out as markdown: plain tables, fixed precision, None shown as "n/a"."""
    sections = [
        _intro(metrics),
        _headline_section(metrics),
        _all_files_section(metrics),
        _tuned_section(metrics),
        _per_group_section(metrics),
        _window_section(metrics),
        _experiments_section(metrics),
        _go_no_go_section(metrics),
        _caveats_section(metrics),
    ]
    return "\n\n".join(sections) + "\n"


def _intro(metrics: dict) -> str:
    """The title and the facts every table rests on."""
    files = metrics["files"]
    cutoffs = metrics["cutoffs"]
    no_text = files["dev_no_text"] + files["test_no_text"]
    return "\n".join(
        [
            "# Phase 0 evaluation report",
            "",
            f"- Model: {metrics['model']}. Group set version: {metrics['set_version']}.",
            (
                f"- Files: {files['dev'] + files['test']} "
                f"(dev {files['dev']}, test {files['test']}). "
                f"With no extractable text: {no_text} "
                f"(dev {files['dev_no_text']}, test {files['test_no_text']})."
            ),
            (
                "- Default cutoffs: Unsorted when fits_any is under "
                f"{cutoffs['fits_any_below']:.2f} or the top group is under "
                f"{cutoffs['top_below']:.2f}; close call when the top two groups are within "
                f"{cutoffs['close_margin']:.2f}."
            ),
            (
                "- Unless a table says otherwise, numbers are for the test split, over its "
                f"{files['test_with_text']} files that have text."
            ),
            "",
            "This report states results. It does not make the go/no-go decision.",
        ]
    )


def _majority_label(metrics: dict) -> str:
    """The name of the majority-class row, with the group it always picks."""
    group_id = metrics["majority"]["group"]
    return f"majority class ({metrics['groups'].get(group_id, group_id)})"


def _headline_section(metrics: dict) -> str:
    """The headline table: one row per configuration that has answers, then the majority row."""
    header = [
        "Configuration",
        "n",
        "Top-1",
        "Macro-F1",
        "Placed",
        "Placed right",
        "Unsorted",
        "Unsorted, top right",
        "Close calls",
        "ECE",
        "Input tokens",
        "USD per file",
        "p50 ms",
        "p95 ms",
    ]

    def cells(label: str, row: dict) -> list[str]:
        """One table row from one configuration's metrics."""
        return [
            label,
            str(row["n"]),
            _pct(row["top1_accuracy"]),
            _num(row["macro_f1"], 3),
            _pct(row["placed_share"]),
            _pct(row["placed_accuracy"]),
            _pct(row["unsorted_share"]),
            _pct(row["unsorted_top1_right_share"]),
            _pct(row["close_call_share"]),
            _num(row["ece"], 3),
            _num(row["mean_input_tokens"]),
            _usd(row["usd_per_file"]),
            _num(row["latency_p50_ms"]),
            _num(row["latency_p95_ms"]),
        ]

    rows = [cells(name, row) for name, row in metrics["headline"].items() if row]
    majority = metrics["majority"]
    if majority and majority["headline"]:
        rows.append(cells(_majority_label(metrics), majority["headline"]))
    absent = [
        f"{name} ({metrics['configurations'][name]['status']})"
        for name, row in metrics["headline"].items()
        if row is None
    ]
    lines = [
        "## Headline",
        "",
        "Test split, files with text, default cutoffs.",
        "",
        _table(header, rows),
        "",
        (
            "Top-1 is the share of files whose top group is their gold group; it ignores the "
            "cutoffs. Placed right is the share of placed files that sit in their gold group. "
            "Unsorted, top right is the share of Unsorted files whose top group was their gold "
            "group. Input tokens and USD per file are means per file."
        ),
    ]
    if absent:
        lines += ["", f"No row for: {', '.join(absent)}. See the caveats."]
    return "\n".join(lines)


def _all_files_section(metrics: dict) -> str:
    """Accuracy over every test file, next to the headline accuracy."""
    rows = [
        [
            name,
            str(metrics["headline"][name]["n"]),
            _pct(metrics["headline"][name]["top1_accuracy"]),
            str(row["n"]),
            _pct(row["top1_accuracy"]),
        ]
        for name, row in metrics["all_files"].items()
        if row
    ]
    majority = metrics["majority"]
    if majority and majority["headline"]:
        rows.append(
            [
                _majority_label(metrics),
                str(majority["headline"]["n"]),
                _pct(majority["headline"]["top1_accuracy"]),
                str(majority["all_files"]["n"]),
                _pct(majority["all_files"]["top1_accuracy"]),
            ]
        )
    header = ["Configuration", "Files with text", "Top-1", "All files", "Top-1, all files"]
    return "\n".join(
        [
            "## All test files",
            "",
            (
                f"Test files with no extractable text: {metrics['files']['test_no_text']} of "
                f"{metrics['files']['test']}. No content-based configuration is asked about "
                "them, so here each counts as wrong for those. The file-name rules and the "
                "majority class need no text, so they are scored on them."
            ),
            "",
            _table(header, rows),
        ]
    )


def _tuned_section(metrics: dict) -> str:
    """For each Jev configuration, the test metrics under the default and the tuned cutoffs."""

    def cells(name: str, label: str, cutoffs: dict, row: dict) -> list[str]:
        """One table row: a configuration under one pair of cutoffs."""
        return [
            name,
            label,
            _num(cutoffs["fits_any_below"], 2),
            _num(cutoffs["top_below"], 2),
            _pct(row["placed_share"]),
            _pct(row["placed_accuracy"]),
            _pct(row["unsorted_share"]),
            _pct(row["unsorted_top1_right_share"]),
            _pct(row["close_call_share"]),
        ]

    rows = []
    for name, tuned in metrics["tuned"].items():
        rows.append(cells(name, "default", metrics["cutoffs"], metrics["headline"][name]))
        label = f"tuned on dev (n = {tuned['dev_n']})"
        rows.append(cells(name, label, tuned["cutoffs"], tuned["metrics"]))
    header = [
        "Configuration",
        "Cutoffs",
        "fits_any under",
        "Top under",
        "Placed",
        "Placed right",
        "Unsorted",
        "Unsorted, top right",
        "Close calls",
    ]
    lines = [
        "## Tuned cutoffs",
        "",
        (
            "The two Unsorted cutoffs are tuned on the dev split for each Jev configuration, to "
            "get the most right placements minus wrong placements, and then applied to the test "
            "split. The defaults stay the headline. Top-1 does not depend on the cutoffs."
        ),
        "",
    ]
    lines.append(_table(header, rows) if rows else "No Jev configuration has answers yet.")
    return "\n".join(lines)


def _per_group_section(metrics: dict) -> str:
    """The card's scores per group, then how every other Jev variant differs from it."""
    names = metrics["groups"]
    lines = [f"## Per group: {CARD}", ""]
    per_group = metrics["per_group"]
    if not per_group:
        return "\n".join([*lines, f"{CARD} has no answers yet."])
    rows = [
        [
            names.get(group_id, group_id),
            str(group["n"]),
            _pct(group["precision"]),
            _pct(group["recall"]),
            _num(group["f1"], 3),
        ]
        for group_id, group in per_group.items()
    ]
    lines.append(_table(["Group", "n", "Precision", "Recall", "F1"], rows))
    differences = metrics["variant_differences"]
    if differences:
        header = ["Variant", "Top-1", *(names.get(group_id, group_id) for group_id in per_group)]
        rows = [
            [
                name,
                _signed(difference["top1_accuracy"]),
                *(_signed(difference["recall"].get(group_id)) for group_id in per_group),
            ]
            for name, difference in differences.items()
        ]
        lines += [
            "",
            f"Each variant minus {CARD}, in points: top-1 accuracy, then recall per group.",
            "",
            _table(header, rows),
        ]
    return "\n".join(lines)


def _window_section(metrics: dict) -> str:
    """How often the windows of one file disagree on the top group."""
    vote = metrics["window_vote"]
    lines = ["## Window vote", ""]
    if not vote:
        return "\n".join([*lines, f"{WINDOW_VOTE} has no answers yet."])
    lines.append(
        f"Test files {WINDOW_VOTE} answered: {vote['n']}. With more than one window: "
        f"{vote['several_windows']}. With windows that disagree on the top group: "
        f"{vote['disagree']}, which is {_pct(vote['disagree_share'])} of the {vote['n']}."
    )
    return "\n".join(lines)


def _group_added_section(metrics: dict) -> list[str]:
    """Experiments 1 and 2: the passes without the held-out group, side by side."""
    tests = metrics["experiments"]
    held_out = tests["held_out_group"]
    lines = [f"### Group added mid-run (held out: {metrics['groups'].get(held_out, held_out)})", ""]
    shown = {base: numbers for base, numbers in tests["group_added"].items() if numbers}
    if not tests["has_held_out_group"]:
        return [*lines, f"Not run: this vault has no group with the id {held_out}."]
    if not shown:
        return [*lines, "Not run: no answers are cached for the passes without the group."]
    measures = [
        ("Files sorted without the group", "n", str),
        ("Held-out files", "held_out_n", str),
        ("Held-out files sent to Unsorted", "held_out_unsorted_share", _pct),
        ("Mean fits_any of the held-out files", "held_out_mean_fits_any", lambda v: _num(v, 3)),
        ("Files of the other groups", "others_n", str),
        ("Their top-1 accuracy", "others_top1_accuracy", _pct),
        ("Of them, wrongly sent to Unsorted", "others_unsorted_share", _pct),
        ("AUROC of fits_any", "fits_any_auroc", lambda v: _num(v, 3)),
        ("Files frozen (placed in their gold group)", "frozen_n", str),
        ("Files rerun against the full set", "rerun_n", str),
        ("Rerun accuracy", "rerun_accuracy", _pct),
    ]
    rows = [
        [label, *(show(numbers[key]) for numbers in shown.values())]
        for label, key, show in measures
    ]
    text = (
        "Every file with text (dev and test) is sorted with the group left out, under the "
        "default cutoffs. The held-out group's files then fit no group, so Unsorted is right "
        "for them. The AUROC says how well fits_any tells the other groups' files from them. "
        "For the rerun, a user freezes each file placed in its gold group and sorts the rest "
        "against the full set: rerun accuracy is the share of those placed right."
    )
    names_only = experiments.HELD_OUT_BASES[1]
    if names_only in shown:
        text += f" The {names_only} column asks with group names only, without descriptions."
    return [*lines, text, "", _table(["Measure", *shown], rows)]


def _edits_section(tests: dict) -> list[str]:
    """Experiment 3: what each description edit changed, against the card's answers before."""
    lines = ["### Description edits", ""]
    edits = tests["description_edits"]
    if edits is None:
        return [*lines, f"Not run: the vault has no {experiments.EDITS_FILE}."]
    if not edits:
        return [*lines, f"Not run: the vault's {experiments.EDITS_FILE} lists no edits."]
    rows = []
    for edit in edits:
        effect = edit["effect"]
        if effect is None:
            rows.append([edit["name"], edit["group_id"], *["n/a"] * 8])
            continue
        rows.append(
            [
                edit["name"],
                edit["group_id"],
                str(effect["n"]),
                str(effect["targets_n"]),
                str(len(effect["targets_fixed"])),
                str(len(effect["wrong_to_right"])),
                str(len(effect["right_to_wrong"])),
                str(len(effect["wrong_to_other_wrong"])),
                _pct(effect["top1_before"]),
                _pct(effect["top1_after"]),
            ]
        )
    header = [
        "Edit",
        "Group",
        "Files",
        "Targets",
        "Targets fixed",
        "Others wrong to right",
        "Others right to wrong",
        "Others between wrong groups",
        "Top-1 before",
        "Top-1 after",
    ]
    return [
        *lines,
        (
            f"Each edit changes one group's description. {experiments.EDIT_BASE} sorts every file "
            "with text (dev and test) again under it, and each file's top group is compared with "
            "the one before (cutoffs play no part). Targets are the files the edit was meant to "
            f"fix. The file ids behind the counts are in {METRICS_FILE}."
        ),
        "",
        _table(header, rows),
    ]


def _experiments_section(metrics: dict) -> str:
    """The experiments: group added mid-run with both wordings, then the description edits."""
    tests = metrics["experiments"]
    lines = ["## Experiments", "", *_group_added_section(metrics), "", *_edits_section(tests)]
    return "\n".join(lines)


def _go_no_go_section(metrics: dict) -> str:
    """The go/no-go table: each condition, what was measured, and whether it is met."""

    def measured(name: str, value: float | None) -> str:
        """One accuracy with the number of files behind it, so uneven rows show here too."""
        row = metrics["headline"].get(name)
        return f"{name} {_pct(value)} (n {row['n']})" if row else f"{name} n/a"

    rows = [
        [
            row["result"],
            row["then"],
            "; ".join(measured(name, value) for name, value in row["measured"].items()),
            _yes_no(row["met"]),
        ]
        for row in metrics["go_no_go"]
    ]
    return "\n".join(
        [
            "## Go/no-go",
            "",
            (
                "The conditions are the proposed defaults in docs/phase0.md. The numbers are the "
                "headline top-1 accuracies, each with the number of files it was scored on. "
                '"n/a" means a number the condition needs is missing.'
            ),
            "",
            _table(["Result", "Then", "Measured", "Condition met"], rows),
        ]
    )


def _caveats_section(metrics: dict) -> str:
    """The caveats, one per line."""
    return "\n".join(["## Caveats", "", *(f"- {note}" for note in metrics["caveats"])])
