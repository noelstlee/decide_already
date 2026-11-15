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

What the report does not print as a result:

- A difference the data cannot carry. A go/no-go condition that needs one configuration above
  another is checked file by file, on the files both answered, and counts as met only when
  an exact sign test on the discordant files separates the two. Otherwise it reads "not
  separable", or "cannot be told" when the card is so close to 100% that no input could gain
  the margin on it.
- A cell that is fixed by construction: a close call under cutoffs that can never flag one,
  cutoffs tuned on a dev split that cannot tell them apart, and placement or calibration
  numbers for a backend whose probabilities are one-hot or set by a temperature.

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
from .metrics import (
    SIGN_TEST_ALPHA,
    Cutoffs,
    Metrics,
    Prediction,
    close_call_can_fire,
    min_discordant,
    paired,
    summarize,
    top1_right,
    top_group,
    tune_cutoffs,
    tuning_ties,
)

RESULTS_DIR = "results"
METRICS_FILE = "metrics.json"
REPORT_FILE = "report.md"

# The configurations the go/no-go table reads.
CARD = "jev:card"
RULES = "rules:card"
EMBEDDINGS = ("embed:card", "embed-chunks:card")
LONG_INPUTS = ("jev:full_text", "jev:window_vote")
WINDOW_VOTE = "jev:window_vote"
NAMES_ONLY = "jev:names_only"

# The go/no-go numbers proposed in docs/phase0.md.
GO_ACCURACY = 0.90
GO_MARGIN_POINTS = 5.0

# What a go/no-go row, or one comparison in it, can come to.
YES = "yes"
NO = "no"  # the numbers do not meet the condition
NOT_SEPARABLE = "not separable"  # the numbers meet it, but the sign test cannot carry it
AT_CEILING = "cannot be told: the card is at the ceiling"
MISSING = "n/a"  # a number the condition needs is missing

# Backends whose probabilities are the model's own. The others put the whole probability on one
# pick (the rules, the LLM) or shape it with a temperature set by hand (the embeddings), so
# where the cutoffs put their files, and how calibrated they look, is fixed by construction.
OWN_PROBABILITIES = ("jev",)

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
    # Why the run was cut short, when it was: the spend cap, or a call log that cannot be written.
    stopped: str | None = None


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


def own_probabilities(name: str) -> bool:
    """Whether a configuration's probabilities are measured: only Jev reports its own."""
    return name.partition(":")[0] in OWN_PROBABILITIES


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
    scored_by: dict[str, list[Prediction]] = {}  # the files each headline row is scored on
    for answers in inputs.answers:
        scored = _pick(answers.predictions, test_text)
        scored_by[answers.name] = scored
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
            ties = tuning_ties(dev_scored, cutoffs)
            tuned[answers.name] = {
                "dev_n": len(dev_scored),
                "cutoffs": best.model_dump(),
                "metrics": _score(scored, best, usd).model_dump(),
                "scored_pairs": ties.pairs,
                "tied_pairs": ties.tied,
                # The defaults among several pairs that tie for best: the dev split told
                # nothing apart, and the cutoffs above are only the defaults again.
                "tunable": not (ties.defaults_tied and ties.tied > 1),
                "close_call_can_fire": close_call_can_fire(best),
            }

    card = headline.get(CARD)
    variants = [a.name for a in inputs.answers if a.backend == "jev" and a.name != CARD]
    everything = test_ids | dev_ids
    text = test_text | dev_text
    ceiling = _ceiling(scored_by.get(CARD, []), cutoffs)
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
            # Files that fit no group. Only they can teach the tuning a higher cutoff.
            "no_gold": sum(label.gold_group is None for label in inputs.labels),
        },
        "configurations": {a.name: _status(a, text, everything) for a in inputs.answers},
        "passes": {a.name: _status(a, text, everything) for a in inputs.passes},
        "headline": headline,
        "majority": _majority(test, test_text, cutoffs),
        "all_files": all_files,
        "by_construction": _by_construction(scored_by, cutoffs),
        "sensitivity": _sensitivity(inputs, test_text, cutoffs),
        "tuned": tuned,
        "per_group": card["per_group"] if card else None,
        "variant_differences": _differences(headline, variants, scored_by, cutoffs),
        "ceiling": ceiling,
        "window_vote": _window_vote(inputs.answers, test_text),
        "go_no_go": _go_no_go(headline, scored_by, cutoffs, ceiling),
        "experiments": _experiments(inputs, text, cutoffs),
        "label_signal": _label_signal(inputs.labels, test_text, scored_by.get(RULES, []), cutoffs),
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
    """The majority-class row: every file is given the largest gold group of the files scored.

    The headline row picks the group among the test files with text, and the all-files number
    picks it among all test files. Each is then the best a constant answer can do on its own
    files; one group picked over all files could score below that on the files with text.
    A tie goes to the lower group id. None when no test file has a gold group.
    """

    def largest(labels: Sequence[GoldLabel]) -> tuple[str | None, int]:
        """The largest gold group among some files and its size; (None, 0) when none has one."""
        counts = Counter(label.gold_group for label in labels if label.gold_group is not None)
        if not counts:
            return None, 0
        group = min(counts, key=lambda group_id: (-counts[group_id], group_id))
        return group, counts[group]

    every, every_count = largest(test)
    if every is None:
        return None
    with_text = [label for label in test if label.file_id in test_text]
    group, _ = largest(with_text)
    scored = [
        Prediction(file_id=label.file_id, gold=label.gold_group, probs={group: 1.0}, fits_any=1.0)
        for label in with_text
        if group is not None
    ]
    return {
        "group": group,
        "headline": _score(scored, cutoffs, (None, None)).model_dump() if scored else None,
        "all_files": {"group": every, "n": len(test), "top1_accuracy": every_count / len(test)},
    }


def _by_construction(scored_by: dict[str, list[Prediction]], cutoffs: Cutoffs) -> dict:
    """What the headline table must not show as a measurement.

    - close_call_can_fire: False when the default cutoffs can never flag a close call.
    - probabilities_not_measured: the configurations with a headline row whose probabilities
      are not the model's own. Their placement and calibration cells are left empty.
    - no_pick: for each of those, how many of its scored files it had no pick for at all (no
      rule matched). Such a file counts as wrong on top-1.
    """
    fixed = [name for name, scored in scored_by.items() if scored and not own_probabilities(name)]
    no_pick = {name: sum(not p.probs for p in scored_by[name]) for name in fixed}
    return {
        "close_call_can_fire": close_call_can_fire(cutoffs),
        "probabilities_not_measured": fixed,
        "no_pick": {name: count for name, count in no_pick.items() if count},
    }


def _sensitivity(inputs: Inputs, test_text: set[str], cutoffs: Cutoffs) -> dict | None:
    """The card's top-1 under other wordings of the same groups, on the same test files.

    The wordings are the vault's descriptions (the headline), group names only, and each
    description edit. Only the files every one of them answered are counted. rules_right is
    how many of those files the file-name rules got right, when they answered them all.
    None when the card has no answers, or no other wording does.
    """
    by_name = {answers.name: answers for answers in [*inputs.answers, *inputs.passes]}
    wordings = [("the vault's descriptions", CARD), ("group names only", NAMES_ONLY)]
    wordings += [(f"edit {edit.name}", experiments.edit_name(edit)) for edit in inputs.edits or []]
    answered = [
        (wording, name, _pick(by_name[name].predictions, test_text))
        for wording, name in wordings
        if name in by_name
    ]
    answered = [item for item in answered if item[2]]
    if len(answered) < 2 or answered[0][1] != CARD:
        return None
    shared = set.intersection(*({p.file_id for p in scored} for _, _, scored in answered))
    if not shared:
        return None

    def right(scored: Sequence[Prediction]) -> int:
        """How many of the shared files some answers got right on top-1."""
        return sum(top1_right(p, cutoffs) for p in scored if p.file_id in shared)

    rules = _pick(by_name[RULES].predictions, shared) if RULES in by_name else []
    return {
        "n": len(shared),
        "rows": [
            {"wording": wording, "name": name, "right": right(scored)}
            for wording, name, scored in answered
        ],
        "rules_right": right(rules) if len(rules) == len(shared) else None,
    }


def _label_signal(
    labels: Sequence[GoldLabel],
    test_text: set[str],
    rules_scored: Sequence[Prediction],
    cutoffs: Cutoffs,
) -> dict:
    """Where a file's path alone gives its gold group away.

    - folders: each top-level folder of the file ids that holds at least two of this run's
      files (dev and test), all with the same gold group. Its name tells the group.
    - test_with_text: how many of the scored test files sit under such a folder.
    - rules: how the file-name rules, which read only names and paths, did on the test files
      with text. None when they have no answers.
    """
    by_folder: dict[str, list[GoldLabel]] = {}
    for label in labels:
        folder, slash, _ = label.file_id.partition("/")
        if slash:
            by_folder.setdefault(folder, []).append(label)
    folders = {
        folder: {"gold_group": members[0].gold_group, "files": len(members)}
        for folder, members in sorted(by_folder.items())
        if len(members) > 1 and len({member.gold_group for member in members}) == 1
    }
    under = sum("/" in file_id and file_id.partition("/")[0] in folders for file_id in test_text)
    rules = None
    if rules_scored:
        right = sum(top1_right(prediction, cutoffs) for prediction in rules_scored)
        rules = {"n": len(rules_scored), "right": right}
    return {"folders": folders, "test_with_text": under, "rules": rules}


def _ceiling(scored: Sequence[Prediction], cutoffs: Cutoffs) -> dict | None:
    """Whether the card is so close to 100% that no input could beat it by the go/no-go margin.

    `scored` holds the card's answers for the test files with text. at_ceiling is True when
    the files it got wrong are fewer than GO_MARGIN_POINTS points of them: then "beats the
    card by 5 points" cannot come out as met, whatever another input does. None with no answers.
    """
    if not scored:
        return None
    right = sum(top1_right(prediction, cutoffs) for prediction in scored)
    room = (len(scored) - right) / len(scored) * 100
    return {
        "right": right,
        "n": len(scored),
        "margin_points": GO_MARGIN_POINTS,
        "at_ceiling": room < GO_MARGIN_POINTS - 1e-9,
    }


def _gap(value: float | None, reference: float | None) -> float | None:
    """value minus reference in percentage points. None when either is missing."""
    if value is None or reference is None:
        return None
    return (value - reference) * 100


def _differences(
    headline: dict[str, dict | None],
    variants: Sequence[str],
    scored_by: dict[str, list[Prediction]],
    cutoffs: Cutoffs,
) -> dict[str, dict]:
    """How each other Jev variant differs from the card.

    For each variant that has a headline row: the gap in top-1 accuracy in points, the gap in
    recall (the top-1 accuracy within a group) for every group both rows hold, and the paired
    count of the files only one of the two got right (the variant first, then the card).
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
            "paired": paired(scored_by[name], scored_by[CARD], cutoffs).model_dump(),
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


def _all(outcomes: Sequence[str]) -> str:
    """Every check must hold: NO when one failed, else what keeps it from a plain YES."""
    for outcome in (NO, MISSING, NOT_SEPARABLE):
        if outcome in outcomes:
            return outcome
    return YES


def _any(outcomes: Sequence[str]) -> str:
    """One check is enough: YES when one held, else what keeps it from a plain NO."""
    for outcome in (YES, AT_CEILING, MISSING, NOT_SEPARABLE):
        if outcome in outcomes:
            return outcome
    return NO


def _comparison(
    scored_by: dict[str, list[Prediction]],
    cutoffs: Cutoffs,
    first: str,
    second: str,
    margin: float = 0.0,
    at_ceiling: bool = False,
) -> dict:
    """One paired comparison of the go/no-go table: is `first` above `second`?

    Both are counted on the files both answered (see metrics.paired). With a margin, `first`
    must be ahead by that many points of those files; without one, by any file at all.

    - NO: the counts do not put `first` ahead by that much.
    - YES: they do, and the sign test on the discordant files is under SIGN_TEST_ALPHA.
    - NOT_SEPARABLE: they do, but on too few discordant files for the test to tell.
    - AT_CEILING: with a margin, when `second` (the card) leaves less room than the margin, so
      the counts could never put `first` ahead by it. Only a loss can still be told there:
      `first` separably behind is a NO.
    - MISSING: the two share no answered file.
    """
    pair = paired(scored_by.get(first) or [], scored_by.get(second) or [], cutoffs)
    lead = pair.first_right - pair.second_right
    separable = pair.p_value < SIGN_TEST_ALPHA
    if pair.n == 0:
        outcome = MISSING
    elif margin and at_ceiling:
        outcome = NO if lead < 0 and separable else AT_CEILING
    elif (lead / pair.n * 100 < margin - 1e-9) if margin else lead <= 0:
        outcome = NO
    else:
        outcome = YES if separable else NOT_SEPARABLE
    return {
        "first": first,
        "second": second,
        "margin_points": margin,
        **pair.model_dump(),
        "outcome": outcome,
    }


def _go_no_go(
    headline: dict[str, dict | None],
    scored_by: dict[str, list[Prediction]],
    cutoffs: Cutoffs,
    ceiling: dict | None,
) -> list[dict]:
    """The three rows of the go/no-go table in docs/phase0.md, with what was measured.

    - measured: the headline top-1 accuracy of each configuration the row names.
    - comparisons: each paired comparison the row needs (see _comparison).
    - verdict: YES, NO, NOT_SEPARABLE, AT_CEILING or MISSING. A condition that needs one
      configuration above another is YES only when the two are separable, so no row is called
      met on a difference the data cannot carry. The second row is AT_CEILING when the card
      leaves less room than its margin, unless both long inputs are separably behind it.
    - met: True for YES, False for NO, and None for every verdict in between.
    """

    def accuracy(name: str) -> float | None:
        """A configuration's headline top-1 accuracy, when it has one."""
        row = headline.get(name)
        return row["top1_accuracy"] if row else None

    card = accuracy(CARD)
    at_ceiling = bool(ceiling and ceiling["at_ceiling"])
    enough = MISSING if card is None else YES if card >= GO_ACCURACY else NO
    over_rules = _comparison(scored_by, cutoffs, CARD, RULES)
    over_embeddings = [_comparison(scored_by, cutoffs, CARD, name) for name in EMBEDDINGS]
    long_inputs = [
        _comparison(scored_by, cutoffs, name, CARD, GO_MARGIN_POINTS, at_ceiling)
        for name in LONG_INPUTS
    ]
    beats_embeddings = _all([item["outcome"] for item in over_embeddings])
    rows = [
        (
            f"Card accuracy of {GO_ACCURACY:.0%} or more, above both non-LLM baselines",
            "Start Phase 1 as designed",
            (CARD, RULES, *EMBEDDINGS),
            [over_rules, *over_embeddings],
            _all([enough, over_rules["outcome"], beats_embeddings]),
        ),
        (
            f"Full text or window vote beats the card by {GO_MARGIN_POINTS:g} points or more",
            "Use that input for D1",
            (*LONG_INPUTS, CARD),
            long_inputs,
            _any([item["outcome"] for item in long_inputs]),
        ),
        (
            "Jev doesn't beat embedding similarity",
            "Rethink Jev's role on the write path before Phase 1",
            (CARD, *EMBEDDINGS),
            over_embeddings,
            {YES: NO, NO: YES}.get(beats_embeddings, beats_embeddings),
        ),
    ]
    return [
        {
            "result": result,
            "then": then,
            "measured": {name: accuracy(name) for name in names},
            "comparisons": comparisons,
            "verdict": verdict,
            "met": {YES: True, NO: False}.get(verdict),
        }
        for result, then, names, comparisons, verdict in rows
    ]


def _experiments(inputs: Inputs, text: set[str], cutoffs: Cutoffs) -> dict:
    """The experiments' numbers, over every file with text (dev and test).

    A pass with no answers gives None. held_out_named_in lists the groups of the reduced set
    whose description still names the held-out group (none, as eval/groups.py words them).
    edits_base says how many files the base of the edits answered and how many of those have
    a wrong top group: the spec's edits fix those. description_edits is None when the vault
    has no edits file, else one entry per edit.
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
    before = predictions(experiments.EDIT_BASE)
    edits = None
    if inputs.edits is not None:
        edits = []
        for edit in inputs.edits:
            after = predictions(experiments.edit_name(edit))
            effect = None
            if before and after:
                effect = experiments.edit_effect(before, after, edit.target_file_ids)
            edits.append({**edit.model_dump(), "effect": effect})
    wrong = sum(top_group(prediction.probs) != prediction.gold for prediction in before)
    return {
        "held_out_group": HELD_OUT_GROUP_ID,
        "has_held_out_group": any(group.id == HELD_OUT_GROUP_ID for group in inputs.groups),
        "held_out_named_in": experiments.naming_held_out(inputs.groups),
        "group_added": group_added,
        "edits_base": {"n": len(before), "wrong": wrong} if before else None,
        "description_edits": edits,
    }


def _listed(file_ids: Sequence[str]) -> str:
    """A few file ids, comma separated, with a count of the rest."""
    shown = ", ".join(file_ids[:LISTED])
    more = len(file_ids) - LISTED
    return f"{shown} and {more} more" if more > 0 else shown


def _count(number: int, noun: str) -> str:
    """A count with its noun: "1 file", "3 files"."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _and(items: Sequence[str]) -> str:
    """Some names as one phrase: "a", "a and b", "a, b and c"."""
    return " and ".join([", ".join(items[:-1]), items[-1]] if len(items) > 1 else items)


def _discordant(pair: dict) -> str:
    """The files only one of two configurations got right: "6 (0 to 6)", or "0" for none.

    The first configuration's count comes first.
    """
    total = pair["only_first"] + pair["only_second"]
    return f"{total} ({pair['only_first']} to {pair['only_second']})" if total else "0"


def _thin_comparisons(rows: Sequence[dict]) -> list[dict]:
    """The go/no-go comparisons that rest on fewer discordant files than a sign test can use.

    Each comparison is listed once, even when two rows share it.
    """
    fewest = min_discordant()
    thin: dict[tuple[str, str], dict] = {}
    for row in rows:
        for item in row["comparisons"]:
            if item["n"] and item["only_first"] + item["only_second"] < fewest:
                thin.setdefault((item["first"], item["second"]), item)
    return list(thin.values())


def _caveats(metrics: dict) -> list[str]:
    """What a reader should know before trusting the numbers. Every line comes from the data."""
    files = metrics["files"]
    names = metrics["groups"]
    notes = []
    if metrics["limit"] is not None:
        notes.append(
            f"This run took only the first files of each split (--limit {metrics['limit']}): "
            f"every number rests on {files['dev'] + files['test']} files in all."
        )
    if metrics["stopped"]:
        notes.append(
            f"The run was stopped before it finished ({metrics['stopped']}). "
            "Whatever had not been asked by then is missing."
        )
    thin = _thin_comparisons(metrics["go_no_go"])
    if thin:
        fewest = min_discordant()
        listed = "; ".join(
            f"{item['first']} against {item['second']}, {_discordant(item)}" for item in thin
        )
        notes.append(
            f"Go/no-go comparisons that rest on fewer than {fewest} discordant files (files "
            f"only one of the two got right): {listed}. It takes {fewest} such files, all one "
            "way, for an exact two-sided sign test to separate two configurations at p under "
            f"{SIGN_TEST_ALPHA:g}. Fewer cannot tell them apart, so one file falling the other "
            "way could change what these comparisons show."
        )
    ceiling = metrics["ceiling"]
    if ceiling and ceiling["at_ceiling"]:
        notes.append(
            f"{CARD} is right on {ceiling['right']} of its {ceiling['n']} test files, which "
            f"leaves another input less than the {ceiling['margin_points']:g} points of the "
            "go/no-go margin to gain. Whether full text, window vote or any other variant "
            "does better than the card cannot be told on these files: a difference of +0.0 "
            "is not evidence that more text does not help."
        )
    signal = metrics["label_signal"]
    if signal["folders"]:
        folders = "; ".join(
            f"{folder}/ ({names.get(found['gold_group'], found['gold_group'] or 'no group')}, "
            f"{_count(found['files'], 'file')})"
            for folder, found in signal["folders"].items()
        )
        note = (
            f"Folders whose files all have the same gold group: {folders}. Test files with "
            f"text under such a folder: {signal['test_with_text']} of "
            f"{files['test_with_text']}. The folder name in their path gives the group away "
            "to every configuration that is shown the path."
        )
        if signal["rules"]:
            rules = signal["rules"]
            note += (
                f" The file-name rules ({RULES}), which read only names and paths, are right "
                f"on {rules['right']} of the {_count(rules['n'], 'test file')} with text they "
                f"were scored on ({_pct(rules['right'] / rules['n'])})."
            )
        notes.append(note)
    named = metrics["experiments"]["held_out_named_in"]
    if named:
        held_out = metrics["experiments"]["held_out_group"]
        notes.append(
            f"In the passes without {names.get(held_out, held_out)}, the descriptions of "
            f"{', '.join(named)} still name that group. That tells the model its files are "
            "excluded, so the column with descriptions may be helped by it."
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
    note = (
        f"Files in the dev split: {files['dev']} ({files['dev_with_text']} with text). "
        "The tuned cutoffs rest on them."
    )
    flat = [name for name, tuned in metrics["tuned"].items() if not tuned["tunable"]]
    if flat:
        note += (
            f" They could not tune the cutoffs of {_and(flat)}: several pairs of cutoffs "
            "tie for best on the dev split, the defaults among them."
        )
    notes.append(note)
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


def _p(value: float) -> str:
    """A p-value with three decimals; anything smaller is shown as under 0.001."""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _close_calls(share: float | None, can_fire: bool) -> str:
    """The close call cell: "n/a" when the flag cannot fire under the cutoffs in use.

    A 0% there would read as a measurement. A share above 0 is always shown: it means the
    probabilities summed to more than 1, which the cutoffs alone cannot rule out.
    """
    return _pct(share) if can_fire or share else "n/a"


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


def _majority_label(metrics: dict, all_files: bool = False) -> str:
    """The name of the majority-class row, with the group it always picks.

    In the all-files table the row also holds a number over all test files, where the largest
    group can be another one. Then both groups are named.
    """
    names = metrics["groups"]
    group_id = metrics["majority"]["group"]
    label = names.get(group_id, group_id)
    every = metrics["majority"]["all_files"]["group"]
    if all_files and every != group_id:
        label = f"{label} with text, {names.get(every, every)} over all files"
    return f"majority class ({label})"


def _headline_notes(metrics: dict, measured: Sequence[str], fixed: Sequence[str]) -> list[str]:
    """The lines under the headline table: what its cells do not show, and the sensitivity.

    `measured` names the rows whose probabilities are the model's own, and `fixed` the rows
    whose placement and calibration cells were left empty.
    """
    cutoffs = metrics["cutoffs"]
    by_construction = metrics["by_construction"]
    notes = []
    if measured and not by_construction["close_call_can_fire"]:
        least = 2 * cutoffs["top_below"] - cutoffs["close_margin"]
        notes.append(
            "Close calls show n/a because the flag cannot fire under these cutoffs. It needs a "
            f"top group at or over {cutoffs['top_below']:.2f} with a second group within "
            f"{cutoffs['close_margin']:.2f} of it, and two such probabilities add up to more "
            f"than {least:.2f}, which probabilities that sum to 1 have no room for. A 0% "
            "there would not be a measurement."
        )
    if fixed:
        note = (
            f"The cells from Placed to ECE are left empty for {_and(fixed)}. Their "
            "probabilities are one-hot (the whole probability on one pick: the file-name "
            "rules, the LLM baseline, the majority class) or follow a softmax temperature set "
            "by hand (the embeddings). Where the cutoffs put such a file, and how calibrated "
            "its top probability looks, is fixed by that construction and not measured."
        )
        no_pick = [
            f"{name} for {count} of {metrics['headline'][name]['n']}"
            for name, count in by_construction["no_pick"].items()
        ]
        if no_pick:
            note += f" Files with no pick at all, counted as wrong: {'; '.join(no_pick)}."
        notes.append(note)
    all_right = [
        name
        for name in measured
        if metrics["headline"][name]["top1_accuracy"] == 1.0
        and metrics["headline"][name]["ece"] is not None
    ]
    if all_right:
        notes.append(
            f"Every file is right for {_and(all_right)}, so the ECE in those rows is only "
            "the mean of 1 minus the top probability: how far the confidence stays under 1. "
            "It does not test calibration."
        )
    sensitivity = metrics["sensitivity"]
    if sensitivity:
        n = sensitivity["n"]
        wordings = "; ".join(
            f"{row['wording']} {_pct(row['right'] / n)} ({row['right']} of {n})"
            for row in sensitivity["rows"]
        )
        note = (
            f"Sensitivity to the wording of the groups: top-1 of {CARD} on the same "
            f"{_count(n, 'test file')}, by wording: {wordings}."
        )
        if sensitivity["rules_right"] is not None:
            note += f" For scale, {RULES} is right on {sensitivity['rules_right']} of them."
        notes.append(
            note + " The headline is the first of these: it holds for the vault's own "
            "descriptions, not for any wording of the same groups."
        )
    return notes


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
    can_fire = metrics["by_construction"]["close_call_can_fire"]

    def cells(label: str, row: dict, measured: bool) -> list[str]:
        """One table row. Placement and calibration stay empty when they are not measured."""
        placement = [""] * 6
        if measured:
            placement = [
                _pct(row["placed_share"]),
                _pct(row["placed_accuracy"]),
                _pct(row["unsorted_share"]),
                _pct(row["unsorted_top1_right_share"]),
                _close_calls(row["close_call_share"], can_fire),
                _num(row["ece"], 3),
            ]
        return [
            label,
            str(row["n"]),
            _pct(row["top1_accuracy"]),
            _num(row["macro_f1"], 3),
            *placement,
            _num(row["mean_input_tokens"]),
            _usd(row["usd_per_file"]),
            _num(row["latency_p50_ms"]),
            _num(row["latency_p95_ms"]),
        ]

    shown = [name for name, row in metrics["headline"].items() if row]
    fixed = list(metrics["by_construction"]["probabilities_not_measured"])
    measured = [name for name in shown if name not in fixed]
    rows = [cells(name, metrics["headline"][name], name in measured) for name in shown]
    majority = metrics["majority"]
    explained = (
        "Top-1 is the share of files whose top group is their gold group; it ignores the "
        "cutoffs. Placed right is the share of placed files that sit in their gold group. "
        "Unsorted, top right is the share of Unsorted files whose top group was their gold "
        "group. Input tokens and USD per file are means per file."
    )
    if majority and majority["headline"]:
        rows.append(cells(_majority_label(metrics), majority["headline"], False))
        fixed.append("the majority class")
        explained += (
            " The majority class gives every file the largest gold group among the files of "
            "its row, so it is the best a constant answer can do on them."
        )
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
        explained,
    ]
    for note in _headline_notes(metrics, measured, fixed):
        lines += ["", note]
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
                _majority_label(metrics, all_files=True),
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
    """For each Jev configuration, the test metrics under the default and the tuned cutoffs.

    A configuration whose dev split cannot tell the cutoffs apart gets no tuned numbers: its
    row says so, and how many pairs of cutoffs tie for best.
    """

    def cells(name: str, label: str, cutoffs: dict, row: dict, can_fire: bool) -> list[str]:
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
            _close_calls(row["close_call_share"], can_fire),
        ]

    default_can_fire = metrics["by_construction"]["close_call_can_fire"]
    rows = []
    flat = False
    for name, tuned in metrics["tuned"].items():
        headline = metrics["headline"][name]
        rows.append([*cells(name, "default", metrics["cutoffs"], headline, default_can_fire), ""])
        label = f"tuned on dev (n = {tuned['dev_n']})"
        ties = f"{tuned['tied_pairs']} of {tuned['scored_pairs']}"
        if tuned["tunable"]:
            row = cells(
                name, label, tuned["cutoffs"], tuned["metrics"], tuned["close_call_can_fire"]
            )
            rows.append([*row, ties])
        else:
            flat = True
            rows.append([name, label, "not tunable on this dev split", *[""] * 6, ties])
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
        "Pairs tied for best",
    ]
    lines = [
        "## Tuned cutoffs",
        "",
        (
            "The two Unsorted cutoffs are tuned on the dev split for each Jev configuration, to "
            "get the most right placements minus wrong placements, and then applied to the test "
            "split. The defaults stay the headline. Top-1 does not depend on the cutoffs. Pairs "
            "tied for best is how many of the pairs of cutoffs tried share the best score on "
            "the dev files."
        ),
        "",
    ]
    lines.append(_table(header, rows) if rows else "No Jev configuration has answers yet.")
    if flat:
        lines += [
            "",
            (
                '"not tunable on this dev split" means the defaults are among several pairs '
                "that tie for best there. The dev files cannot tell those cutoffs apart, so a "
                "tuned row would only repeat the defaults: it would not show that the "
                "defaults are right."
            ),
        ]
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
        header = [
            "Variant",
            "Top-1",
            *(names.get(group_id, group_id) for group_id in per_group),
            "Files only one got right",
        ]
        rows = [
            [
                name,
                _signed(difference["top1_accuracy"]),
                *(_signed(difference["recall"].get(group_id)) for group_id in per_group),
                _discordant(difference["paired"]) if difference["paired"]["n"] else "n/a",
            ]
            for name, difference in differences.items()
        ]
        lines += [
            "",
            (
                f"Each variant minus {CARD}, in points: top-1 accuracy, then recall per group. "
                "The last column counts, on the files both answered, the files only one of "
                "the two got right (the variant's count, then the card's)."
            ),
            "",
            _table(header, rows),
        ]
        ceiling = metrics["ceiling"]
        if ceiling and ceiling["at_ceiling"]:
            lines += [
                "",
                (
                    f"Whether a variant does better than {CARD} {AT_CEILING} "
                    f"({ceiling['right']} of {ceiling['n']} right). That leaves a variant less "
                    f"than the {ceiling['margin_points']:g} points of the go/no-go margin to "
                    "gain, so a difference of +0.0 does not show that the extra input does "
                    "not help. A negative difference is a measured loss."
                ),
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
    """Experiments 1 and 2: the passes without the held-out group, side by side.

    The AUROC of fits_any is the comparison between the columns, because it needs no cutoff.
    The shares that do depend on the cutoffs sit in a second table, under a line that says the
    fits_any cutoff could not be tuned on this data set.
    """
    tests = metrics["experiments"]
    files = metrics["files"]
    cutoffs = metrics["cutoffs"]
    held_out = tests["held_out_group"]
    held_out_name = metrics["groups"].get(held_out, held_out)
    lines = [f"### Group added mid-run (held out: {held_out_name})", ""]
    shown = {base: numbers for base, numbers in tests["group_added"].items() if numbers}
    if not tests["has_held_out_group"]:
        return [*lines, f"Not run: this vault has no group with the id {held_out}."]
    if not shown:
        return [*lines, "Not run: no answers are cached for the passes without the group."]

    def table(first: str, measures: Sequence[tuple]) -> str:
        """One table of the section: a row per measure, a column per pass."""
        rows = [
            [label, *(show(numbers[key]) for numbers in shown.values())]
            for label, key, show in measures
        ]
        return _table([first, *shown], rows)

    free = [
        ("Files sorted without the group", "n", str),
        ("Held-out files", "held_out_n", str),
        ("Files of the other groups", "others_n", str),
        ("AUROC of fits_any", "fits_any_auroc", lambda v: _num(v, 3)),
        ("Mean fits_any of the held-out files", "held_out_mean_fits_any", lambda v: _num(v, 3)),
        ("Top-1 accuracy of the other groups' files", "others_top1_accuracy", _pct),
    ]
    cut = [
        ("Held-out files sent to Unsorted", "held_out_unsorted_share", _pct),
        ("Other groups' files wrongly sent to Unsorted", "others_unsorted_share", _pct),
        ("Files frozen (placed in their gold group)", "frozen_n", str),
        ("Files rerun against the full set", "rerun_n", str),
        ("Rerun accuracy", "rerun_accuracy", _pct),
    ]
    text = (
        "Every file with text (dev and test) is sorted with the group left out. The held-out "
        "group's files then fit no group of the reduced set, so Unsorted is right for them."
    )
    named = tests["held_out_named_in"]
    if named:
        text += (
            " Careful: the reduced set's descriptions still name the held-out group "
            f"(in {', '.join(named)}), which tells the model that its files are excluded."
        )
    else:
        text += (
            " The reduced set's descriptions do not name the held-out group: what the "
            "remaining groups exclude is worded as it would be before anyone thought of it."
        )
    names_only = experiments.HELD_OUT_BASES[1]
    if names_only in shown:
        text += f" The {names_only} column asks with group names only, without descriptions."
    compared = (
        "The AUROC of fits_any is the comparison between the columns: how well fits_any "
        "alone tells the other groups' files from the held-out ones, whatever the cutoff "
        "(1.000 is a clean split, 0.500 is chance)."
    )
    total = files["dev"] + files["test"]
    if files["no_gold"]:
        tunable = (
            f"Files that fit no group in this data set: {files['no_gold']} of {total}, so the "
            "fits_any cutoff had next to nothing to be tuned on."
        )
    else:
        tunable = (
            f"No file in this data set fits no group (none of the {total} has that as its "
            "gold label), so the fits_any cutoff could not be tuned on it."
        )
    at_cutoffs = (
        f"The shares below are taken at the default cutoffs (fits_any under "
        f"{cutoffs['fits_any_below']:.2f}, top group under {cutoffs['top_below']:.2f}), the "
        f"same for every column. {tunable} A column can do worse at that one cutoff than its "
        "fits_any allows. For the rerun, a user freezes each file placed in its gold group "
        "and sorts the rest against the full set: rerun accuracy is the share of those placed "
        "right. The rerun asks nothing new: it reads the answers the same configuration gave "
        "under the full set."
    )
    return [
        *lines,
        text,
        "",
        compared,
        "",
        table("Measure", free),
        "",
        at_cutoffs,
        "",
        table("At the default cutoffs", cut),
    ]


def _edits_section(tests: dict) -> list[str]:
    """Experiment 3: what each description edit changed, against the card's answers before.

    When no edit names a target, the section says that the spec's experiment (fix each wrong
    placement with an edit) was not run, and the table drops its target columns.
    """
    lines = ["### Description edits", ""]
    edits = tests["description_edits"]
    if edits is None:
        return [*lines, f"Not run: the vault has no {experiments.EDITS_FILE}."]
    if not edits:
        return [*lines, f"Not run: the vault's {experiments.EDITS_FILE} lists no edits."]
    targeted = any(edit["target_file_ids"] for edit in edits)
    rows = []
    for edit in edits:
        effect = edit["effect"]
        if effect is None:
            rows.append([edit["name"], edit["group_id"], *["n/a"] * (8 if targeted else 6)])
            continue
        targets = [str(effect["targets_n"]), str(len(effect["targets_fixed"]))]
        rows.append(
            [
                edit["name"],
                edit["group_id"],
                str(effect["n"]),
                *(targets if targeted else []),
                str(len(effect["wrong_to_right"])),
                str(len(effect["right_to_wrong"])),
                str(len(effect["wrong_to_other_wrong"])),
                _pct(effect["top1_before"]),
                _pct(effect["top1_after"]),
            ]
        )
    how = (
        f"{experiments.EDIT_BASE} sorts every file with text (dev and test) again under the "
        "edited description, and each file's top group is compared with the one before "
        f"(cutoffs play no part). The file ids behind the counts are in {METRICS_FILE}."
    )
    if targeted:
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
        text = (
            f"Each edit changes one group's description. {how} Targets are the files the "
            "edit was meant to fix."
        )
    else:
        header = [
            "Edit",
            "Group",
            "Files",
            "Wrong to right",
            "Right to wrong",
            "Between wrong groups",
            "Top-1 before",
            "Top-1 after",
        ]
        base = tests["edits_base"]
        if base and not base["wrong"]:
            why = (
                f"{experiments.EDIT_BASE} has the right top group for all "
                f"{_count(base['n'], 'file')} with text, so the base run had no wrong "
                "placement to fix."
            )
        elif base:
            why = (
                f"no edit names a file to fix, although {experiments.EDIT_BASE} has a wrong "
                f"top group for {base['wrong']} of the {_count(base['n'], 'file')} with text."
            )
        else:
            why = "no edit names a file to fix."
        text = (
            "The spec's experiment (fix each wrong placement with a description edit, then "
            f"count which other files changed) was not run: {why} The edits below name no "
            "target. Each changes one group's description, and they measure how far one "
            f"description edit reaches. {how}"
        )
    lines += [text, "", _table(header, rows)]
    # A note is free text, so it is put on one line to stay inside its list item.
    notes = {edit["name"]: " ".join((edit["note"] or "").split()) for edit in edits}
    noted = [f"- {name}: {note}" for name, note in notes.items() if note]
    bare = [name for name, note in notes.items() if not note]
    if noted:
        lines += ["", "Notes on the edits (who wrote them, when and why):", "", *noted]
    if bare:
        lines += ["", f"No note says who wrote these edits, when or why: {', '.join(bare)}."]
    return lines


def _experiments_section(metrics: dict) -> str:
    """The experiments: group added mid-run with both wordings, then the description edits."""
    tests = metrics["experiments"]
    lines = ["## Experiments", "", *_group_added_section(metrics), "", *_edits_section(tests)]
    return "\n".join(lines)


def _verdict(row: dict) -> str:
    """A go/no-go row's verdict. "not separable" comes with the counts it rests on."""
    if row["verdict"] != NOT_SEPARABLE:
        return row["verdict"]
    thin = [
        f"{item['first']} against {item['second']}: {item['only_first']} to "
        f"{item['only_second']} on "
        f"{_count(item['only_first'] + item['only_second'], 'discordant file')}"
        for item in row["comparisons"]
        if item["outcome"] == NOT_SEPARABLE
    ]
    return f"{NOT_SEPARABLE} ({'; '.join(thin)})"


def _go_no_go_section(metrics: dict) -> str:
    """The go/no-go table, then the paired comparisons its verdicts rest on."""

    def measured(name: str, value: float | None) -> str:
        """One accuracy with the number of files behind it, so uneven rows show here too."""
        row = metrics["headline"].get(name)
        return f"{name} {_pct(value)} (n {row['n']})" if row else f"{name} n/a"

    rows = [
        [
            row["result"],
            row["then"],
            "; ".join(measured(name, value) for name, value in row["measured"].items()),
            _verdict(row),
        ]
        for row in metrics["go_no_go"]
    ]
    pairs: dict[tuple[str, str], dict] = {}
    for row in metrics["go_no_go"]:
        for item in row["comparisons"]:
            pairs.setdefault((item["first"], item["second"]), item)
    compared = []
    for item in pairs.values():
        needs = f"above by {item['margin_points']:g} points" if item["margin_points"] else "above"
        cells = ["n/a"] * 4
        if item["n"]:
            cells = [
                str(item["n"]),
                f"{item['first_right']} to {item['second_right']}",
                _discordant(item),
                _p(item["p_value"]),
            ]
        compared.append(
            [f"{item['first']} against {item['second']}", needs, *cells, item["outcome"]]
        )
    fewest = min_discordant()
    return "\n".join(
        [
            "## Go/no-go",
            "",
            (
                "The conditions are the proposed defaults in docs/phase0.md. Measured holds the "
                "headline top-1 accuracies, each with the number of files it was scored on. "
                "A condition that needs one configuration above another is checked file by "
                "file, on the files both answered, and counts as met only when the two are "
                "separable: an exact two-sided sign test on the discordant files (the files "
                f"only one of the two got right) gives p under {SIGN_TEST_ALPHA:g}. "
                f"That takes at least {fewest} discordant files, all one way. The test counts "
                "every file as independent, so on a set with near copies (an exam and its "
                "solutions, say) it is easier to pass than it should be."
            ),
            "",
            (
                'Condition met reads "yes", "no" (the numbers do not meet the condition), '
                f'"{NOT_SEPARABLE}" (the numbers meet it, but the discordant files are too '
                f'few, or too evenly split, to carry it), "{AT_CEILING}" (the card leaves '
                f'less room than the margin the condition needs) or "{MISSING}" (a number the '
                "condition needs is missing)."
            ),
            "",
            _table(["Result", "Then", "Measured", "Condition met"], rows),
            "",
            (
                "The paired comparisons behind the table. Right and Discordant files give "
                "the first configuration's count, then the second's."
            ),
            "",
            _table(
                [
                    "Comparison",
                    "Needs",
                    "Files both answered",
                    "Right",
                    "Discordant files",
                    "Sign test p",
                    "Outcome",
                ],
                compared,
            ),
        ]
    )


def _caveats_section(metrics: dict) -> str:
    """The caveats, one per line."""
    return "\n".join(["## Caveats", "", *(f"- {note}" for note in metrics["caveats"])])
