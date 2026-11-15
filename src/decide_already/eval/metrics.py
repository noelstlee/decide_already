"""Evaluation metrics: pure functions over predictions and their gold labels.

Nothing here calls a model or reads a vault. The runner turns each cached answer and its gold
label into a Prediction, and these functions turn lists of predictions into numbers.
A number that cannot be computed (nothing placed, no token counts) is None, never a guess.
"""

import math
from collections.abc import Sequence

from ..config import Config
from ..types import StrictModel

# The values tried for each tuned cutoff: 0.0, 0.05, and so on up to 0.9.
CUTOFF_GRID = [round(step * 0.05, 2) for step in range(19)]


class Prediction(StrictModel):
    """One file's answer from one configuration, next to its gold label."""

    file_id: str
    gold: str | None  # the gold group id; None for a file that fits no group
    probs: dict[str, float]  # by group id
    fits_any: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


class Cutoffs(StrictModel):
    """The three numbers that turn probabilities into a placement."""

    fits_any_below: float  # fits_any under this: Unsorted
    top_below: float  # top group's probability under this: Unsorted
    close_margin: float  # top two groups closer than this: placed, but flagged as a close call

    @classmethod
    def from_config(cls, config: Config) -> "Cutoffs":
        """The cutoffs a vault's config holds."""
        return cls(
            fits_any_below=config.unsorted_fits_any_below,
            top_below=config.unsorted_top_below,
            close_margin=config.close_call_margin,
        )


class GroupMetrics(StrictModel):
    """Top-1 scores for one gold group."""

    n: int  # files whose gold label is this group
    precision: float | None  # None when no file was scored as this group
    recall: float  # share of this group's files with the right top group: its top-1 accuracy
    f1: float


class Metrics(StrictModel):
    """Every number reported for one configuration. See summarize for the definitions."""

    n: int
    top1_accuracy: float | None
    per_group: dict[str, GroupMetrics]
    macro_f1: float | None
    placed_share: float | None
    placed_accuracy: float | None
    unsorted_share: float | None
    unsorted_top1_right_share: float | None
    close_call_share: float | None
    ece: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    usd_per_file: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None


def top_group(probs: dict[str, float]) -> str | None:
    """The group with the highest probability; a tie goes to the lower group id. None if empty."""
    if not probs:
        return None
    return min(probs, key=lambda group_id: (-probs[group_id], group_id))


def outcome(prediction: Prediction, cutoffs: Cutoffs) -> tuple[str | None, bool]:
    """Apply the placement rules: (group id, or None for Unsorted; close call flag).

    This is the same truth table as sort.decide, written separately so the two can be checked
    against each other. "Under" is strict: a value equal to its cutoff passes. With one group,
    the second probability counts as 0. An Unsorted file is never a close call.
    """
    top = top_group(prediction.probs)
    if top is None or prediction.fits_any < cutoffs.fits_any_below:
        return None, False
    ranked = sorted(prediction.probs.values(), reverse=True)
    top_prob = ranked[0]
    second_prob = ranked[1] if len(ranked) > 1 else 0.0
    if top_prob < cutoffs.top_below:
        return None, False
    return top, top_prob - second_prob < cutoffs.close_margin


def expected_calibration_error(
    confidences: Sequence[float], corrects: Sequence[bool], bins: int = 10
) -> float | None:
    """Expected calibration error: how far stated confidence is from the share that was right.

    Confidences are cut into `bins` equal bins over 0 to 1. Bin k holds k/bins up to, but not
    including, (k+1)/bins; 1.0 goes in the last bin. In each bin, the gap between the mean
    confidence and the share right is weighted by the bin's share of all items.
    0 is perfectly calibrated. None when there is nothing to score.
    A confidence outside 0 to 1 is not an error: it goes in the first or the last bin.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if len(confidences) != len(corrects):
        raise ValueError("confidences and corrects must have the same length")
    if not confidences:
        return None
    counts = [0] * bins
    confidence_sums = [0.0] * bins
    right_counts = [0] * bins
    for confidence, correct in zip(confidences, corrects):
        # Clamped at both ends: a negative index would count from the end of the list.
        index = min(max(int(confidence * bins), 0), bins - 1)
        counts[index] += 1
        confidence_sums[index] += confidence
        right_counts[index] += bool(correct)
    error = 0.0
    for count, confidence_sum, right in zip(counts, confidence_sums, right_counts):
        if count:
            gap = abs(right / count - confidence_sum / count)
            error += count / len(confidences) * gap
    return error


def summarize(
    predictions: Sequence[Prediction],
    cutoffs: Cutoffs,
    usd_per_million_tokens: float | None,
    usd_per_million_output_tokens: float | None = None,
) -> Metrics:
    """Compute every metric for one configuration's predictions under one set of cutoffs.

    Top-1 looks at the top group and ignores the cutoffs, so a file sent to Unsorted still
    counts as right when its top group was right. The one exception is a file whose gold label
    is "no group": no top group can be right for it, so it counts as right only when the
    cutoffs send it to Unsorted, and as a wrong pick of its top group when it is placed.

    - top1_accuracy: share of files that are right in that sense.
    - per_group, macro_f1: precision, recall and F1 on top-1 for each group that appears as a
      gold label, and the plain mean of those F1 scores.
    - placed_share, placed_accuracy: share of files the cutoffs place, and the share of those
      placed in their gold group.
    - unsorted_share, unsorted_top1_right_share: share of files sent to Unsorted, and the share
      of those whose top group was their gold group (placements the cutoffs gave up). A file
      with no gold group never adds to the second number.
    - close_call_share: share of all files placed with the close call flag.
    - ece: calibration error of the top probability against top-1 being right.
    - mean_input_tokens, mean_output_tokens, usd_per_file: over the predictions that report
      tokens. The cost is input tokens at usd_per_million_tokens, plus output tokens at
      usd_per_million_output_tokens when both that price and output counts are given.
    - latency_p50_ms, latency_p95_ms: over the predictions that report a latency.
    """
    n = len(predictions)
    outcomes = [outcome(prediction, cutoffs) for prediction in predictions]
    # The label each file is scored on for top-1: its top group, or for a file with no gold
    # group, where the cutoffs put it.
    labels = [
        placed if prediction.gold is None else top_group(prediction.probs)
        for prediction, (placed, _) in zip(predictions, outcomes)
    ]
    rights = [label == prediction.gold for prediction, label in zip(predictions, labels)]

    per_group: dict[str, GroupMetrics] = {}
    for group_id in sorted({p.gold for p in predictions if p.gold is not None}):
        support = sum(p.gold == group_id for p in predictions)
        scored_as = sum(label == group_id for label in labels)
        hits = sum(
            p.gold == group_id and label == group_id for p, label in zip(predictions, labels)
        )
        per_group[group_id] = GroupMetrics(
            n=support,
            precision=_share(hits, scored_as),
            recall=hits / support,
            f1=2 * hits / (support + scored_as),
        )

    placed = [
        (p, group_id) for p, (group_id, _) in zip(predictions, outcomes) if group_id is not None
    ]
    unsorted = [p for p, (group_id, _) in zip(predictions, outcomes) if group_id is None]
    # Unsorted files whose top group was their gold group: placements the cutoffs gave up.
    given_up = sum(p.gold is not None and top_group(p.probs) == p.gold for p in unsorted)
    # (top probability, top-1 right) for every prediction that has probabilities.
    scored = [(max(p.probs.values()), right) for p, right in zip(predictions, rights) if p.probs]
    mean_tokens = _mean([p.input_tokens for p in predictions if p.input_tokens is not None])
    latencies = [p.latency_ms for p in predictions if p.latency_ms is not None]
    mean_output = _mean([p.output_tokens for p in predictions if p.output_tokens is not None])
    usd_per_file = None
    if mean_tokens is not None and usd_per_million_tokens is not None:
        usd_per_file = mean_tokens / 1_000_000 * usd_per_million_tokens
        if mean_output is not None and usd_per_million_output_tokens is not None:
            usd_per_file += mean_output / 1_000_000 * usd_per_million_output_tokens

    return Metrics(
        n=n,
        top1_accuracy=_share(sum(rights), n),
        per_group=per_group,
        macro_f1=_mean([group.f1 for group in per_group.values()]),
        placed_share=_share(len(placed), n),
        placed_accuracy=_share(sum(group_id == p.gold for p, group_id in placed), len(placed)),
        unsorted_share=_share(len(unsorted), n),
        unsorted_top1_right_share=_share(given_up, len(unsorted)),
        close_call_share=_share(sum(close for _, close in outcomes), n),
        ece=expected_calibration_error([c for c, _ in scored], [right for _, right in scored]),
        mean_input_tokens=mean_tokens,
        mean_output_tokens=mean_output,
        usd_per_file=usd_per_file,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
    )


def tune_cutoffs(dev_predictions: Sequence[Prediction], defaults: Cutoffs) -> Cutoffs:
    """Pick fits_any_below and top_below on the dev split by grid search.

    Every pair of values on CUTOFF_GRID is scored by its utility: (right placements minus
    wrong placements) / n. Placing a file in its gold group is right and placing it anywhere
    else is wrong. For a file with no gold group, Unsorted is right and any placement is wrong.
    Unsorted for a file that has a gold group counts 0.

    Among the pairs with the best utility, the one closest to the defaults wins (straight-line
    distance; if that also ties, the lower values). The defaults themselves are scored too and
    win every tie. So defaults that are not on the grid are kept unless a grid pair does
    better, and the result never scores worse on the dev split than the defaults do.
    close_margin is not tuned. With no dev predictions, the defaults come back unchanged.
    """
    if not dev_predictions:
        return defaults
    best = defaults
    best_key = (-_net_right(dev_predictions, defaults), 0.0)
    for fits_any_below in CUTOFF_GRID:
        for top_below in CUTOFF_GRID:
            candidate = defaults.model_copy(
                update={"fits_any_below": fits_any_below, "top_below": top_below}
            )
            distance = (fits_any_below - defaults.fits_any_below) ** 2 + (
                top_below - defaults.top_below
            ) ** 2
            # n is the same for every pair, so the count is compared instead of the ratio.
            # The distance is rounded so float noise cannot break a real tie.
            key = (-_net_right(dev_predictions, candidate), round(distance, 9))
            if key < best_key:
                best, best_key = candidate, key
    return best


def _net_right(predictions: Sequence[Prediction], cutoffs: Cutoffs) -> int:
    """Right placements minus wrong placements under these cutoffs (see tune_cutoffs)."""
    net = 0
    for prediction in predictions:
        group_id, _ = outcome(prediction, cutoffs)
        if group_id == prediction.gold:  # in its gold group, or Unsorted with no gold group
            net += 1
        elif group_id is not None:  # placed anywhere else
            net -= 1
    return net


def _share(count: int, total: int) -> float | None:
    """count / total, or None when total is 0."""
    return count / total if total else None


def _mean(values: Sequence[float]) -> float | None:
    """The mean of the values, or None when there are none."""
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], percent: float) -> float | None:
    """The given percentile (0 to 100), interpolating between neighbors. None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
