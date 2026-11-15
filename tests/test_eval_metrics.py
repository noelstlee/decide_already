"""Tests for eval/metrics.py. Every expected number here was worked out by hand."""

import pytest

from decide_already.config import Config
from decide_already.eval.metrics import (
    CUTOFF_GRID,
    SIGN_TEST_ALPHA,
    Cutoffs,
    GroupMetrics,
    Paired,
    Prediction,
    TuningTies,
    close_call_can_fire,
    expected_calibration_error,
    min_discordant,
    outcome,
    paired,
    sign_test,
    summarize,
    top1_right,
    top_group,
    tune_cutoffs,
    tuning_ties,
)

DEFAULTS = Cutoffs(fits_any_below=0.5, top_below=0.6, close_margin=0.15)


def make(file_id, gold, probs, fits_any=0.9, **extra):
    """Build a Prediction with less typing."""
    return Prediction(file_id=file_id, gold=gold, probs=probs, fits_any=fits_any, **extra)


def utility(predictions, cutoffs):
    """(right placements minus wrong placements) / n, written straight from the definition."""
    net = 0
    for prediction in predictions:
        group_id, _ = outcome(prediction, cutoffs)
        if group_id is None:
            net += 1 if prediction.gold is None else 0
        else:
            net += 1 if group_id == prediction.gold else -1
    return net / len(predictions)


# Cutoffs


def test_cutoffs_come_from_the_config():
    """from_config copies the three cutoffs out of a vault's config."""
    assert Cutoffs.from_config(Config()) == DEFAULTS
    custom = Config(unsorted_fits_any_below=0.3, unsorted_top_below=0.4, close_call_margin=0.1)
    assert Cutoffs.from_config(custom) == Cutoffs(
        fits_any_below=0.3, top_below=0.4, close_margin=0.1
    )


def test_the_grid_runs_from_0_to_0_9_in_steps_of_0_05():
    """The tuning grid has 19 clean values."""
    assert len(CUTOFF_GRID) == 19
    assert CUTOFF_GRID[:3] == [0.0, 0.05, 0.1]
    assert CUTOFF_GRID[-1] == 0.9


# top_group


def test_top_group_picks_the_highest_probability():
    """The highest probability wins."""
    assert top_group({"a": 0.2, "b": 0.7, "c": 0.1}) == "b"


def test_top_group_tie_goes_to_the_lower_id():
    """Equal probabilities: the lower group id wins, whatever the dict order."""
    assert top_group({"b": 0.4, "a": 0.4, "c": 0.2}) == "a"


def test_top_group_of_no_probs_is_none():
    """No probabilities, no top group."""
    assert top_group({}) is None


# outcome: the truth table in docs/implementation.md, row by row

TABLE = Cutoffs(fits_any_below=0.5, top_below=0.4, close_margin=0.15)


@pytest.mark.parametrize(
    ("fits_any", "probs", "expected"),
    [
        # Row 1: fits_any under its cutoff. Unsorted, however sure the top group is.
        (0.49, {"a": 0.95, "b": 0.05}, (None, False)),
        # Row 2: fits_any passes, top probability under its cutoff. Unsorted.
        (0.5, {"a": 0.39, "b": 0.31, "c": 0.30}, (None, False)),
        # Row 3: both pass, top minus second (0.1) under the margin. Placed, close call.
        (0.9, {"a": 0.5, "b": 0.4, "c": 0.1}, ("a", True)),
        # Row 4: both pass, top minus second (0.6) at or over the margin. Placed.
        (0.9, {"a": 0.8, "b": 0.2}, ("a", False)),
    ],
)
def test_outcome_follows_the_truth_table(fits_any, probs, expected):
    """Each row of the placement table gives the outcome the contract names."""
    assert outcome(make("f", "a", probs, fits_any), TABLE) == expected


def test_outcome_a_value_equal_to_its_cutoff_passes():
    """Under means strictly under. These numbers are exact in binary, so no rounding hides it."""
    cutoffs = Cutoffs(fits_any_below=0.5, top_below=0.5, close_margin=0.25)
    # fits_any 0.5, top 0.5 and top minus second 0.25 all sit exactly on their cutoffs.
    on_the_line = make("f", "a", {"a": 0.5, "b": 0.25, "c": 0.25}, fits_any=0.5)
    assert outcome(on_the_line, cutoffs) == ("a", False)
    # A margin of 0.125 is under 0.25: still placed, now a close call.
    assert outcome(make("f", "a", {"a": 0.5, "b": 0.375, "c": 0.125}), cutoffs) == ("a", True)
    # Just under either cutoff: Unsorted.
    assert outcome(make("f", "a", {"a": 0.5, "b": 0.5}, fits_any=0.4999), cutoffs) == (None, False)
    assert outcome(make("f", "a", {"a": 0.49, "b": 0.26, "c": 0.25}), cutoffs) == (None, False)


def test_outcome_with_one_group_the_second_probability_counts_as_0():
    """A lone group is compared against 0, so its own probability is the margin."""
    assert outcome(make("f", "a", {"a": 1.0}), DEFAULTS) == ("a", False)
    loose = Cutoffs(fits_any_below=0.5, top_below=0.0, close_margin=0.15)
    assert outcome(make("f", "a", {"a": 0.1}), loose) == ("a", True)


def test_outcome_tie_is_placed_in_the_lower_id_as_a_close_call():
    """Two groups level at the top: the lower id gets the file, flagged."""
    cutoffs = Cutoffs(fits_any_below=0.5, top_below=0.5, close_margin=0.15)
    assert outcome(make("f", "a", {"b": 0.5, "a": 0.5}), cutoffs) == ("a", True)


def test_outcome_unsorted_is_never_a_close_call():
    """The close call flag only applies to placed files."""
    level = make("f", "a", {"a": 0.5, "b": 0.5}, fits_any=0.1)
    assert outcome(level, TABLE) == (None, False)


def test_outcome_with_no_probs_is_unsorted():
    """Nothing to pick from means Unsorted."""
    assert outcome(make("f", "a", {}), DEFAULTS) == (None, False)


def test_outcome_agrees_with_sort_decide():
    """outcome and sort.decide are two writings of one truth table, so they must agree.

    Probabilities step by eighths and the cutoffs are exact in binary, so the cases that sit
    right on a cutoff, and the ties, are compared without rounding getting in the way.
    Skipped until sort.py exists.
    """
    sort = pytest.importorskip("decide_already.sort")
    config = Config(unsorted_fits_any_below=0.5, unsorted_top_below=0.5, close_call_margin=0.25)
    cutoffs = Cutoffs.from_config(config)
    cases = [{"a": a / 8, "b": b / 8, "c": (8 - a - b) / 8} for a in range(9) for b in range(9 - a)]
    cases += [{"a": a / 8} for a in range(9)]  # one group: the second probability counts as 0
    for probs in cases:
        for fits_any in (0.0, 0.25, 0.5, 0.75, 1.0):
            decision = sort.decide(probs, fits_any, config)
            expected = outcome(make("f", None, probs, fits_any), cutoffs)
            assert (decision.group_id, decision.close_call) == expected, (probs, fits_any)


# close_call_can_fire


def test_a_close_call_cannot_fire_under_the_default_cutoffs():
    """A top of 0.6 leaves at most 0.4 for the second group: a gap of 0.2, over the 0.15 margin."""
    assert close_call_can_fire(DEFAULTS) is False
    # Every distribution over two or three groups in steps of 0.05, under the defaults.
    steps = [step / 20 for step in range(21)]
    spreads = [{"a": a, "b": 1 - a} for a in steps]
    spreads += [{"a": a, "b": b, "c": 1 - a - b} for a in steps for b in steps if a + b <= 1]
    assert not any(outcome(make("f", "a", probs), DEFAULTS)[1] for probs in spreads)


def test_a_close_call_can_fire_once_the_top_cutoff_is_under_0_575():
    """Twice the top cutoff minus the margin has to stay under 1 for the flag to have room."""
    lower = Cutoffs(fits_any_below=0.5, top_below=0.55, close_margin=0.15)
    assert close_call_can_fire(lower) is True
    assert outcome(make("f", "a", {"a": 0.55, "b": 0.45}), lower) == ("a", True)
    # Exactly at 0.575 the gap is exactly the margin, and "within" is strict.
    on_the_line = Cutoffs(fits_any_below=0.5, top_below=0.575, close_margin=0.15)
    assert close_call_can_fire(on_the_line) is False
    wider = Cutoffs(fits_any_below=0.5, top_below=0.6, close_margin=0.25)
    assert close_call_can_fire(wider) is True


# top1_right, sign_test and paired


def test_top1_right_is_what_summarize_counts():
    """The same five of the eight files below are right: f1, f2, f4, f5 and f7."""
    rights = [top1_right(prediction, LOOSE) for prediction in EIGHT]
    assert rights == [True, True, False, True, True, False, True, False]
    assert sum(rights) / len(EIGHT) == summarize(EIGHT, LOOSE, None).top1_accuracy


def test_sign_test_on_hand_computed_cases():
    """Twice the binomial tail of the smaller count, out of 2 to the number of pairs."""
    assert sign_test(6, 0) == 2 / 64  # 0.03125
    assert sign_test(5, 0) == 2 / 32  # 0.0625
    assert sign_test(7, 1) == 2 * (1 + 8) / 256  # 0.0703125
    assert sign_test(8, 1) == 2 * (1 + 9) / 512  # 0.0390625
    assert sign_test(1, 8) == sign_test(8, 1)
    # One file either way, an even split and no discordant file at all say nothing.
    assert sign_test(1, 0) == 1.0
    assert sign_test(3, 3) == 1.0
    assert sign_test(0, 0) == 1.0


def test_six_files_one_way_are_the_fewest_the_sign_test_separates():
    """Five to none is 0.0625, six to none 0.031: the first count under 0.05."""
    assert SIGN_TEST_ALPHA == 0.05
    assert min_discordant() == 6
    assert sign_test(5, 0) >= SIGN_TEST_ALPHA > sign_test(6, 0)
    assert min_discordant(0.01) == 8  # 2 / 128 is 0.016, 2 / 256 is 0.008


def test_paired_counts_the_discordant_files_each_way():
    """Five files both answered: a1 and a2 only the first got right, b1 only the second."""
    first = [
        make("a1", "a", {"a": 0.9, "b": 0.1}),
        make("a2", "a", {"a": 0.9, "b": 0.1}),
        make("b1", "b", {"a": 0.9, "b": 0.1}),
        make("both", "a", {"a": 0.9, "b": 0.1}),
        make("neither", "b", {"a": 0.9, "b": 0.1}),
    ]
    second = [
        make("a1", "a", {"a": 0.1, "b": 0.9}),
        make("a2", "a", {"a": 0.1, "b": 0.9}),
        make("b1", "b", {"a": 0.1, "b": 0.9}),
        make("both", "a", {"a": 0.9, "b": 0.1}),
        make("neither", "b", {"a": 0.9, "b": 0.1}),
    ]
    assert paired(first, second, DEFAULTS) == Paired(
        n=5, first_right=3, second_right=2, only_first=2, only_second=1, p_value=1.0
    )
    assert paired(second, first, DEFAULTS).only_first == 1


def test_paired_leaves_out_a_file_only_one_of_the_two_answered():
    """The first failed on f3, the one file the second got wrong: on f1 and f2 they tie."""
    first = [make("f1", "a", {"a": 1.0}), make("f2", "a", {"a": 1.0})]
    second = [*first, make("f3", "a", {"b": 1.0})]
    assert paired(first, second, DEFAULTS) == Paired(
        n=2, first_right=2, second_right=2, only_first=0, only_second=0, p_value=1.0
    )
    assert paired([], second, DEFAULTS).n == 0


def test_paired_counts_a_file_with_no_gold_group_by_where_the_cutoffs_put_it():
    """Unsorted is right for it, so the configuration that holds it back gets it right."""
    held_back = [make("x", None, {"a": 0.9, "b": 0.1}, fits_any=0.1)]
    placed = [make("x", None, {"a": 0.9, "b": 0.1}, fits_any=0.9)]
    result = paired(held_back, placed, DEFAULTS)
    assert (result.first_right, result.second_right, result.only_first) == (1, 0, 1)


# expected_calibration_error


def test_ece_on_a_hand_computed_case():
    """Three bins are used: 0.9s (gap 0.05), 0.8s (gap 0.35), 0.5s (gap 0.55).

    Weighted by 1/4, 2/4 and 1/4: 0.0125 + 0.175 + 0.1375 = 0.325.
    """
    confidences = [0.95, 0.85, 0.85, 0.55]
    corrects = [True, True, False, False]
    assert expected_calibration_error(confidences, corrects) == pytest.approx(0.325)


def test_ece_respects_the_number_of_bins():
    """Two bins. Low half: mean 0.3, half right (gap 0.2). High half: mean 0.7, all right
    (gap 0.3). Each holds half the items: 0.1 + 0.15 = 0.25."""
    confidences = [0.2, 0.4, 0.6, 0.8]
    corrects = [False, True, True, True]
    assert expected_calibration_error(confidences, corrects, bins=2) == pytest.approx(0.25)


def test_ece_is_zero_when_sure_and_right():
    """Always sure and always right is a gap of 0."""
    assert expected_calibration_error([1.0, 1.0], [True, True]) == 0.0


def test_ece_puts_a_confidence_of_1_in_the_last_bin():
    """1.0 shares the 0.9s bin with 0.9: mean confidence 0.95, half right, gap 0.45.

    (In a bin of its own, the wrong 1.0 and the right 0.9 would give 0.5 + 0.05 = 0.55.)
    """
    assert expected_calibration_error([1.0, 0.9], [False, True]) == pytest.approx(0.45)


def test_ece_of_nothing_is_none():
    """No items, no number."""
    assert expected_calibration_error([], []) is None


def test_ece_rejects_lists_of_different_lengths():
    """Each confidence needs its own right or wrong."""
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.6], [True])


def test_ece_needs_at_least_one_bin():
    """Zero bins is a mistake in the call, and says so."""
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error([0.5], [True], bins=0)


def test_ece_puts_out_of_range_confidences_in_the_end_bins():
    """A confidence under 0 goes in the first bin, and one over 1 in the last.

    -0.15 alone in the first bin and wrong: gap 0.15. 0.95 alone in the last bin and wrong:
    gap 0.95. Each holds half the items: 0.075 + 0.475 = 0.55.
    (Counted from the end of the list, -0.15 would share the last bin with 0.95: mean
    confidence 0.4, none right, 0.4.)
    """
    assert expected_calibration_error([-0.15, 0.95], [False, False]) == pytest.approx(0.55)
    assert expected_calibration_error([1.2], [True]) == pytest.approx(0.2)


# summarize

LOOSE = Cutoffs(fits_any_below=0.5, top_below=0.5, close_margin=0.15)

# Eight files under LOOSE. What happens to each:
#   f1 gold a: top a 0.9, placed in a, right.
#   f2 gold a: top a 0.55 over b 0.45, placed in a, right, a close call (margin 0.1).
#   f3 gold a: top b 0.7, placed in b, wrong.
#   f4 gold b: top b 0.8, placed in b, right.
#   f5 gold b: top b 0.45 is under 0.5, Unsorted, although the top group was right.
#   f6 gold c: fits_any 0.3 is under 0.5, Unsorted, and the top group (a) was wrong anyway.
#   f7 no gold group: fits_any 0.2, Unsorted, which is the right answer for it.
#   f8 no gold group: placed in b, which is wrong.
EIGHT = [
    make("f1", "a", {"a": 0.9, "b": 0.1}, input_tokens=1000, latency_ms=100),
    make("f2", "a", {"a": 0.55, "b": 0.45}, input_tokens=2000, latency_ms=200),
    make("f3", "a", {"b": 0.7, "a": 0.3}, input_tokens=3000, latency_ms=300),
    make("f4", "b", {"b": 0.8, "c": 0.2}, latency_ms=400),
    make("f5", "b", {"b": 0.45, "a": 0.35, "c": 0.2}),
    make("f6", "c", {"a": 0.6, "c": 0.4}, fits_any=0.3),
    make("f7", None, {"a": 0.4, "b": 0.35, "c": 0.25}, fits_any=0.2),
    make("f8", None, {"b": 0.9, "c": 0.1}, fits_any=0.8),
]


def test_summarize_top1_accuracy():
    """Right on top-1: f1, f2, f4, f5 (top group right) and f7 (no gold group, Unsorted)."""
    metrics = summarize(EIGHT, LOOSE, 0.042)
    assert metrics.n == 8
    assert metrics.top1_accuracy == pytest.approx(5 / 8)


def test_summarize_per_group_and_macro_f1():
    """Scored as a: f1, f2, f6. Scored as b: f3, f4, f5, f8. Scored as c: nothing.

    a: 2 hits of 3 scored, 3 gold. b: 2 hits of 4 scored, 2 gold. c: 0 hits, 1 gold.
    """
    metrics = summarize(EIGHT, LOOSE, 0.042)
    assert list(metrics.per_group) == ["a", "b", "c"]
    a, b, c = (metrics.per_group[group_id] for group_id in "abc")
    assert (a.n, a.precision, a.recall, a.f1) == pytest.approx((3, 2 / 3, 2 / 3, 2 / 3))
    assert (b.n, b.precision, b.recall, b.f1) == pytest.approx((2, 0.5, 1.0, 2 / 3))
    assert c == GroupMetrics(n=1, precision=None, recall=0.0, f1=0.0)
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)


def test_summarize_placed_unsorted_and_close_calls():
    """Placed: f1, f2, f3, f4, f8, of which f1, f2, f4 are in their gold group.

    Unsorted: f5, f6, f7, of which only f5 had its gold group on top. Close call: f2.
    """
    metrics = summarize(EIGHT, LOOSE, 0.042)
    assert metrics.placed_share == pytest.approx(5 / 8)
    assert metrics.placed_accuracy == pytest.approx(3 / 5)
    assert metrics.unsorted_share == pytest.approx(3 / 8)
    assert metrics.unsorted_top1_right_share == pytest.approx(1 / 3)
    assert metrics.close_call_share == pytest.approx(1 / 8)


def test_summarize_ece_of_the_top_probability():
    """Top probabilities by bin, with top-1 right or wrong:

    0.9s: f1 right, f8 wrong, gap 0.4, weight 2/8.   0.8s: f4 right, gap 0.2, weight 1/8.
    0.7s: f3 wrong, gap 0.7, weight 1/8.              0.6s: f6 wrong, gap 0.6, weight 1/8.
    0.5s: f2 right, gap 0.45, weight 1/8.             0.4s: f5 and f7 right, gap 0.575, 2/8.
    Sum: 0.1 + 0.025 + 0.0875 + 0.075 + 0.05625 + 0.14375 = 0.4875.
    """
    assert summarize(EIGHT, LOOSE, 0.042).ece == pytest.approx(0.4875)


def test_summarize_cost_and_latency():
    """Tokens: mean of 1000, 2000, 3000. Cost: 2000 tokens at 0.042 USD per million.

    Latency over 100, 200, 300, 400: the median is halfway between 200 and 300, and the
    95th percentile is 85% of the way from 300 to 400.
    """
    metrics = summarize(EIGHT, LOOSE, 0.042)
    assert metrics.mean_input_tokens == pytest.approx(2000)
    assert metrics.usd_per_file == pytest.approx(0.000084)
    assert metrics.latency_p50_ms == pytest.approx(250)
    assert metrics.latency_p95_ms == pytest.approx(385)


def test_summarize_leaves_out_what_cannot_be_computed():
    """No tokens, no latency and no price give None, not zero."""
    bare = [make("f1", "a", {"a": 0.9, "b": 0.1})]
    metrics = summarize(bare, DEFAULTS, 0.042)
    assert metrics.mean_input_tokens is None
    assert metrics.usd_per_file is None
    assert metrics.latency_p50_ms is None
    assert metrics.latency_p95_ms is None
    assert metrics.unsorted_top1_right_share is None  # nothing was sent to Unsorted
    priced = [make("f1", "a", {"a": 0.9, "b": 0.1}, input_tokens=500)]
    assert summarize(priced, DEFAULTS, None).usd_per_file is None
    assert summarize(priced, DEFAULTS, 0.042).mean_output_tokens is None


def test_summarize_adds_output_tokens_to_the_cost_when_priced():
    """3,000 input tokens at 8 USD and 200 output tokens at 40 USD per million: 0.032 USD."""
    billed = [make("f1", "a", {"a": 1.0}, input_tokens=3000, output_tokens=200)]
    metrics = summarize(billed, DEFAULTS, 8.0, 40.0)
    assert metrics.mean_output_tokens == pytest.approx(200)
    assert metrics.usd_per_file == pytest.approx(0.032)
    # Without an output price, only the input is counted.
    assert summarize(billed, DEFAULTS, 8.0).usd_per_file == pytest.approx(0.024)


def test_summarize_one_latency_is_both_percentiles():
    """With a single latency there is nothing to interpolate."""
    metrics = summarize([make("f1", "a", {"a": 1.0}, latency_ms=42.0)], DEFAULTS, None)
    assert metrics.latency_p50_ms == 42.0
    assert metrics.latency_p95_ms == 42.0


def test_summarize_of_no_predictions():
    """An empty run has a count of 0 and no made-up numbers."""
    metrics = summarize([], DEFAULTS, 0.042)
    assert metrics.n == 0
    assert metrics.per_group == {}
    for field in type(metrics).model_fields:
        if field not in ("n", "per_group"):
            assert getattr(metrics, field) is None, field


def test_a_file_with_no_gold_group_is_right_only_when_unsorted():
    """Unsorted is the only right answer for a file that fits no group."""
    unsorted = summarize([make("f", None, {"a": 0.9, "b": 0.1}, fits_any=0.1)], DEFAULTS, None)
    assert unsorted.top1_accuracy == 1.0
    assert unsorted.unsorted_share == 1.0
    # It had no right top group, so it is not a placement the cutoffs gave up.
    assert unsorted.unsorted_top1_right_share == 0.0
    assert unsorted.placed_accuracy is None
    assert unsorted.per_group == {}
    assert unsorted.macro_f1 is None

    placed = summarize([make("f", None, {"a": 0.9, "b": 0.1}, fits_any=0.9)], DEFAULTS, None)
    assert placed.top1_accuracy == 0.0
    assert placed.placed_share == 1.0
    assert placed.placed_accuracy == 0.0


def test_a_placed_file_with_no_gold_group_costs_its_top_group_precision():
    """The file placed in a is a wrong pick of a: 1 hit out of 2 scored as a."""
    predictions = [
        make("f1", "a", {"a": 0.9, "b": 0.1}),
        make("f2", None, {"a": 0.8, "b": 0.2}),
    ]
    a = summarize(predictions, DEFAULTS, None).per_group["a"]
    assert (a.n, a.precision, a.recall) == (1, 0.5, 1.0)


def test_top1_ignores_the_cutoffs_for_a_file_with_a_gold_group():
    """Sent to Unsorted with the right group on top: still right on top-1."""
    metrics = summarize([make("f", "a", {"a": 0.55, "b": 0.45})], DEFAULTS, None)
    assert metrics.unsorted_share == 1.0
    assert metrics.top1_accuracy == 1.0
    assert metrics.unsorted_top1_right_share == 1.0


def test_a_prediction_with_no_probs_is_wrong_and_left_out_of_the_calibration():
    """No probabilities means no top group and no confidence to score.

    It still counts in n and as a miss. The ECE comes from the other file alone: 0.9 and
    right, a gap of 0.1.
    """
    predictions = [make("f1", "a", {"a": 0.9, "b": 0.1}), make("f2", "a", {})]
    metrics = summarize(predictions, DEFAULTS, None)
    assert metrics.n == 2
    assert metrics.top1_accuracy == 0.5
    assert metrics.unsorted_share == 0.5
    assert metrics.ece == pytest.approx(0.1)


# tune_cutoffs

# Five dev files. Under DEFAULTS (0.5, 0.6): d1 and d2 are Unsorted (0 each), d3 is Unsorted (0),
# d4 is placed (-1), d5 is placed in its gold group (+1). Utility 0 / 5.
# The best any cutoffs can do is +4: place d1, d2 and d5, keep d3 and d4 out. That needs
# fits_any_below over 0.55 and at most 0.7, and top_below over 0.45 and at most 0.55.
DEV = [
    make("d1", "a", {"a": 0.55, "b": 0.45}),  # right, but under the default top cutoff
    make("d2", "a", {"a": 0.55, "b": 0.45}),
    make("d3", "b", {"a": 0.45, "b": 0.35, "c": 0.2}),  # wrong on top: better left Unsorted
    make("d4", None, {"a": 0.9, "b": 0.1}, fits_any=0.55),  # fits no group: must stay out
    make("d5", "a", {"a": 0.95, "b": 0.05}, fits_any=0.7),
]


def test_tune_cutoffs_finds_better_cutoffs_than_the_defaults():
    """Of the best pairs, (0.6, 0.55) is the nearest to the defaults (0.5, 0.6)."""
    tuned = tune_cutoffs(DEV, DEFAULTS)
    assert (tuned.fits_any_below, tuned.top_below) == (0.6, 0.55)
    assert utility(DEV, DEFAULTS) == 0.0
    assert utility(DEV, tuned) == pytest.approx(4 / 5)


def test_tune_cutoffs_result_is_the_best_on_the_grid():
    """No pair on the grid beats the tuned one."""
    tuned = tune_cutoffs(DEV, DEFAULTS)
    best = max(
        utility(DEV, Cutoffs(fits_any_below=fits, top_below=top, close_margin=0.15))
        for fits in CUTOFF_GRID
        for top in CUTOFF_GRID
    )
    assert utility(DEV, tuned) == best


def test_tune_cutoffs_does_not_tune_the_close_margin():
    """close_margin comes back as it went in."""
    defaults = Cutoffs(fits_any_below=0.5, top_below=0.6, close_margin=0.2)
    assert tune_cutoffs(DEV, defaults).close_margin == 0.2


def test_tune_cutoffs_keeps_the_defaults_on_a_tie():
    """One sure, right file is placed under every pair, so nothing beats the defaults."""
    sure = [make("d1", "a", {"a": 1.0, "b": 0.0}, fits_any=1.0)]
    assert tune_cutoffs(sure, DEFAULTS) == DEFAULTS


def test_tune_cutoffs_keeps_defaults_that_are_not_on_the_grid_on_a_tie():
    """Defaults between two grid values come back as they are when nothing beats them."""
    off_grid = Cutoffs(fits_any_below=0.52, top_below=0.61, close_margin=0.15)
    sure = [make("d1", "a", {"a": 1.0, "b": 0.0}, fits_any=1.0)]
    assert tune_cutoffs(sure, off_grid) == off_grid


def test_tune_cutoffs_never_does_worse_than_the_defaults():
    """A fits_any cutoff of 0.52 splits these two files, and no value on the grid can.

    d1 fits no group and must stay out (fits_any 0.51). d2 is right and must be placed (0.53).
    The defaults do both: +2. On the grid, 0.5 or under places both (-1 + 1) and 0.55 or over
    places neither (+1 + 0), so the defaults stay.
    """
    off_grid = Cutoffs(fits_any_below=0.52, top_below=0.6, close_margin=0.15)
    close_pair = [
        make("d1", None, {"a": 0.9, "b": 0.1}, fits_any=0.51),
        make("d2", "a", {"a": 0.9, "b": 0.1}, fits_any=0.53),
    ]
    tuned = tune_cutoffs(close_pair, off_grid)
    assert tuned == off_grid
    assert utility(close_pair, tuned) == 1.0


def test_tune_cutoffs_moves_off_grid_defaults_to_a_better_grid_pair():
    """Defaults off the grid still move when a grid pair scores higher.

    top_below has to come down to 0.55 to place this file. fits_any_below goes to 0.5, the
    grid value nearest to 0.52.
    """
    off_grid = Cutoffs(fits_any_below=0.52, top_below=0.61, close_margin=0.15)
    tuned = tune_cutoffs([make("d1", "a", {"a": 0.55, "b": 0.45})], off_grid)
    assert (tuned.fits_any_below, tuned.top_below) == (0.5, 0.55)


def test_tune_cutoffs_moves_no_further_than_it_has_to():
    """Only top_below has to move (down to 0.55) to place this file. fits_any_below stays."""
    tuned = tune_cutoffs([make("d1", "a", {"a": 0.55, "b": 0.45})], DEFAULTS)
    assert (tuned.fits_any_below, tuned.top_below) == (0.5, 0.55)


def test_tune_cutoffs_counts_unsorted_as_right_for_a_file_with_no_gold_group():
    """Two files with the same numbers, both placed under the defaults: one right (+1) and
    one that fits no group (-1), utility 0. Sending both to Unsorted scores +1 and 0, so
    fits_any_below goes up just past their 0.6."""
    twins = [
        make("d1", "a", {"a": 0.9, "b": 0.1}, fits_any=0.6),
        make("d2", None, {"a": 0.9, "b": 0.1}, fits_any=0.6),
    ]
    tuned = tune_cutoffs(twins, DEFAULTS)
    assert (tuned.fits_any_below, tuned.top_below) == (0.65, 0.6)


def test_tune_cutoffs_charges_for_wrong_placements():
    """Lowering top_below to 0.55 would place one right file and two wrong ones (+1 - 2).

    Leaving all three in Unsorted scores 0, which is better, so the defaults stay.
    """
    risky = [
        make("d1", "a", {"a": 0.55, "b": 0.45}),
        make("d2", "b", {"a": 0.55, "b": 0.45}),
        make("d3", "b", {"a": 0.55, "b": 0.45}),
    ]
    assert tune_cutoffs(risky, DEFAULTS) == DEFAULTS


def test_tune_cutoffs_equal_distance_goes_to_the_lower_value():
    """fits_any_below 0.45 and 0.55 both score +1 here and are equally far from 0.5.

    At 0.45 or under: d1 and d2 are placed right (+2), d3 is placed wrong (-1).
    At 0.5: d1 and d2 are Unsorted, d3 is still placed (-1).
    At 0.55 or over: all three are Unsorted, which is right for d3 only (+1).
    """
    split = [
        make("d1", "a", {"a": 0.9, "b": 0.1}, fits_any=0.45),
        make("d2", "a", {"a": 0.9, "b": 0.1}, fits_any=0.45),
        make("d3", None, {"a": 0.9, "b": 0.1}, fits_any=0.5),
    ]
    tuned = tune_cutoffs(split, DEFAULTS)
    assert (tuned.fits_any_below, tuned.top_below) == (0.45, 0.6)


def test_tune_cutoffs_equal_distance_is_not_decided_by_float_noise():
    """The same tie around a default of 0.4: 0.35 and 0.45 both score +1.

    In floating point (0.35 - 0.4) ** 2 comes out a hair larger than (0.45 - 0.4) ** 2, so an
    unrounded distance would pick 0.45. The two are equally far, and the lower value wins.
    """
    around_04 = Cutoffs(fits_any_below=0.4, top_below=0.6, close_margin=0.15)
    assert (0.35 - 0.4) ** 2 > (0.45 - 0.4) ** 2
    split = [
        make("d1", "a", {"a": 0.9, "b": 0.1}, fits_any=0.35),
        make("d2", "a", {"a": 0.9, "b": 0.1}, fits_any=0.35),
        make("d3", None, {"a": 0.9, "b": 0.1}, fits_any=0.4),
    ]
    tuned = tune_cutoffs(split, around_04)
    assert (tuned.fits_any_below, tuned.top_below) == (0.35, 0.6)


def test_tune_cutoffs_with_no_dev_predictions_returns_the_defaults():
    """Nothing to tune on: the defaults stand, even ones that are not on the grid."""
    assert tune_cutoffs([], DEFAULTS) == DEFAULTS
    off_grid = Cutoffs(fits_any_below=0.52, top_below=0.61, close_margin=0.15)
    assert tune_cutoffs([], off_grid) == off_grid


# tuning_ties


def test_tuning_ties_counts_the_pairs_that_share_the_best_score():
    """On DEV six pairs reach +4: fits_any 0.6, 0.65 or 0.7 with top 0.5 or 0.55."""
    assert tuning_ties(DEV, DEFAULTS) == TuningTies(pairs=361, tied=6, defaults_tied=False)


def test_tuning_ties_when_the_dev_split_cannot_tell_the_cutoffs_apart():
    """One sure, right file is placed under every pair on the grid, so all 361 tie for best."""
    sure = [make("d1", "a", {"a": 1.0, "b": 0.0}, fits_any=1.0)]
    assert tuning_ties(sure, DEFAULTS) == TuningTies(pairs=361, tied=361, defaults_tied=True)
    # With no dev predictions at all every pair scores 0.
    assert tuning_ties([], DEFAULTS) == TuningTies(pairs=361, tied=361, defaults_tied=True)


def test_tuning_ties_scores_defaults_that_are_not_on_the_grid_too():
    """Off-grid defaults are one more pair, as tune_cutoffs scores them."""
    off_grid = Cutoffs(fits_any_below=0.52, top_below=0.61, close_margin=0.15)
    sure = [make("d1", "a", {"a": 1.0, "b": 0.0}, fits_any=1.0)]
    assert tuning_ties(sure, off_grid) == TuningTies(pairs=362, tied=362, defaults_tied=True)
    # Two files that only a fits_any cutoff of 0.52 splits: the defaults score +2. On the grid,
    # 0.5 or under places both (0) and 0.55 or over places neither (+1), so nothing ties them.
    close_pair = [
        make("d1", None, {"a": 0.9, "b": 0.1}, fits_any=0.51),
        make("d2", "a", {"a": 0.9, "b": 0.1}, fits_any=0.53),
    ]
    alone = Cutoffs(fits_any_below=0.52, top_below=0.6, close_margin=0.15)
    assert tuning_ties(close_pair, alone) == TuningTies(pairs=362, tied=1, defaults_tied=True)
