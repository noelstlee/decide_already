"""sort.py: the placement rules, the judgment of one card, and the run over a vault's dump."""

import hashlib

import pytest

from decide_already import vault
from decide_already.backends.base import BackendError, BudgetExceeded, LogFailed, SortResult
from decide_already.card import VARIANTS, Card, build_card, render_card
from decide_already.config import JEV_LOG_FILE, LABELS_FILE, Config, decide_path
from decide_already.corrections import active_labels, freeze, move_file, sync, unfreeze
from decide_already.log import append_jsonl, read_jsonl
from decide_already.sort import (
    CHANGED_BY_HAND,
    NO_NOTE,
    NOT_REACHED,
    STOPPED_BEFORE,
    Decision,
    SortReport,
    decide,
    judge,
    sort_vault,
)
from decide_already.types import File, Group
from fakes import FakeBackend

# Answers for the fake backend: (probabilities by group id, fits_any).
TO_HW = ({"hw": 0.9, "logistics": 0.05, "project": 0.05}, 0.95)
TO_LOGISTICS = ({"hw": 0.05, "logistics": 0.9, "project": 0.05}, 0.95)
NO_FIT = ({"hw": 0.9, "logistics": 0.05, "project": 0.05}, 0.2)
WEAK = ({"hw": 0.35, "logistics": 0.33, "project": 0.32}, 0.9)
CLOSE = ({"hw": 0.5, "logistics": 0.4, "project": 0.1}, 0.9)  # needs a top cutoff of 0.5 or less

GROUPS = [
    Group(id="hw", name="HW", covers="homework"),
    Group(id="logistics", name="Logistics", covers="syllabus and deadlines"),
]


class ScriptedBackend(FakeBackend):
    """A fake that answers each request in turn from a script.

    An entry is a dict of SortResult fields that replace the fake's usual answer, or an
    exception to raise for that request.
    """

    def __init__(self, script):
        """Keep the script. The first request gets the first entry, and so on."""
        super().__init__()
        self.script = list(script)

    def sort(self, request):
        """Answer the next request from the script."""
        result = super().sort(request)
        entry = self.script[len(self.calls) - 1]
        if isinstance(entry, Exception):
            raise entry
        return result.model_copy(update=entry)


class CappedBackend(FakeBackend):
    """A fake with a spend cap: it answers a set number of requests and refuses the rest."""

    def __init__(self, allowed, **answers):
        """Answer `allowed` requests as FakeBackend would, then raise BudgetExceeded."""
        super().__init__(**answers)
        self.allowed = allowed

    def sort(self, request):
        """Answer while under the cap. A refused request is not sent, so it is not recorded."""
        if len(self.calls) >= self.allowed:
            raise BudgetExceeded("spend cap reached: $5.0000 spent, the cap is $5")
        return super().sort(request)


@pytest.fixture
def root(tmp_path):
    """A vault with three groups (HW, Logistics, Project), at set version 3, and an empty dump."""
    vault.init_vault(tmp_path)
    vault.add_group(tmp_path, "HW", "homework specs, starter code and solutions")
    vault.add_group(tmp_path, "Logistics", "syllabus, schedule and deadlines")
    vault.add_group(tmp_path, "Project", "project specs and examples")
    return tmp_path


def drop(root, file_id, text="Homework 3: implement the kernel functions.\n"):
    """Put a file in the dump folder and return its path."""
    path = root / "_dump" / file_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def note_of(root, file_id):
    """The one card note of a file. Fails when the file has no note, or more than one."""
    (note,) = [note for note in vault.read_card_notes(root) if note.placement.file_id == file_id]
    return note


def freeze_by_hand(root, file_id):
    """Set `frozen: true` in a file's card note, the way a person would in Obsidian."""
    path = note_of(root, file_id).path
    text = path.read_text(encoding="utf-8")
    assert "\nfrozen: false\n" in text
    path.write_text(text.replace("\nfrozen: false\n", "\nfrozen: true\n"), encoding="utf-8")


def set_config(root, **changes):
    """Change settings in the vault's config file."""
    Config.load(root).model_copy(update=changes).save(root)


def card_with(states, file_id="notes/long.txt"):
    """A card with the given states, as build_card would make for a window vote."""
    file = File(id=file_id, content_hash="0" * 64, file_type="Text", card_tokens=12)
    return Card(file=file, title=None, states=states)


def unsorted(reason):
    """The decision that sends a file to Unsorted for a reason."""
    return Decision(group_id=None, close_call=False, reason=reason)


def placed(group_id, close_call=False):
    """The decision that places a file in a group."""
    return Decision(group_id=group_id, close_call=close_call, reason=None)


# decide: the placement table in docs/implementation.md

# Close calls can only happen when the top cutoff is low enough: a top group at 0.6 or more
# leaves at most 0.4 for the second, which is never within 0.15.
TABLE = Config(unsorted_fits_any_below=0.5, unsorted_top_below=0.4, close_call_margin=0.15)
# Cutoffs that are exact in binary, so a value can sit right on one without rounding.
EXACT = Config(unsorted_fits_any_below=0.5, unsorted_top_below=0.5, close_call_margin=0.25)


@pytest.mark.parametrize(
    ("fits_any", "probs", "expected"),
    [
        # Row 1: fits_any under its cutoff. Unsorted, however sure the top group is.
        (0.49, {"a": 0.95, "b": 0.05}, unsorted("no_group_fits")),
        # Row 2: fits_any passes, the top probability is under its cutoff. Unsorted.
        (0.9, {"a": 0.39, "b": 0.31, "c": 0.30}, unsorted("weak_top")),
        # Row 3: both pass, top minus second (0.1) is under the margin. Placed, close call.
        (0.9, {"a": 0.5, "b": 0.4, "c": 0.1}, placed("a", close_call=True)),
        # Row 4: both pass, top minus second (0.6) is at or over the margin. Placed.
        (0.9, {"a": 0.8, "b": 0.2}, placed("a")),
    ],
)
def test_decide_follows_the_placement_table(fits_any, probs, expected):
    """Each row of the table gives the outcome the contract names."""
    assert decide(probs, fits_any, TABLE) == expected


def test_decide_with_the_default_cutoffs():
    """The starting values: fits_any under 0.5 or a top group under 0.6 means Unsorted."""
    config = Config()
    sure = {"hw": 0.9, "logistics": 0.1}
    assert decide(sure, 0.95, config) == placed("hw")
    assert decide(sure, 0.49, config) == unsorted("no_group_fits")
    assert decide({"hw": 0.59, "logistics": 0.41}, 0.95, config) == unsorted("weak_top")
    assert decide({"hw": 0.6, "logistics": 0.4}, 0.5, config) == placed("hw")


def test_decide_a_value_exactly_at_a_cutoff_counts_as_at_or_over():
    """fits_any 0.5, top 0.5 and top minus second 0.25 all sit on their cutoffs and all pass."""
    assert decide({"a": 0.5, "b": 0.25, "c": 0.25}, 0.5, EXACT) == placed("a")


def test_decide_just_under_each_cutoff():
    """One step under each cutoff in turn changes the outcome."""
    # fits_any just under: Unsorted, with the same probabilities that were placed above.
    assert decide({"a": 0.5, "b": 0.25, "c": 0.25}, 0.4999, EXACT) == unsorted("no_group_fits")
    # The top probability just under: Unsorted.
    assert decide({"a": 0.4375, "b": 0.3125, "c": 0.25}, 0.5, EXACT) == unsorted("weak_top")
    # The margin just under (0.125): still placed, now flagged.
    assert decide({"a": 0.5, "b": 0.375, "c": 0.125}, 0.5, EXACT) == placed("a", close_call=True)


def test_decide_fits_any_is_checked_before_the_top_probability():
    """When both cutoffs fail, the reason is the first row's: no group fits."""
    assert decide({"a": 0.2, "b": 0.2}, 0.1, EXACT) == unsorted("no_group_fits")


def test_decide_with_one_group_the_second_probability_counts_as_0():
    """A lone group is compared against 0, so its own probability is the margin."""
    assert decide({"a": 1.0}, 0.9, Config()) == placed("a")
    assert decide({"a": 0.7}, 0.9, Config()) == placed("a")
    assert decide({"a": 0.59}, 0.9, Config()) == unsorted("weak_top")
    loose = Config(unsorted_top_below=0.0)
    assert decide({"a": 0.1}, 0.9, loose) == placed("a", close_call=True)


def test_decide_a_tie_goes_to_the_lower_group_id():
    """Level at the top: the lower id gets the file, whatever the order of the answer."""
    assert decide({"b": 0.5, "a": 0.5}, 0.9, EXACT) == placed("a", close_call=True)
    assert decide({"a": 0.5, "b": 0.5}, 0.9, EXACT) == placed("a", close_call=True)
    three = Config(unsorted_top_below=0.25)
    assert decide({"c": 0.3125, "b": 0.3125, "a": 0.3125}, 0.9, three) == placed("a", True)


def test_decide_a_tie_for_second_place_does_not_matter():
    """Only the top two probabilities are compared."""
    assert decide({"c": 0.2, "a": 0.6, "b": 0.2}, 0.9, Config()) == placed("a")


def test_decide_with_no_probs_is_unsorted_because_no_group_fits():
    """A backend with no pick gives nothing to place by, however high fits_any is."""
    assert decide({}, 1.0, Config()) == unsorted("no_group_fits")
    assert decide({}, 0.0, Config()) == unsorted("no_group_fits")


def test_decide_unsorted_is_never_a_close_call():
    """The close call flag is for placed files only."""
    assert decide({"a": 0.5, "b": 0.5}, 0.1, EXACT).close_call is False
    assert decide({"a": 0.26, "b": 0.25, "c": 0.49}, 0.9, EXACT).close_call is False


# judge


def test_judge_sends_one_request_per_state():
    """Each state of the card becomes its own request, in order, with the same groups."""
    states = [{"text": "first window"}, {"text": "second window"}, {"text": "third window"}]
    backend = FakeBackend()
    result = judge(card_with(states), GROUPS, 7, backend, "window_vote")
    assert [request.state for request in backend.calls] == states
    for request in backend.calls:
        assert request.file_id == "notes/long.txt"
        assert request.groups == GROUPS
        assert request.set_version == 7
        assert request.variant == "window_vote"
        assert request.with_descriptions is True
    assert result.file_id == "notes/long.txt"


def test_judge_averages_probs_and_fits_any_over_the_states():
    """Three windows that disagree: the answer is the mean of the three."""
    backend = ScriptedBackend(
        [
            {"probs": {"hw": 0.75, "logistics": 0.25}, "fits_any": 1.0},
            {"probs": {"hw": 0.25, "logistics": 0.75}, "fits_any": 0.5},
            {"probs": {"hw": 0.5, "logistics": 0.5}, "fits_any": 0.75},
        ]
    )
    result = judge(
        card_with([{"text": "a"}, {"text": "b"}, {"text": "c"}]), GROUPS, 2, backend, "x"
    )
    assert result.probs == pytest.approx({"hw": 0.5, "logistics": 0.5})
    assert result.fits_any == pytest.approx(0.75)


def test_judge_appends_each_states_own_answer_to_per_state():
    """A caller that wants the single votes passes a list and gets one answer per state."""
    script = [
        {"probs": {"hw": 1.0, "logistics": 0.0}, "fits_any": 1.0},
        {"probs": {"hw": 0.0, "logistics": 1.0}, "fits_any": 0.5},
    ]
    earlier = SortResult(file_id="x", probs={}, fits_any=0.0, model="fake-0", option_order=[])
    votes = [earlier]
    result = judge(
        card_with([{"text": "a"}, {"text": "b"}]), GROUPS, 2, ScriptedBackend(script), "x", votes
    )
    assert votes[0] is earlier  # what was already in the list stays
    assert [vote.probs for vote in votes[1:]] == [entry["probs"] for entry in script]
    assert [vote.fits_any for vote in votes[1:]] == [1.0, 0.5]
    assert result.probs == pytest.approx({"hw": 0.5, "logistics": 0.5})


def test_judge_adds_up_tokens_and_latency_and_keeps_the_first_requests_details():
    """Cost and time are totals over the requests. Order, model and id are the first one's."""
    backend = ScriptedBackend(
        [
            {
                "input_tokens": 100,
                "output_tokens": 5,
                "latency_ms": 10.5,
                "request_id": "req_1",
                "option_order": ["logistics", "hw"],
                "model": "fake-1",
            },
            {
                "input_tokens": 200,
                "output_tokens": 7,
                "latency_ms": 20.25,
                "request_id": "req_2",
                "option_order": ["hw", "logistics"],
                "model": "fake-2",
            },
        ]
    )
    result = judge(card_with([{"text": "a"}, {"text": "b"}]), GROUPS, 2, backend, "window_vote")
    assert result.input_tokens == 300
    assert result.output_tokens == 12
    assert result.latency_ms == pytest.approx(30.75)
    assert result.request_id == "req_1"
    assert result.option_order == ["logistics", "hw"]
    assert result.model == "fake-1"


def test_judge_keeps_none_when_the_backend_reports_no_tokens_or_latency():
    """The fake reports no cost and no time, so the totals stay None, not 0."""
    result = judge(card_with([{"text": "a"}, {"text": "b"}]), GROUPS, 2, FakeBackend(), "card")
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.latency_ms is None
    assert result.request_id is None


def test_judge_with_one_state_returns_the_backends_own_answer():
    """The usual case: one state, one request, and nothing is changed by the averaging."""
    backend = ScriptedBackend(
        [
            {
                "probs": {"logistics": 0.1, "hw": 0.9},
                "fits_any": 0.3,
                "input_tokens": 2710,
                "latency_ms": 412.3,
                "request_id": "req_1",
            }
        ]
    )
    per_state = []
    result = judge(card_with([{"opening": "text"}]), GROUPS, 2, backend, "card", per_state)
    assert per_state == [result]
    assert list(result.probs) == ["logistics", "hw"]
    assert (result.probs, result.fits_any) == ({"logistics": 0.1, "hw": 0.9}, 0.3)
    assert (result.input_tokens, result.latency_ms, result.output_tokens) == (2710, 412.3, None)


@pytest.mark.parametrize("variant", VARIANTS)
def test_judge_leaves_out_the_descriptions_only_for_names_only(variant):
    """names_only is the one variant that sends the groups without their descriptions."""
    backend = FakeBackend()
    judge(card_with([{"opening": "text"}]), GROUPS, 2, backend, variant)
    (request,) = backend.calls
    assert request.variant == variant
    assert request.with_descriptions is (variant != "names_only")


def test_judge_counts_a_group_missing_from_one_answer_as_0():
    """A backend that names only its pick: the other groups count as 0 in that state."""
    backend = ScriptedBackend(
        [
            {"probs": {"hw": 1.0}},
            {"probs": {"logistics": 1.0}},
            {"probs": {}},
            {"probs": {"hw": 1.0}},
        ]
    )
    result = judge(card_with([{"text": str(n)} for n in range(4)]), GROUPS, 2, backend, "x")
    assert result.probs == pytest.approx({"hw": 0.5, "logistics": 0.25})


def test_judge_keeps_probs_empty_when_no_state_had_a_pick():
    """No pick in any state is still no pick, so decide sends the file to Unsorted."""
    backend = ScriptedBackend([{"probs": {}, "fits_any": 0.0}, {"probs": {}, "fits_any": 0.0}])
    result = judge(card_with([{"text": "a"}, {"text": "b"}]), GROUPS, 2, backend, "x")
    assert result.probs == {}
    assert decide(result.probs, result.fits_any, Config()) == unsorted("no_group_fits")


def test_judge_refuses_a_card_with_no_states():
    """A file with no text has nothing to ask about. The backend is not called."""
    backend = FakeBackend()
    with pytest.raises(ValueError, match="no text"):
        judge(card_with([]), GROUPS, 2, backend, "card")
    assert backend.calls == []


def test_judge_passes_a_backend_error_on():
    """One failed request fails the file. The votes so far are in per_state, the answer is not."""
    backend = ScriptedBackend([{}, BackendError("timed out"), {}])
    votes = []
    with pytest.raises(BackendError, match="timed out"):
        judge(
            card_with([{"text": "a"}, {"text": "b"}, {"text": "c"}]), GROUPS, 2, backend, "x", votes
        )
    assert len(votes) == 1
    assert len(backend.calls) == 2  # the third window was never sent


# sort_vault


def test_sort_writes_a_card_note_into_each_files_group_folder(root):
    """The whole path: card, request, decision, and a note whose frontmatter is the placement."""
    solution = drop(
        root, "hw3/hw3_solution.py", '"""Homework 3 solution."""\n\ndef kernel():\n    pass\n'
    )
    drop(root, "syllabus.md", "# Syllabus\n\nGrading, deadlines and office hours.\n")
    backend = FakeBackend(answers={"hw3/hw3_solution.py": TO_HW, "syllabus.md": TO_LOGISTICS})
    report = sort_vault(root, backend)
    assert report == SortReport(
        set_version=3,
        asked=2,
        placed=2,
        unsorted=0,
        close_calls=0,
        frozen_skipped=0,
        errors={},
        spent_usd=None,
    )

    note = note_of(root, "hw3/hw3_solution.py")
    assert note.path == root / "HW" / "hw3__hw3_solution.py.md"
    assert note.placement.model_dump() == {
        "source": "[[_dump/hw3/hw3_solution.py]]",
        "file_id": "hw3/hw3_solution.py",
        "content_hash": hashlib.sha256(solution.read_bytes()).hexdigest(),
        "file_type": "Python code",
        "group_id": "hw",
        "placed_by": "jev",
        "frozen": False,
        "close_call": False,
        "reason": None,
        "set_version": 3,
        "probs": TO_HW[0],
        "fits_any": 0.95,
        "variant": "card",
        "model": "fake-0",
    }
    card = build_card(solution, "hw3/hw3_solution.py", Config.load(root))
    assert note.body == render_card(card)
    assert note_of(root, "syllabus.md").path == root / "Logistics" / "syllabus.md.md"

    # What the backend was asked: the card as the only state, against the full current set.
    request = backend.calls[0]
    assert request.file_id == "hw3/hw3_solution.py"
    assert request.state == card.states[0]
    assert request.groups == vault.read_groups(root)
    assert [group.id for group in request.groups] == ["hw", "logistics", "project"]
    assert (request.set_version, request.variant, request.with_descriptions) == (3, "card", True)


def test_sort_sends_a_file_to_unsorted_by_the_rules(root):
    """A low fits_any or a weak top group: the note goes to Unsorted and says why."""
    drop(root, "apartment_checklist.txt", "Boxes, tape, movers, change of address.\n")
    drop(root, "mixed.txt", "A bit of everything.\n")
    backend = FakeBackend(answers={"apartment_checklist.txt": NO_FIT, "mixed.txt": WEAK})
    report = sort_vault(root, backend)
    assert (report.asked, report.placed, report.unsorted) == (2, 0, 2)

    no_fit = note_of(root, "apartment_checklist.txt")
    assert no_fit.folder == "Unsorted"
    assert (no_fit.placement.group_id, no_fit.placement.reason) == (None, "no_group_fits")
    weak = note_of(root, "mixed.txt")
    assert weak.folder == "Unsorted"
    assert (weak.placement.group_id, weak.placement.reason) == (None, "weak_top")
    # The model's answer is kept, so the board can still show what it leaned toward.
    assert (weak.placement.probs, weak.placement.fits_any) == WEAK
    assert (weak.placement.placed_by, weak.placement.model) == ("jev", "fake-0")


def test_sort_flags_a_close_call(root):
    """Top two groups within the margin: placed in the top group, and flagged."""
    set_config(root, unsorted_top_below=0.4)
    drop(root, "project_logistics_sheet.md", "# Project deadlines\n\nProposal due in week 4.\n")
    report = sort_vault(root, FakeBackend(default=CLOSE))
    assert (report.placed, report.close_calls, report.unsorted) == (1, 1, 0)
    note = note_of(root, "project_logistics_sheet.md")
    assert note.folder == "HW"
    assert (note.placement.group_id, note.placement.close_call) == ("hw", True)
    assert note.placement.reason is None


def test_a_file_with_no_text_goes_to_unsorted_without_asking_the_backend(root):
    """Nothing to read means nothing to send: code places the file, and says why."""
    drop(root, "empty.txt", "")
    drop(root, "blank.md", "   \n\n\t\n")
    (root / "_dump" / "photo.bin").write_bytes(b"\x89PNG\x00\x00\x01\x02" * 50)
    backend = FakeBackend(default=TO_HW)
    report = sort_vault(root, backend)
    assert backend.calls == []
    assert report == SortReport(
        set_version=3,
        asked=0,
        placed=0,
        unsorted=3,
        close_calls=0,
        frozen_skipped=0,
        errors={},
        spent_usd=None,
    )
    for file_id in ("empty.txt", "blank.md", "photo.bin"):
        note = note_of(root, file_id)
        assert note.folder == "Unsorted"
        placement = note.placement
        assert (placement.group_id, placement.reason, placement.placed_by) == (
            None,
            "no_text",
            "code",
        )
        assert (placement.probs, placement.fits_any, placement.model) == (None, None, None)
        assert (placement.set_version, placement.variant, placement.frozen) == (3, "card", False)
        assert "No text could be extracted from this file." in note.body
    assert note_of(root, "photo.bin").placement.file_type == "Binary file"


def test_a_frozen_file_is_skipped(root):
    """A frozen file never goes back to the model, and its note is left exactly as it is."""
    drop(root, "a.txt")
    drop(root, "b.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze_by_hand(root, "a.txt")
    frozen_note = note_of(root, "a.txt").path
    before = frozen_note.read_bytes()

    backend = FakeBackend(default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert (report.frozen_skipped, report.asked, report.placed) == (1, 1, 1)
    assert [request.file_id for request in backend.calls] == ["b.txt"]
    assert frozen_note.read_bytes() == before
    assert note_of(root, "a.txt").folder == "HW"
    assert note_of(root, "b.txt").folder == "Logistics"

    # A new set version does not thaw it either: the user's label stands until they unfreeze.
    vault.add_group(root, "Exams", "midterms and finals")
    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert (report.set_version, report.frozen_skipped, report.asked) == (4, 1, 1)
    assert frozen_note.read_bytes() == before
    assert note_of(root, "a.txt").placement.set_version == 3


def test_a_note_left_in_a_deleted_groups_folder_is_replaced_by_the_rerun(root):
    """An unfrozen file whose group was deleted is sorted again, and its old note goes."""
    drop(root, "a.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    vault.delete_group(root, "HW")  # the folder is set aside, since the note is still in it
    old = note_of(root, "a.txt").path
    assert old == root / "HW (old)" / "a.txt.md"

    answer = ({"logistics": 0.9, "project": 0.1}, 0.95)
    report = sort_vault(root, FakeBackend(default=answer))
    assert (report.set_version, report.asked, report.placed) == (4, 1, 1)
    assert note_of(root, "a.txt").path == root / "Logistics" / "a.txt.md"
    assert not old.exists()
    assert not (root / "HW (old)").exists()  # emptied by the sort's sync, so it is removed


def test_a_frozen_file_whose_content_changed_is_sorted_again(root):
    """The freeze covered the old content. New content goes back to the model."""
    path = drop(root, "a.txt", "Homework 3: implement the kernel functions.\n")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze_by_hand(root, "a.txt")
    assert sync(root).confirmed == ["a.txt"]  # the freeze is on record before the change
    old_hash = note_of(root, "a.txt").placement.content_hash
    path.write_text("Syllabus: grading, deadlines and office hours.\n", encoding="utf-8")

    backend = FakeBackend(default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert (report.frozen_skipped, report.asked, report.placed) == (0, 1, 1)
    note = note_of(root, "a.txt")
    assert note.folder == "Logistics"
    assert (note.placement.frozen, note.placement.placed_by) == (False, "jev")
    assert note.placement.content_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert note.placement.content_hash != old_hash
    assert "Syllabus: grading" in note.body
    assert not (root / "HW" / "a.txt.md").exists()


def test_an_unfrozen_file_is_sorted_again_and_its_note_moves(root):
    """A rerun asks about every unfrozen file. A new answer moves the note; the old one goes."""
    drop(root, "hw3/notes.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    old = note_of(root, "hw3/notes.txt").path
    assert old == root / "HW" / "hw3__notes.txt.md"

    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert (report.asked, report.placed) == (1, 1)
    note = note_of(root, "hw3/notes.txt")  # exactly one note is left
    assert note.path == root / "Logistics" / "hw3__notes.txt.md"
    assert note.placement.group_id == "logistics"
    assert not old.exists()

    sort_vault(root, FakeBackend(default=NO_FIT))
    assert note_of(root, "hw3/notes.txt").path == root / "Unsorted" / "hw3__notes.txt.md"
    assert not note.path.exists()


def test_sorting_twice_with_the_same_answers_changes_nothing(root):
    """A rerun with no new information leaves one note per file, byte for byte the same."""
    for file_id in ("a.txt", "hw3/a.txt", "empty.txt"):
        drop(root, file_id, "" if file_id == "empty.txt" else "Homework text.\n")
    first = sort_vault(root, FakeBackend(default=TO_HW))
    before = {note.path: note.path.read_bytes() for note in vault.read_card_notes(root)}
    second = sort_vault(root, FakeBackend(default=TO_HW))
    after = {note.path: note.path.read_bytes() for note in vault.read_card_notes(root)}
    assert first == second
    assert before == after
    assert len(after) == 3


def test_a_sort_does_not_read_the_card_notes_once_per_file(root, monkeypatch):
    """Reading every note again for each file written would make a long sort crawl.

    A run reads them twice in all, however many files there are: once to pick up edits made
    by hand (corrections.sync), and once for the sort itself.
    """
    for number in range(6):
        drop(root, f"file{number}.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    reads = []
    read_card_notes = vault.read_card_notes

    def counting(root):
        """read_card_notes, with every call counted."""
        reads.append(root)
        return read_card_notes(root)

    monkeypatch.setattr(vault, "read_card_notes", counting)
    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert report.placed == 6
    assert len(reads) == 2
    monkeypatch.undo()
    notes = vault.read_card_notes(root)
    assert [note.folder for note in notes] == ["Logistics"] * 6  # moved, with no copy left behind


def test_a_rerun_asks_against_the_full_current_set(root):
    """A group added or edited since the last sort is part of the next one."""
    drop(root, "lecture09.txt", "Lecture 9: kernels and the dual.\n")
    assert sort_vault(root, FakeBackend(default=TO_HW)).set_version == 3
    vault.add_group(root, "Lecture notes", "lecture slides and notes")
    note_path = root / "_groups" / "HW.md"
    edited = note_path.read_text(encoding="utf-8").replace("homework specs", "problem sets")
    note_path.write_text(edited, encoding="utf-8")  # an edit by hand, picked up by the sort

    answer = ({"hw": 0.05, "lecture-notes": 0.9, "logistics": 0.03, "project": 0.02}, 0.9)
    backend = FakeBackend(default=answer)
    report = sort_vault(root, backend)
    assert report.set_version == 5
    (request,) = backend.calls
    assert [group.id for group in request.groups] == ["hw", "lecture-notes", "logistics", "project"]
    assert request.groups[0].covers.startswith("problem sets")
    assert request.set_version == 5
    note = note_of(root, "lecture09.txt")
    assert note.path == root / "Lecture notes" / "lecture09.txt.md"
    assert note.placement.set_version == 5


def test_a_backend_error_keeps_the_previous_note_and_is_reported(root):
    """A failure is never an answer: the old note stays, a new file gets no note."""
    drop(root, "a.txt")
    drop(root, "b.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    drop(root, "c.txt")
    kept = note_of(root, "b.txt").path
    before = kept.read_bytes()

    backend = FakeBackend(default=TO_LOGISTICS, fail={"b.txt", "c.txt"})
    report = sort_vault(root, backend)
    assert report.errors == {
        "b.txt": "the fake backend was told to fail for b.txt",
        "c.txt": "the fake backend was told to fail for c.txt",
    }
    # All three were sent to the backend; one came back with an answer.
    assert (report.asked, report.placed, report.unsorted) == (3, 1, 0)
    assert kept.read_bytes() == before
    assert note_of(root, "b.txt").folder == "HW"
    assert note_of(root, "a.txt").folder == "Logistics"
    assert [note.placement.file_id for note in vault.read_card_notes(root)] == ["a.txt", "b.txt"]


def test_the_spend_cap_stops_the_run_and_lists_the_files_not_reached(root):
    """Once the backend refuses for the cap, nothing after that file is handled at all."""
    for file_id in ("a.txt", "b.txt", "c.txt"):
        drop(root, file_id)
    drop(root, "d.txt", "")  # even a file that needs no model is left for the next run
    progress = []
    backend = CappedBackend(1, default=TO_HW)
    report = sort_vault(root, backend, lambda done, total: progress.append((done, total)))
    assert report.errors == {
        "b.txt": "spend cap reached: $5.0000 spent, the cap is $5",
        "c.txt": NOT_REACHED,
        "d.txt": NOT_REACHED,
    }
    assert "spend cap" in NOT_REACHED
    assert (report.asked, report.placed, report.unsorted) == (1, 1, 0)
    assert [request.file_id for request in backend.calls] == ["a.txt"]
    assert [note.placement.file_id for note in vault.read_card_notes(root)] == ["a.txt"]
    assert progress == [(1, 4)]


def test_the_report_counts_every_kind_of_outcome(root):
    """One file of each kind, and each lands in the right count."""
    set_config(root, unsorted_top_below=0.4)
    drop(root, "f_frozen.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze_by_hand(root, "f_frozen.txt")
    for file_id in ("a_placed.txt", "b_close.txt", "c_no_fit.txt", "d_weak.txt", "g_fails.txt"):
        drop(root, file_id)
    drop(root, "e_empty.txt", "")
    drop(root, ".hidden.txt")

    backend = FakeBackend(
        answers={
            "a_placed.txt": TO_LOGISTICS,
            "b_close.txt": CLOSE,
            "c_no_fit.txt": NO_FIT,
            "d_weak.txt": WEAK,
        },
        fail={"g_fails.txt"},
    )
    report = sort_vault(root, backend)
    assert report == SortReport(
        set_version=3,
        asked=5,  # a, b, c, d and the one that failed
        placed=2,  # a and b
        unsorted=3,  # c, d and the empty file
        close_calls=1,  # b
        frozen_skipped=1,
        errors={"g_fails.txt": "the fake backend was told to fail for g_fails.txt"},
        spent_usd=None,
    )
    folders = {note.placement.file_id: note.folder for note in vault.read_card_notes(root)}
    assert folders == {
        "a_placed.txt": "Logistics",
        "b_close.txt": "HW",
        "c_no_fit.txt": "Unsorted",
        "d_weak.txt": "Unsorted",
        "e_empty.txt": "Unsorted",
        "f_frozen.txt": "HW",
    }


def test_on_progress_is_called_after_each_file_skipped_ones_included(root):
    """Placed, without text, frozen or failed: each file moves the progress on by one."""
    drop(root, "c_frozen.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze_by_hand(root, "c_frozen.txt")
    drop(root, "a_placed.txt")
    drop(root, "b_empty.txt", "")
    drop(root, "d_fails.txt")
    progress = []
    backend = FakeBackend(default=TO_HW, fail={"d_fails.txt"})
    sort_vault(root, backend, lambda done, total: progress.append((done, total)))
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_spent_usd_is_the_vaults_total_from_the_jev_log(root):
    """Input tokens in the log times the config's price. No log means None, not 0."""
    drop(root, "a.txt")
    assert sort_vault(root, FakeBackend(default=TO_HW)).spent_usd is None
    log_path = decide_path(root, JEV_LOG_FILE)
    append_jsonl(log_path, {"file_id": "a.txt", "input_tokens": 1_000_000, "status": "ok"})
    append_jsonl(log_path, {"file_id": "b.txt", "input_tokens": 500_000, "status": "error"})
    append_jsonl(log_path, {"file_id": "c.txt", "input_tokens": None, "status": "error"})
    assert sort_vault(root, FakeBackend(default=TO_HW)).spent_usd == pytest.approx(0.063)
    set_config(root, usd_per_million_tokens=2.0)
    assert sort_vault(root, FakeBackend(default=TO_HW)).spent_usd == pytest.approx(3.0)


def test_an_unknown_variant_in_the_config_stops_the_sort_before_it_starts(root):
    """A typo in the config is an error up front, even when there is nothing to sort yet."""
    set_config(root, variant="cards")
    backend = FakeBackend(default=TO_HW)
    with pytest.raises(ValueError, match="unknown variant 'cards' in the vault config"):
        sort_vault(root, backend)
    drop(root, "a.txt")
    with pytest.raises(ValueError, match="unknown variant 'cards' in the vault config"):
        sort_vault(root, backend)
    assert backend.calls == []
    assert vault.read_card_notes(root) == []


def test_sorting_without_groups_is_an_error(tmp_path):
    """With nothing to choose from there is nothing to ask."""
    vault.init_vault(tmp_path)
    drop(tmp_path, "a.txt")
    backend = FakeBackend()
    with pytest.raises(ValueError, match="no groups"):
        sort_vault(tmp_path, backend)
    assert backend.calls == []


def test_sorting_an_empty_dump_reports_nothing(root):
    """No files: no requests, no notes, no progress, all counts at 0."""
    progress = []
    report = sort_vault(root, FakeBackend(), lambda done, total: progress.append((done, total)))
    assert (report.asked, report.placed, report.unsorted, report.errors) == (0, 0, 0, {})
    assert progress == []


def test_the_configs_variant_decides_what_is_sent(root):
    """names_only sends the same card with the descriptions left out, and the note says so."""
    set_config(root, variant="names_only")
    drop(root, "a.txt")
    backend = FakeBackend(default=TO_HW)
    sort_vault(root, backend)
    (request,) = backend.calls
    assert (request.variant, request.with_descriptions) == ("names_only", False)
    assert note_of(root, "a.txt").placement.variant == "names_only"


def test_a_window_vote_is_several_requests_but_one_file_asked(root):
    """A long file under window_vote: one request per window, one answer, one file counted."""
    set_config(root, variant="window_vote")
    path = drop(root, "long.txt", " ".join(f"word{number}" for number in range(3000)))
    windows = build_card(path, "long.txt", Config.load(root)).states
    assert len(windows) == 4  # about 26,000 characters in windows of at most 8,000
    backend = FakeBackend(default=TO_HW)
    report = sort_vault(root, backend)
    assert [request.state for request in backend.calls] == windows
    assert (report.asked, report.placed) == (1, 1)
    placement = note_of(root, "long.txt").placement
    assert placement.variant == "window_vote"
    assert placement.probs == pytest.approx(TO_HW[0])
    assert placement.fits_any == pytest.approx(0.95)


def test_a_file_that_cannot_be_read_is_reported_and_the_run_goes_on(root, monkeypatch):
    """An OSError while building a card is that file's error, not the end of the run."""
    for file_id in ("a.txt", "b.txt", "c.txt"):
        drop(root, file_id)

    def build_or_fail(path, file_id, config, variant=None):
        """Build the card, except for b.txt, which cannot be opened."""
        if file_id == "b.txt":
            raise PermissionError(13, "Permission denied", str(path))
        return build_card(path, file_id, config, variant)

    monkeypatch.setattr("decide_already.sort.build_card", build_or_fail)
    backend = FakeBackend(default=TO_HW)
    report = sort_vault(root, backend)
    assert list(report.errors) == ["b.txt"]
    assert "Permission denied" in report.errors["b.txt"]
    assert (report.asked, report.placed) == (2, 2)
    assert [request.file_id for request in backend.calls] == ["a.txt", "c.txt"]


def test_a_note_of_the_users_in_the_way_is_reported_not_overwritten(root):
    """The user's own file sits where the card note would go: it stays, and the file is listed."""
    drop(root, "a.txt")
    drop(root, "b.txt")
    own = root / "HW" / "a.txt.md"
    own.write_text("My own note.\n", encoding="utf-8")
    report = sort_vault(root, FakeBackend(default=TO_HW))
    assert list(report.errors) == ["a.txt"]
    assert "HW/a.txt.md exists and is not the card note of a.txt" in report.errors["a.txt"]
    assert own.read_text(encoding="utf-8") == "My own note.\n"
    assert (report.asked, report.placed) == (2, 1)
    assert [note.placement.file_id for note in vault.read_card_notes(root)] == ["b.txt"]


def test_hidden_files_and_the_users_own_notes_are_left_out(root):
    """Hidden files in the dump are not sorted. The user's notes in the vault are not touched."""
    drop(root, "a.txt")
    drop(root, ".DS_Store", "not for sorting")
    drop(root, ".git/config", "not for sorting")
    own = {
        root / "HW" / "my notes.md": "Things to remember.\n",
        root / "Unsorted" / "todo.md": "---\ntags: [todo]\n---\nSort these later.\n",
        root / "README.md": "# My vault\n",
    }
    for path, text in own.items():
        path.write_text(text, encoding="utf-8")
    backend = FakeBackend(default=TO_HW)
    report = sort_vault(root, backend)
    assert [request.file_id for request in backend.calls] == ["a.txt"]
    assert (report.asked, report.placed, report.unsorted) == (1, 1, 0)
    assert [note.placement.file_id for note in vault.read_card_notes(root)] == ["a.txt"]
    for path, text in own.items():
        assert path.read_text(encoding="utf-8") == text


def test_two_files_with_the_same_name_get_their_own_notes(root):
    """hw1/README.md and hw2/README.md are two files, with two notes, in one folder."""
    drop(root, "hw1/README.md", "# Homework 1\n\nDue in week 2.\n")
    drop(root, "hw2/README.md", "# Homework 2\n\nDue in week 4.\n")
    sort_vault(root, FakeBackend(default=TO_HW))
    assert note_of(root, "hw1/README.md").path == root / "HW" / "hw1__README.md.md"
    assert note_of(root, "hw2/README.md").path == root / "HW" / "hw2__README.md.md"
    assert "Homework 1" in note_of(root, "hw1/README.md").body
    assert "Homework 2" in note_of(root, "hw2/README.md").body


def test_the_originals_are_never_touched(root):
    """Sorting reads the dump and writes elsewhere: same files, same bytes, same times."""
    drop(root, "a.txt")
    drop(root, "hw3/b.py", "def kernel():\n    pass\n")
    drop(root, "empty.txt", "")

    def snapshot():
        """Every path under the dump folder with its size, bytes and modification time."""
        paths = sorted((root / "_dump").rglob("*"))
        return [
            (path, path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
            for path in paths
        ]

    before = snapshot()
    sort_vault(root, FakeBackend(default=TO_HW))
    sort_vault(root, FakeBackend(default=TO_LOGISTICS, fail={"a.txt"}))
    assert snapshot() == before


# Corrections in a sort: edits made by hand are picked up first, and new content ends a freeze


def labels(root):
    """Every line of the vault's labels file as a dict, without its time."""
    lines = read_jsonl(decide_path(root, LABELS_FILE))
    return [{key: value for key, value in line.items() if key != "at"} for line in lines]


def label(file_id, kind, from_group, to_group, set_version=3, jev_probs=TO_HW[0]):
    """One expected line of the labels file, for a file the model placed under set version 3."""
    return {
        "file_id": file_id,
        "kind": kind,
        "from_group": from_group,
        "to_group": to_group,
        "jev_probs": jev_probs,
        "jev_set_version": 3,
        "set_version": set_version,
    }


def test_a_note_dragged_by_hand_is_not_overwritten_by_the_next_sort(root):
    """sort_vault syncs first: the drag is a move, and a moved file is frozen where it was put."""
    drop(root, "a.txt")
    drop(root, "b.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    dragged = note_of(root, "a.txt").path
    dragged.rename(root / "Logistics" / dragged.name)  # in Obsidian, with no `p0 sync` after it

    backend = FakeBackend(default=TO_HW)  # the model would put it straight back in HW
    report = sort_vault(root, backend)
    assert (report.frozen_skipped, report.asked, report.placed) == (1, 1, 1)
    assert [request.file_id for request in backend.calls] == ["b.txt"]
    note = note_of(root, "a.txt")
    assert note.path == root / "Logistics" / "a.txt.md"
    placement = note.placement
    assert (placement.group_id, placement.placed_by, placement.frozen) == ("logistics", "you", True)
    assert labels(root) == [label("a.txt", "moved", "hw", "logistics")]


def test_a_flag_set_by_hand_is_confirmed_once_however_many_sorts_follow(root):
    """The first sort after the edit writes the label. Later sorts find nothing new."""
    drop(root, "a.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze_by_hand(root, "a.txt")
    for _ in range(3):
        backend = FakeBackend(default=TO_LOGISTICS)
        assert sort_vault(root, backend).frozen_skipped == 1
        assert backend.calls == []
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw")]


def test_a_frozen_file_in_a_deleted_group_is_unfrozen_and_sorted_again(root):
    """The sort's sync sends it to Unsorted without its freeze, so the sort asks about it."""
    drop(root, "a.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze(root, "a.txt")
    vault.delete_group(root, "HW")

    backend = FakeBackend(default=({"logistics": 0.9, "project": 0.1}, 0.95))
    report = sort_vault(root, backend)
    assert (report.set_version, report.frozen_skipped, report.asked, report.placed) == (4, 0, 1, 1)
    note = note_of(root, "a.txt")
    assert note.path == root / "Logistics" / "a.txt.md"
    assert (note.placement.frozen, note.placement.placed_by) == (False, "jev")
    assert labels(root) == [
        label("a.txt", "confirmed", "hw", "hw"),
        label("a.txt", "unfrozen", "hw", None, set_version=4),
    ]


def test_a_frozen_file_whose_content_changed_gets_the_unfrozen_label(root):
    """The freeze was for the old content: the label is withdrawn, then the file is sorted."""
    confirmed = drop(root, "a.txt", "Homework 3: implement the kernel functions.\n")
    moved = drop(root, "b.txt", "Homework 4: implement the dual.\n")
    drop(root, "c.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze(root, "HW")
    move_file(root, "b.txt", "Project")
    vault.add_group(root, "Exams", "midterms and finals")  # set version 4
    confirmed.write_text("Syllabus: grading, deadlines and office hours.\n", encoding="utf-8")
    moved.write_text("Syllabus: the same, for the other section.\n", encoding="utf-8")

    backend = FakeBackend(default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert [request.file_id for request in backend.calls] == ["a.txt", "b.txt"]
    assert (report.frozen_skipped, report.asked, report.placed) == (1, 2, 2)
    # What the model said back then, and the set version the user sees now.
    assert labels(root)[4:] == [
        label("a.txt", "unfrozen", "hw", "hw", set_version=4),
        label("b.txt", "unfrozen", "project", "project", set_version=4),
    ]
    for file_id in ("a.txt", "b.txt"):
        note = note_of(root, file_id)
        assert note.path == root / "Logistics" / f"{file_id}.md"
        assert (note.placement.frozen, note.placement.placed_by) == (False, "jev")
        assert note.placement.set_version == 4
    assert note_of(root, "c.txt").placement.frozen is True

    # Sorted again like any unfrozen file from here on, with no label for it.
    sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert len(labels(root)) == 6


def test_a_changed_frozen_file_stays_unfrozen_when_the_backend_fails(root):
    """The freeze comes off before the model is asked. A failure does not put it back."""
    path = drop(root, "a.txt", "Homework 3: implement the kernel functions.\n")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze(root, "a.txt")
    old_hash = note_of(root, "a.txt").placement.content_hash
    path.write_text("Syllabus: grading, deadlines and office hours.\n", encoding="utf-8")

    report = sort_vault(root, FakeBackend(fail={"a.txt"}))
    assert list(report.errors) == ["a.txt"]
    note = note_of(root, "a.txt")
    assert note.path == root / "HW" / "a.txt.md"  # the previous note, in its previous place
    assert (note.placement.frozen, note.placement.content_hash) == (False, old_hash)
    expected = [label("a.txt", "confirmed", "hw", "hw"), label("a.txt", "unfrozen", "hw", "hw")]
    assert labels(root) == expected

    # The next run sorts it as an unfrozen file, and writes no second "unfrozen" line.
    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert (report.asked, report.placed, report.errors) == (1, 1, {})
    assert note_of(root, "a.txt").path == root / "Logistics" / "a.txt.md"
    assert labels(root) == expected


def test_a_frozen_file_with_the_same_content_keeps_its_label_through_a_rerun(root):
    """Nothing changed, so nothing is written: no label, and the note stays as it is."""
    drop(root, "a.txt")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze(root, "a.txt")
    note = note_of(root, "a.txt").path
    before = note.read_bytes()
    vault.add_group(root, "Exams", "midterms and finals")
    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert (report.frozen_skipped, report.asked) == (1, 0)
    assert note.read_bytes() == before
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw")]


def test_a_file_frozen_by_hand_after_its_content_changed_stays_frozen(root):
    """The freeze is for the content the user had in front of them, so the sort leaves it."""
    path = drop(root, "a.txt", "Homework 3: implement the kernel functions.\n")
    sort_vault(root, FakeBackend(default=TO_HW))
    path.write_text("Syllabus: grading, deadlines and office hours.\n", encoding="utf-8")
    freeze_by_hand(root, "a.txt")

    backend = FakeBackend(default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert (report.frozen_skipped, report.asked, report.errors) == (1, 0, {})
    assert backend.calls == []
    note = note_of(root, "a.txt")
    assert (note.folder, note.placement.frozen, note.placement.placed_by) == ("HW", True, "jev")
    assert note.placement.content_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "Syllabus: grading" in note.body  # the card shows what the freeze is about
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw")]

    # A change after the freeze still takes it off.
    path.write_text("Project proposal: due in week 4.\n", encoding="utf-8")
    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert (report.frozen_skipped, report.asked, report.placed) == (0, 1, 1)
    assert labels(root)[1:] == [label("a.txt", "unfrozen", "hw", "hw")]
    assert note_of(root, "a.txt").folder == "Logistics"


# Card notes that Obsidian touched


def test_a_frozen_file_whose_note_gained_a_property_is_still_skipped(root):
    """A tag added in Obsidian does not send a frozen file back to the model."""
    drop(root, "sheet.txt")
    drop(root, "other.txt")
    sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    move_file(root, "sheet.txt", "HW")
    path = note_of(root, "sheet.txt").path
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("\n---\n", "\ntags: [review]\n---\n", 1), encoding="utf-8")
    before = path.read_bytes()

    backend = FakeBackend(default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert [request.file_id for request in backend.calls] == ["other.txt"]
    assert (report.frozen_skipped, report.asked, report.errors) == (1, 1, {})
    assert path.read_bytes() == before
    assert active_labels(root)["sheet.txt"].kind == "moved"
    assert labels(root) == [
        label("sheet.txt", "moved", "logistics", "hw", jev_probs=TO_LOGISTICS[0])
    ]

    # Unfrozen and sorted again, the note keeps the property.
    unfreeze(root, "sheet.txt")
    sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    note = note_of(root, "sheet.txt")
    assert (note.folder, note.placement.placed_by, note.extras) == (
        "Logistics",
        "jev",
        {"tags": ["review"]},
    )


def test_a_frozen_file_whose_note_was_filed_into_a_folder_of_its_group_is_still_skipped(root):
    """HW/week 3 is still HW: the file is not sent again, and no second note is written."""
    drop(root, "sheet.txt")
    sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    move_file(root, "sheet.txt", "HW")
    path = note_of(root, "sheet.txt").path
    (root / "HW" / "week 3").mkdir()
    nested = path.rename(root / "HW" / "week 3" / path.name)

    backend = FakeBackend(default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert backend.calls == []
    assert (report.frozen_skipped, report.errors) == (1, {})
    assert note_of(root, "sheet.txt").path == nested  # one note, where the user filed it
    assert active_labels(root)["sheet.txt"].kind == "moved"


def test_a_labeled_file_without_a_card_note_is_not_sent_to_the_model(root):
    """The note is gone or broken, but the user's label stands: the sort reports the file."""
    drop(root, "deleted.txt")
    drop(root, "broken.txt")
    drop(root, "plain.txt")
    sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    move_file(root, "deleted.txt", "HW")
    freeze(root, "broken.txt")
    note_of(root, "deleted.txt").path.unlink()
    broken = note_of(root, "broken.txt").path
    text = broken.read_text(encoding="utf-8")
    broken.write_text(text.replace("placed_by: jev", "placed_by: [oops"), encoding="utf-8")
    note_of(root, "plain.txt").path.unlink()  # no label: a deleted note just means a new file

    backend = FakeBackend(default=TO_LOGISTICS)
    progress = []
    report = sort_vault(root, backend, lambda done, total: progress.append((done, total)))
    assert [request.file_id for request in backend.calls] == ["plain.txt"]
    assert report.errors == {"broken.txt": NO_NOTE, "deleted.txt": NO_NOTE}
    assert "p0 unfreeze" in NO_NOTE
    assert (report.asked, report.placed, report.frozen_skipped) == (1, 1, 0)
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert [note.placement.file_id for note in vault.read_card_notes(root)] == ["plain.txt"]
    assert sorted(active_labels(root)) == ["broken.txt", "deleted.txt"]
    # Reported every time, never asked, until the user decides.
    again = FakeBackend(default=TO_LOGISTICS)
    assert sort_vault(root, again).errors == report.errors
    assert [request.file_id for request in again.calls] == ["plain.txt"]


def test_unfreezing_a_labeled_file_without_a_card_note_starts_it_from_scratch(root):
    """The way out: the label is withdrawn, and the next sort treats the file as new."""
    drop(root, "sheet.txt")
    sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    move_file(root, "sheet.txt", "HW")
    note_of(root, "sheet.txt").path.unlink()
    assert sort_vault(root, FakeBackend(default=TO_LOGISTICS)).errors == {"sheet.txt": NO_NOTE}

    assert unfreeze(root, "sheet.txt") == ["sheet.txt"]
    assert labels(root)[1:] == [
        {
            "file_id": "sheet.txt",
            "kind": "unfrozen",
            "from_group": "hw",
            "to_group": "hw",
            "jev_probs": None,
            "jev_set_version": None,
            "set_version": 3,
        }
    ]
    assert active_labels(root) == {}
    backend = FakeBackend(default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert [request.file_id for request in backend.calls] == ["sheet.txt"]
    assert (report.asked, report.placed, report.errors) == (1, 1, {})
    note = note_of(root, "sheet.txt")
    assert (note.folder, note.placement.placed_by, note.placement.frozen) == (
        "Logistics",
        "jev",
        False,
    )
    # With the page's exact reading, and for a file that has nothing to withdraw.
    move_file(root, "sheet.txt", "HW")
    note_of(root, "sheet.txt").path.unlink()
    assert unfreeze(root, "sheet.txt", kind="file") == ["sheet.txt"]
    with pytest.raises(ValueError, match="no sorted file with the id 'sheet.txt'"):
        unfreeze(root, "sheet.txt", kind="file")
    with pytest.raises(ValueError, match="no group or sorted file with the id or name"):
        unfreeze(root, "sheet.txt")


# Edits made by hand while a sort is running


class EditingBackend(FakeBackend):
    """A fake that makes an edit by hand when it is asked about one file.

    That is what a person does in Obsidian while a long sort is under way: the sort has
    read the card notes already, and has not reached every file yet.
    """

    def __init__(self, when, edit, **answers):
        """Run `edit()` once, when the request for the file `when` arrives."""
        super().__init__(**answers)
        self.when = when
        self.edit = edit

    def sort(self, request):
        """Make the edit when its moment comes, then answer as FakeBackend would."""
        if request.file_id == self.when and self.edit is not None:
            edit, self.edit = self.edit, None
            edit()
        return super().sort(request)


def three_files_in(root, group_answer):
    """Sort a.txt, b.txt and c.txt into one group, and return c.txt's card note path."""
    for file_id in ("a.txt", "b.txt", "c.txt"):
        drop(root, file_id)
    sort_vault(root, FakeBackend(default=group_answer))
    return note_of(root, "c.txt").path


@pytest.mark.parametrize(
    ("target", "group_id"),
    [("HW", "hw"), ("Project", "project")],
    ids=["a folder before the model's", "a folder after the model's"],
)
def test_a_note_dragged_while_the_sort_runs_is_left_where_the_user_put_it(root, target, group_id):
    """The sort read c.txt's note in Logistics. The user drags it away before its turn."""
    note = three_files_in(root, TO_LOGISTICS)

    def drag():
        """Move c.txt's card note into another group's folder."""
        note.rename(root / target / note.name)

    backend = EditingBackend("b.txt", drag, default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert report.errors == {"c.txt": CHANGED_BY_HAND}
    # The model is not asked about a file whose note the user has just changed.
    assert [request.file_id for request in backend.calls] == ["a.txt", "b.txt"]
    assert (report.asked, report.placed) == (2, 2)
    assert note_of(root, "c.txt").path == root / target / "c.txt.md"  # one note, not two
    assert labels(root) == []

    # The next sync reads the drag as what it is: a move by the user.
    assert sync(root).moved == ["c.txt"]
    placement = note_of(root, "c.txt").placement
    assert (placement.group_id, placement.placed_by, placement.frozen) == (group_id, "you", True)
    assert labels(root) == [
        label("c.txt", "moved", "logistics", group_id, jev_probs=TO_LOGISTICS[0])
    ]
    later = FakeBackend(default=TO_LOGISTICS)
    assert sort_vault(root, later).errors == {}
    assert [request.file_id for request in later.calls] == ["a.txt", "b.txt"]
    assert note_of(root, "c.txt").path == root / target / "c.txt.md"


def test_a_note_frozen_by_hand_while_the_sort_runs_keeps_its_freeze(root):
    """frozen: true is set on c.txt's note mid-run. The sort does not write over it."""
    three_files_in(root, TO_HW)
    backend = EditingBackend("b.txt", lambda: freeze_by_hand(root, "c.txt"), default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert report.errors == {"c.txt": CHANGED_BY_HAND}
    assert [request.file_id for request in backend.calls] == ["a.txt", "b.txt"]
    note = note_of(root, "c.txt")
    assert (note.folder, note.placement.frozen, note.placement.placed_by) == ("HW", True, "jev")

    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert (report.frozen_skipped, report.errors) == (1, {})
    assert labels(root) == [label("c.txt", "confirmed", "hw", "hw")]
    assert note_of(root, "c.txt").folder == "HW"


def test_a_note_changed_while_the_model_is_asked_about_its_file_is_not_written_over(root):
    """The edit lands during the file's own request: the answer is dropped, not the edit."""
    three_files_in(root, TO_HW)
    backend = EditingBackend("c.txt", lambda: freeze_by_hand(root, "c.txt"), default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert report.errors == {"c.txt": CHANGED_BY_HAND}
    assert (report.asked, report.placed) == (3, 2)  # the request for c.txt did go out
    note = note_of(root, "c.txt")
    assert (note.folder, note.placement.frozen) == ("HW", True)
    assert [note.folder for note in vault.read_card_notes(root)] == ["Logistics", "Logistics", "HW"]


def test_the_sorts_own_rewrite_of_a_note_is_not_taken_for_an_edit_by_hand(root):
    """A frozen file whose content changed: the freeze comes off, then the new note is written."""
    path = drop(root, "a.txt", "Homework 3: implement the kernel functions.\n")
    sort_vault(root, FakeBackend(default=TO_HW))
    freeze(root, "a.txt")
    path.write_text("Syllabus: grading, deadlines and office hours.\n", encoding="utf-8")
    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert (report.asked, report.placed, report.errors) == (1, 1, {})
    assert note_of(root, "a.txt").folder == "Logistics"


# A backend that stops the run


def test_a_call_log_that_cannot_be_written_stops_the_run_like_the_spend_cap(root):
    """Every call must be logged, so nothing more is sent, and the files not reached are listed."""
    for file_id in ("a.txt", "b.txt", "c.txt"):
        drop(root, file_id)
    reason = "the call log cannot be written: [Errno 28] No space left on device"
    backend = ScriptedBackend([{}, LogFailed(reason), {}])
    progress = []
    report = sort_vault(root, backend, lambda done, total: progress.append((done, total)))
    assert report.errors == {"b.txt": reason, "c.txt": STOPPED_BEFORE}
    assert "spend cap" not in STOPPED_BEFORE
    assert [request.file_id for request in backend.calls] == ["a.txt", "b.txt"]
    assert [note.placement.file_id for note in vault.read_card_notes(root)] == ["a.txt"]
    assert progress == [(1, 3)]
