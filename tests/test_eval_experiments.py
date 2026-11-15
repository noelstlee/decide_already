"""eval/experiments.py: the extra passes, and the numbers of the three experiments.

The first half checks the pure functions on cases worked out by hand. The second half runs the
passes through the runner with a stand-in for Jev, so nothing is asked of a real model.
"""

import hashlib
import json
import random

import pytest

from decide_already import vault
from decide_already.backends.base import SortResult
from decide_already.eval import run
from decide_already.eval.dataset import GoldLabel
from decide_already.eval.experiments import (
    EDIT_BASE,
    HELD_OUT_BASES,
    SET_VERSION,
    Edit,
    auroc,
    edit_effect,
    edit_name,
    group_added,
    held_out_name,
    naming_held_out,
    passes,
    read_edits,
    reduced_groups,
    with_edit,
    without_group,
)
from decide_already.eval.groups import EVAL_GROUPS, HELD_OUT_EXCLUDES, HELD_OUT_GROUP_ID
from decide_already.eval.metrics import Cutoffs, Prediction
from decide_already.questions import build_questions
from decide_already.types import Group
from fakes import FakeBackend

DEFAULTS = Cutoffs(fits_any_below=0.5, top_below=0.6, close_margin=0.15)

GROUPS = [
    Group(id="exams", name="Exams", covers="quizzes and exams", excludes="homework"),
    Group(id="homework", name="Homework", covers="problem sets"),
    Group(id="lecture-notes", name="Lecture notes", covers="lecture notes"),
]


def pred(file_id, gold, probs, fits_any=0.9):
    """Build a Prediction with less typing."""
    return Prediction(file_id=file_id, gold=gold, probs=probs, fits_any=fits_any)


def two(homework, fits_any=0.9):
    """An answer over the two groups left when Exams is held out."""
    return {"homework": homework, "lecture-notes": 1 - homework}, fits_any


# The passes


def test_the_held_out_group_is_exams():
    """The numbers below are worked out for this group."""
    assert HELD_OUT_GROUP_ID == "exams"
    assert SET_VERSION == 0


def test_pass_names_are_built_from_the_configuration_they_copy():
    """A pass is named after its base configuration, then what was changed."""
    edit = Edit(name="wider", group_id="exams", covers="anything graded")
    assert held_out_name("jev:card") == "jev:card@without-exams"
    assert held_out_name("jev:names_only") == "jev:names_only@without-exams"
    assert edit_name(edit) == "jev:card@edit-wider"


def test_without_group_drops_one_group():
    """The other groups stay as they are, in order."""
    assert without_group(GROUPS, "exams") == GROUPS[1:]
    assert without_group(GROUPS, "nope") == GROUPS


def test_with_edit_replaces_the_description_of_one_group():
    """covers is replaced. excludes stays unless the edit names it; an empty one removes it."""
    covers_only = Edit(name="a", group_id="exams", covers="  anything graded ")
    assert with_edit(GROUPS, covers_only) == [
        Group(id="exams", name="Exams", covers="anything graded", excludes="homework"),
        *GROUPS[1:],
    ]
    both = Edit(name="b", group_id="exams", covers="anything graded", excludes="ungraded work")
    assert with_edit(GROUPS, both)[0].excludes == "ungraded work"
    removed = Edit(name="c", group_id="exams", covers="anything graded", excludes=" ")
    assert with_edit(GROUPS, removed)[0].excludes is None
    assert GROUPS[0].covers == "quizzes and exams"  # the vault's own groups are not touched


def test_with_edit_rejects_a_group_the_vault_does_not_have():
    """An edit has to name one of the groups."""
    edit = Edit(name="a", group_id="projects", covers="course projects")
    with pytest.raises(ValueError, match="edit 'a' is for group 'projects'"):
        with_edit(GROUPS, edit)


def test_the_reduced_set_drops_the_group_and_every_mention_of_it():
    """Exams goes, and what the others exclude is worded as before anyone thought of Exams."""
    reduced = reduced_groups(EVAL_GROUPS)
    assert {group.id: group.excludes for group in reduced} == {
        "lecture-notes": "homework",
        "homework": "course projects",
        "projects": "homework problem sets",
        "logistics": None,
    }
    assert HELD_OUT_EXCLUDES == {"lecture-notes": "homework", "homework": "course projects"}
    # Nothing else about a group changes, and the vault's own groups are not touched.
    by_id = {group.id: group for group in EVAL_GROUPS}
    for group in reduced:
        assert (group.name, group.covers) == (by_id[group.id].name, by_id[group.id].covers)
    assert by_id["homework"].excludes == "exams and course projects"
    assert by_id["lecture-notes"].excludes == "homework and exams"


def test_no_question_of_a_held_out_pass_names_the_held_out_group():
    """A description that says "exams" would tell the model those files are excluded."""
    for one in passes(EVAL_GROUPS, None):
        with_descriptions = one.base != "jev:names_only"
        questions, order = build_questions(one.groups, random.Random(0), with_descriptions)
        text = json.dumps(questions).lower().replace("example", "")
        assert "exam" not in text
        assert "exams" not in order
    assert naming_held_out(EVAL_GROUPS) == []
    # Dropping the group alone is not enough: two descriptions would still name it.
    left = without_group(EVAL_GROUPS, HELD_OUT_GROUP_ID)
    assert sum("exams" in (group.excludes or "") for group in left) == 2


def test_a_description_that_still_names_the_held_out_group_is_found():
    """Whole words only, with or without the final s, in what a group covers or excludes."""
    groups = [
        Group(id="exams", name="Exams", covers="quizzes and exams"),
        # Homework is listed in HELD_OUT_EXCLUDES, so its own excludes text is replaced.
        Group(id="homework", name="Homework", covers="problem sets", excludes="exams"),
        Group(id="projects", name="Projects", covers="project work", excludes="any exam"),
        Group(id="readings", name="Readings", covers="papers, for example surveys"),
        Group(id="review", name="Review", covers="sheets for the Exams"),
    ]
    assert naming_held_out(groups) == ["Projects", "Review"]
    assert naming_held_out(groups[1:]) == []  # the set has no held-out group to name


def test_passes_without_edits_are_the_two_held_out_passes():
    """The card and the names-only configuration, each under the reduced set."""
    extra = passes(GROUPS, None)
    assert [(one.name, one.base) for one in extra] == [
        ("jev:card@without-exams", "jev:card"),
        ("jev:names_only@without-exams", "jev:names_only"),
    ]
    reduced = [
        Group(id="homework", name="Homework", covers="problem sets", excludes="course projects"),
        Group(
            id="lecture-notes", name="Lecture notes", covers="lecture notes", excludes="homework"
        ),
    ]
    assert all(one.groups == reduced == reduced_groups(GROUPS) for one in extra)
    assert passes(GROUPS, []) == extra
    assert HELD_OUT_BASES == ("jev:card", "jev:names_only")


def test_both_held_out_passes_get_a_new_fingerprint():
    """The cache follows the groups, so the reworded set is asked again, names only as well."""
    card_pass, names_pass = passes(EVAL_GROUPS, None)
    assert run.groups_hash(card_pass.groups) == run.groups_hash(names_pass.groups)
    dropped_only = without_group(EVAL_GROUPS, HELD_OUT_GROUP_ID)
    assert run.groups_hash(card_pass.groups) != run.groups_hash(dropped_only)


def test_passes_add_one_card_pass_per_edit():
    """Each edit sorts the card configuration again under the edited groups."""
    edit = Edit(name="wider", group_id="homework", covers="problem sets and labs")
    extra = passes(GROUPS, [edit])
    assert [one.name for one in extra][2:] == ["jev:card@edit-wider"]
    assert extra[2].base == EDIT_BASE == "jev:card"
    assert [group.covers for group in extra[2].groups] == [
        "quizzes and exams",
        "problem sets and labs",
        "lecture notes",
    ]


def test_no_held_out_pass_when_the_vault_lacks_the_group_or_has_no_other():
    """Without Exams there is nothing to hold out; with only Exams nothing would be left."""
    assert passes(GROUPS[1:], None) == []
    assert passes(GROUPS[:1], None) == []


# The edits file


def test_no_edits_file_gives_none(tmp_path):
    """None says the experiment was not set up, which is different from an empty list."""
    assert read_edits(tmp_path) is None
    (tmp_path / "description_edits.yaml").write_text("", encoding="utf-8")
    assert read_edits(tmp_path) == []


def test_edits_are_read_from_yaml(tmp_path):
    """Each entry has a name, a group id and covers; excludes, targets and a note are optional."""
    (tmp_path / "description_edits.yaml").write_text(
        "- name: wider\n"
        "  group_id: exams\n"
        "  covers: anything graded\n"
        "- name: narrow-2\n"
        "  group_id: homework\n"
        "  covers: problem sets only\n"
        "  excludes: labs\n"
        "  target_file_ids: [labs/lab1.py, hw9.txt]\n"
        "  note: Written by a TA before the first run, to keep the labs out.\n",
        encoding="utf-8",
    )
    assert read_edits(tmp_path) == [
        Edit(name="wider", group_id="exams", covers="anything graded"),
        Edit(
            name="narrow-2",
            group_id="homework",
            covers="problem sets only",
            excludes="labs",
            target_file_ids=["labs/lab1.py", "hw9.txt"],
            note="Written by a TA before the first run, to keep the labs out.",
        ),
    ]
    assert read_edits(tmp_path)[0].note is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("name: wider\ngroup_id: exams\ncovers: x\n", "must hold a list of edits"),
        ("- name: Wider One\n  group_id: exams\n  covers: x\n", "cannot name an edit"),
        ("- name: ../up\n  group_id: exams\n  covers: x\n", "cannot name an edit"),
        ("- name: wider\n  group_id: exams\n  covers: ' '\n", "needs a covers description"),
        ("- name: wider\n  group_id: exams\n", "cannot be read"),
        ("- name: wider\n  group_id: exams\n  covers: x\n  extra: 1\n", "cannot be read"),
        ("- name: [unclosed\n", "is not valid YAML"),
        (
            (
                "- name: wider\n  group_id: exams\n  covers: x\n"
                "- name: wider\n  group_id: homework\n  covers: y\n"
            ),
            "has two edits named wider",
        ),
    ],
)
def test_a_bad_edits_file_is_a_value_error_that_names_the_file(tmp_path, text, message):
    """Every problem with the file is reported as a ValueError with the file's path in it."""
    (tmp_path / "description_edits.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message) as error:
        read_edits(tmp_path)
    assert "description_edits.yaml" in str(error.value)


# AUROC


def test_auroc_on_a_hand_computed_case():
    """Six pairs: four wins, one tie (0.4 against 0.4), one loss. (4 + 0.5) / 6 = 0.75."""
    assert auroc([0.9, 0.8, 0.4], [0.5, 0.4]) == 0.75


def test_auroc_at_its_ends_and_in_the_middle():
    """1.0 when every positive is above every negative, 0.0 the other way, 0.5 for all ties."""
    assert auroc([0.8, 0.9], [0.1, 0.7]) == 1.0
    assert auroc([0.1, 0.2], [0.3, 0.9]) == 0.0
    assert auroc([0.5, 0.5], [0.5]) == 0.5


def test_auroc_needs_both_kinds():
    """With no positives or no negatives there is nothing to compare."""
    assert auroc([], [0.5]) is None
    assert auroc([0.5], []) is None


# Group added mid-run: nine files sorted without Exams, then the rerun against the full set

REDUCED = [
    pred("e1", "exams", *two(0.7, fits_any=0.2)),  # Unsorted: no group fits
    pred("e2", "exams", *two(0.8, fits_any=0.7)),  # placed in Homework
    pred("e3", "exams", *two(0.55, fits_any=0.6)),  # Unsorted: weak top
    pred("h1", "homework", *two(0.9)),  # placed right, so frozen
    pred("h2", "homework", *two(0.2, fits_any=0.8)),  # placed in Lecture notes
    pred("h3", "homework", *two(0.58)),  # right top, but too weak: Unsorted
    pred("l1", "lecture-notes", *two(0.05, fits_any=0.6)),  # placed right, so frozen
    pred("l2", "lecture-notes", *two(0.3, fits_any=0.4)),  # right top, no group fits: Unsorted
    pred("x1", None, *two(0.9)),  # no gold group, and placed
]
FULL = [
    pred("e1", "exams", {"exams": 0.9, "homework": 0.05, "lecture-notes": 0.05}),  # right
    pred("e2", "exams", {"exams": 0.5, "homework": 0.5, "lecture-notes": 0.0}),  # weak: Unsorted
    pred("e3", "exams", {"exams": 0.3, "homework": 0.7, "lecture-notes": 0.0}),  # wrong group
    pred("h1", "homework", {"exams": 0.9, "homework": 0.1, "lecture-notes": 0.0}),  # frozen
    pred("h2", "homework", {"exams": 0.1, "homework": 0.8, "lecture-notes": 0.1}),  # right
    pred("h3", "homework", {"exams": 0.0, "homework": 0.9, "lecture-notes": 0.1}, 0.3),  # Unsorted
    pred("l1", "lecture-notes", {"exams": 0.9, "homework": 0.1, "lecture-notes": 0.0}),  # frozen
    # l2 has no answer under the full set.
    pred("x1", None, {"exams": 0.9, "homework": 0.1, "lecture-notes": 0.0}, 0.1),  # Unsorted: right
]


def test_group_added_numbers_on_a_constructed_case():
    """Each number checked against the nine files above, under the default cutoffs."""
    numbers = group_added(REDUCED, FULL, DEFAULTS)
    assert numbers == {
        "n": 9,
        # e1 and e3 go to Unsorted, e2 is placed.
        "held_out_n": 3,
        "held_out_unsorted_share": pytest.approx(2 / 3),
        "held_out_mean_fits_any": pytest.approx((0.2 + 0.7 + 0.6) / 3),
        # h2 has the wrong top group; h3 and l2 are right on top but sent to Unsorted.
        "others_n": 5,
        "others_top1_accuracy": pytest.approx(4 / 5),
        "others_unsorted_share": pytest.approx(2 / 5),
        # fits_any of the others: 0.9, 0.8, 0.9, 0.6, 0.4. Of the held-out: 0.2, 0.7, 0.6.
        # Wins: 3 + 3 + 3 + 1.5 (a tie at 0.6) + 1 = 11.5 of 15 pairs.
        "fits_any_auroc": pytest.approx(11.5 / 15),
        # h1 and l1 sit in their gold group. l2 is rerun too, but has no full-set answer.
        "frozen_n": 2,
        "rerun_n": 6,
        # Right after the rerun: e1, h2, and x1 (Unsorted, with no gold group).
        "rerun_accuracy": pytest.approx(3 / 6),
    }


def test_rerun_accuracy_ignores_the_frozen_files():
    """A frozen file is not rerun, so a wrong full-set answer for it changes nothing."""
    # h1 and l1 are both wrong in FULL (Exams on top), and both are frozen.
    without_them = [p for p in FULL if p.file_id not in ("h1", "l1")]
    assert group_added(REDUCED, without_them, DEFAULTS) == group_added(REDUCED, FULL, DEFAULTS)


def test_group_added_follows_the_cutoffs():
    """With no cutoffs at all nothing is Unsorted, so no held-out file is caught."""
    wide_open = Cutoffs(fits_any_below=0.0, top_below=0.0, close_margin=0.15)
    numbers = group_added(REDUCED, FULL, wide_open)
    assert numbers["held_out_unsorted_share"] == 0.0
    assert numbers["others_unsorted_share"] == 0.0
    assert numbers["frozen_n"] == 4  # h1, h3, l1 and l2 are placed in their gold group
    assert numbers["others_top1_accuracy"] == pytest.approx(4 / 5)  # top-1 ignores the cutoffs


def test_group_added_without_a_full_pass_has_no_rerun():
    """The reduced pass still gives its numbers; the rerun has nothing to read."""
    numbers = group_added(REDUCED, [], DEFAULTS)
    assert (numbers["frozen_n"], numbers["rerun_n"], numbers["rerun_accuracy"]) == (2, 0, None)
    assert numbers["held_out_n"] == 3


def test_group_added_with_no_files_is_all_none():
    """Shares of nothing are None, never a guess."""
    assert group_added([], [], DEFAULTS) == {
        "n": 0,
        "held_out_n": 0,
        "held_out_unsorted_share": None,
        "held_out_mean_fits_any": None,
        "others_n": 0,
        "others_top1_accuracy": None,
        "others_unsorted_share": None,
        "fits_any_auroc": None,
        "frozen_n": 0,
        "rerun_n": 0,
        "rerun_accuracy": None,
    }


# Description edits


def top(group_id):
    """Probabilities with one group on top."""
    return {group_id: 0.8, "other": 0.2}


def test_edit_effect_on_a_constructed_case():
    """Targets and other files are counted apart; only changed answers are listed."""
    base = [
        pred("t1", "logistics", top("projects")),  # target: wrong, then right
        pred("t2", "logistics", top("projects")),  # target: wrong, then wrong elsewhere
        pred("t3", "logistics", top("logistics")),  # target: right both times
        pred("a", "homework", top("exams")),  # wrong, then right
        pred("b", "homework", top("homework")),  # right, then wrong
        pred("c", "exams", top("homework")),  # wrong, then wrong in another group
        pred("d", "exams", top("homework")),  # wrong in the same group both times
        pred("e", "exams", top("exams")),  # right both times
        pred("only-before", "exams", top("exams")),
    ]
    edited = [
        pred("t1", "logistics", top("logistics")),
        pred("t2", "logistics", top("homework")),
        pred("t3", "logistics", top("logistics")),
        pred("a", "homework", top("homework")),
        pred("b", "homework", top("logistics")),
        pred("c", "exams", top("projects")),
        pred("d", "exams", top("homework")),
        pred("e", "exams", top("exams")),
        pred("only-after", "exams", top("exams")),
    ]
    assert edit_effect(base, edited, ["t1", "t2", "t3", "gone"]) == {
        "n": 8,
        "top1_before": 3 / 8,  # t3, b, e
        "top1_after": 4 / 8,  # t1, t3, a, e
        "targets_n": 4,
        "targets_missing": ["gone"],
        "targets_fixed": ["t1"],
        "wrong_to_right": ["a"],
        "right_to_wrong": ["b"],
        "wrong_to_other_wrong": ["c"],
    }


def test_edit_effect_looks_at_the_top_group_and_not_at_the_cutoffs():
    """A weak answer still counts by its top group, as in the headline accuracy."""
    base = [pred("a", "homework", {"exams": 0.4, "homework": 0.35, "other": 0.25}, fits_any=0.1)]
    edited = [pred("a", "homework", {"exams": 0.35, "homework": 0.4, "other": 0.25}, fits_any=0.1)]
    effect = edit_effect(base, edited, [])
    assert effect["wrong_to_right"] == ["a"]
    assert (effect["top1_before"], effect["top1_after"]) == (0.0, 1.0)


def test_edit_effect_with_nothing_to_compare():
    """No file answered both times: no shares, and every target is missing."""
    assert edit_effect([], [pred("a", "exams", top("exams"))], ["a"]) == {
        "n": 0,
        "top1_before": None,
        "top1_after": None,
        "targets_n": 1,
        "targets_missing": ["a"],
        "targets_fixed": [],
        "wrong_to_right": [],
        "right_to_wrong": [],
        "wrong_to_other_wrong": [],
    }


# Through the runner, with a stand-in for Jev

# (file id, content, gold group, split). The zip is all NUL bytes, so it has no text.
FILES = [
    ("exam1.txt", "Midterm exam. Closed book.\n", "exams", "test"),
    ("exam2.txt", "Final exam with worked solutions.\n", "exams", "test"),
    ("hw1.txt", "Problem set 1.\n", "homework", "test"),
    ("hw2.txt", "Problem set 2, with reading notes.\n", "homework", "test"),
    ("lec1.md", "# Lecture 1\n\nLinear regression.\n", "lecture-notes", "test"),
    ("hw0.txt", "Problem set 0.\n", "homework", "dev"),
    ("scan.zip", b"PK\x03\x04" + b"\x00" * 60, "exams", "test"),
]
TEXT_IDS = sorted(file_id for file_id, *_ in FILES if file_id != "scan.zip")
EDITED_COVERS = "quizzes, exams and every solution set"


def three(top_id, prob=0.9, fits_any=0.9):
    """An answer over the three groups with `prob` on one of them."""
    rest = (1 - prob) / 2
    ids = ("exams", "homework", "lecture-notes")
    return {group_id: prob if group_id == top_id else rest for group_id in ids}, fits_any


# Under the full set: exam2 and hw2 are placed in the wrong group, the rest are right.
FULL_TABLE = {
    "exam1.txt": three("exams"),
    "exam2.txt": three("homework", 0.7),
    "hw1.txt": three("homework"),
    "hw2.txt": three("lecture-notes", 0.7),
    "lec1.md": three("lecture-notes"),
    "hw0.txt": three("homework"),
}
# Without Exams, asked with descriptions.
REDUCED_TABLE = {
    "exam1.txt": two(0.6, fits_any=0.1),  # Unsorted, as it should be
    "exam2.txt": two(0.9, fits_any=0.8),  # placed in Homework
    "hw1.txt": two(0.9),  # frozen
    "hw2.txt": two(0.2, fits_any=0.5),  # placed in Lecture notes
    "lec1.md": two(0.1),  # frozen
    "hw0.txt": two(0.9),  # frozen
}
# Without Exams, asked with group names only.
NAMES_TABLE = {
    "exam1.txt": two(0.6, fits_any=0.7),  # placed in Homework
    "exam2.txt": two(0.9),  # placed in Homework
    "hw1.txt": two(0.9),  # frozen
    "hw2.txt": two(0.9),  # frozen
    "lec1.md": two(0.1),  # frozen
    "hw0.txt": two(0.55),  # right top, but too weak: Unsorted
}
# Under the edited description of Exams: exam2 is fixed, hw1 breaks, hw2 moves to Exams.
EDITED_TABLE = {
    **FULL_TABLE,
    "exam2.txt": three("exams", 0.8),
    "hw1.txt": three("exams", 0.6),
    "hw2.txt": three("exams", 0.6),
}


class TableBackend(FakeBackend):
    """A stand-in for Jev that answers from one table per kind of request."""

    def sort(self, request):
        """Record the request and answer from the table that fits its groups."""
        self.calls.append(request)
        covers = {group.id: group.covers for group in request.groups}
        if "exams" not in covers:
            table = REDUCED_TABLE if request.with_descriptions else NAMES_TABLE
        elif covers["exams"] == EDITED_COVERS:
            table = EDITED_TABLE
        else:
            table = FULL_TABLE
        probs, fits_any = table[request.file_id]
        return SortResult(
            file_id=request.file_id,
            probs=probs,
            fits_any=fits_any,
            model="fake-0",
            option_order=list(covers),
        )


def add_file(root, file_id, content, gold, split="test"):
    """Put a file in the dump folder and add its gold label to gold.jsonl."""
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    path = root / "_dump" / file_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    label = GoldLabel(
        file_id=file_id,
        gold_group=gold,
        source_url=f"https://example.org/{file_id}",
        source_type="made up",
        license="CC0",
        sha256=hashlib.sha256(data).hexdigest(),
        split=split,
    )
    with (root / "gold.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label.model_dump()) + "\n")


@pytest.fixture
def root(tmp_path):
    """A small evaluation vault with Exams, Homework and Lecture notes, and the files above."""
    root = tmp_path / "eval"
    vault.init_vault(root)
    vault.add_group(root, "Exams", "quizzes, midterm and final exams")
    vault.add_group(root, "Homework", "problem sets and their solutions")
    vault.add_group(root, "Lecture notes", "lecture notes and slides")
    for file_id, content, gold, split in FILES:
        add_file(root, file_id, content, gold, split)
    return root


@pytest.fixture
def jev(monkeypatch):
    """Stand in for Jev, and note every backend the runner builds. Others get a plain fake."""
    backend = TableBackend()
    backend.built = []

    def build(key, root, config):
        """Stand in for run.build_backend."""
        backend.built.append(key)
        return backend if key == "jev" else FakeBackend()

    monkeypatch.setattr(run, "build_backend", build)
    return backend


def write_edits(root, covers=EDITED_COVERS, group_id="exams"):
    """Write a description_edits.yaml with one edit that is meant to fix exam2.txt."""
    (root / "description_edits.yaml").write_text(
        f"- name: solutions\n"
        f"  group_id: {group_id}\n"
        f"  covers: {covers}\n"
        f"  target_file_ids: [exam2.txt]\n",
        encoding="utf-8",
    )


def evaluate(root, *args):
    """Run the command line on a vault and return its exit status."""
    return run.main(["--root", str(root), *args])


def experiments_of(root):
    """The experiments section of the metrics.json a run wrote."""
    metrics = json.loads((root / "results" / "metrics.json").read_text(encoding="utf-8"))
    return metrics["experiments"]


def report_of(root):
    """The report.md a run wrote."""
    return (root / "results" / "report.md").read_text(encoding="utf-8")


def calls_of(backend, kind):
    """The requests of one kind: "full", "reduced" (without Exams), or "edited"."""
    picked = []
    for call in backend.calls:
        covers = {group.id: group.covers for group in call.groups}
        if "exams" not in covers:
            found = "reduced"
        else:
            found = "edited" if covers["exams"] == EDITED_COVERS else "full"
        if found == kind:
            picked.append(call)
    return picked


def test_the_held_out_passes_go_through_the_same_backend_and_cache(root, jev):
    """Both passes ask the one Jev backend, without Exams, logged under set version 0."""
    assert evaluate(root, "--configs", "jev:card,jev:names_only") == 0
    assert jev.built == ["jev"]
    reduced = calls_of(jev, "reduced")
    assert len(reduced) == 2 * len(TEXT_IDS)
    assert {call.set_version for call in reduced} == {0}
    assert {tuple(group.id for group in call.groups) for call in reduced} == {
        ("homework", "lecture-notes")
    }
    assert sorted(call.file_id for call in reduced if call.variant == "card") == TEXT_IDS
    assert sorted(call.file_id for call in reduced if call.variant == "names_only") == TEXT_IDS
    assert all(call.with_descriptions is (call.variant == "card") for call in reduced)
    # What the remaining groups exclude is the reduced set's wording, not the vault's.
    for call in reduced:
        assert {group.id: group.excludes for group in call.groups} == {
            "homework": "course projects",
            "lecture-notes": "homework",
        }
    # The vault's own set is not changed by leaving a group out.
    assert vault.latest_set(root).version == 3
    assert {call.set_version for call in calls_of(jev, "full")} == {3}

    for name in ("jev:card@without-exams", "jev:names_only@without-exams"):
        rows = run.read_rows(root, name)
        assert sorted(rows) == sorted(file_id for file_id, *_ in FILES)
        assert {row.set_version for row in rows.values()} == {0}
        assert rows["scan.zip"].no_text is True
        assert rows["scan.zip"].fits_any is None
    assert (
        run.read_rows(root, "jev:card@without-exams")["exam1.txt"].probs
        == (REDUCED_TABLE["exam1.txt"][0])
    )
    assert run.cache_path(root, "jev:card@without-exams").name == "jev_card@without-exams.jsonl"


def test_the_group_added_numbers_of_the_constructed_run(root, jev):
    """Worked out from the tables above, over the six files with text, default cutoffs."""
    evaluate(root, "--configs", "jev:card,jev:names_only")
    tests = experiments_of(root)
    assert tests["held_out_group"] == "exams"
    assert tests["has_held_out_group"] is True
    assert tests["held_out_named_in"] == []
    assert tests["group_added"]["jev:card"] == {
        "n": 6,
        "held_out_n": 2,
        "held_out_unsorted_share": 0.5,  # exam1 is caught, exam2 is placed
        "held_out_mean_fits_any": pytest.approx(0.45),
        "others_n": 4,
        "others_top1_accuracy": 0.75,  # hw2 has Lecture notes on top
        "others_unsorted_share": 0.0,
        # fits_any of the others: 0.9, 0.5, 0.9, 0.9. Of the held-out: 0.1, 0.8. 7 of 8 pairs.
        "fits_any_auroc": 0.875,
        "frozen_n": 3,  # hw1, lec1, hw0
        "rerun_n": 3,  # exam1, exam2, hw2
        "rerun_accuracy": pytest.approx(1 / 3),  # only exam1 is placed right by jev:card
    }
    assert tests["group_added"]["jev:names_only"] == {
        "n": 6,
        "held_out_n": 2,
        "held_out_unsorted_share": 0.0,  # with names only, both exams are placed
        "held_out_mean_fits_any": pytest.approx(0.8),
        "others_n": 4,
        "others_top1_accuracy": 1.0,
        "others_unsorted_share": 0.25,  # hw0
        # Every other file has 0.9. Against 0.7 that wins, against 0.9 it ties: 6 of 8.
        "fits_any_auroc": 0.75,
        "frozen_n": 3,  # hw1, hw2, lec1
        "rerun_n": 3,  # exam1, exam2, hw0
        "rerun_accuracy": pytest.approx(2 / 3),  # exam1 and hw0 under jev:names_only
    }
    report = report_of(root)
    assert "### Group added mid-run (held out: Exams)" in report
    assert "| Measure | jev:card | jev:names_only |" in report
    assert "| Held-out files sent to Unsorted | 50.0% | 0.0% |" in report
    assert "| Mean fits_any of the held-out files | 0.450 | 0.800 |" in report
    assert "| AUROC of fits_any | 0.875 | 0.750 |" in report
    assert "| Rerun accuracy | 33.3% | 66.7% |" in report


def test_the_group_added_section_says_what_its_numbers_can_carry(root, jev):
    """No description names the group, the AUROC compares the columns, the cutoff is untuned."""
    evaluate(root, "--configs", "jev:card,jev:names_only")
    report = report_of(root)
    assert "The reduced set's descriptions do not name the held-out group" in report
    assert "The AUROC of fits_any is the comparison between the columns" in report
    # The shares that depend on a cutoff sit in their own table, under the line about tuning.
    assert (
        "The shares below are taken at the default cutoffs (fits_any under 0.50, top group "
        "under 0.60), the same for every column. No file in this data set fits no group (none "
        "of the 7 has that as its gold label), so the fits_any cutoff could not be tuned on it."
    ) in report
    assert "| At the default cutoffs | jev:card | jev:names_only |" in report
    assert "| Other groups' files wrongly sent to Unsorted | 0.0% | 25.0% |" in report
    cut = report.index("| At the default cutoffs |")
    assert report.index("| AUROC of fits_any |") < cut < report.index("| Held-out files sent to")
    assert "The rerun asks nothing new" in report


def test_a_file_that_fits_no_group_is_counted_next_to_the_cutoff_shares(root, jev, monkeypatch):
    """With such a file in the data set, the report says how little the cutoff had to go on."""
    add_file(root, "menu.txt", "Lunch menu for the week.\n", None)
    for table in (FULL_TABLE, REDUCED_TABLE, NAMES_TABLE):
        monkeypatch.setitem(table, "menu.txt", (table["hw1.txt"][0], 0.2))
    evaluate(root, "--configs", "jev:card,jev:names_only")
    report = report_of(root)
    assert (
        "Files that fit no group in this data set: 1 of 8, so the fits_any cutoff had next to "
        "nothing to be tuned on."
    ) in report
    assert "could not be tuned on it" not in report


def test_the_rerun_costs_no_extra_request(root, jev):
    """The rerun reads the cached full-set answers: one request per file and configuration."""
    evaluate(root, "--configs", "jev:card,jev:names_only")
    assert len(jev.calls) == 4 * len(TEXT_IDS)  # two configurations and their two passes


def test_a_second_run_asks_nothing_for_the_passes(root, jev):
    """The passes are cached like every other configuration."""
    write_edits(root)
    evaluate(root, "--configs", "jev:card,jev:names_only")
    asked = len(jev.calls)
    assert asked == 5 * len(TEXT_IDS)
    assert evaluate(root, "--configs", "jev:card,jev:names_only") == 0
    assert len(jev.calls) == asked


def test_no_experiments_skips_every_pass(root, jev):
    """With --no-experiments only the configurations themselves are asked."""
    write_edits(root)
    assert evaluate(root, "--configs", "jev:card", "--no-experiments") == 0
    assert len(jev.calls) == len(TEXT_IDS)
    assert not run.cache_path(root, "jev:card@without-exams").exists()
    tests = experiments_of(root)
    assert tests["group_added"] == {"jev:card": None, "jev:names_only": None}
    assert tests["description_edits"][0]["effect"] is None
    assert "Not run: no answers are cached for the passes without the group." in report_of(root)


def test_a_pass_runs_only_with_the_configuration_it_copies(root, jev):
    """jev:card alone brings its own pass, not the names-only one. No Jev: no pass at all."""
    assert evaluate(root, "--configs", "jev:card") == 0
    assert run.cache_path(root, "jev:card@without-exams").exists()
    assert not run.cache_path(root, "jev:names_only@without-exams").exists()
    tests = experiments_of(root)
    assert tests["group_added"]["jev:card"]["n"] == 6
    assert tests["group_added"]["jev:names_only"] is None
    assert "| Measure | jev:card |\n" in report_of(root)

    jev.calls.clear()
    jev.built.clear()
    assert evaluate(root, "--configs", "rules:card") == 0
    assert jev.built == ["rules"]
    assert jev.calls == []


def test_cached_pass_answers_are_reported_even_when_the_pass_is_not_run_again(root, jev):
    """The report is made from the cache, so an earlier pass still shows up."""
    evaluate(root, "--configs", "jev:card")
    assert evaluate(root, "--configs", "jev:card", "--no-experiments") == 0
    assert experiments_of(root)["group_added"]["jev:card"]["n"] == 6


def test_without_an_edits_file_the_report_says_the_experiment_was_not_run(root, jev):
    """No description_edits.yaml: no pass, and one plain line in the report."""
    evaluate(root, "--configs", "jev:card")
    assert experiments_of(root)["description_edits"] is None
    assert calls_of(jev, "edited") == []
    assert "Not run: the vault has no description_edits.yaml." in report_of(root)


def test_an_edit_sorts_every_file_again_under_the_edited_groups(root, jev):
    """The pass asks jev:card with the new description, under set version 0."""
    write_edits(root)
    assert evaluate(root, "--configs", "jev:card") == 0
    edited = calls_of(jev, "edited")
    assert sorted(call.file_id for call in edited) == TEXT_IDS
    assert {call.set_version for call in edited} == {0}
    assert {call.variant for call in edited} == {"card"}
    assert {tuple(group.id for group in call.groups) for call in edited} == {
        ("exams", "homework", "lecture-notes")
    }
    # The vault's own group note and set are left alone.
    assert vault.read_groups(root)[0].covers == "quizzes, midterm and final exams"
    assert vault.latest_set(root).version == 3
    assert sorted(run.read_rows(root, "jev:card@edit-solutions")) == sorted(
        file_id for file_id, *_ in FILES
    )


def test_the_edit_comparison_of_the_constructed_run(root, jev):
    """exam2 is fixed, hw1 goes from right to wrong, hw2 moves between wrong groups."""
    write_edits(root)
    evaluate(root, "--configs", "jev:card")
    (edit,) = experiments_of(root)["description_edits"]
    assert edit == {
        "name": "solutions",
        "group_id": "exams",
        "covers": EDITED_COVERS,
        "excludes": None,
        "target_file_ids": ["exam2.txt"],
        "note": None,
        "effect": {
            "n": 6,
            "top1_before": pytest.approx(4 / 6),  # all but exam2 and hw2
            "top1_after": pytest.approx(4 / 6),  # all but hw1 and hw2
            "targets_n": 1,
            "targets_missing": [],
            "targets_fixed": ["exam2.txt"],
            "wrong_to_right": [],
            "right_to_wrong": ["hw1.txt"],
            "wrong_to_other_wrong": ["hw2.txt"],
        },
    }
    assert experiments_of(root)["edits_base"] == {"n": 6, "wrong": 2}
    report = report_of(root)
    assert "### Description edits" in report
    assert "Targets are the files the edit was meant to fix." in report
    assert "| solutions | exams | 6 | 1 | 1 | 0 | 1 | 1 | 66.7% | 66.7% |" in report
    assert "No note says who wrote these edits, when or why: solutions." in report


def write_untargeted_edits(root):
    """Write two edits that name no target; only the first says where it comes from."""
    (root / "description_edits.yaml").write_text(
        f"- name: solutions\n"
        f"  group_id: exams\n"
        f"  covers: {EDITED_COVERS}\n"
        f"  note: Written after the first full run, to see how far one edit reaches.\n"
        f"- name: plain\n"
        f"  group_id: homework\n"
        f"  covers: problem sets and their solutions\n",
        encoding="utf-8",
    )


def test_edits_with_no_target_are_not_shown_as_the_specs_experiment(root, jev, monkeypatch):
    """The base run had nothing to fix, so the section says what the edits measure instead."""
    monkeypatch.setitem(FULL_TABLE, "exam2.txt", three("exams"))
    monkeypatch.setitem(FULL_TABLE, "hw2.txt", three("homework"))
    write_untargeted_edits(root)
    evaluate(root, "--configs", "jev:card")
    assert experiments_of(root)["edits_base"] == {"n": 6, "wrong": 0}
    report = report_of(root)
    assert (
        "The spec's experiment (fix each wrong placement with a description edit, then count "
        "which other files changed) was not run: jev:card has the right top group for all 6 "
        "files with text, so the base run had no wrong placement to fix. The edits below name "
        "no target. Each changes one group's description, and they measure how far one "
        "description edit reaches."
    ) in report
    # No target columns, and no "others": every file is one of the others.
    assert (
        "| Edit | Group | Files | Wrong to right | Right to wrong | Between wrong groups "
        "| Top-1 before | Top-1 after |"
    ) in report
    assert "| solutions | exams | 6 | 0 | 2 | 0 | 100.0% | 66.7% |" in report  # hw1 and hw2
    assert "| plain | homework | 6 | 0 | 0 | 0 | 100.0% | 100.0% |" in report
    assert "Targets" not in report


def test_edits_with_no_target_say_so_when_the_base_run_had_wrong_placements(root, jev):
    """Then the reason is that no edit names a file, not that there was nothing to fix."""
    write_untargeted_edits(root)
    evaluate(root, "--configs", "jev:card")
    report = report_of(root)
    assert (
        "was not run: no edit names a file to fix, although jev:card has a wrong top group for "
        "2 of the 6 files with text. The edits below name no target."
    ) in report
    # exam2 goes from wrong to right, hw1 from right to wrong, hw2 between two wrong groups.
    assert "| solutions | exams | 6 | 1 | 1 | 1 | 66.7% | 66.7% |" in report


def test_an_edit_with_no_target_that_was_not_run_fills_the_shorter_table(root, jev):
    """Without the pass there is nothing to compare, and the row still has eight cells."""
    write_untargeted_edits(root)
    evaluate(root, "--configs", "jev:card", "--no-experiments")
    assert "| solutions | exams | n/a | n/a | n/a | n/a | n/a | n/a |\n" in report_of(root)


def test_the_notes_of_the_edits_are_printed_under_the_table(root, jev):
    """Who wrote an edit, when and why is in the report, and so is an edit that does not say."""
    write_untargeted_edits(root)
    evaluate(root, "--configs", "jev:card")
    edits = experiments_of(root)["description_edits"]
    assert [edit["note"] for edit in edits] == [
        "Written after the first full run, to see how far one edit reaches.",
        None,
    ]
    report = report_of(root)
    table = report.index("| plain | homework |")
    notes = report.index("Notes on the edits (who wrote them, when and why):")
    assert table < notes
    assert (
        "- solutions: Written after the first full run, to see how far one edit reaches."
    ) in report
    assert "No note says who wrote these edits, when or why: plain." in report


def test_a_note_does_not_change_what_is_asked(root, jev):
    """The cache follows the groups, so adding a note to an edit asks nothing again."""
    write_edits(root)
    evaluate(root, "--configs", "jev:card")
    asked = len(jev.calls)
    with (root / "description_edits.yaml").open("a", encoding="utf-8") as handle:
        handle.write("  note: Added later.\n")
    assert evaluate(root, "--configs", "jev:card") == 0
    assert len(jev.calls) == asked
    assert "- solutions: Added later." in report_of(root)


def test_a_reworded_edit_is_asked_again_under_the_same_name(root, jev):
    """The cache follows the groups, not the edit's name, so new wording means new answers."""
    write_edits(root)
    evaluate(root, "--configs", "jev:card")
    jev.calls.clear()
    write_edits(root, covers="quizzes, exams and graded solutions")
    assert evaluate(root, "--configs", "jev:card") == 0
    assert sorted(call.file_id for call in jev.calls) == TEXT_IDS
    assert {call.groups[0].covers for call in jev.calls} == {"quizzes, exams and graded solutions"}
    jev.calls.clear()
    assert evaluate(root, "--configs", "jev:card") == 0
    assert jev.calls == []


def test_an_edit_for_an_unknown_group_stops_the_run_before_anything_is_asked(root, jev, capsys):
    """A mistake in the edits file is an error line and exit 1."""
    write_edits(root, group_id="projects")
    assert evaluate(root, "--configs", "jev:card") == 1
    assert "error: edit 'solutions' is for group 'projects'" in capsys.readouterr().err
    assert jev.calls == []
    assert not (root / "results").exists()


def test_a_target_that_was_not_compared_is_named_in_the_caveats(root, jev):
    """A target id that matches no answered file is reported, not dropped in silence."""
    (root / "description_edits.yaml").write_text(
        f"- name: solutions\n  group_id: exams\n  covers: {EDITED_COVERS}\n"
        "  target_file_ids: [exam2.txt, typo.txt]\n",
        encoding="utf-8",
    )
    evaluate(root, "--configs", "jev:card")
    metrics = json.loads((root / "results" / "metrics.json").read_text(encoding="utf-8"))
    effect = metrics["experiments"]["description_edits"][0]["effect"]
    assert (effect["targets_n"], effect["targets_missing"]) == (2, ["typo.txt"])
    assert (
        "Edit solutions lists target files that could not be compared: typo.txt."
        in metrics["caveats"]
    )


def test_a_vault_without_the_held_out_group_has_no_held_out_pass(tmp_path, jev):
    """With no Exams group there is nothing to hold out, and the report says so."""
    root = tmp_path / "eval"
    vault.init_vault(root)
    vault.add_group(root, "Homework", "problem sets and their solutions")
    vault.add_group(root, "Lecture notes", "lecture notes and slides")
    add_file(root, "hw1.txt", "Problem set 1.\n", "homework")
    add_file(root, "lec1.md", "# Lecture 1\n", "lecture-notes")
    assert evaluate(root, "--configs", "jev:card") == 0
    assert {call.set_version for call in jev.calls} == {2}
    assert len(jev.calls) == 2
    tests = experiments_of(root)
    assert tests["has_held_out_group"] is False
    assert tests["group_added"] == {"jev:card": None, "jev:names_only": None}
    assert "Not run: this vault has no group with the id exams." in report_of(root)
