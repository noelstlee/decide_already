"""The walkthrough in docs/phase0.md, replayed end to end with the fake backend.

Six files, three groups, then a fourth group and one correction. Each test runs twice: once
with the commands' functions (freeze, move_file, unfreeze, edit_group) and once with every
correction made by hand, the way Obsidian leaves the files: a frozen flag in the frontmatter,
a note dragged into another folder, an edited group note. Both ways must end with the same
labels, line for line.

Every `add_group` is a new set version, so the walkthrough's "set v1" (three groups) is
version 3 here, and its "v2" (four groups) is version 4.
"""

import pytest

from decide_already import vault
from decide_already.config import LABELS_FILE, decide_path
from decide_already.corrections import active_labels, freeze, move_file, sync, unfreeze
from decide_already.log import read_jsonl
from decide_already.sort import sort_vault
from decide_already.types import Label
from fakes import FakeBackend

SKELETON = "hw3/hw3_skeleton.py"
SOLUTION = "hw3/hw3_solution.py"
SLIDES = "lectures/lecture01_slides.md"
EXAMPLES = "project/project_examples.md"
SHEET = "project/project_logistics_sheet.md"
SYLLABUS = "syllabus.md"

FILES = {
    SKELETON: '"""Homework 3 starter code."""\n\ndef rbf_kernel(x, y):\n    pass  # fill this in\n',
    SOLUTION: '"""Homework 3 solution."""\n\ndef rbf_kernel(x, y):\n    return exp(-gamma(x, y))\n',
    SLIDES: "# Lecture 1: What is learning?\n\n## Supervised learning\n\n## Generalization\n",
    EXAMPLES: "# Project examples\n\nThree past projects with their reports and code.\n",
    SHEET: "# Project logistics\n\nProposal due in week 4. Late policy: two free days.\n",
    SYLLABUS: "# Syllabus\n\nSchedule, grading, office hours and the late policy.\n",
}

LECTURES = "lecture-slides-and-notes"  # the id of the group added in step 5

# What the model says against the first three groups. Without a group for lectures, the
# slides land in Logistics. The project logistics sheet goes to Project.
FIRST = {
    SKELETON: ({"hw": 0.91, "logistics": 0.03, "project": 0.06}, 0.96),
    SOLUTION: ({"hw": 0.93, "logistics": 0.02, "project": 0.05}, 0.97),
    SLIDES: ({"hw": 0.12, "logistics": 0.7, "project": 0.18}, 0.62),
    EXAMPLES: ({"hw": 0.1, "logistics": 0.04, "project": 0.86}, 0.94),
    SHEET: ({"hw": 0.03, "logistics": 0.3, "project": 0.67}, 0.9),
    SYLLABUS: ({"hw": 0.02, "logistics": 0.95, "project": 0.03}, 0.98),
}
# What it says about the slides once the fourth group exists.
SLIDES_WITH_FOUR = ({"hw": 0.03, LECTURES: 0.9, "logistics": 0.05, "project": 0.02}, 0.97)
# What it says about the sheet once Logistics covers deadlines and policies for anything.
SHEET_AFTER_EDIT = ({"hw": 0.02, LECTURES: 0.02, "logistics": 0.84, "project": 0.12}, 0.95)

OLD_COVERS = "syllabus, schedule, deadlines, grading and submission rules"
NEW_COVERS = "syllabus, schedule, deadlines and policies for anything, including projects"


def where(root):
    """The folder each file's card note sits in: the board, in short."""
    return {note.placement.file_id: note.folder for note in vault.read_card_notes(root)}


def note_path(root, file_id):
    """The path of a file's one card note."""
    (note,) = [note for note in vault.read_card_notes(root) if note.placement.file_id == file_id]
    return note.path


def edit_by_hand(path, old, new):
    """Change a piece of text in a note, the way a person would in Obsidian."""
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def set_frozen_by_hand(root, file_id, frozen):
    """Flip `frozen` in the frontmatter of a file's card note."""
    old, new = (f"\nfrozen: {str(not frozen).lower()}\n", f"\nfrozen: {str(frozen).lower()}\n")
    edit_by_hand(note_path(root, file_id), old, new)


def labels(root):
    """labels.jsonl line by line, each line without its time and checked to be a Label."""
    lines = read_jsonl(decide_path(root, LABELS_FILE))
    times = [Label.model_validate(line).at for line in lines]
    assert times == sorted(times)  # lines are only ever added, so the times never go back
    return [{key: value for key, value in line.items() if key != "at"} for line in lines]


def label(kind, file_id, from_group, to_group, said, set_version):
    """One expected line. `said` is what the model said and the set version it said it under."""
    return {
        "file_id": file_id,
        "kind": kind,
        "from_group": from_group,
        "to_group": to_group,
        "jev_probs": said[0],
        "jev_set_version": said[1],
        "set_version": set_version,
    }


def sent(backend, since=0):
    """The file ids the backend was asked about, from request number `since` on."""
    return [request.file_id for request in backend.calls[since:]]


def start(root, by_hand):
    """Steps 1 to 6 of the walkthrough. Returns the backend, which has had seven requests."""
    # 1. Name the groups: HW, Logistics and Project, each with a description.
    vault.init_vault(root)
    vault.add_group(root, "HW", "homework specs, starter code, solutions", "project files")
    vault.add_group(root, "Logistics", OLD_COVERS)
    vault.add_group(root, "Project", "project specs, examples", "deadlines and policies")

    # 2. Dump the six files.
    for file_id, text in FILES.items():
        path = root / "_dump" / file_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    # 3 and 4. Code builds a card per file, and the model sorts all six against three groups.
    backend = FakeBackend(answers=dict(FIRST))
    report = sort_vault(root, backend)
    assert (report.set_version, report.asked, report.placed, report.unsorted) == (3, 6, 6, 0)
    assert sent(backend) == sorted(FILES)
    assert where(root) == {
        SKELETON: "HW",
        SOLUTION: "HW",
        SLIDES: "Logistics",
        EXAMPLES: "Project",
        SHEET: "Project",
        SYLLABUS: "Logistics",
    }
    assert labels(root) == []  # nothing is a label until the user moves or freezes

    # 5. Lecture slides do not belong in Logistics: add a group, and freeze the other five.
    assert vault.add_group(root, "Lecture slides and notes", "lecture slides, notes").version == 4
    assert where(root)[SLIDES] == "Logistics"  # nothing moves until the rerun
    if by_hand:
        for file_id in (SKELETON, SOLUTION, EXAMPLES, SHEET, SYLLABUS):
            set_frozen_by_hand(root, file_id, True)  # picked up by the rerun, which syncs first
    else:
        assert freeze(root, "HW") == [SKELETON, SOLUTION]  # a whole group
        assert freeze(root, "project") == [EXAMPLES, SHEET]  # a group by its id
        assert freeze(root, "syllabus.md") == [SYLLABUS]  # one file

    # 6. Rerun: only the slides go back to the model, against all four groups.
    backend.answers[SLIDES] = SLIDES_WITH_FOUR
    report = sort_vault(root, backend)
    assert (report.set_version, report.asked, report.frozen_skipped) == (4, 1, 5)
    assert sent(backend, since=6) == [SLIDES]
    rerun = backend.calls[6]
    assert [group.id for group in rerun.groups] == ["hw", LECTURES, "logistics", "project"]
    assert rerun.set_version == 4
    assert where(root)[SLIDES] == "Lecture slides and notes"
    assert labels(root) == five_confirmed()
    return backend


def five_confirmed():
    """The labels after step 5: the model's pick accepted for five files, under set version 4."""
    return [
        label("confirmed", SKELETON, "hw", "hw", (FIRST[SKELETON][0], 3), 4),
        label("confirmed", SOLUTION, "hw", "hw", (FIRST[SOLUTION][0], 3), 4),
        label("confirmed", EXAMPLES, "project", "project", (FIRST[EXAMPLES][0], 3), 4),
        label("confirmed", SHEET, "project", "project", (FIRST[SHEET][0], 3), 4),
        label("confirmed", SYLLABUS, "logistics", "logistics", (FIRST[SYLLABUS][0], 3), 4),
    ]


def freeze_everything(root, by_hand, unfrozen):
    """Step 8: freeze every file. `unfrozen` lists the files that are not frozen yet."""
    if by_hand:
        for file_id in unfrozen:
            set_frozen_by_hand(root, file_id, True)
        assert sync(root).confirmed == unfrozen
    else:
        newly_frozen = []
        for group in ("HW", "Lecture slides and notes", "Logistics", "Project", "Unsorted"):
            newly_frozen += freeze(root, group)
        assert newly_frozen == unfrozen
    assert all(note.placement.frozen for note in vault.read_card_notes(root))


def finish(root, backend, requests):
    """After step 8: a rerun asks about nothing, and the originals are as they were dumped."""
    report = sort_vault(root, backend)
    assert (report.asked, report.frozen_skipped, report.errors) == (0, 6, {})
    assert len(backend.calls) == requests
    for file_id, text in FILES.items():
        assert (root / "_dump" / file_id).read_text(encoding="utf-8") == text


@pytest.mark.parametrize("by_hand", [False, True], ids=["with commands", "by hand in Obsidian"])
def test_the_walkthrough_with_the_sheet_fixed_by_a_move(tmp_path, by_hand):
    """Step 7, first way: move the project logistics sheet to Logistics."""
    root = tmp_path
    backend = start(root, by_hand)

    # 7. Move the sheet. That fixes one file, freezes it, and logs a correction.
    if by_hand:
        dragged = note_path(root, SHEET)
        dragged.rename(root / "Logistics" / dragged.name)
        assert sync(root).moved == [SHEET]
    else:
        placement = move_file(root, "project_logistics_sheet.md", "Logistics")
        assert (placement.file_id, placement.group_id) == (SHEET, "logistics")
    (sheet,) = [note for note in vault.read_card_notes(root) if note.placement.file_id == SHEET]
    assert sheet.path == root / "Logistics" / "project__project_logistics_sheet.md.md"
    assert (sheet.placement.placed_by, sheet.placement.frozen) == ("you", True)
    assert sheet.placement.probs == FIRST[SHEET][0]  # what the model said is kept

    # 8. Freeze everything: only the slides were still unfrozen.
    freeze_everything(root, by_hand, unfrozen=[SLIDES])

    # The final groups are the labels, and the log shows what the model got right on its own.
    assert where(root) == {
        SKELETON: "HW",
        SOLUTION: "HW",
        SLIDES: "Lecture slides and notes",
        EXAMPLES: "Project",
        SHEET: "Logistics",
        SYLLABUS: "Logistics",
    }
    assert labels(root) == [
        *five_confirmed(),
        label("moved", SHEET, "project", "logistics", (FIRST[SHEET][0], 3), 4),
        label("confirmed", SLIDES, LECTURES, LECTURES, (SLIDES_WITH_FOUR[0], 4), 4),
    ]
    kinds = {file_id: line.kind for file_id, line in active_labels(root).items()}
    assert kinds == {
        SKELETON: "confirmed",
        SOLUTION: "confirmed",
        SLIDES: "confirmed",
        EXAMPLES: "confirmed",
        SHEET: "moved",  # the one correction: one file in six
        SYLLABUS: "confirmed",
    }

    # A frozen file was never sent again: six requests, then the slides once more.
    finish(root, backend, requests=7)
    assert sent(backend) == [*sorted(FILES), SLIDES]


@pytest.mark.parametrize("by_hand", [False, True], ids=["with commands", "by hand in Obsidian"])
def test_the_walkthrough_with_the_sheet_fixed_by_a_description_edit(tmp_path, by_hand):
    """Step 7, second way: edit Logistics, unfreeze the sheet and rerun."""
    root = tmp_path
    backend = start(root, by_hand)

    # 7. Make Logistics cover deadlines and policies for anything, and unfreeze the sheet.
    if by_hand:
        edit_by_hand(root / "_groups" / "Logistics.md", OLD_COVERS, NEW_COVERS)
        set_frozen_by_hand(root, SHEET, False)  # both are picked up by the rerun
    else:
        assert vault.edit_group(root, "Logistics", covers=NEW_COVERS).version == 5
        assert unfreeze(root, SHEET) == [SHEET]

    # Rerun: the two unfrozen files go back to the model, against the edited set.
    backend.answers[SHEET] = SHEET_AFTER_EDIT
    report = sort_vault(root, backend)
    assert (report.set_version, report.asked, report.frozen_skipped) == (5, 2, 4)
    assert sent(backend, since=7) == [SLIDES, SHEET]
    covers = {group.id: group.covers for group in backend.calls[8].groups}
    assert covers["logistics"] == NEW_COVERS
    (sheet,) = [note for note in vault.read_card_notes(root) if note.placement.file_id == SHEET]
    assert sheet.path == root / "Logistics" / "project__project_logistics_sheet.md.md"
    assert (sheet.placement.placed_by, sheet.placement.frozen) == ("jev", False)
    assert sheet.placement.set_version == 5

    # 8. Freeze everything: the slides and the sheet were still unfrozen.
    freeze_everything(root, by_hand, unfrozen=[SLIDES, SHEET])

    assert where(root) == {
        SKELETON: "HW",
        SOLUTION: "HW",
        SLIDES: "Lecture slides and notes",
        EXAMPLES: "Project",
        SHEET: "Logistics",
        SYLLABUS: "Logistics",
    }
    assert labels(root) == [
        *five_confirmed(),
        label("unfrozen", SHEET, "project", "project", (FIRST[SHEET][0], 3), 5),
        label("confirmed", SLIDES, LECTURES, LECTURES, (SLIDES_WITH_FOUR[0], 5), 5),
        label("confirmed", SHEET, "logistics", "logistics", (SHEET_AFTER_EDIT[0], 5), 5),
    ]
    # No file was moved by the user this time: the rule was fixed, and the model did the rest.
    active = active_labels(root)
    assert sorted(active) == sorted(FILES)
    assert {line.kind for line in active.values()} == {"confirmed"}

    # Frozen files were never sent again: six requests, the slides, then the slides and sheet.
    finish(root, backend, requests=9)
    assert sent(backend) == [*sorted(FILES), SLIDES, SLIDES, SHEET]
