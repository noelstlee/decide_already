"""corrections.py: sync of edits made by hand, move, freeze, unfreeze, suggest, and the labels."""

import pytest

from decide_already import vault
from decide_already.backends.base import BudgetExceeded
from decide_already.card import hash_file
from decide_already.config import LABELS_FILE, SUGGESTIONS_FILE, Config, decide_path
from decide_already.corrections import (
    SyncReport,
    active_labels,
    freeze,
    move_file,
    read_labels,
    suggest,
    sync,
    unfreeze,
)
from decide_already.log import read_jsonl
from decide_already.sort import sort_vault
from decide_already.types import Label, Placement, Suggestion
from fakes import FakeBackend

# What the model said about every file that `place` writes a note for.
PROBS = {"hw": 0.75, "logistics": 0.2, "project": 0.05}

# Answers for the fake backend: (probabilities by group id, fits_any).
TO_HW = ({"hw": 0.9, "logistics": 0.05, "project": 0.05}, 0.95)
TO_LOGISTICS = ({"hw": 0.05, "logistics": 0.9, "project": 0.05}, 0.95)
NO_FIT = ({"hw": 0.9, "logistics": 0.05, "project": 0.05}, 0.2)


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


def write_note(root, file_id, group_id="hw", **changes):
    """Write a card note as the sorter would, with any placement fields changed."""
    fields = {
        "source": f"[[_dump/{file_id}]]",
        "file_id": file_id,
        "content_hash": "0" * 64,
        "file_type": "Text",
        "group_id": group_id,
        "placed_by": "jev",
        "set_version": 3,
        "probs": PROBS,
        "fits_any": 0.9,
        "model": "fake-0",
    }
    return vault.write_card_note(root, Placement(**{**fields, **changes}), "a card\n")


def place(root, file_id, group_id="hw", **changes):
    """Put a file in the dump and write its card note, as if a sort had placed it."""
    path = drop(root, file_id)
    return write_note(root, file_id, group_id, content_hash=hash_file(path), **changes)


def note_of(root, file_id):
    """The one card note of a file. Fails when the file has no note, or more than one."""
    (note,) = [note for note in vault.read_card_notes(root) if note.placement.file_id == file_id]
    return note


def drag(root, file_id, folder):
    """Move a file's card note into another folder by hand, as a drag in Obsidian does."""
    path = note_of(root, file_id).path
    (root / folder).mkdir(exist_ok=True)
    return path.rename(root / folder / path.name)


def set_frozen_by_hand(root, file_id, frozen=True):
    """Flip `frozen` in a card note's frontmatter, the way a person would in Obsidian."""
    path = note_of(root, file_id).path
    old = f"\nfrozen: {str(not frozen).lower()}\n"
    new = f"\nfrozen: {str(frozen).lower()}\n"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def labels(root):
    """Every line of labels.jsonl as a dict without its time, checked to be a Label."""
    lines = read_jsonl(decide_path(root, LABELS_FILE))
    for line in lines:
        assert list(line) == list(Label.model_fields)
        assert Label.model_validate(line).at.tzinfo is not None
    return [{key: value for key, value in line.items() if key != "at"} for line in lines]


def label(file_id, kind, from_group, to_group, set_version=3, jev_probs=PROBS, jev_set_version=3):
    """One expected line of labels.jsonl, without its time."""
    return {
        "file_id": file_id,
        "kind": kind,
        "from_group": from_group,
        "to_group": to_group,
        "jev_probs": jev_probs,
        "jev_set_version": jev_set_version,
        "set_version": set_version,
    }


def picked_up(set_version=3, **lists):
    """The report of a sync that picked up only what is named here."""
    empty = {"moved": [], "confirmed": [], "unfrozen": [], "orphaned": [], "missing_sources": []}
    return SyncReport(**{**empty, **lists}, set_version=set_version)


def snapshot(root):
    """Every file in the vault with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def state(root, file_id):
    """Where a file's note sits and what its frontmatter says about the user's hand."""
    note = note_of(root, file_id)
    placement = note.placement
    return (note.folder, placement.group_id, placement.placed_by, placement.frozen)


class CappedBackend(FakeBackend):
    """A fake with a spend cap: it answers a set number of requests and refuses the rest."""

    def __init__(self, allowed, **answers):
        """Answer `allowed` requests as FakeBackend would, then raise BudgetExceeded."""
        super().__init__(**answers)
        self.allowed = allowed
        self.refused = 0

    def sort(self, request):
        """Answer while under the cap. A refused request is not sent, so it is only counted."""
        if len(self.calls) >= self.allowed:
            self.refused += 1
            raise BudgetExceeded("spend cap reached: $5.0000 spent, the cap is $5")
        return super().sort(request)


# sync: nothing to do


def test_sync_of_a_vault_nobody_touched_picks_up_nothing(root):
    """Notes that sit where their frontmatter says, unfrozen and unlabeled: nothing is written."""
    place(root, "a.txt", "hw")
    place(root, "hw3/b.py", "logistics", close_call=True)
    place(root, "c.txt", None, reason="weak_top")
    before = snapshot(root)
    assert sync(root) == picked_up()
    assert snapshot(root) == before  # no note rewritten, and no labels file made
    assert read_labels(root) == []
    assert active_labels(root) == {}


def test_sync_of_an_empty_vault(tmp_path):
    """No groups and no notes yet: the report is empty, at set version 0."""
    vault.init_vault(tmp_path)
    assert sync(tmp_path) == picked_up(set_version=0)


def test_sync_leaves_the_users_own_notes_alone(root):
    """A markdown file that is not a card note is not synced, wherever it sits."""
    own = {
        root / "HW" / "my notes.md": "Things to remember.\n",
        root / "Old stuff" / "todo.md": "---\ntags: [todo]\n---\nSort these later.\n",
    }
    for path, text in own.items():
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert sync(root) == picked_up()
    for path, text in own.items():
        assert path.read_text(encoding="utf-8") == text


# sync rule 2: a drag


def test_a_note_dragged_to_another_groups_folder_is_a_move(root):
    """The folder is the group: the note is frozen there, and the move is a label."""
    place(root, "hw3/a.txt", "hw", close_call=True)
    drag(root, "hw3/a.txt", "Logistics")
    assert sync(root) == picked_up(moved=["hw3/a.txt"])

    note = note_of(root, "hw3/a.txt")
    assert note.path == root / "Logistics" / "hw3__a.txt.md"
    placement = note.placement
    assert (placement.group_id, placement.placed_by, placement.frozen) == ("logistics", "you", True)
    assert (placement.close_call, placement.reason) == (False, None)
    # What the model said stays in the note, with the set version it said it under.
    assert (placement.probs, placement.fits_any) == (PROBS, 0.9)
    assert (placement.set_version, placement.model, placement.variant) == (3, "fake-0", "card")
    assert note.body == "a card\n"
    assert labels(root) == [label("hw3/a.txt", "moved", "hw", "logistics")]


def test_a_note_dragged_into_unsorted_says_no_group_fits(root):
    """Unsorted is a place a file can be moved to: the label's to_group is null."""
    place(root, "a.txt", "hw")
    drag(root, "a.txt", "Unsorted")
    assert sync(root) == picked_up(moved=["a.txt"])
    assert state(root, "a.txt") == ("Unsorted", None, "you", True)
    assert labels(root) == [label("a.txt", "moved", "hw", None)]


def test_a_note_dragged_out_of_unsorted_loses_its_reason(root):
    """The rules sent it to Unsorted for a reason. The user's move replaces that."""
    place(root, "a.txt", None, reason="weak_top")
    drag(root, "a.txt", "Project")
    assert sync(root) == picked_up(moved=["a.txt"])
    assert state(root, "a.txt") == ("Project", "project", "you", True)
    assert note_of(root, "a.txt").placement.reason is None
    assert labels(root) == [label("a.txt", "moved", None, "project")]


def test_a_dragged_note_that_was_renamed_gets_its_own_name_back(root):
    """The note is written as a move writes it: under the file's note name, once."""
    path = place(root, "a.txt", "hw")
    (root / "Logistics" / "My favorite.md").write_bytes(path.read_bytes())
    path.unlink()
    assert sync(root) == picked_up(moved=["a.txt"])
    assert note_of(root, "a.txt").path == root / "Logistics" / "a.txt.md"
    assert not (root / "Logistics" / "My favorite.md").exists()


def test_a_folder_added_with_its_group_note_by_hand_is_a_group(root):
    """A group made in Obsidian counts: a note dragged into its folder is a move, not an orphan."""
    place(root, "midterm.txt", "hw")
    (root / "_groups" / "Exams.md").write_text("---\ncovers: midterms\n---\n", encoding="utf-8")
    drag(root, "midterm.txt", "Exams")
    assert sync(root) == picked_up(moved=["midterm.txt"], set_version=4)
    assert state(root, "midterm.txt") == ("Exams", "exams", "you", True)
    assert labels(root) == [label("midterm.txt", "moved", "hw", "exams", set_version=4)]


def test_a_group_renamed_by_hand_keeps_its_files_and_their_labels(root):
    """Renaming the group note in Obsidian is a rename, not a delete: nothing goes to Unsorted."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    freeze(root, "b.txt")
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Homework.md")
    assert sync(root) == picked_up(set_version=4)
    assert state(root, "a.txt") == ("Homework", "hw", "jev", False)
    assert state(root, "b.txt") == ("Homework", "hw", "jev", True)
    assert labels(root) == [label("b.txt", "confirmed", "hw", "hw")]
    assert not (root / "HW").exists()


def test_a_note_dragged_out_of_a_deleted_groups_folder_is_a_move(root):
    """The user got to it before sync did: it is where they put it, not in Unsorted."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    vault.delete_group(root, "HW")
    drag(root, "a.txt", "Logistics")
    assert sync(root) == picked_up(moved=["a.txt"], orphaned=["b.txt"], set_version=4)
    assert state(root, "a.txt") == ("Logistics", "logistics", "you", True)
    assert state(root, "b.txt") == ("Unsorted", None, "code", False)
    # The label still names the group the model had it in, though that group is gone.
    assert labels(root) == [label("a.txt", "moved", "hw", "logistics", set_version=4)]


# sync rules 3 and 4: the frozen flag


def test_frozen_set_by_hand_is_a_confirmation(root):
    """frozen: true on a note the model placed means its pick was accepted."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", None, reason="no_group_fits")
    set_frozen_by_hand(root, "a.txt")
    set_frozen_by_hand(root, "b.txt")
    notes = snapshot(root)
    assert sync(root) == picked_up(confirmed=["a.txt", "b.txt"])
    assert labels(root) == [
        label("a.txt", "confirmed", "hw", "hw"),
        label("b.txt", "confirmed", None, None),
    ]
    assert sorted(active_labels(root)) == ["a.txt", "b.txt"]
    # Only the label is written. The notes are as the user left them.
    after = snapshot(root)
    del after[".decide/labels.jsonl"]
    assert after == notes


def test_frozen_cleared_by_hand_withdraws_the_label(root):
    """frozen: false on a file that has a label means the user took the label back."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    freeze(root, "a.txt")
    move_file(root, "b.txt", "Logistics")
    set_frozen_by_hand(root, "a.txt", False)
    set_frozen_by_hand(root, "b.txt", False)
    assert sync(root) == picked_up(unfrozen=["a.txt", "b.txt"])
    assert labels(root)[2:] == [
        label("a.txt", "unfrozen", "hw", "hw"),
        label("b.txt", "unfrozen", "logistics", "logistics"),
    ]
    assert active_labels(root) == {}
    # The moved note stays where the user put it, until a rerun sorts it again.
    assert state(root, "b.txt") == ("Logistics", "logistics", "you", False)


# sync rule 1: a deleted group


def test_a_deleted_groups_notes_go_to_unsorted_and_lose_their_freeze(root):
    """Every note left in the folder moves to Unsorted, unfrozen. Labels are withdrawn."""
    place(root, "confirmed.txt", "hw")
    place(root, "moved.txt", "logistics")
    place(root, "plain.txt", "hw", close_call=True)
    place(root, "hand.txt", "hw")
    place(root, "elsewhere.txt", "logistics")
    freeze(root, "confirmed.txt")
    move_file(root, "moved.txt", "HW")
    freeze(root, "elsewhere.txt")
    set_frozen_by_hand(root, "hand.txt")  # frozen by hand, and not synced before the delete
    vault.delete_group(root, "HW")
    assert (root / "HW").is_dir()  # the folder stays while notes are in it

    report = sync(root)
    assert report == picked_up(
        orphaned=["confirmed.txt", "hand.txt", "moved.txt", "plain.txt"], set_version=4
    )
    for file_id in report.orphaned:
        assert state(root, file_id) == ("Unsorted", None, "code", False)
        placement = note_of(root, file_id).placement
        assert (placement.close_call, placement.probs, placement.set_version) == (False, PROBS, 3)
    assert list((root / "HW").iterdir()) == []
    # Only the two files that had a label get an "unfrozen" line.
    assert labels(root)[3:] == [
        label("confirmed.txt", "unfrozen", "hw", None, set_version=4),
        label("moved.txt", "unfrozen", "hw", None, set_version=4),
    ]
    assert list(active_labels(root)) == ["elsewhere.txt"]
    assert state(root, "elsewhere.txt") == ("Logistics", "logistics", "jev", True)

    # They are unfrozen, so the next rerun asks about all four again.
    backend = FakeBackend(default=({"logistics": 0.9, "project": 0.1}, 0.95))
    sort_vault(root, backend)
    assert [request.file_id for request in backend.calls] == report.orphaned


def test_a_note_in_a_folder_that_was_never_a_group_goes_to_unsorted_too(root):
    """A folder without a group note is no group, whatever it is called."""
    place(root, "a.txt", "hw")
    drag(root, "a.txt", "Old stuff")
    assert sync(root) == picked_up(orphaned=["a.txt"])
    assert state(root, "a.txt") == ("Unsorted", None, "code", False)
    assert labels(root) == []


def test_a_renamed_group_takes_its_notes_along(root):
    """Renaming is not deleting: the id stays, the folder moves, and sync picks up nothing."""
    place(root, "a.txt", "hw")
    freeze(root, "a.txt")
    vault.rename_group(root, "HW", "Homework")
    assert sync(root) == picked_up(set_version=4)
    assert state(root, "a.txt") == ("Homework", "hw", "jev", True)
    assert list(active_labels(root)) == ["a.txt"]


# sync: the order of the rules


def test_rule_1_wins_over_rule_2_a_note_dragged_into_a_folder_that_is_no_group(root):
    """Dragged and frozen, but the folder is no group: Unsorted and unfrozen, with no move."""
    place(root, "a.txt", "hw")
    freeze(root, "a.txt")
    drag(root, "a.txt", "Old stuff")
    assert sync(root) == picked_up(orphaned=["a.txt"])
    assert state(root, "a.txt") == ("Unsorted", None, "code", False)
    assert labels(root) == [
        label("a.txt", "confirmed", "hw", "hw"),
        label("a.txt", "unfrozen", "hw", None),
    ]


def test_rule_2_wins_over_rule_3_a_note_dragged_and_frozen_by_hand(root):
    """One thing the user did, one label: the move. No confirmation on top of it."""
    place(root, "a.txt", "hw")
    drag(root, "a.txt", "Logistics")
    set_frozen_by_hand(root, "a.txt")
    assert sync(root) == picked_up(moved=["a.txt"])
    assert labels(root) == [label("a.txt", "moved", "hw", "logistics")]
    assert state(root, "a.txt") == ("Logistics", "logistics", "you", True)


def test_rule_2_wins_over_rule_4_a_note_dragged_with_its_freeze_cleared(root):
    """A move freezes, so clearing the flag while dragging does not withdraw anything."""
    place(root, "a.txt", "hw")
    freeze(root, "a.txt")
    drag(root, "a.txt", "Logistics")
    set_frozen_by_hand(root, "a.txt", False)
    assert sync(root) == picked_up(moved=["a.txt"])
    assert labels(root) == [
        label("a.txt", "confirmed", "hw", "hw"),
        label("a.txt", "moved", "hw", "logistics"),
    ]
    assert state(root, "a.txt") == ("Logistics", "logistics", "you", True)
    assert active_labels(root)["a.txt"].kind == "moved"


def test_one_sync_applies_each_rule_to_the_note_it_fits(root):
    """Four notes, four different edits by hand, one sync. Labels come in order of file id."""
    place(root, "1_orphan.txt", "project")
    place(root, "2_dragged.txt", "hw")
    place(root, "3_frozen.txt", "hw")
    place(root, "4_thawed.txt", "logistics")
    place(root, "5_untouched.txt", "logistics")
    freeze(root, "Project")
    freeze(root, "4_thawed.txt")
    vault.delete_group(root, "Project")
    drag(root, "2_dragged.txt", "Logistics")
    set_frozen_by_hand(root, "3_frozen.txt")
    set_frozen_by_hand(root, "4_thawed.txt", False)

    assert sync(root) == picked_up(
        orphaned=["1_orphan.txt"],
        moved=["2_dragged.txt"],
        confirmed=["3_frozen.txt"],
        unfrozen=["4_thawed.txt"],
        set_version=4,
    )
    assert labels(root)[2:] == [
        label("1_orphan.txt", "unfrozen", "project", None, set_version=4),
        label("2_dragged.txt", "moved", "hw", "logistics", set_version=4),
        label("3_frozen.txt", "confirmed", "hw", "hw", set_version=4),
        label("4_thawed.txt", "unfrozen", "logistics", "logistics", set_version=4),
    ]
    assert sorted(active_labels(root)) == ["2_dragged.txt", "3_frozen.txt"]


def test_a_second_sync_right_away_picks_up_nothing(root):
    """Every rule leaves its note settled, so sync can run as often as anyone likes."""
    for file_id in ("a.txt", "b.txt", "c.txt", "d.txt"):
        place(root, file_id, "hw")
    freeze(root, "c.txt")
    drag(root, "a.txt", "Logistics")
    set_frozen_by_hand(root, "b.txt")
    set_frozen_by_hand(root, "c.txt", False)
    drag(root, "d.txt", "Old stuff")
    assert sync(root) != picked_up()
    after_first = snapshot(root)
    assert sync(root) == picked_up()
    assert sync(root) == picked_up()
    assert snapshot(root) == after_first


def test_sync_reads_the_card_notes_once_however_many_it_rewrites(root, monkeypatch):
    """Rewriting a note must not mean reading all of them again, or a big sync would crawl."""
    for number in range(6):
        place(root, f"file{number}.txt", "hw")
        drag(root, f"file{number}.txt", "Logistics" if number % 2 else "Old stuff")
    reads = []
    read_card_notes = vault.read_card_notes

    def counting(root):
        """read_card_notes, with every call counted."""
        reads.append(root)
        return read_card_notes(root)

    monkeypatch.setattr(vault, "read_card_notes", counting)
    report = sync(root)
    assert (len(report.moved), len(report.orphaned)) == (3, 3)
    assert len(reads) == 1


def test_a_copy_of_a_note_does_not_make_sync_write_labels_forever(root):
    """Two notes for one file that disagree about frozen: the last one counts, every time."""
    path = place(root, "a.txt", "hw")
    copy = root / "HW" / "a copy.md"  # comes before a.txt.md, so a.txt.md is the one that counts
    copy.write_bytes(path.read_bytes())
    set_frozen = path.read_text(encoding="utf-8").replace("\nfrozen: false\n", "\nfrozen: true\n")
    path.write_text(set_frozen, encoding="utf-8")
    assert sync(root) == picked_up(confirmed=["a.txt"])
    assert sync(root) == picked_up()
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw")]
    # The next move writes the file's note once and removes the copy.
    move_file(root, "a.txt", "Logistics")
    assert [note.path for note in vault.read_card_notes(root)] == [root / "Logistics" / "a.txt.md"]


# sync: group notes and missing originals


def test_a_group_note_edited_by_hand_becomes_a_new_set_version_through_sync(root):
    """Sync reads the group notes too, and labels written after it carry the new version."""
    place(root, "a.txt", "hw")
    group_note = root / "_groups" / "Logistics.md"
    text = group_note.read_text(encoding="utf-8")
    edited = text.replace(
        "syllabus, schedule and deadlines",
        "deadlines and policies for anything, including projects",
    )
    assert edited != text
    group_note.write_text(edited, encoding="utf-8")

    assert sync(root) == picked_up(set_version=4)
    latest = vault.latest_set(root)
    assert latest.version == 4
    covers = {group.id: group.covers for group in latest.groups}
    assert covers["logistics"] == "deadlines and policies for anything, including projects"
    assert sync(root) == picked_up(set_version=4)  # nothing new, so no version 5

    set_frozen_by_hand(root, "a.txt")
    assert sync(root) == picked_up(confirmed=["a.txt"], set_version=4)
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw", set_version=4)]


def test_sync_reports_a_broken_group_note(root):
    """A group note without covers is the user's to fix before anything is picked up."""
    place(root, "a.txt", "hw")
    drag(root, "a.txt", "Logistics")
    (root / "_groups" / "Empty.md").write_text("---\nid: empty\n---\n", encoding="utf-8")
    with pytest.raises(ValueError, match="group 'Empty' has no covers description"):
        sync(root)
    assert labels(root) == []


def test_notes_whose_original_is_gone_are_listed_and_left_alone(root):
    """No rule is applied to them, whatever was done to them by hand."""
    place(root, "kept.txt", "hw")
    for file_id in ("gone.txt", "gone_dragged.txt", "gone_frozen.txt", "hw3/gone_orphan.txt"):
        place(root, file_id, "hw")
        (root / "_dump" / file_id).unlink()
    drag(root, "gone_dragged.txt", "Logistics")
    set_frozen_by_hand(root, "gone_frozen.txt")
    drag(root, "hw3/gone_orphan.txt", "Old stuff")
    before = snapshot(root)

    gone = ["gone.txt", "gone_dragged.txt", "gone_frozen.txt", "hw3/gone_orphan.txt"]
    assert sync(root) == picked_up(missing_sources=gone)
    assert snapshot(root) == before  # no note changed, no label written
    assert sync(root) == picked_up(missing_sources=gone)  # listed every time, as long as it lasts

    # Once an original is back, what was done to its note by hand is picked up.
    drop(root, "gone_dragged.txt")
    drop(root, "gone_frozen.txt")
    assert sync(root) == picked_up(
        moved=["gone_dragged.txt"],
        confirmed=["gone_frozen.txt"],
        missing_sources=["gone.txt", "hw3/gone_orphan.txt"],
    )


def test_sync_never_touches_the_originals(root):
    """Sync reads the dump's file list and nothing more: same files, same bytes, same times."""
    for file_id in ("a.txt", "hw3/b.py", "c.txt"):
        place(root, file_id, "hw")
    drag(root, "a.txt", "Logistics")
    set_frozen_by_hand(root, "hw3/b.py")
    drag(root, "c.txt", "Old stuff")

    def originals():
        """Every path under the dump folder with its bytes and modification time."""
        paths = sorted((root / "_dump").rglob("*"))
        return [
            (path, path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
            for path in paths
        ]

    before = originals()
    assert sync(root) == picked_up(moved=["a.txt"], confirmed=["hw3/b.py"], orphaned=["c.txt"])
    assert originals() == before


# File and group references


def test_a_file_reference_is_an_id_or_a_name_only_one_file_has(root):
    """The full id always works. The bare name works when no other sorted file has it."""
    place(root, "syllabus.md", "logistics")
    place(root, "hw3/hw3_solution.py", "hw")
    place(root, "hw3/notes.txt", "hw")
    place(root, "hw4/notes.txt", "hw")
    assert freeze(root, "hw3/hw3_solution.py") == ["hw3/hw3_solution.py"]
    assert unfreeze(root, "hw3_solution.py") == ["hw3/hw3_solution.py"]
    assert freeze(root, "syllabus.md") == ["syllabus.md"]
    assert move_file(root, "hw3_solution.py", "Project").file_id == "hw3/hw3_solution.py"
    assert move_file(root, "hw4/notes.txt", "Project").file_id == "hw4/notes.txt"


def test_a_file_name_that_several_files_share_is_refused_with_their_ids(root):
    """The message lists the ids to pick from, and nothing is changed."""
    place(root, "hw3/notes.txt", "hw")
    place(root, "hw4/notes.txt", "hw")
    place(root, "hw4/more/notes.txt", "logistics")
    before = snapshot(root)
    message = (
        "3 files are named 'notes.txt': hw3/notes.txt, hw4/more/notes.txt, hw4/notes.txt; "
        "use the full id of one"
    )
    for action in (freeze, unfreeze):
        with pytest.raises(ValueError) as error:
            action(root, "notes.txt")
        assert str(error.value) == message
    with pytest.raises(ValueError) as error:
        move_file(root, "notes.txt", "Project")
    assert str(error.value) == message
    assert snapshot(root) == before


def test_an_exact_file_id_wins_over_a_name_other_files_share(root):
    """notes.txt at the top of the dump is an id, so it is not ambiguous with hw3/notes.txt."""
    place(root, "notes.txt", "logistics")
    place(root, "hw3/notes.txt", "hw")
    assert freeze(root, "notes.txt") == ["notes.txt"]
    assert state(root, "hw3/notes.txt") == ("HW", "hw", "jev", False)


def test_an_unknown_file_reference_says_what_was_not_found(root):
    """A name no sorted file has is an error, and so is a file that was not sorted yet."""
    place(root, "a.txt", "hw")
    drop(root, "new.txt")  # in the dump, but it has no card note yet
    for ref in ("nope.txt", "new.txt", "A.TXT", "a"):
        with pytest.raises(ValueError, match=f"no sorted file with the id or name '{ref}'"):
            move_file(root, ref, "Logistics")
        with pytest.raises(
            ValueError, match=f"no group or sorted file with the id or name '{ref}'"
        ):
            freeze(root, ref)
        with pytest.raises(ValueError, match="p0 sort"):
            unfreeze(root, ref)
    assert labels(root) == []
    assert state(root, "a.txt") == ("HW", "hw", "jev", False)


@pytest.mark.parametrize(
    ("ref", "group_id", "folder"),
    [
        ("Logistics", "logistics", "Logistics"),
        ("logistics", "logistics", "Logistics"),
        ("Lecture slides and notes", "lecture-slides-and-notes", "Lecture slides and notes"),
        ("lecture-slides-and-notes", "lecture-slides-and-notes", "Lecture slides and notes"),
        ("Unsorted", None, "Unsorted"),
        (None, None, "Unsorted"),
    ],
)
def test_a_group_reference_is_an_id_a_name_or_unsorted(root, ref, group_id, folder):
    """A group can be named by its name or its id. None and "Unsorted" mean Unsorted."""
    vault.add_group(root, "Lecture slides and notes", "lecture slides, lecture notes")
    place(root, "a.txt", "hw")
    placement = move_file(root, "a.txt", ref)
    assert placement.group_id == group_id
    assert note_of(root, "a.txt").folder == folder
    assert labels(root) == [label("a.txt", "moved", "hw", group_id, set_version=4)]


def test_an_unknown_group_reference_says_what_was_not_found(root):
    """The message names the groups there are, and nothing is changed."""
    place(root, "a.txt", "hw")
    before = snapshot(root)
    for ref in ("Exams", "Hw", "unsorted", ""):
        with pytest.raises(ValueError) as error:
            move_file(root, "a.txt", ref)
        assert str(error.value) == (
            f"no group with the id or name {ref!r}; pick one of: HW, Logistics, Project, Unsorted"
        )
    assert snapshot(root) == before


def test_freeze_and_unfreeze_with_a_group_reference_apply_to_every_note_in_it(root):
    """A group's name or id, or Unsorted, stands for all the files in it."""
    place(root, "a.txt", "hw")
    place(root, "hw3/b.py", "hw")
    place(root, "c.txt", "logistics")
    place(root, "d.txt", None, reason="weak_top")
    assert freeze(root, "HW") == ["a.txt", "hw3/b.py"]
    assert freeze(root, "hw") == []  # the same group by its id: all frozen already
    assert freeze(root, "Unsorted") == ["d.txt"]
    assert freeze(root, "Project") == []  # a group with no files in it
    assert state(root, "c.txt") == ("Logistics", "logistics", "jev", False)
    assert labels(root) == [
        label("a.txt", "confirmed", "hw", "hw"),
        label("hw3/b.py", "confirmed", "hw", "hw"),
        label("d.txt", "confirmed", None, None),
    ]
    assert unfreeze(root, "hw") == ["a.txt", "hw3/b.py"]
    assert unfreeze(root, "Unsorted") == ["d.txt"]
    assert unfreeze(root, "Logistics") == []
    assert active_labels(root) == {}


def test_a_group_wins_over_a_file_with_the_same_name(root):
    """A file called Project does not hide the group Project."""
    place(root, "Project", "hw")
    place(root, "spec.md", "project")
    assert freeze(root, "Project") == ["spec.md"]
    assert state(root, "Project") == ("HW", "hw", "jev", False)


# move_file


def test_move_puts_the_note_in_the_group_freezes_it_and_logs_a_correction(root):
    """The effect on the note, the placement that is returned, and the one label."""
    place(root, "project_logistics_sheet.md", "project", close_call=True)
    placement = move_file(root, "project_logistics_sheet.md", "Logistics")
    note = note_of(root, "project_logistics_sheet.md")
    assert note.path == root / "Logistics" / "project_logistics_sheet.md.md"
    assert not (root / "Project" / "project_logistics_sheet.md.md").exists()
    assert note.placement == placement
    assert placement == Placement(
        source="[[_dump/project_logistics_sheet.md]]",
        file_id="project_logistics_sheet.md",
        content_hash=hash_file(root / "_dump" / "project_logistics_sheet.md"),
        file_type="Text",
        group_id="logistics",
        placed_by="you",
        frozen=True,
        close_call=False,
        reason=None,
        set_version=3,
        probs=PROBS,
        fits_any=0.9,
        variant="card",
        model="fake-0",
    )
    assert note.body == "a card\n"
    assert labels(root) == [label("project_logistics_sheet.md", "moved", "project", "logistics")]
    assert active_labels(root)["project_logistics_sheet.md"].kind == "moved"


def test_moving_a_file_to_the_group_it_is_in_is_a_freeze(root):
    """No "moved" label: the file is frozen where it is, and its group is confirmed."""
    place(root, "a.txt", "hw", close_call=True)
    placement = move_file(root, "a.txt", "HW")
    assert state(root, "a.txt") == ("HW", "hw", "jev", True)  # still the model's pick
    assert placement == note_of(root, "a.txt").placement
    assert placement.close_call is True
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw")]

    # Again: it is frozen and labeled, so there is nothing left to do.
    before = snapshot(root)
    assert move_file(root, "a.txt", "hw") == placement
    assert snapshot(root) == before


def test_moving_a_file_that_sits_in_unsorted_to_unsorted_confirms_it_there(root):
    """The user agrees that no group fits: frozen in Unsorted, with its reason kept."""
    place(root, "a.txt", None, reason="no_group_fits")
    placement = move_file(root, "a.txt", None)
    assert (placement.group_id, placement.frozen, placement.reason) == (None, True, "no_group_fits")
    assert labels(root) == [label("a.txt", "confirmed", None, None)]


def test_moving_a_frozen_file_keeps_it_frozen_and_the_move_is_its_label(root):
    """A confirmed file can still be moved. Moving it back is one more move."""
    place(root, "a.txt", "hw")
    freeze(root, "a.txt")
    move_file(root, "a.txt", "Logistics")
    assert state(root, "a.txt") == ("Logistics", "logistics", "you", True)
    move_file(root, "a.txt", "HW")
    assert state(root, "a.txt") == ("HW", "hw", "you", True)
    assert labels(root) == [
        label("a.txt", "confirmed", "hw", "hw"),
        label("a.txt", "moved", "hw", "logistics"),
        label("a.txt", "moved", "logistics", "hw"),
    ]
    active = active_labels(root)["a.txt"]
    assert (active.kind, active.from_group, active.to_group) == ("moved", "logistics", "hw")


def test_a_move_that_cannot_be_written_leaves_no_label(root):
    """A file of the user's sits where the note would go: nothing moves, nothing is logged."""
    place(root, "a.txt", "hw")
    own = root / "Logistics" / "a.txt.md"
    own.write_text("My own note.\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="exists and is not the card note of a.txt"):
        move_file(root, "a.txt", "Logistics")
    assert own.read_text(encoding="utf-8") == "My own note.\n"
    assert state(root, "a.txt") == ("HW", "hw", "jev", False)
    assert labels(root) == []


# freeze and unfreeze


def test_freeze_sets_the_flag_and_confirms_the_group(root):
    """frozen: true in the note, a "confirmed" label, and nothing else about the note changes."""
    place(root, "a.txt", "hw", close_call=True)
    before = note_of(root, "a.txt")
    assert freeze(root, "a.txt") == ["a.txt"]
    after = note_of(root, "a.txt")
    assert after.path == before.path
    assert after.body == before.body
    assert after.placement == before.placement.model_copy(update={"frozen": True})
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw")]


def test_freezing_a_file_that_is_frozen_and_labeled_does_nothing(root):
    """Confirmed or moved already: no second label, and the note is not written again."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    freeze(root, "a.txt")
    move_file(root, "b.txt", "Logistics")
    before = snapshot(root)
    assert freeze(root, "a.txt") == []
    assert freeze(root, "b.txt") == []
    assert freeze(root, "Logistics") == []
    assert snapshot(root) == before


def test_unfreeze_clears_the_flag_and_withdraws_the_label(root):
    """frozen: false in the note and an "unfrozen" label. The file stays where it is."""
    place(root, "a.txt", "hw")
    move_file(root, "a.txt", "Logistics")
    assert unfreeze(root, "a.txt") == ["a.txt"]
    assert state(root, "a.txt") == ("Logistics", "logistics", "you", False)
    assert labels(root) == [
        label("a.txt", "moved", "hw", "logistics"),
        label("a.txt", "unfrozen", "logistics", "logistics"),
    ]
    assert active_labels(root) == {}
    # Frozen again: a new confirmation, since the old label was withdrawn.
    assert freeze(root, "a.txt") == ["a.txt"]
    assert labels(root)[2:] == [label("a.txt", "confirmed", "logistics", "logistics")]


def test_unfreezing_a_file_that_is_not_frozen_does_nothing(root):
    """Nothing to withdraw: no label, and the note is not written again."""
    place(root, "a.txt", "hw")
    before = snapshot(root)
    assert unfreeze(root, "a.txt") == []
    assert unfreeze(root, "HW") == []
    assert snapshot(root) == before


def test_an_action_picks_up_edits_made_by_hand_before_it_acts(root):
    """Every action syncs first, so it sees the vault as the user left it."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    place(root, "c.txt", "hw")
    drag(root, "a.txt", "Logistics")  # not synced yet
    set_frozen_by_hand(root, "c.txt")  # not synced yet
    assert freeze(root, "HW") == ["b.txt"]  # a.txt is in Logistics now, c.txt is frozen already
    assert labels(root) == [
        label("a.txt", "moved", "hw", "logistics"),
        label("c.txt", "confirmed", "hw", "hw"),
        label("b.txt", "confirmed", "hw", "hw"),
    ]
    # Moving a.txt to where it was dragged finds it there, frozen and labeled.
    before = snapshot(root)
    assert move_file(root, "a.txt", "Logistics").group_id == "logistics"
    assert snapshot(root) == before


def test_move_and_unfreeze_pick_up_edits_made_by_hand_too(root):
    """A drag before a move is two moves. A flag set by hand is confirmed before it is cleared."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    drag(root, "a.txt", "Logistics")  # not synced yet
    move_file(root, "a.txt", "Project")
    set_frozen_by_hand(root, "b.txt")  # not synced yet
    assert unfreeze(root, "b.txt") == ["b.txt"]
    assert labels(root) == [
        label("a.txt", "moved", "hw", "logistics"),
        label("a.txt", "moved", "logistics", "project"),
        label("b.txt", "confirmed", "hw", "hw"),
        label("b.txt", "unfrozen", "hw", "hw"),
    ]


def test_the_actions_still_work_on_a_note_whose_original_is_gone(root):
    """Sync leaves such a note alone. The user can still freeze it, move it and unfreeze it."""
    write_note(root, "gone.txt", "hw")
    write_note(root, "hand.txt", "hw", frozen=True)  # frozen by hand, which sync passes over
    assert sync(root) == picked_up(missing_sources=["gone.txt", "hand.txt"])
    assert freeze(root, "hand.txt") == ["hand.txt"]  # frozen already, but not confirmed yet
    assert freeze(root, "hand.txt") == []
    assert move_file(root, "gone.txt", "Logistics").group_id == "logistics"
    assert unfreeze(root, "Logistics") == ["gone.txt"]
    assert labels(root) == [
        label("hand.txt", "confirmed", "hw", "hw"),
        label("gone.txt", "moved", "hw", "logistics"),
        label("gone.txt", "unfrozen", "logistics", "logistics"),
    ]
    assert state(root, "gone.txt") == ("Logistics", "logistics", "you", False)


# Labels


def test_a_label_records_what_the_model_said_and_the_set_the_user_saw(root):
    """jev_probs and jev_set_version come from the note. set_version is the current one."""
    drop(root, "a.txt")
    answer = ({"hw": 0.8, "logistics": 0.15, "project": 0.05}, 0.9)
    sort_vault(root, FakeBackend(default=answer))  # sorted under set version 3
    vault.add_group(root, "Exams", "midterms and finals")  # version 4
    vault.edit_group(root, "HW", covers="problem sets and their solutions")  # version 5

    freeze(root, "a.txt")
    move_file(root, "a.txt", "Exams")
    unfreeze(root, "a.txt")
    said = {"jev_probs": answer[0], "jev_set_version": 3, "set_version": 5}
    assert labels(root) == [
        label("a.txt", "confirmed", "hw", "hw", **said),
        label("a.txt", "moved", "hw", "exams", **said),
        label("a.txt", "unfrozen", "exams", "exams", **said),
    ]
    # The move did not touch what the model said, so later labels can still record it.
    placement = note_of(root, "a.txt").placement
    assert (placement.probs, placement.set_version, placement.model) == (answer[0], 3, "fake-0")


def test_a_file_no_model_was_asked_about_has_no_model_answer_in_its_label(root):
    """A file with no text was placed by code: no probabilities, and no set version for them."""
    drop(root, "scan.txt", "")
    sort_vault(root, FakeBackend())
    assert note_of(root, "scan.txt").placement.placed_by == "code"
    move_file(root, "scan.txt", "HW")
    assert labels(root) == [
        label("scan.txt", "moved", None, "hw", jev_probs=None, jev_set_version=None)
    ]
    assert note_of(root, "scan.txt").placement.reason is None


def test_a_files_active_label_is_its_last_line_when_that_is_moved_or_confirmed(root):
    """Lines are only ever added. The last one per file decides."""
    for file_id in ("a.txt", "b.txt", "c.txt", "d.txt"):
        place(root, file_id, "hw")
    freeze(root, "a.txt")
    move_file(root, "b.txt", "Logistics")
    freeze(root, "c.txt")
    unfreeze(root, "c.txt")
    move_file(root, "a.txt", "Project")
    kinds = [(line.file_id, line.kind) for line in read_labels(root)]
    assert kinds == [
        ("a.txt", "confirmed"),
        ("b.txt", "moved"),
        ("c.txt", "confirmed"),
        ("c.txt", "unfrozen"),
        ("a.txt", "moved"),
    ]
    active = active_labels(root)
    assert {file_id: line.kind for file_id, line in active.items()} == {
        "a.txt": "moved",
        "b.txt": "moved",
    }
    assert active["a.txt"].to_group == "project"
    assert all(isinstance(line, Label) for line in active.values())


# suggest


def frozen_in(root, file_id, group_id):
    """Put a file in the dump with a card note in a group, and freeze it there."""
    place(root, file_id, group_id)
    freeze(root, file_id)


def suggestions(root):
    """Every line of suggestions.jsonl, checked to be a Suggestion."""
    lines = read_jsonl(decide_path(root, SUGGESTIONS_FILE))
    for line in lines:
        assert list(line) == list(Suggestion.model_fields)
    return [Suggestion.model_validate(line) for line in lines]


def test_suggest_lists_the_frozen_files_the_model_would_place_elsewhere(root):
    """Each difference is a Suggestion, returned and written. Nothing moves, no label is added."""
    frozen_in(root, "agrees.txt", "hw")
    frozen_in(root, "elsewhere.txt", "hw")
    frozen_in(root, "no_fit.txt", "logistics")
    frozen_in(root, "out.txt", None)
    place(root, "unfrozen.txt", "project")
    backend = FakeBackend(
        answers={
            "agrees.txt": TO_HW,
            "elsewhere.txt": TO_LOGISTICS,
            "no_fit.txt": NO_FIT,
            "out.txt": TO_HW,
        }
    )
    before = snapshot(root)

    found = suggest(root, backend)
    assert [request.file_id for request in backend.calls] == [
        "agrees.txt",
        "elsewhere.txt",
        "no_fit.txt",
        "out.txt",
    ]
    assert [item.model_dump(exclude={"at"}) for item in found] == [
        {
            "file_id": "elsewhere.txt",
            "current_group": "hw",
            "suggested_group": "logistics",
            "probs": TO_LOGISTICS[0],
            "fits_any": 0.95,
            "set_version": 3,
            "model": "fake-0",
        },
        {
            "file_id": "no_fit.txt",
            "current_group": "logistics",
            "suggested_group": None,
            "probs": NO_FIT[0],
            "fits_any": 0.2,
            "set_version": 3,
            "model": "fake-0",
        },
        {
            "file_id": "out.txt",
            "current_group": None,
            "suggested_group": "hw",
            "probs": TO_HW[0],
            "fits_any": 0.95,
            "set_version": 3,
            "model": "fake-0",
        },
    ]
    assert all(item.at.tzinfo is not None for item in found)
    assert suggestions(root) == found

    # Apart from the suggestions file, the vault is as it was: notes and labels alike.
    after = snapshot(root)
    del after[".decide/suggestions.jsonl"]
    assert after == before


def test_suggest_with_nothing_to_report_writes_nothing(root):
    """The model agrees with every frozen file: an empty list, and no suggestions file."""
    frozen_in(root, "a.txt", "hw")
    frozen_in(root, "b.txt", "logistics")
    place(root, "c.txt", "project")
    backend = FakeBackend(answers={"a.txt": TO_HW, "b.txt": TO_LOGISTICS})
    before = snapshot(root)
    assert suggest(root, backend) == []
    assert len(backend.calls) == 2
    assert snapshot(root) == before
    # With no frozen files at all, the backend is not even asked.
    unfreeze(root, "HW")
    unfreeze(root, "Logistics")
    quiet = FakeBackend()
    assert suggest(root, quiet) == []
    assert quiet.calls == []


def test_suggest_asks_with_the_current_set_and_the_configs_variant(root):
    """The request is built as a sort would build it, against the groups as they are now."""
    frozen_in(root, "a.txt", "hw")
    vault.add_group(root, "Exams", "midterms and finals")
    Config.load(root).model_copy(update={"variant": "names_only"}).save(root)
    answer = ({"exams": 0.9, "hw": 0.04, "logistics": 0.03, "project": 0.03}, 0.9)
    backend = FakeBackend(default=answer)
    (found,) = suggest(root, backend)
    (request,) = backend.calls
    assert [group.id for group in request.groups] == ["exams", "hw", "logistics", "project"]
    assert (request.set_version, request.variant, request.with_descriptions) == (
        4,
        "names_only",
        False,
    )
    assert request.state["file"]["name"] == "a.txt"
    assert (found.suggested_group, found.set_version) == ("exams", 4)


def test_suggest_applies_the_placement_rules_from_the_config(root):
    """A top pick under the cutoff is a suggestion for Unsorted, as it would be in a sort."""
    frozen_in(root, "a.txt", "hw")
    weak = ({"hw": 0.5, "logistics": 0.3, "project": 0.2}, 0.9)
    (found,) = suggest(root, FakeBackend(default=weak))
    assert (found.current_group, found.suggested_group) == ("hw", None)
    Config.load(root).model_copy(update={"unsorted_top_below": 0.4}).save(root)
    assert suggest(root, FakeBackend(default=weak)) == []


def test_suggest_passes_over_a_file_the_backend_fails_on(root):
    """One failure does not lose what was found before it, or stop what comes after it."""
    for file_id in ("a.txt", "b.txt", "c.txt"):
        frozen_in(root, file_id, "hw")
    backend = FakeBackend(default=TO_LOGISTICS, fail={"b.txt"})
    failed = {}
    found = suggest(root, backend, failed)
    assert [item.file_id for item in found] == ["a.txt", "c.txt"]
    assert len(backend.calls) == 3
    assert suggestions(root) == found
    # The failure is not agreement: a caller that asks is told which file had no answer.
    assert failed == {"b.txt": "the fake backend was told to fail for b.txt"}


def test_suggest_stops_at_the_spend_cap_and_returns_what_it_found(root):
    """Once the backend refuses for the cap, no more files are asked about."""
    for file_id in ("a.txt", "b.txt", "c.txt", "d.txt"):
        frozen_in(root, file_id, "hw")
    backend = CappedBackend(2, default=TO_LOGISTICS)
    failed = {}
    found = suggest(root, backend, failed)
    assert [item.file_id for item in found] == ["a.txt", "b.txt"]
    assert [request.file_id for request in backend.calls] == ["a.txt", "b.txt"]
    assert backend.refused == 1  # c.txt was refused, and d.txt was never tried
    assert suggestions(root) == found
    assert failed == {"c.txt": "spend cap reached: $5.0000 spent, the cap is $5"}


def test_suggest_passes_over_files_with_no_text_and_files_whose_original_is_gone(root):
    """There is nothing to send for either, so the backend is not asked about them."""
    drop(root, "empty.txt", "")
    write_note(root, "empty.txt", None, placed_by="code", reason="no_text", probs=None)
    freeze(root, "empty.txt")
    write_note(root, "gone.txt", "hw", frozen=True)
    frozen_in(root, "a.txt", "hw")
    backend = FakeBackend(default=TO_LOGISTICS)
    failed = {}
    found = suggest(root, backend, failed)
    assert [request.file_id for request in backend.calls] == ["a.txt"]
    assert [item.file_id for item in found] == ["a.txt"]
    assert failed == {}  # nothing to ask is not a failure


def test_suggest_picks_up_edits_made_by_hand_before_it_asks(root):
    """A file dragged since the last sync is compared with the group the user put it in."""
    frozen_in(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    drag(root, "a.txt", "Logistics")  # not synced yet
    set_frozen_by_hand(root, "b.txt")  # not synced yet
    backend = FakeBackend(answers={"a.txt": TO_LOGISTICS, "b.txt": TO_HW})
    assert suggest(root, backend) == []  # the model agrees with where both files are now
    assert [request.file_id for request in backend.calls] == ["a.txt", "b.txt"]
    assert labels(root)[1:] == [
        label("a.txt", "moved", "hw", "logistics"),
        label("b.txt", "confirmed", "hw", "hw"),
    ]


def test_suggest_in_a_vault_without_groups_has_nothing_to_suggest(root):
    """No groups to choose from: an empty list, and the backend is not asked."""
    place(root, "a.txt", "hw")
    move_file(root, "a.txt", "Unsorted")
    for name in ("HW", "Logistics", "Project"):
        vault.delete_group(root, name)
    backend = FakeBackend()
    assert suggest(root, backend) == []
    assert backend.calls == []


def test_suggest_reports_an_unknown_variant_in_the_config(root):
    """A typo in the config is an error, not a file to pass over."""
    frozen_in(root, "a.txt", "hw")
    Config.load(root).model_copy(update={"variant": "cards"}).save(root)
    backend = FakeBackend()
    with pytest.raises(ValueError, match="unknown variant 'cards'"):
        suggest(root, backend)
    assert backend.calls == []
