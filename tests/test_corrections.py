"""corrections.py: sync of edits made by hand, move, freeze, unfreeze, suggest, and the labels."""

import unicodedata

import pytest

from decide_already import vault
from decide_already.backends.base import BudgetExceeded, LogFailed
from decide_already.board import build_board
from decide_already.card import build_card, hash_file, render_card
from decide_already.config import LABELS_FILE, SUGGESTIONS_FILE, Config, decide_path
from decide_already.corrections import (
    SyncReport,
    active_labels,
    delete_file,
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
    names = ("moved", "confirmed", "unfrozen", "orphaned", "rehomed", "missing_sources")
    return SyncReport(**{**{name: [] for name in names}, **lists}, set_version=set_version)


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


def top_folders(root):
    """The visible folders at the top of the vault, apart from the dump and _groups."""
    names = (path.name for path in root.iterdir() if path.is_dir())
    return sorted(name for name in names if name[0] not in "._")


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
    """Renaming the group note in Obsidian is a rename, not a delete: nothing goes to Unsorted.

    The old folder is set aside, and sync puts each note in the folder of the new name.
    """
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    freeze(root, "b.txt")
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Homework.md")
    assert sync(root) == picked_up(rehomed=["a.txt", "b.txt"], set_version=4)
    assert state(root, "a.txt") == ("Homework", "hw", "jev", False)
    assert state(root, "b.txt") == ("Homework", "hw", "jev", True)
    assert labels(root) == [label("b.txt", "confirmed", "hw", "hw")]
    assert top_folders(root) == ["Homework", "Logistics", "Project", "Unsorted"]
    assert sync(root) == picked_up(set_version=4)


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
    # The folder is set aside while notes are in it, so the name HW is free at once.
    assert not (root / "HW").exists()
    assert len(list((root / "HW (old)").iterdir())) == 4

    report = sync(root)
    assert report == picked_up(
        orphaned=["confirmed.txt", "hand.txt", "moved.txt", "plain.txt"], set_version=4
    )
    for file_id in report.orphaned:
        assert state(root, file_id) == ("Unsorted", None, "code", False)
        placement = note_of(root, file_id).placement
        assert (placement.close_call, placement.probs, placement.set_version) == (False, PROBS, 3)
    assert not (root / "HW (old)").exists()  # nothing is left in it, so it is removed
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


def test_a_note_dragged_into_a_folder_that_is_no_group_is_put_back(root):
    """A folder without a group note says nothing: the note goes back to its own group."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", None, reason="weak_top")
    before = snapshot(root)
    drag(root, "a.txt", "Old stuff")
    drag(root, "b.txt", "Old stuff")
    assert sync(root) == picked_up(rehomed=["a.txt", "b.txt"])
    assert state(root, "a.txt") == ("HW", "hw", "jev", False)
    assert state(root, "b.txt") == ("Unsorted", None, "jev", False)
    assert snapshot(root) == before  # the notes are back as they were, and nothing is logged
    assert (root / "Old stuff").is_dir()  # the user's folder, so it stays


def test_a_note_of_a_deleted_group_goes_to_unsorted_from_any_folder(root):
    """The note's own group is gone, so no folder can keep it: Unsorted, wherever it sat."""
    place(root, "a.txt", "hw")
    freeze(root, "a.txt")
    drag(root, "a.txt", "Old stuff")
    vault.delete_group(root, "HW")
    assert sync(root) == picked_up(orphaned=["a.txt"], set_version=4)
    assert state(root, "a.txt") == ("Unsorted", None, "code", False)
    assert labels(root)[1:] == [label("a.txt", "unfrozen", "hw", None, set_version=4)]


def test_a_renamed_group_takes_its_notes_along(root):
    """Renaming is not deleting: the id stays, the folder moves, and sync picks up nothing."""
    place(root, "a.txt", "hw")
    freeze(root, "a.txt")
    vault.rename_group(root, "HW", "Homework")
    assert sync(root) == picked_up(set_version=4)
    assert state(root, "a.txt") == ("Homework", "hw", "jev", True)
    assert list(active_labels(root)) == ["a.txt"]


# sync: the order of the rules


def test_a_frozen_note_dragged_into_a_folder_that_is_no_group_keeps_its_freeze(root):
    """Dragged and frozen, but the folder is no group: no move, and no label is lost."""
    place(root, "a.txt", "hw")
    freeze(root, "a.txt")
    drag(root, "a.txt", "Old stuff")
    assert sync(root) == picked_up(rehomed=["a.txt"])
    assert state(root, "a.txt") == ("HW", "hw", "jev", True)
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw")]
    assert list(active_labels(root)) == ["a.txt"]


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
    assert (len(report.moved), len(report.rehomed)) == (3, 3)
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
    assert sync(root) == picked_up(moved=["a.txt"], confirmed=["hw3/b.py"], rehomed=["c.txt"])
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


def test_a_file_named_like_a_group_can_be_frozen_on_its_own_by_kind(root):
    """The page says what it names, so a file called HW is never read as the group HW."""
    place(root, "HW", "hw")
    place(root, "a.txt", "hw")
    assert freeze(root, "HW", kind="file") == ["HW"]
    assert state(root, "a.txt") == ("HW", "hw", "jev", False)
    assert unfreeze(root, "HW", kind="file") == ["HW"]
    assert freeze(root, "hw", kind="group") == ["HW", "a.txt"]


def test_kind_refuses_a_reference_of_the_other_kind(root):
    """With a kind, a file is matched by its id only, and a group is never read as a file."""
    place(root, "notes/a.txt", "hw")
    with pytest.raises(ValueError, match="no sorted file with the id 'a.txt'"):
        freeze(root, "a.txt", kind="file")
    with pytest.raises(ValueError, match="no group with the id or name 'notes/a.txt'"):
        unfreeze(root, "notes/a.txt", kind="group")
    assert labels(root) == []


def test_an_exact_move_takes_ids_only(root):
    """With exact, a name that only another file has is not enough, and neither is a group name."""
    place(root, "notes/a.txt", "hw")
    with pytest.raises(ValueError, match="no sorted file with the id 'a.txt'"):
        move_file(root, "a.txt", "logistics", exact=True)
    with pytest.raises(ValueError, match="no group with the id or name 'Logistics'"):
        move_file(root, "notes/a.txt", "Logistics", exact=True)
    assert labels(root) == []
    move_file(root, "notes/a.txt", "logistics", exact=True)
    assert state(root, "notes/a.txt") == ("Logistics", "logistics", "you", True)
    move_file(root, "notes/a.txt", None, exact=True)
    assert state(root, "notes/a.txt") == ("Unsorted", None, "you", True)


def test_suggest_stops_when_the_call_log_cannot_be_written(root):
    """A failure about the run, not about one file, ends the run as the spend cap does."""
    for file_id in ("a.txt", "b.txt", "c.txt"):
        frozen_in(root, file_id, "hw")

    class NoLog(FakeBackend):
        """A fake whose second call finds that the log cannot be written."""

        def sort(self, request):
            """Answer the first request, then fail the way JevBackend does for its log."""
            if self.calls:
                raise LogFailed("the call log cannot be written: disk full")
            return super().sort(request)

    backend = NoLog(default=TO_LOGISTICS)
    failed = {}
    found = suggest(root, backend, failed)
    assert [item.file_id for item in found] == ["a.txt"]
    assert failed == {"b.txt": "the call log cannot be written: disk full"}  # c.txt was not tried


# Group notes renamed or deleted by hand, and folders that are no group's


def two_groups(tmp_path):
    """A vault with Homework (hw1, hw2) and Exams (exam1, exam2); hw2 and exam2 are confirmed."""
    vault.init_vault(tmp_path)
    vault.add_group(tmp_path, "Homework", "homework specs and solutions")
    vault.add_group(tmp_path, "Exams", "exams and their solutions")
    for file_id, group_id in (
        ("hw1.txt", "homework"),
        ("hw2.txt", "homework"),
        ("exam1.txt", "exams"),
        ("exam2.txt", "exams"),
    ):
        place(tmp_path, file_id, group_id, set_version=2)
    freeze(tmp_path, "hw2.txt")
    freeze(tmp_path, "exam2.txt")
    return tmp_path


def where(root):
    """Each file's folder, group, who placed it and whether it is frozen."""
    return {note.placement.file_id: state(root, note.placement.file_id) for note in notes_of(root)}


def notes_of(root):
    """Every card note in the vault."""
    return vault.read_card_notes(root)


def kinds(root):
    """Every label line as (file id, kind, from group, to group)."""
    return [
        (line["file_id"], line["kind"], line["from_group"], line["to_group"])
        for line in labels(root)
    ]


CONFIRMED = [
    ("hw2.txt", "confirmed", "homework", "homework"),
    ("exam2.txt", "confirmed", "exams", "exams"),
]


def settled(root):
    """Check that the vault is at rest: a second sync finds nothing, and the board agrees."""
    report = sync(root)
    assert report == picked_up(set_version=report.set_version)
    board = build_board(root)
    on_board = {file.file_id: column.group.id for column in board.groups for file in column.files}
    on_board.update({file.file_id: None for file in board.unsorted})
    assert on_board == {note.placement.file_id: note.placement.group_id for note in notes_of(root)}
    for note in notes_of(root):
        assert note.path.parent == vault.folder_for(root, note.placement.group_id)
        assert note.placement.frozen == (note.placement.file_id in active_labels(root))


def test_one_group_note_renamed_by_hand(tmp_path):
    """Homework.md becomes Problem sets.md in Obsidian: the group's files follow, labels stay."""
    root = two_groups(tmp_path)
    (root / "_groups" / "Homework.md").rename(root / "_groups" / "Problem sets.md")
    report = sync(root)
    assert (report.rehomed, report.moved, report.orphaned) == (["hw1.txt", "hw2.txt"], [], [])
    assert where(root) == {
        "exam1.txt": ("Exams", "exams", "jev", False),
        "exam2.txt": ("Exams", "exams", "jev", True),
        "hw1.txt": ("Problem sets", "homework", "jev", False),
        "hw2.txt": ("Problem sets", "homework", "jev", True),
    }
    assert kinds(root) == CONFIRMED
    assert top_folders(root) == ["Exams", "Problem sets", "Unsorted"]
    settled(root)


@pytest.mark.parametrize(
    ("renames", "folders"),
    [
        ([("Exams", "Tests"), ("Homework", "Exams")], {"exams": "Tests", "homework": "Exams"}),
        (
            [("Homework", "Problem sets"), ("Exams", "Homework")],
            {"exams": "Homework", "homework": "Problem sets"},
        ),
        (
            [("Exams", "tmp"), ("Homework", "Exams"), ("tmp", "Homework")],
            {"exams": "Homework", "homework": "Exams"},
        ),
    ],
    ids=["a chain", "a chain, the other way", "a swap"],
)
def test_group_notes_renamed_by_hand_in_a_chain_or_a_swap(tmp_path, renames, folders):
    """One group takes the name another just gave up: each keeps its own files and labels.

    No folder is merged into another, so no file is read as moved by the user.
    """
    root = two_groups(tmp_path)
    for old, new in renames:
        (root / "_groups" / f"{old}.md").rename(root / "_groups" / f"{new}.md")
    report = sync(root)
    assert report.rehomed == ["exam1.txt", "exam2.txt", "hw1.txt", "hw2.txt"]
    assert (report.moved, report.orphaned, report.confirmed, report.unfrozen) == ([], [], [], [])
    assert where(root) == {
        "exam1.txt": (folders["exams"], "exams", "jev", False),
        "exam2.txt": (folders["exams"], "exams", "jev", True),
        "hw1.txt": (folders["homework"], "homework", "jev", False),
        "hw2.txt": (folders["homework"], "homework", "jev", True),
    }
    assert kinds(root) == CONFIRMED
    assert top_folders(root) == sorted([*folders.values(), "Unsorted"])
    settled(root)


def test_a_chain_with_one_rename_by_hand_and_one_by_command(tmp_path):
    """Exams.md is renamed in Obsidian, then `p0 group rename Homework Exams` takes the name."""
    root = two_groups(tmp_path)
    (root / "_groups" / "Exams.md").rename(root / "_groups" / "Tests.md")
    vault.rename_group(root, "Homework", "Exams")
    report = sync(root)
    assert (report.rehomed, report.moved) == (["exam1.txt", "exam2.txt"], [])
    assert where(root)["exam2.txt"] == ("Tests", "exams", "jev", True)
    assert where(root)["hw2.txt"] == ("Exams", "homework", "jev", True)
    assert kinds(root) == CONFIRMED
    settled(root)


@pytest.mark.parametrize("by_hand", [False, True], ids=["with commands", "by hand in Obsidian"])
def test_a_group_deleted_and_its_name_taken_by_a_rename_before_any_sync(tmp_path, by_hand):
    """Exams is deleted and Homework renamed to Exams. The old Exams files are not Homework's.

    They go to Unsorted and lose their labels, as after any delete. They are never read as
    files the user moved into the group that now has the name.
    """
    root = two_groups(tmp_path)
    if by_hand:
        (root / "_groups" / "Exams.md").unlink()
        (root / "_groups" / "Homework.md").rename(root / "_groups" / "Exams.md")
    else:
        vault.delete_group(root, "Exams")
        vault.rename_group(root, "Homework", "Exams")
    report = sync(root)
    assert (report.orphaned, report.moved) == (["exam1.txt", "exam2.txt"], [])
    assert report.rehomed == (["hw1.txt", "hw2.txt"] if by_hand else [])
    assert where(root) == {
        "exam1.txt": ("Unsorted", None, "code", False),
        "exam2.txt": ("Unsorted", None, "code", False),
        "hw1.txt": ("Exams", "homework", "jev", False),
        "hw2.txt": ("Exams", "homework", "jev", True),
    }
    assert kinds(root) == [*CONFIRMED, ("exam2.txt", "unfrozen", "exams", None)]
    assert top_folders(root) == ["Exams", "Unsorted"]
    settled(root)


def test_a_group_deleted_and_added_again_under_a_new_id(tmp_path):
    """HW was renamed to Homework, deleted, and Homework added again: a new group, a new id."""
    root = two_groups(tmp_path)  # the group Homework has the id homework
    vault.rename_group(root, "Homework", "Problem sets")
    vault.delete_group(root, "Problem sets")
    vault.add_group(root, "Problem sets", "problem sets")  # its id is problem-sets
    report = sync(root)
    assert (report.orphaned, report.moved, report.rehomed) == (["hw1.txt", "hw2.txt"], [], [])
    assert where(root)["hw2.txt"] == ("Unsorted", None, "code", False)
    assert kinds(root) == [*CONFIRMED, ("hw2.txt", "unfrozen", "homework", None)]
    assert top_folders(root) == ["Exams", "Problem sets", "Unsorted"]
    settled(root)


def test_a_group_id_edited_by_hand_under_the_same_name(tmp_path):
    """`id: homework` becomes `id: hw` in the note: to the tool, one group went and one came.

    The files of the old id go to Unsorted. They are not read as moved to the new id.
    """
    root = two_groups(tmp_path)
    note = root / "_groups" / "Homework.md"
    text = note.read_text(encoding="utf-8")
    assert "id: homework\n" in text
    note.write_text(text.replace("id: homework\n", "id: hw\n"), encoding="utf-8")
    report = sync(root)
    assert (report.orphaned, report.moved) == (["hw1.txt", "hw2.txt"], [])
    assert where(root)["hw1.txt"] == ("Unsorted", None, "code", False)
    assert where(root)["hw2.txt"] == ("Unsorted", None, "code", False)
    assert kinds(root) == [*CONFIRMED, ("hw2.txt", "unfrozen", "homework", None)]
    assert [group.id for group in vault.read_groups(root)] == ["exams", "hw"]
    settled(root)


def test_a_group_deleted_and_added_again_in_other_capitals(root):
    """Delete HW, add Hw: one folder on a Mac. Nothing loops, and no correction is undone."""
    place(root, "hw1.txt", "hw")
    place(root, "hw2.txt", "hw")
    freeze(root, "hw2.txt")
    vault.delete_group(root, "HW")
    vault.add_group(root, "Hw", "homework again")  # the id is hw again
    assert sync(root).rehomed == ["hw1.txt", "hw2.txt"]
    assert state(root, "hw1.txt") == ("Hw", "hw", "jev", False)
    assert state(root, "hw2.txt") == ("Hw", "hw", "jev", True)
    assert "Hw" in top_folders(root)
    assert "HW" not in top_folders(root)
    settled(root)

    # A sort places a file in it, and the next sync and sort leave it there.
    drop(root, "hw3.txt")
    to_hw = ({"hw": 0.9, "logistics": 0.05, "project": 0.05}, 0.95)
    sort_vault(root, FakeBackend(default=to_hw))
    assert state(root, "hw3.txt") == ("Hw", "hw", "jev", False)
    settled(root)
    move_file(root, "hw1.txt", "Hw")
    assert state(root, "hw1.txt") == ("Hw", "hw", "jev", True)
    settled(root)
    assert list(active_labels(root)) == ["hw2.txt", "hw1.txt"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("HW", "hw"),
        ("HW", "Hw"),
        ("HW", "Homework"),
        ("Übungen", "ÜBUNGEN"),
        ("Übungen", unicodedata.normalize("NFD", "Übungen")),
    ],
    ids=["small letters", "other capitals", "another name", "capitals, not ASCII", "NFD"],
)
def test_a_group_folder_renamed_by_hand(root, old, new):
    """The folder gets another name or spelling in the file explorer. The group note stays.

    Every file is still in its group afterwards, and no label is lost.
    """
    vault.add_group(root, "Übungen", "exercises")
    for group_id in ("hw", "übungen"):
        place(root, f"{group_id}1.txt", group_id)
        place(root, f"{group_id}2.txt", group_id)
        freeze(root, f"{group_id}2.txt")
    (root / old).rename(root / new)
    report = sync(root)
    assert (report.orphaned, report.moved, report.unfrozen, report.confirmed) == ([], [], [], [])
    for group_id, folder in (("hw", "HW"), ("übungen", "Übungen")):
        for number, frozen in ((1, False), (2, True)):
            on_disk, in_group, placed_by, is_frozen = state(root, f"{group_id}{number}.txt")
            assert (in_group, placed_by, is_frozen) == (group_id, "jev", frozen)
            assert vault.folder_key(on_disk) == vault.folder_key(folder)
    assert len(labels(root)) == 2
    assert sorted(active_labels(root)) == ["hw2.txt", "übungen2.txt"]
    settled_by_key(root)


def settled_by_key(root):
    """Like settled, for a vault whose folder may be spelled otherwise than its group."""
    report = sync(root)
    assert not any(
        [report.moved, report.confirmed, report.unfrozen, report.orphaned, report.rehomed]
    )
    board = build_board(root)
    on_board = {file.file_id: column.group.id for column in board.groups for file in column.files}
    assert on_board == {note.placement.file_id: note.placement.group_id for note in notes_of(root)}
    assert board.unsorted == []


def test_a_note_dragged_from_a_set_aside_folder_into_a_group_is_a_move(root):
    """The user rescues a file of a deleted group by hand: that is a move, as from any folder."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    vault.delete_group(root, "HW")
    assert note_of(root, "a.txt").folder == "HW (old)"
    drag(root, "a.txt", "Logistics")
    assert sync(root) == picked_up(moved=["a.txt"], orphaned=["b.txt"], set_version=4)
    assert state(root, "a.txt") == ("Logistics", "logistics", "you", True)
    assert not (root / "HW (old)").exists()


def test_a_set_aside_folder_goes_once_nothing_is_left_in_it(root):
    """Empty, or with only the Finder's .DS_Store: removed. With a file of the user's: kept."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "project")
    (root / "HW" / ".DS_Store").write_bytes(b"Finder")
    (root / "Project" / "my notes.md").write_text("Things to remember.\n", encoding="utf-8")
    (root / "Notes (old)").mkdir()  # a folder of the user's that happens to look like one
    (root / "Notes (old)" / "keep.md").write_text("mine\n", encoding="utf-8")
    vault.delete_group(root, "HW")
    vault.delete_group(root, "Project")
    assert top_folders(root) == [
        "HW (old)",
        "Logistics",
        "Notes (old)",
        "Project (old)",
        "Unsorted",
    ]
    assert sync(root).orphaned == ["a.txt", "b.txt"]
    assert top_folders(root) == ["Logistics", "Notes (old)", "Project (old)", "Unsorted"]
    assert [path.name for path in (root / "Project (old)").iterdir()] == ["my notes.md"]
    assert (root / "Notes (old)" / "keep.md").read_text(encoding="utf-8") == "mine\n"


def test_a_note_put_back_in_its_group_has_its_frozen_flag_checked_in_the_same_sync(root):
    """Rehomed and frozen by hand: confirmed at once, so the next sync has nothing left to do."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    freeze(root, "b.txt")
    set_frozen_by_hand(root, "a.txt")
    set_frozen_by_hand(root, "b.txt", False)
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Homework.md")
    assert sync(root) == picked_up(
        rehomed=["a.txt", "b.txt"], confirmed=["a.txt"], unfrozen=["b.txt"], set_version=4
    )
    assert state(root, "a.txt") == ("Homework", "hw", "jev", True)
    assert state(root, "b.txt") == ("Homework", "hw", "jev", False)
    assert labels(root)[1:] == [
        label("a.txt", "confirmed", "hw", "hw", set_version=4),
        label("b.txt", "unfrozen", "hw", "hw", set_version=4),
    ]
    assert sync(root) == picked_up(set_version=4)


def test_a_confirmed_file_in_unsorted_dragged_into_a_folder_that_is_no_group_goes_back(root):
    """Its own place is Unsorted, which is always there: it is put back, label and all."""
    place(root, "a.txt", "hw")
    move_file(root, "a.txt", "Unsorted")
    drag(root, "a.txt", "Old stuff")
    assert sync(root) == picked_up(rehomed=["a.txt"])
    assert state(root, "a.txt") == ("Unsorted", None, "you", True)
    assert labels(root) == [label("a.txt", "moved", "hw", None)]


# Card notes that Obsidian touched


def add_property(root, file_id, line="tags: [review]"):
    """Add one frontmatter key to a file's card note by hand."""
    path = note_of(root, file_id).path
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("\n---\n", f"\n{line}\n---\n", 1), encoding="utf-8")


def test_a_property_added_by_hand_survives_every_correction(root):
    """Freeze, unfreeze, move and a drag picked up by sync all keep the user's own keys."""
    place(root, "a.txt", "hw")
    add_property(root, "a.txt")
    assert sync(root) == picked_up()
    freeze(root, "a.txt")
    assert note_of(root, "a.txt").extras == {"tags": ["review"]}
    unfreeze(root, "a.txt")
    assert note_of(root, "a.txt").extras == {"tags": ["review"]}
    move_file(root, "a.txt", "Logistics")
    assert note_of(root, "a.txt").extras == {"tags": ["review"]}
    drag(root, "a.txt", "Project")
    assert sync(root).moved == ["a.txt"]
    note = note_of(root, "a.txt")
    assert (note.folder, note.placement.group_id, note.extras) == (
        "Project",
        "project",
        {"tags": ["review"]},
    )
    frontmatter, _ = vault.read_note(note.path)
    assert list(frontmatter) == [*Placement.model_fields, "tags"]
    assert [line["kind"] for line in labels(root)] == ["confirmed", "unfrozen", "moved", "moved"]


def test_a_note_filed_into_a_folder_inside_its_group_is_left_there_by_sync(root):
    """HW/week 3 is HW: nothing to pick up. A freeze rewrites the note where it is."""
    place(root, "a.txt", "hw")
    path = note_of(root, "a.txt").path
    (root / "HW" / "week 3").mkdir()
    nested = path.rename(root / "HW" / "week 3" / path.name)
    assert sync(root) == picked_up()
    assert freeze(root, "HW") == ["a.txt"]
    note = note_of(root, "a.txt")
    assert (note.path, note.folder, note.placement.frozen) == (nested, "HW", True)
    # Filed into another group's folder, it is a drag like any other.
    (root / "Logistics" / "later").mkdir()
    nested.rename(root / "Logistics" / "later" / nested.name)
    assert sync(root).moved == ["a.txt"]
    assert note_of(root, "a.txt").path == root / "Logistics" / "a.txt.md"


# A freeze given after the original changed


def changed_original(root, file_id, group_id="hw"):
    """Place a file, then replace its original, as an edit after the sort does."""
    place(root, file_id, group_id)
    path = drop(root, file_id, "# Syllabus\n\nGrading, deadlines and office hours.\n")
    return hash_file(path), render_card(build_card(path, file_id, Config.load(root)))


def test_a_move_after_the_original_changed_covers_the_new_content(root):
    """The note takes the hash and card of the file as it is now, so the next sort skips it."""
    new_hash, new_card = changed_original(root, "sheet.txt", "logistics")
    old_hash = note_of(root, "sheet.txt").placement.content_hash
    placement = move_file(root, "sheet.txt", "HW")
    note = note_of(root, "sheet.txt")
    assert note.placement == placement
    assert (placement.content_hash, placement.file_type) == (new_hash, "Text")
    assert placement.content_hash != old_hash
    assert note.body == new_card
    assert (placement.probs, placement.set_version, placement.model) == (PROBS, 3, "fake-0")

    backend = FakeBackend(default=TO_LOGISTICS)
    report = sort_vault(root, backend)
    assert (report.frozen_skipped, backend.calls) == (1, [])
    assert state(root, "sheet.txt") == ("HW", "hw", "you", True)
    assert labels(root) == [label("sheet.txt", "moved", "logistics", "hw")]

    # A change after the move still takes the freeze off.
    drop(root, "sheet.txt", "Homework 9: something new.\n")
    report = sort_vault(root, FakeBackend(default=TO_LOGISTICS))
    assert (report.frozen_skipped, report.asked) == (0, 1)
    assert state(root, "sheet.txt") == ("Logistics", "logistics", "jev", False)
    assert labels(root)[1:] == [label("sheet.txt", "unfrozen", "hw", "hw")]


def test_a_freeze_after_the_original_changed_covers_the_new_content(root):
    """By command for a file and for a group, and for a move to the group the file is in."""
    new = {file_id: changed_original(root, file_id) for file_id in ("a.txt", "b.txt", "c.txt")}
    place(root, "same.txt", "hw")
    untouched = note_of(root, "same.txt").path.read_bytes()
    assert freeze(root, "a.txt") == ["a.txt"]
    assert move_file(root, "b.txt", "HW").content_hash == new["b.txt"][0]
    assert freeze(root, "HW") == ["c.txt", "same.txt"]
    for file_id, (new_hash, new_card) in new.items():
        note = note_of(root, file_id)
        assert (note.placement.content_hash, note.body) == (new_hash, new_card)
        assert (note.placement.frozen, note.placement.placed_by) == (True, "jev")
    # A file that did not change gets its flag and nothing else.
    assert note_of(root, "same.txt").path.read_bytes() == untouched.replace(
        b"frozen: false", b"frozen: true"
    )
    backend = FakeBackend(default=TO_LOGISTICS)
    assert sort_vault(root, backend).frozen_skipped == 4
    assert backend.calls == []
    assert [line["kind"] for line in labels(root)] == ["confirmed"] * 4


def test_a_drag_or_a_flag_set_by_hand_after_the_original_changed_covers_it_too(root):
    """Sync rules 2 and 3 freeze a note as the commands do: for the file as it is now."""
    dragged_hash, dragged_card = changed_original(root, "dragged.txt")
    frozen_hash, frozen_card = changed_original(root, "frozen.txt")
    drag(root, "dragged.txt", "Logistics")
    set_frozen_by_hand(root, "frozen.txt")
    assert sync(root) == picked_up(moved=["dragged.txt"], confirmed=["frozen.txt"])
    dragged, frozen = note_of(root, "dragged.txt"), note_of(root, "frozen.txt")
    assert (dragged.placement.content_hash, dragged.body) == (dragged_hash, dragged_card)
    assert (frozen.placement.content_hash, frozen.body) == (frozen_hash, frozen_card)
    assert state(root, "frozen.txt") == ("HW", "hw", "jev", True)
    assert sync(root) == picked_up()
    backend = FakeBackend(default=TO_HW)
    assert sort_vault(root, backend).frozen_skipped == 2
    assert backend.calls == []


def test_a_freeze_keeps_the_notes_values_when_the_original_cannot_be_read(root, monkeypatch):
    """No hash can be taken, so the note keeps the one it has. The freeze is still given."""
    place(root, "a.txt", "hw")
    drop(root, "a.txt", "New content.\n")
    old = note_of(root, "a.txt")

    def unreadable(path):
        """Stand in for hash_file: the original cannot be opened."""
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr("decide_already.corrections.hash_file", unreadable)
    assert freeze(root, "a.txt") == ["a.txt"]
    note = note_of(root, "a.txt")
    assert note.placement == old.placement.model_copy(update={"frozen": True})
    assert note.body == old.body
    assert labels(root) == [label("a.txt", "confirmed", "hw", "hw")]


def test_notes_filed_into_a_folder_of_a_group_renamed_by_hand_come_along(root):
    """HW/week 3 held a note when HW.md was renamed: the note goes to the new folder too.

    The set-aside folder is removed once only the empty week 3 folder is left in it.
    """
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    freeze(root, "b.txt")
    (root / "HW" / "week 3").mkdir()
    for file_id in ("a.txt", "b.txt"):
        path = note_of(root, file_id).path
        path.rename(root / "HW" / "week 3" / path.name)
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Homework.md")
    assert sync(root) == picked_up(rehomed=["a.txt", "b.txt"], set_version=4)
    assert note_of(root, "a.txt").path == root / "Homework" / "a.txt.md"
    assert state(root, "b.txt") == ("Homework", "hw", "jev", True)
    assert top_folders(root) == ["Homework", "Logistics", "Project", "Unsorted"]
    assert labels(root) == [label("b.txt", "confirmed", "hw", "hw")]


def test_unfreezing_a_labeled_file_without_a_note_goes_by_its_exact_id(root):
    """Another sorted file has the same name: the id that was typed is the one that counts."""
    place(root, "notes.txt", "hw")
    place(root, "week1/notes.txt", "hw")
    place(root, "Logistics", "hw")  # a file named like a group
    for file_id in ("notes.txt", "week1/notes.txt", "Logistics"):
        freeze(root, file_id, kind="file")
    note_of(root, "notes.txt").path.unlink()
    note_of(root, "Logistics").path.unlink()

    assert unfreeze(root, "notes.txt") == ["notes.txt"]
    assert state(root, "week1/notes.txt") == ("HW", "hw", "jev", True)  # not this one
    assert labels(root)[3:] == [
        label("notes.txt", "unfrozen", "hw", "hw", jev_probs=None, jev_set_version=None)
    ]
    # On the command line a group wins over a file of the same name, as for any unfreeze.
    assert unfreeze(root, "Logistics") == []
    assert unfreeze(root, "Logistics", kind="group") == []
    assert unfreeze(root, "Logistics", kind="file") == ["Logistics"]
    assert sorted(active_labels(root)) == ["week1/notes.txt"]
    assert unfreeze(root, "week1/notes.txt") == ["week1/notes.txt"]
    assert active_labels(root) == {}


def test_deleting_a_corrected_file_withdraws_its_label_so_a_new_one_is_sorted(root):
    """Replacing a file is delete, then add: the new file must not inherit the old correction."""
    place(root, "hw3.txt", "hw")
    move_file(root, "hw3.txt", "Logistics")
    assert delete_file(root, "hw3.txt") == 1
    assert labels(root) == [
        label("hw3.txt", "moved", "hw", "logistics"),
        label(
            "hw3.txt", "unfrozen", "logistics", "logistics", jev_probs=None, jev_set_version=None
        ),
    ]
    assert active_labels(root) == {}

    drop(root, "hw3.txt", "Homework 3, the new version.\n")
    report = sort_vault(root, FakeBackend(default=TO_HW))
    assert report.errors == {}
    assert state(root, "hw3.txt") == ("HW", "hw", "jev", False)


def test_deleting_a_file_with_no_label_writes_none(root):
    """A file nobody corrected has nothing to withdraw."""
    place(root, "a.txt", "hw")
    assert delete_file(root, "a.txt") == 1
    assert labels(root) == []
