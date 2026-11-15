"""board.py: the vault read into groups, Unsorted and the files not sorted yet."""

import pytest

from decide_already import vault
from decide_already.board import Board, BoardFile, build_board
from decide_already.sort import sort_vault
from decide_already.types import Placement
from fakes import FakeBackend


@pytest.fixture
def root(tmp_path):
    """A vault with two groups (HW, Logistics), at set version 2, and an empty dump."""
    vault.init_vault(tmp_path)
    vault.add_group(tmp_path, "HW", "homework specs, starter code and solutions")
    vault.add_group(tmp_path, "Logistics", "syllabus, schedule and deadlines")
    return tmp_path


def drop(root, file_id, text="Homework 3: implement the kernel functions.\n"):
    """Put a file in the dump folder and return its path."""
    path = root / "_dump" / file_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def note(root, file_id, group_id="hw", **changes):
    """Write a card note for a file, as the sorter would, with any placement fields changed."""
    fields = {
        "source": f"[[_dump/{file_id}]]",
        "file_id": file_id,
        "content_hash": "0" * 64,
        "file_type": "Text",
        "group_id": group_id,
        "placed_by": "jev",
        "set_version": 2,
        "probs": {"hw": 0.75, "logistics": 0.25},
        "fits_any": 0.9,
        "model": "fake-0",
    }
    return vault.write_card_note(root, Placement(**{**fields, **changes}), "a card\n")


def names(files):
    """The names of a list of board files, in order."""
    return [file.name for file in files]


def test_files_are_grouped_by_the_folder_their_note_sits_in(root):
    """Each group lists the notes in its folder; Unsorted lists the ones in Unsorted."""
    note(root, "hw3/hw3_solution.py", "hw", file_type="Python code")
    note(root, "syllabus.md", "logistics", file_type="Markdown")
    note(root, "checklist.txt", None, reason="no_group_fits")
    board = build_board(root)
    assert board.set_version == 2
    assert [column.group.id for column in board.groups] == ["hw", "logistics"]
    assert [column.group.name for column in board.groups] == ["HW", "Logistics"]
    assert names(board.groups[0].files) == ["hw3_solution.py"]
    assert names(board.groups[1].files) == ["syllabus.md"]
    assert names(board.unsorted) == ["checklist.txt"]
    assert board.groups[0].files[0] == BoardFile(
        file_id="hw3/hw3_solution.py",
        name="hw3_solution.py",
        file_type="Python code",
        group_id="hw",
        frozen=False,
        close_call=False,
        placed_by="jev",
        reason=None,
        top_prob=0.75,
        probs={"hw": 0.75, "logistics": 0.25},
        fits_any=0.9,
        stale=False,
    )
    assert (board.unsorted[0].group_id, board.unsorted[0].reason) == (None, "no_group_fits")


def test_a_group_with_no_files_is_still_on_the_board(root):
    """Every current group gets a column, empty or not, in order of group id."""
    vault.add_group(root, "Exams", "midterms and finals")
    note(root, "a.txt", "logistics", set_version=3)
    board = build_board(root)
    assert [(column.group.id, names(column.files)) for column in board.groups] == [
        ("exams", []),
        ("hw", []),
        ("logistics", ["a.txt"]),
    ]


def test_a_note_dragged_to_another_folder_is_listed_there(root):
    """The folder is the file's current group, even before sync has recorded the move."""
    path = note(root, "a.txt", "hw")
    path.rename(root / "Logistics" / path.name)
    unsorted_path = note(root, "b.txt", "hw")
    unsorted_path.rename(root / "Unsorted" / unsorted_path.name)
    board = build_board(root)
    assert board.groups[0].files == []
    (moved,) = board.groups[1].files
    assert (moved.file_id, moved.group_id) == ("a.txt", "logistics")
    (to_unsorted,) = board.unsorted
    assert (to_unsorted.file_id, to_unsorted.group_id) == ("b.txt", None)


def test_close_calls_come_first_then_files_by_name(root):
    """Within a column: the flagged files, then the rest, each part ordered by name."""
    note(root, "c.txt")
    note(root, "Beta.txt")
    note(root, "week2/a.txt")
    note(root, "week1/a.txt")
    note(root, "z_close.txt", close_call=True)
    note(root, "m_close.txt", close_call=True)
    note(root, "b.txt", None)
    note(root, "z/a.txt", None)
    board = build_board(root)
    assert [file.file_id for file in board.groups[0].files] == [
        "m_close.txt",
        "z_close.txt",
        "week1/a.txt",  # the same name twice: ordered by the full id
        "week2/a.txt",
        "Beta.txt",  # capitals do not sort ahead of small letters
        "c.txt",
    ]
    # Unsorted is ordered the same way: by name, not by the folder the file came from.
    assert [file.file_id for file in board.unsorted] == ["z/a.txt", "b.txt"]


def test_stale_marks_a_file_sorted_under_an_older_set(root):
    """A note written under set version 1 is stale once the current version is 2."""
    note(root, "old.txt", set_version=1)
    note(root, "current.txt", set_version=2)
    stale = {file.file_id: file.stale for file in build_board(root).groups[0].files}
    assert stale == {"old.txt": True, "current.txt": False}


def test_a_new_set_version_makes_every_earlier_note_stale(root):
    """Adding or editing a group bumps the version; a rerun brings a file up to date."""
    drop(root, "a.txt")
    drop(root, "b.txt")
    answer = ({"hw": 0.9, "logistics": 0.1}, 0.95)
    sort_vault(root, FakeBackend(default=answer))
    assert [file.stale for file in build_board(root).groups[0].files] == [False, False]

    vault.edit_group(root, "HW", covers="problem sets and their solutions")
    board = build_board(root)
    assert board.set_version == 3
    assert [file.stale for file in board.groups[0].files] == [True, True]

    sort_vault(root, FakeBackend(default=answer, fail={"b.txt"}))
    stale = {file.file_id: file.stale for file in build_board(root).groups[0].files}
    assert stale == {"a.txt": False, "b.txt": True}


def test_a_group_note_edited_by_hand_counts_as_a_new_set(root):
    """The board shows the set as it is now, so a hand edit makes the earlier notes stale."""
    note(root, "a.txt", set_version=2)
    group_note = root / "_groups" / "HW.md"
    text = group_note.read_text(encoding="utf-8").replace("homework specs", "problem sets")
    group_note.write_text(text, encoding="utf-8")
    board = build_board(root)
    assert board.set_version == 3
    assert board.groups[0].group.covers.startswith("problem sets")
    assert board.groups[0].files[0].stale is True


def test_new_files_are_the_dump_files_without_a_card_note(root):
    """Files dropped since the last sort are listed by id; hidden files are not."""
    drop(root, "sorted.txt")
    drop(root, "hw4/new.py")
    drop(root, "also_new.txt")
    drop(root, ".DS_Store")
    note(root, "sorted.txt")
    note(root, "gone_from_the_dump.txt")
    board = build_board(root)
    assert board.new_files == ["also_new.txt", "hw4/new.py"]
    assert names(board.groups[0].files) == ["gone_from_the_dump.txt", "sorted.txt"]


def test_top_prob_is_the_models_highest_probability(root):
    """The top probability is the highest one, or None when no model was asked."""
    note(root, "a.txt", probs={"hw": 0.2, "logistics": 0.8})
    note(root, "no_pick.txt", None, probs={}, fits_any=0.0, reason="no_group_fits")
    note(
        root,
        "scan.pdf",
        None,
        placed_by="code",
        reason="no_text",
        probs=None,
        fits_any=None,
        model=None,
    )
    board = build_board(root)
    assert board.groups[0].files[0].top_prob == 0.8
    by_id = {file.file_id: file for file in board.unsorted}
    assert (by_id["no_pick.txt"].top_prob, by_id["no_pick.txt"].probs) == (None, {})
    scan = by_id["scan.pdf"]
    assert (scan.top_prob, scan.probs, scan.fits_any) == (None, None, None)
    assert (scan.reason, scan.placed_by) == ("no_text", "code")


def test_frozen_and_moved_files_carry_their_marks(root):
    """What the user did to a file shows on the board."""
    note(root, "moved.txt", "logistics", placed_by="you", frozen=True)
    note(root, "confirmed.txt", "hw", frozen=True)
    board = build_board(root)
    (confirmed,) = board.groups[0].files
    (moved,) = board.groups[1].files
    assert (confirmed.frozen, confirmed.placed_by) == (True, "jev")
    assert (moved.frozen, moved.placed_by) == (True, "you")


def test_a_note_in_the_folder_of_a_deleted_group_is_listed_under_unsorted(root):
    """A deleted group has no column. Its notes show under Unsorted, where sync will move them."""
    note(root, "a.txt", "hw")
    vault.delete_group(root, "HW")
    board = build_board(root)
    assert board.set_version == 3
    assert [column.group.id for column in board.groups] == ["logistics"]
    (left_behind,) = board.unsorted
    assert (left_behind.file_id, left_behind.group_id, left_behind.stale) == ("a.txt", None, True)


def test_the_users_own_notes_are_not_on_the_board(root):
    """Only card notes are files on the board."""
    (root / "HW" / "my notes.md").write_text("Things to remember.\n", encoding="utf-8")
    (root / "Unsorted" / "todo.md").write_text("---\ntags: [todo]\n---\n", encoding="utf-8")
    board = build_board(root)
    assert board.groups[0].files == []
    assert board.unsorted == []


def test_the_board_of_a_vault_with_no_groups(tmp_path):
    """Before any group exists: version 0, no columns, and every dump file is new."""
    vault.init_vault(tmp_path)
    drop(tmp_path, "a.txt")
    assert build_board(tmp_path) == Board(
        set_version=0, groups=[], unsorted=[], new_files=["a.txt"]
    )


def test_building_the_board_changes_no_note_and_no_original(root):
    """Reading is all it does: every file in the vault is the same afterwards."""
    drop(root, "a.txt")
    drop(root, "b.txt")
    sort_vault(root, FakeBackend(default=({"hw": 0.9, "logistics": 0.1}, 0.95)))
    drop(root, "c.txt")

    def snapshot():
        """Every file in the vault with its bytes."""
        return {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}

    before = snapshot()
    build_board(root)
    assert snapshot() == before


def test_the_board_can_be_sent_as_json(root):
    """The UI gets the board as JSON, so every part of it must dump cleanly."""
    note(root, "a.txt", close_call=True)
    note(root, "b.txt", None, reason="weak_top")
    drop(root, "c.txt")
    data = build_board(root).model_dump(mode="json")
    assert Board.model_validate(data) == build_board(root)
    assert data["groups"][0]["files"][0]["file_id"] == "a.txt"
    assert data["new_files"] == ["c.txt"]
