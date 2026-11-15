"""Card notes in vault.py: the dump listing, note names, and reading and writing the notes."""

import unicodedata

import pytest

from decide_already import vault
from decide_already.config import Config
from decide_already.types import Placement


@pytest.fixture
def root(tmp_path):
    """A vault with two groups: HW (id hw) and Logistics (id logistics)."""
    vault.init_vault(tmp_path)
    vault.add_group(tmp_path, "HW", "homework specs, starter code and solutions")
    vault.add_group(tmp_path, "Logistics", "syllabus, schedule and deadlines")
    return tmp_path


def drop(root, file_id, text="some text\n"):
    """Put a file in the dump folder and return its path."""
    path = root / "_dump" / file_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def placement(file_id, group_id="hw", **changes):
    """A Placement as the sorter writes one, with any fields changed."""
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
    return Placement(**{**fields, **changes})


def write_by_hand(path, note_placement, body="a card\n"):
    """Put a card note at any path, the way a person could by copying a file there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vault.write_note(path, note_placement.model_dump(mode="json"), body)
    return path


def ids(notes):
    """The file ids of a list of card notes, in order."""
    return [note.placement.file_id for note in notes]


# dump_files


def test_dump_files_lists_nested_files_sorted_by_id(root):
    """Every file under the dump folder is listed, with its path from there as the id."""
    for file_id in ("syllabus.md", "hw3/hw3_solution.py", "hw3/data/train.csv", "b.txt"):
        drop(root, file_id)
    (root / "_dump" / "empty folder").mkdir()
    listed = vault.dump_files(root, Config.load(root))
    assert [file_id for file_id, _ in listed] == [
        "b.txt",
        "hw3/data/train.csv",
        "hw3/hw3_solution.py",
        "syllabus.md",
    ]
    assert [path for _, path in listed] == [root / "_dump" / file_id for file_id, _ in listed]


def test_dump_files_skips_hidden_files_and_folders(root):
    """A name that starts with a dot hides the file, or the whole folder under it."""
    for file_id in (
        ".DS_Store",
        ".git/config",
        "hw3/.ipynb_checkpoints/hw3-checkpoint.ipynb",
        "hw3/.hidden.txt",
        "hw3/.cache/deep/down/file.txt",
    ):
        drop(root, file_id)
    drop(root, "hw3/a.b.txt")  # a dot inside a name hides nothing
    drop(root, "hw3/notes")  # no extension at all
    listed = vault.dump_files(root, Config.load(root))
    assert [file_id for file_id, _ in listed] == ["hw3/a.b.txt", "hw3/notes"]


def test_dump_files_uses_the_dump_folder_from_the_config(tmp_path):
    """The dump folder is the one the vault was set up with."""
    (tmp_path / "inbox").mkdir()
    config = vault.init_vault(tmp_path, "inbox")
    (tmp_path / "inbox" / "a.txt").write_text("a", encoding="utf-8")
    assert vault.dump_files(tmp_path, config) == [("a.txt", tmp_path / "inbox" / "a.txt")]


def test_dump_files_of_an_empty_or_missing_dump_folder_is_empty(root):
    """Nothing in the dump, or no dump folder at all, is an empty list, not an error."""
    config = Config.load(root)
    assert vault.dump_files(root, config) == []
    (root / "_dump").rmdir()
    assert vault.dump_files(root, config) == []


# note_name and source_link


def test_note_name_writes_slashes_as_double_underscores():
    """The note is named after the whole path under the dump folder."""
    assert vault.note_name("syllabus.md") == "syllabus.md.md"
    assert vault.note_name("hw3/hw3_solution.py") == "hw3__hw3_solution.py.md"
    assert vault.note_name("6.867/assignments/hw1/hw1.pdf") == "6.867__assignments__hw1__hw1.pdf.md"


def test_same_file_name_in_two_folders_gives_two_notes(root):
    """README.md in hw1/ and in hw2/ get their own notes, side by side in one group folder."""
    assert vault.note_name("hw1/README.md") != vault.note_name("hw2/README.md")
    first = vault.write_card_note(root, placement("hw1/README.md"), "first\n")
    second = vault.write_card_note(root, placement("hw2/README.md"), "second\n")
    assert first == root / "HW" / "hw1__README.md.md"
    assert second == root / "HW" / "hw2__README.md.md"
    notes = vault.read_card_notes(root)
    assert ids(notes) == ["hw1/README.md", "hw2/README.md"]
    assert [note.body for note in notes] == ["first\n", "second\n"]


def test_source_link_points_into_the_dump_folder():
    """The link is an Obsidian link to the original, under whatever the dump folder is called."""
    assert vault.source_link("hw3/hw3_solution.py", Config()) == "[[_dump/hw3/hw3_solution.py]]"
    assert vault.source_link("a.txt", Config(dump_dir="inbox")) == "[[inbox/a.txt]]"


# The round trip through a note


def test_a_placement_survives_the_round_trip_through_a_note(root):
    """What is written is what is read back: every field, and a body with Unicode in it."""
    body = "# `과제 3.py`\n\n- Type: Python code\n\n```text\nnaïve café, 한글, 日本語, 😀\n```\n"
    written = placement(
        "hw3/과제 3.py",
        file_type="Python code",
        close_call=True,
        probs={"hw": 0.5625, "logistics": 0.4375},
        fits_any=0.875,
        variant="window_vote",
    )
    path = vault.write_card_note(root, written, body)
    assert path == root / "HW" / "hw3__과제 3.py.md"
    (note,) = vault.read_card_notes(root)
    assert note.placement == written
    assert note.body == body
    assert note.path == path
    assert note.folder == "HW"
    # Written as UTF-8, with the characters themselves and not escapes of them.
    text = path.read_bytes().decode("utf-8")
    assert "file_id: hw3/과제 3.py" in text
    assert "한글, 日本語, 😀" in text


def test_the_frontmatter_is_exactly_a_placement_in_field_order(root):
    """Every Placement field is in the frontmatter, in the order of the model, and nothing else."""
    path = vault.write_card_note(root, placement("a.txt"), "a card\n")
    frontmatter, body = vault.read_note(path)
    assert list(frontmatter) == list(Placement.model_fields)
    assert frontmatter == placement("a.txt").model_dump(mode="json")
    assert body == "a card\n"
    assert path.read_text(encoding="utf-8").startswith("---\nsource: '[[_dump/a.txt]]'\nfile_id: ")


def test_an_unsorted_placement_without_an_answer_survives_the_round_trip(root):
    """A file with no text has no probabilities and no model: the empty fields come back empty."""
    written = placement(
        "scan.pdf",
        group_id=None,
        placed_by="code",
        reason="no_text",
        probs=None,
        fits_any=None,
        model=None,
    )
    path = vault.write_card_note(root, written, "No text could be extracted from this file.\n")
    assert path == root / "Unsorted" / "scan.pdf.md"
    (note,) = vault.read_card_notes(root)
    assert note.placement == written
    assert note.folder == "Unsorted"


@pytest.mark.parametrize(
    "file_id",
    ["123", "true", "null", "2026-11-15", "notes/#1 - [draft] 'final'.txt", "a b/c d.txt", "- x"],
)
def test_file_ids_that_look_like_yaml_survive_the_round_trip(root, file_id):
    """A file name that YAML could read as a number, a date or a list still comes back as text."""
    written = placement(file_id, probs={"hw": 1e-05, "logistics": 0.99999}, content_hash="1" * 64)
    vault.write_card_note(root, written, "a card\n")
    (note,) = vault.read_card_notes(root)
    assert note.placement == written


def test_a_long_file_id_survives_the_round_trip(root):
    """YAML folds long values over several lines; the id still comes back in one piece."""
    file_id = "week 1/" + " ".join(["a long file name with many words"] * 6) + ".txt"
    written = placement(file_id)
    vault.write_card_note(root, written, "---\nnot frontmatter, just a body that starts oddly\n")
    (note,) = vault.read_card_notes(root)
    assert note.placement == written
    assert note.body == "---\nnot frontmatter, just a body that starts oddly\n"


# read_card_notes


def test_read_card_notes_ignores_the_users_own_notes(root):
    """A markdown file whose frontmatter holds no Placement is left out, and left alone."""
    card = vault.write_card_note(root, placement("a.txt"), "a card\n")
    card_text = card.read_text(encoding="utf-8")
    own = {
        "plain.md": "Just my own notes about the homework.\n",
        "tagged.md": "---\ntags: [todo]\n---\nA note with other frontmatter.\n",
        "broken.md": "---\nfrozen: [unclosed\n---\nFrontmatter that is not YAML.\n",
        "list.md": "---\n- one\n- two\n---\nFrontmatter that is a list.\n",
        "word.md": "---\njust a word\n---\nFrontmatter that is a string.\n",
        "zero.md": "---\n0\n---\nFrontmatter that is a number.\n",
        "missing field.md": card_text.replace("file_id: a.txt\n", ""),
        "wrong type.md": card_text.replace("set_version: 2", "set_version: two"),
        "wrong value.md": card_text.replace("placed_by: jev", "placed_by: me"),
    }
    for name, text in own.items():
        (root / "HW" / name).write_text(text, encoding="utf-8")
    (root / "HW" / "latin1.md").write_bytes("---\ntitle: caf\xe9\n---\n".encode("latin-1"))
    (root / "HW" / "a folder.md").mkdir()
    assert ids(vault.read_card_notes(root)) == ["a.txt"]
    for name, text in own.items():
        assert (root / "HW" / name).read_text(encoding="utf-8") == text


def test_read_card_notes_looks_in_the_visible_folders_all_the_way_down(root):
    """Group folders, Unsorted and any other visible folder are read, with the folders in them.

    A note filed into a folder inside its group's folder is still in that group: its folder
    is the top-level one.
    """
    seen = {
        "HW/in-hw.md": "in-hw",
        "HW/week 1/nested.md": "nested",
        "HW/week 1/drafts/nested-twice.md": "nested-twice",
        "Old group/left-behind.md": "left-behind",
        "Unsorted/in-unsorted.md": "in-unsorted",
        "Unsorted/later/unsorted-nested.md": "unsorted-nested",
    }
    unseen = {
        "_dump/in-dump.md": "in-dump",
        "_dump/week 1/in-dump-nested.md": "in-dump-nested",
        "_groups/in-groups.md": "in-groups",
        ".decide/in-decide.md": "in-decide",
        ".obsidian/in-obsidian.md": "in-obsidian",
        ".trash/in-trash.md": "in-trash",
        "at-the-top.md": "at-the-top",
        "HW/.hidden/in-a-hidden-folder.md": "in-a-hidden-folder",
        "HW/not-markdown.txt": "not-markdown",
    }
    for where, file_id in {**seen, **unseen}.items():
        write_by_hand(root / where, placement(file_id))
    notes = vault.read_card_notes(root)
    assert sorted(ids(notes)) == sorted(seen.values())
    by_id = {note.placement.file_id: note for note in notes}
    assert {file_id: by_id[file_id].path for file_id in by_id} == {
        file_id: root / where for where, file_id in seen.items()
    }
    assert {file_id: note.folder for file_id, note in by_id.items()} == {
        file_id: where.split("/")[0] for where, file_id in seen.items()
    }


def test_read_card_notes_leaves_out_the_custom_dump_folder(tmp_path):
    """The dump folder is skipped under whatever name the config gives it."""
    (tmp_path / "inbox").mkdir()
    vault.init_vault(tmp_path, "inbox")
    write_by_hand(tmp_path / "inbox" / "a.md", placement("in-the-dump"))
    write_by_hand(tmp_path / "_dump" / "b.md", placement("in-a-folder-that-is-not-the-dump"))
    assert ids(vault.read_card_notes(tmp_path)) == ["in-a-folder-that-is-not-the-dump"]


def test_read_card_notes_leaves_out_the_folder_a_nested_dump_sits_in(tmp_path):
    """With the dump at course/files, nothing under course is read: no group can be there."""
    (tmp_path / "course" / "files").mkdir(parents=True)
    vault.init_vault(tmp_path, "course/files/")
    write_by_hand(tmp_path / "course" / "files" / "a.md", placement("an-original"))
    write_by_hand(tmp_path / "course" / "b.md", placement("next-to-the-dump"))
    write_by_hand(tmp_path / "Unsorted" / "c.md", placement("a-card-note", group_id=None))
    assert ids(vault.read_card_notes(tmp_path)) == ["a-card-note"]


def test_read_card_notes_is_sorted_by_file_id(root):
    """The order is by file id, whatever folder each note is in."""
    vault.write_card_note(root, placement("c.txt", group_id=None), "")
    vault.write_card_note(root, placement("a.txt", group_id="logistics"), "")
    vault.write_card_note(root, placement("b.txt", group_id="hw"), "")
    assert ids(vault.read_card_notes(root)) == ["a.txt", "b.txt", "c.txt"]


def test_a_vault_without_card_notes_has_none(root):
    """Before the first sort there is nothing to read."""
    assert vault.read_card_notes(root) == []


# folder_for and group_id_of_folder


def test_folder_for_is_the_groups_folder_or_unsorted(root):
    """A group id maps to the folder named after the group; None maps to Unsorted."""
    assert vault.folder_for(root, "hw") == root / "HW"
    assert vault.folder_for(root, "logistics") == root / "Logistics"
    assert vault.folder_for(root, None) == root / "Unsorted"
    with pytest.raises(KeyError):
        vault.folder_for(root, "project")


def test_group_id_of_folder_is_the_reverse(root):
    """A folder name maps back to its group's id; Unsorted maps to None."""
    assert vault.group_id_of_folder(root, "HW") == "hw"
    assert vault.group_id_of_folder(root, "Logistics") == "logistics"
    assert vault.group_id_of_folder(root, "Unsorted") is None
    for folder in ("Old group", "HW (old)", "homework", "_dump", "_groups"):
        with pytest.raises(KeyError):
            vault.group_id_of_folder(root, folder)


def test_a_folder_is_its_groups_under_any_spelling_the_file_system_may_give_it(root):
    """On a Mac, hw is the folder HW. It counts as HW's on every file system."""
    vault.add_group(root, "Übungen", "exercises")
    assert vault.group_id_of_folder(root, "hw") == "hw"
    assert vault.group_id_of_folder(root, "LOGISTICS") == "logistics"
    assert vault.group_id_of_folder(root, "unsorted") is None
    assert vault.group_id_of_folder(root, unicodedata.normalize("NFD", "Übungen")) == "übungen"
    assert vault.folder_ids(vault.read_groups(root)) == {
        "hw": "hw",
        "logistics": "logistics",
        "übungen": "übungen",
        "unsorted": None,
    }


def test_a_renamed_group_keeps_its_id_and_changes_its_folder(root):
    """The id is stable; the folder follows the group's current name."""
    vault.rename_group(root, "HW", "Homework")
    assert vault.folder_for(root, "hw") == root / "Homework"
    assert vault.group_id_of_folder(root, "Homework") == "hw"
    with pytest.raises(KeyError):
        vault.group_id_of_folder(root, "HW")


def test_a_deleted_groups_folder_is_no_group(root):
    """After a delete, neither the id nor the folder name is known any more."""
    vault.delete_group(root, "HW")
    with pytest.raises(KeyError):
        vault.folder_for(root, "hw")
    with pytest.raises(KeyError):
        vault.group_id_of_folder(root, "HW")


# write_card_note


def test_write_card_note_moves_the_note_when_the_group_changes(root):
    """Writing a file's note under a new group removes the note in the old folder."""
    old = vault.write_card_note(root, placement("hw3/a.py", group_id="hw"), "first\n")
    new = vault.write_card_note(root, placement("hw3/a.py", group_id="logistics"), "second\n")
    assert old == root / "HW" / "hw3__a.py.md"
    assert new == root / "Logistics" / "hw3__a.py.md"
    assert not old.exists()
    (note,) = vault.read_card_notes(root)
    assert (note.folder, note.placement.group_id, note.body) == (
        "Logistics",
        "logistics",
        "second\n",
    )
    gone = vault.write_card_note(root, placement("hw3/a.py", group_id=None), "third\n")
    assert gone == root / "Unsorted" / "hw3__a.py.md"
    assert not new.exists()
    assert len(vault.read_card_notes(root)) == 1


def test_write_card_note_overwrites_its_own_note_in_place(root):
    """Writing again under the same group replaces the note where it is."""
    first = vault.write_card_note(root, placement("a.txt"), "first\n")
    second = vault.write_card_note(root, placement("a.txt", frozen=True), "second\n")
    assert first == second
    (note,) = vault.read_card_notes(root)
    assert note.placement.frozen is True
    assert note.body == "second\n"


def test_write_card_note_removes_a_note_that_was_renamed_by_hand(root):
    """A file has one note. A copy under another name is found by its file id and removed."""
    path = vault.write_card_note(root, placement("a.txt"), "first\n")
    renamed = path.rename(root / "HW" / "My favorite homework.md")
    copy = write_by_hand(root / "Logistics" / "a copy.md", placement("a.txt"))
    again = vault.write_card_note(root, placement("a.txt"), "second\n")
    assert again == path
    assert not renamed.exists()
    assert not copy.exists()
    assert [note.path for note in vault.read_card_notes(root)] == [path]


def test_write_card_note_leaves_other_files_notes_alone(root):
    """Only notes for the same file id are removed."""
    other = vault.write_card_note(root, placement("b.txt"), "b\n")
    vault.write_card_note(root, placement("a.txt", group_id="hw"), "a\n")
    vault.write_card_note(root, placement("a.txt", group_id="logistics"), "a\n")
    assert other.read_text(encoding="utf-8").endswith("---\nb\n")
    assert ids(vault.read_card_notes(root)) == ["a.txt", "b.txt"]


def test_write_card_note_never_overwrites_the_users_own_note(root):
    """A file of the user's that sits where the card note would go is not touched."""
    own = root / "HW" / "a.txt.md"
    own.write_text("My own note, with the name a card note would get.\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="HW/a.txt.md exists and is not the card note"):
        vault.write_card_note(root, placement("a.txt"), "a card\n")
    assert own.read_text(encoding="utf-8") == "My own note, with the name a card note would get.\n"
    assert vault.read_card_notes(root) == []


def test_write_card_note_never_overwrites_another_files_note(root):
    """a/b.txt and a__b.txt would get the same note name. The second one is refused."""
    first = vault.write_card_note(root, placement("a/b.txt"), "first\n")
    with pytest.raises(FileExistsError):
        vault.write_card_note(root, placement("a__b.txt"), "second\n")
    (note,) = vault.read_card_notes(root)
    assert (note.path, note.placement.file_id, note.body) == (first, "a/b.txt", "first\n")


def test_write_card_note_keeps_the_old_note_when_the_new_one_is_refused(root):
    """A refused write changes nothing: the note in the old folder stays."""
    old = vault.write_card_note(root, placement("a.txt", group_id="hw"), "first\n")
    (root / "Logistics" / "a.txt.md").write_text("in the way\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        vault.write_card_note(root, placement("a.txt", group_id="logistics"), "second\n")
    assert [note.path for note in vault.read_card_notes(root)] == [old]


def test_write_card_note_makes_the_folder_when_it_is_missing(root):
    """A group folder that was removed by hand is made again."""
    (root / "HW").rmdir()
    (root / "Unsorted").rmdir()
    assert vault.write_card_note(root, placement("a.txt"), "").exists()
    assert vault.write_card_note(root, placement("b.txt", group_id=None), "").exists()


def test_write_card_note_for_an_unknown_group_is_an_error(root):
    """A placement can only go to a current group or to Unsorted."""
    with pytest.raises(KeyError):
        vault.write_card_note(root, placement("a.txt", group_id="project"), "")
    assert vault.read_card_notes(root) == []


def test_a_folder_whose_name_differs_only_in_case_keeps_exactly_one_note(root):
    """The group is HW and its folder was renamed to hw by hand.

    Where the file system ignores case (macOS, Windows) the two are one folder, and writing
    the note must not then remove it as an "other" note. Where case counts, they are two
    folders, and the note moves from hw/ to HW/. Either way one note is left.
    """
    (root / "HW").rename(root / "hw")
    write_by_hand(root / "hw" / "a.txt.md", placement("a.txt"), "first\n")
    path = vault.write_card_note(root, placement("a.txt"), "second\n")
    assert path.exists()
    (note,) = vault.read_card_notes(root)
    assert note.body == "second\n"


def test_write_card_note_can_be_given_the_notes_it_would_otherwise_read(root, monkeypatch):
    """A caller that has just read the card notes passes them on: the vault is not read again."""
    old = vault.write_card_note(root, placement("a.txt", group_id="hw"), "first\n")
    other = vault.write_card_note(root, placement("x/y.txt", group_id="hw"), "other\n")
    notes = vault.read_card_notes(root)

    def no_reading(root):
        """Stand in for read_card_notes: the notes were passed in, so nothing should ask."""
        raise AssertionError("the vault was read again")

    monkeypatch.setattr(vault, "read_card_notes", no_reading)
    new = vault.write_card_note(root, placement("a.txt", group_id="logistics"), "second\n", notes)
    assert new == root / "Logistics" / "a.txt.md"
    assert new.exists()
    assert not old.exists()
    # The same guard as without the list: a note that is not this file's is not overwritten.
    with pytest.raises(FileExistsError):
        vault.write_card_note(root, placement("x__y.txt", group_id="hw"), "", notes)
    (root / "HW" / "c.txt.md").write_text("My own note.\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        vault.write_card_note(root, placement("c.txt", group_id="hw"), "", notes)
    assert other.read_text(encoding="utf-8").endswith("---\nother\n")


def test_write_card_note_passes_over_a_listed_note_that_is_gone(root):
    """The list can be a moment old: a note deleted by hand since then is not an error."""
    old = vault.write_card_note(root, placement("a.txt", group_id="hw"), "first\n")
    notes = vault.read_card_notes(root)
    old.unlink()
    new = vault.write_card_note(root, placement("a.txt", group_id="logistics"), "second\n", notes)
    assert [note.path for note in vault.read_card_notes(root)] == [new]


# Card notes that Obsidian touched: added properties, and notes filed into a folder


def add_properties(path, extra="tags:\n- review\n- week3\ndue: 2026-11-20\nrating: 4\n"):
    """Add frontmatter keys to a card note by hand, the way Obsidian's properties do."""
    text = path.read_text(encoding="utf-8")
    assert text.count("\n---\n") == 1
    path.write_text(text.replace("\n---\n", f"\n{extra}---\n"), encoding="utf-8")


def test_a_card_note_with_properties_added_by_hand_is_still_a_card_note(root):
    """Keys that are not placement fields do not turn a card note into one of the user's notes."""
    path = vault.write_card_note(root, placement("a.txt", frozen=True), "a card\n")
    add_properties(path)
    (note,) = vault.read_card_notes(root)
    assert note.placement == placement("a.txt", frozen=True)
    assert note.body == "a card\n"
    assert list(note.extras) == ["tags", "due", "rating"]
    assert note.extras["tags"] == ["review", "week3"]
    assert note.extras["rating"] == 4
    # A key added in the middle of the placement fields is found as well.
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("\nfrozen: true\n", "\naliases: [sheet]\nfrozen: true\n"))
    (note,) = vault.read_card_notes(root)
    assert note.placement.frozen is True
    assert note.extras["aliases"] == ["sheet"]


def test_a_rewrite_keeps_the_properties_after_the_placement_fields(root):
    """Whatever the user added is written back, in place and when the note moves to a new group."""
    path = vault.write_card_note(root, placement("a.txt"), "first\n")
    add_properties(path)
    again = vault.write_card_note(root, placement("a.txt", frozen=True), "second\n")
    assert again == path
    frontmatter, body = vault.read_note(path)
    assert list(frontmatter) == [*Placement.model_fields, "tags", "due", "rating"]
    assert frontmatter["tags"] == ["review", "week3"]
    assert str(frontmatter["due"]) == "2026-11-20"
    assert body == "second\n"

    moved = vault.write_card_note(root, placement("a.txt", group_id="logistics"), "third\n")
    assert moved == root / "Logistics" / "a.txt.md"
    assert not path.exists()
    (note,) = vault.read_card_notes(root)
    assert note.placement == placement("a.txt", group_id="logistics")
    assert list(note.extras) == ["tags", "due", "rating"]
    # The list a caller passes in carries the properties too.
    notes = vault.read_card_notes(root)
    vault.write_card_note(root, placement("a.txt", group_id=None), "fourth\n", notes)
    (note,) = vault.read_card_notes(root)
    assert (note.folder, note.extras["rating"]) == ("Unsorted", 4)


def test_a_note_without_properties_is_written_as_before(root):
    """No extras means the frontmatter is the placement and nothing else."""
    path = vault.write_card_note(root, placement("a.txt"), "a card\n")
    (note,) = vault.read_card_notes(root)
    assert note.extras == {}
    assert vault.card_frontmatter(note.placement, note.extras) == vault.read_note(path)[0]


def test_a_rewrite_puts_a_nested_note_at_the_top_of_its_groups_folder(root):
    """A note filed into HW/week 3 is read as HW's. Writing it again leaves one note, in HW."""
    path = vault.write_card_note(root, placement("a.txt", frozen=True), "first\n")
    add_properties(path, "tags: [review]\n")
    (root / "HW" / "week 3").mkdir()
    nested = path.rename(root / "HW" / "week 3" / path.name)
    (note,) = vault.read_card_notes(root)
    assert (note.path, note.folder) == (nested, "HW")
    assert note.placement == placement("a.txt", frozen=True)

    again = vault.write_card_note(root, placement("a.txt"), "second\n")
    assert again == path
    assert not nested.exists()
    (note,) = vault.read_card_notes(root)
    assert (note.path, note.body, note.extras) == (path, "second\n", {"tags": ["review"]})


def test_unchanged_tells_whether_a_note_is_as_it_was_read(root):
    """Same place, size and time: unchanged. Edited, moved or deleted: not."""
    for file_id in ("edited.txt", "moved.txt", "deleted.txt", "kept.txt", "rewritten.txt"):
        vault.write_card_note(root, placement(file_id), "a card\n")
    notes = {note.placement.file_id: note for note in vault.read_card_notes(root)}
    assert all(vault.unchanged(note) for note in notes.values())

    edited = notes["edited.txt"].path
    text = edited.read_text(encoding="utf-8")
    edited.write_text(text.replace("frozen: false", "frozen: true"), encoding="utf-8")
    notes["moved.txt"].path.rename(root / "Logistics" / "moved.txt.md")
    notes["deleted.txt"].path.unlink()
    changed = {file_id: vault.unchanged(note) for file_id, note in notes.items()}
    assert changed == {
        "edited.txt": False,
        "moved.txt": False,
        "deleted.txt": False,
        "kept.txt": True,
        "rewritten.txt": True,
    }
    # After the tool's own rewrite, restamp makes the note count as unchanged again.
    rewritten = notes["rewritten.txt"]
    vault.write_note(rewritten.path, {"frozen": True}, "written by the tool\n")
    assert not vault.unchanged(rewritten)
    assert vault.unchanged(vault.restamp(rewritten))
