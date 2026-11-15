import os
import unicodedata

import pytest
import yaml

from decide_already import vault
from decide_already.config import Config
from decide_already.types import Placement


@pytest.fixture
def root(tmp_path):
    """A new vault with no groups yet."""
    vault.init_vault(tmp_path)
    return tmp_path


def top_folders(root):
    """The visible folders at the top of the vault, apart from the dump and _groups."""
    names = (path.name for path in root.iterdir() if path.is_dir())
    return sorted(name for name in names if name[0] not in "._")


def card_note(root, file_id, group_id):
    """Write a real card note for a file into its group's folder, and return the note's path."""
    placement = Placement(
        source=f"[[_dump/{file_id}]]",
        file_id=file_id,
        content_hash="0" * 64,
        file_type="Text",
        group_id=group_id,
        placed_by="jev",
        set_version=1,
    )
    return vault.write_card_note(root, placement, "a card\n")


def originals(root, dump="_dump"):
    """Every file under the dump folder with its bytes."""
    folder = root / dump
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def test_init_creates_layout_and_default_config(tmp_path):
    """init makes the four folders and writes the default config."""
    config = vault.init_vault(tmp_path)
    for folder in ("_dump", "_groups", "Unsorted", ".decide/sets"):
        assert (tmp_path / folder).is_dir()
    assert Config.load(tmp_path) == config
    assert config.model == "jev-1.14.0"
    assert config.dump_dir == "_dump"


def test_init_with_an_existing_dump_folder(tmp_path):
    """A folder that is already there can be the dump folder, and no _dump is made."""
    (tmp_path / "mock_data").mkdir()
    config = vault.init_vault(tmp_path, "mock_data")
    assert config.dump_dir == "mock_data"
    assert not (tmp_path / "_dump").exists()


def test_init_refuses_a_missing_custom_dump_folder(tmp_path):
    """Only the default dump folder is created for you."""
    with pytest.raises(FileNotFoundError):
        vault.init_vault(tmp_path, "nope")


def test_init_twice_keeps_the_existing_config(tmp_path):
    """A second init changes nothing in a vault that is already set up."""
    vault.init_vault(tmp_path)
    Config(model="jev-1.13.0").save(tmp_path)
    assert vault.init_vault(tmp_path).model == "jev-1.13.0"


def test_add_writes_note_folder_and_first_set(root):
    """Adding a group writes its note, makes its folder and records set version 1."""
    result = vault.add_group(root, "HW", "homework specs, starter code, solutions", "project files")
    assert result.version == 1
    assert (root / "_groups" / "HW.md").exists()
    assert (root / "HW").is_dir()
    data = yaml.safe_load((root / ".decide" / "sets" / "v1.yaml").read_text())
    assert data["groups"] == [
        {
            "id": "hw",
            "name": "HW",
            "covers": "homework specs, starter code, solutions",
            "excludes": "project files",
        }
    ]


def test_every_change_bumps_the_version(root):
    """Add, edit, rename and delete each write one new set version."""
    assert vault.add_group(root, "HW", "homework").version == 1
    assert vault.add_group(root, "Logistics", "syllabus, deadlines").version == 2
    assert vault.edit_group(root, "Logistics", covers="syllabus, deadlines, policies").version == 3
    assert vault.rename_group(root, "Logistics", "Course logistics").version == 4
    assert vault.delete_group(root, "HW").version == 5
    assert sorted(p.name for p in (root / ".decide" / "sets").iterdir()) == [
        f"v{i}.yaml" for i in range(1, 6)
    ]


def test_no_change_means_no_bump(root):
    """Reading the groups again, or an edit that changes nothing, writes no version."""
    vault.add_group(root, "HW", "homework")
    assert vault.sync_group_set(root).version == 1
    assert vault.edit_group(root, "HW", covers="homework").version == 1


def test_rename_keeps_id_and_moves_note_and_folder(root):
    """A rename by command keeps the id and moves the note and the folder."""
    vault.add_group(root, "Logistics", "syllabus")
    (root / "Logistics" / "syllabus.pdf.md").write_text("a card note")
    result = vault.rename_group(root, "Logistics", "Course logistics")
    assert [g.id for g in result.groups] == ["logistics"]
    assert result.groups[0].name == "Course logistics"
    assert not (root / "_groups" / "Logistics.md").exists()
    assert (root / "_groups" / "Course logistics.md").exists()
    assert (root / "Course logistics" / "syllabus.pdf.md").exists()


def test_add_requires_a_covers_description(root):
    """A group without a covers text is refused."""
    with pytest.raises(ValueError):
        vault.add_group(root, "HW", "   ")


def test_duplicate_names_and_ids_are_rejected(root):
    """A second group with the same name, or a name that gives the same id, is refused."""
    vault.add_group(root, "HW", "homework")
    with pytest.raises((FileExistsError, ValueError)):
        vault.add_group(root, "HW", "again")
    with pytest.raises((FileExistsError, ValueError)):
        vault.add_group(root, "hw", "same slug")


@pytest.mark.parametrize(
    "name",
    [
        "Unsorted",
        "_groups",
        "_dump",
        ".decide",
        "a/b",
        " HW",
        "",
        # The same folders under another spelling: on a Mac, unsorted is Unsorted.
        "unsorted",
        "UNSORTED",
        "_Groups",
        "_Dump",
        "_DUMP",
        # Names the tool gives to folders it sets aside.
        "HW (old)",
        "Hw (OLD)",
        "HW (old) 2",
    ],
)
def test_reserved_and_unsafe_names_are_rejected(root, name):
    """No group is named like a tool folder, the dump folder or a set-aside folder."""
    vault.add_group(root, "HW", "homework")
    before = sorted(path.name for path in root.rglob("*"))
    with pytest.raises(ValueError, match="group name"):
        vault.add_group(root, name, "anything")
    with pytest.raises(ValueError, match="group name"):
        vault.rename_group(root, "HW", name)
    assert sorted(path.name for path in root.rglob("*")) == before


def test_hand_edited_notes_are_picked_up_and_given_an_id(root):
    """A group note made in Obsidian becomes a group, and gets an id written into it."""
    vault.add_group(root, "HW", "homework")
    (root / "_groups" / "Project.md").write_text(
        "---\ncovers: project specs and examples\n---\nMy notes.\n"
    )
    result = vault.sync_group_set(root)
    assert result.version == 2
    assert [g.id for g in result.groups] == ["hw", "project"]
    frontmatter, body = vault.read_note(root / "_groups" / "Project.md")
    assert frontmatter["id"] == "project"
    assert body == "My notes.\n"


def test_note_without_covers_is_an_error(root):
    """A group note with no covers text stops the set from being read."""
    (root / "_groups" / "Empty.md").write_text("---\nid: empty\n---\n")
    with pytest.raises(ValueError):
        vault.sync_group_set(root)


def test_find_vault_walks_up(root, tmp_path_factory):
    """A folder inside a vault finds it. A folder outside any vault finds none."""
    (root / "HW").mkdir()
    assert vault.find_vault(root / "HW") == root.resolve()
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    for above in [elsewhere.resolve(), *elsewhere.resolve().parents]:
        if Config.path(above).exists():
            pytest.skip(f"{above} is a vault, so no temp folder under it is outside one")
    with pytest.raises(FileNotFoundError):
        vault.find_vault(elsewhere)


def test_an_empty_vault_has_no_set_version_yet(root):
    """With no groups there is nothing to record: version 0, and no set file."""
    result = vault.sync_group_set(root)
    assert result.version == 0
    assert result.groups == []
    assert list((root / ".decide" / "sets").iterdir()) == []
    assert vault.add_group(root, "HW", "homework").version == 1


def test_a_group_note_renamed_by_hand_has_its_old_folder_set_aside(root):
    """The group keeps its id. Its old folder is set aside, which frees the old name at once."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "hw1.pdf.md").write_text("a card note", encoding="utf-8")
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Homework.md")
    result = vault.sync_group_set(root)
    assert (result.version, result.groups[0].id, result.groups[0].name) == (2, "hw", "Homework")
    assert (root / "HW (old)" / "hw1.pdf.md").read_text(encoding="utf-8") == "a card note"
    assert not (root / "HW").exists()
    # Nothing changed since, so nothing more is set aside and no version is written.
    assert vault.sync_group_set(root).version == 2
    assert top_folders(root) == ["HW (old)", "Unsorted"]


def test_an_old_folder_with_nothing_in_it_is_removed(root):
    """A renamed or deleted group whose folder is empty leaves no folder behind."""
    vault.add_group(root, "HW", "homework")
    vault.add_group(root, "Exams", "midterms")
    (root / "Exams" / ".DS_Store").write_bytes(b"Finder")  # does not count as something in it
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Homework.md")
    (root / "_groups" / "Exams.md").unlink()
    vault.sync_group_set(root)
    assert top_folders(root) == ["Unsorted"]


def test_a_set_aside_name_that_is_taken_gets_a_number(root):
    """HW (old) is there already, so the next folder to set aside is HW (old) 2."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "a.md").write_text("first", encoding="utf-8")
    vault.delete_group(root, "HW")
    vault.add_group(root, "HW", "homework again")
    (root / "HW" / "b.md").write_text("second", encoding="utf-8")
    vault.delete_group(root, "HW")
    assert top_folders(root) == ["HW (old)", "HW (old) 2", "Unsorted"]
    assert (root / "HW (old)" / "a.md").read_text(encoding="utf-8") == "first"
    assert (root / "HW (old) 2" / "b.md").read_text(encoding="utf-8") == "second"


def test_a_group_note_respelled_by_hand_has_its_folder_respelled(root):
    """HW to Hw is the same folder on a Mac: it gets the new spelling and is not set aside."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "a.md").write_text("in it", encoding="utf-8")
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Hw.md")
    result = vault.sync_group_set(root)
    assert [(group.id, group.name) for group in result.groups] == [("hw", "Hw")]
    assert top_folders(root) == ["Hw", "Unsorted"]
    assert (root / "Hw" / "a.md").read_text(encoding="utf-8") == "in it"


def test_setting_a_folder_aside_is_safe_to_repeat(root, monkeypatch):
    """The set file cannot be written the first time: the second try sets nothing aside twice."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "a.md").write_text("in it", encoding="utf-8")
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Homework.md")
    safe_dump = yaml.safe_dump

    def fail_once(*args, **kwargs):
        """Stand in for yaml.safe_dump: fail like a full disk, once."""
        monkeypatch.setattr(yaml, "safe_dump", safe_dump)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(yaml, "safe_dump", fail_once)
    with pytest.raises(OSError, match="No space left"):
        vault.sync_group_set(root)
    assert vault.latest_set(root).version == 1  # the new version was not recorded
    assert vault.sync_group_set(root).version == 2
    assert top_folders(root) == ["HW (old)", "Unsorted"]


def test_a_rename_onto_a_leftover_folder_moves_the_files_into_it(root):
    """A folder of the new name that holds no card note is merged into."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "hw1.pdf.md").write_text("a card note", encoding="utf-8")
    (root / "Old").mkdir()  # as a deleted group leaves it
    (root / "Old" / "mine.md").write_text("my own note", encoding="utf-8")
    vault.rename_group(root, "HW", "Old")
    assert sorted(path.name for path in (root / "Old").iterdir()) == ["hw1.pdf.md", "mine.md"]
    assert not (root / "HW").exists()


def test_a_rename_never_writes_over_a_file_in_the_other_folder(root, monkeypatch):
    """A file of the same name in both folders stops the rename before anything is renamed."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "notes.md").write_text("from HW", encoding="utf-8")
    (root / "Old").mkdir()
    (root / "Old" / "notes.md").write_text("already here", encoding="utf-8")

    def not_reached(root, old, new):
        """Stand in for the folder move: the clash has to be found before it is tried."""
        raise AssertionError("the note was renamed before the clash was found")

    monkeypatch.setattr(vault, "_move_folder", not_reached)
    with pytest.raises(FileExistsError, match="notes.md"):
        vault.rename_group(root, "HW", "Old")
    assert (root / "HW" / "notes.md").read_text(encoding="utf-8") == "from HW"
    assert (root / "Old" / "notes.md").read_text(encoding="utf-8") == "already here"
    # Nothing is half done: the note has its old name, and every command still works.
    assert [path.name for path in (root / "_groups").iterdir()] == ["HW.md"]
    assert [(group.id, group.name) for group in vault.sync_group_set(root).groups] == [("hw", "HW")]


def test_a_hidden_file_in_both_folders_does_not_stop_a_rename(root):
    """The Finder leaves a .DS_Store in every folder it shows. The old folder's copy is dropped."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "hw1.pdf.md").write_text("a card note", encoding="utf-8")
    (root / "HW" / ".DS_Store").write_bytes(b"from HW")
    (root / "Old").mkdir()
    (root / "Old" / ".DS_Store").write_bytes(b"already here")
    result = vault.rename_group(root, "HW", "Old")
    assert [(group.id, group.name) for group in result.groups] == [("hw", "Old")]
    assert sorted(path.name for path in (root / "Old").iterdir()) == [".DS_Store", "hw1.pdf.md"]
    assert (root / "Old" / ".DS_Store").read_bytes() == b"already here"
    assert not (root / "HW").exists()


def test_a_rename_onto_the_name_of_a_file_is_refused_before_anything_is_renamed(root):
    """A plain file named like the new group is where the group's folder has to go."""
    vault.add_group(root, "HW", "homework")
    (root / "Problem sets").write_text("a file, not a folder", encoding="utf-8")
    with pytest.raises(FileExistsError, match="'Problem sets' cannot be a group name: a file"):
        vault.rename_group(root, "HW", "Problem sets")
    with pytest.raises(FileExistsError, match="'Problem sets' cannot be a group name: a file"):
        vault.add_group(root, "Problem sets", "more homework")
    assert [path.name for path in (root / "_groups").iterdir()] == ["HW.md"]
    assert (root / "HW").is_dir()
    assert (root / "Problem sets").read_text(encoding="utf-8") == "a file, not a folder"
    assert vault.sync_group_set(root).version == 1


def test_a_rename_whose_folder_cannot_follow_gives_the_note_its_name_back(root, monkeypatch):
    """The folder move fails after every check passed: the note is renamed back."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "hw1.pdf.md").write_text("a card note", encoding="utf-8")

    def cannot_move(root, old, new):
        """Stand in for the folder move: fail the way a locked folder does."""
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(vault, "_move_folder", cannot_move)
    with pytest.raises(PermissionError):
        vault.rename_group(root, "HW", "Homework")
    assert [path.name for path in (root / "_groups").iterdir()] == ["HW.md"]
    assert (root / "HW" / "hw1.pdf.md").exists()
    assert [(group.id, group.name) for group in vault.sync_group_set(root).groups] == [("hw", "HW")]
    assert vault.latest_set(root).version == 1


def test_a_rename_is_stopped_by_a_broken_group_note_before_anything_is_renamed(root):
    """Another group's note cannot be read: the rename changes nothing."""
    vault.add_group(root, "HW", "homework")
    (root / "_groups" / "Draft.md").write_text("---\nid: draft\n---\n", encoding="utf-8")
    with pytest.raises(ValueError, match="group 'Draft' has no covers description"):
        vault.rename_group(root, "HW", "Homework")
    assert sorted(path.name for path in (root / "_groups").iterdir()) == ["Draft.md", "HW.md"]
    assert (root / "HW").is_dir()
    assert not (root / "Homework").exists()


def test_a_rename_never_merges_into_a_folder_that_holds_card_notes(root):
    """Card notes in a folder of the new name are another group's: a merge would take them over."""
    vault.add_group(root, "HW", "homework")
    vault.add_group(root, "Exams", "midterms")
    (root / "Archive").mkdir()
    card_note(root, "exam1.txt", "exams").rename(root / "Archive" / "exam1.txt.md")
    with pytest.raises(FileExistsError, match="the folder 'Archive' already holds card notes"):
        vault.rename_group(root, "HW", "Archive")
    assert sorted(path.name for path in (root / "_groups").iterdir()) == ["Exams.md", "HW.md"]
    assert (root / "HW").is_dir()


def test_a_rename_that_only_changes_the_case(root):
    """HW to Hw by command: the note and the folder get the new spelling."""
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "hw1.pdf.md").write_text("a card note", encoding="utf-8")
    result = vault.rename_group(root, "HW", "Hw")
    assert [(group.id, group.name) for group in result.groups] == [("hw", "Hw")]
    assert [path.name for path in (root / "_groups").iterdir()] == ["Hw.md"]
    assert "Hw" in [path.name for path in root.iterdir()]
    assert "HW" not in [path.name for path in root.iterdir()]
    assert (root / "Hw" / "hw1.pdf.md").exists()


def test_two_groups_that_swap_names_by_hand_have_their_folders_set_aside(root):
    """Neither folder goes to the other group: both names are freed, and nothing is merged."""
    vault.add_group(root, "A", "first")
    vault.add_group(root, "B", "second")
    (root / "A" / "a.md").write_text("in A", encoding="utf-8")
    (root / "B" / "b.md").write_text("in B", encoding="utf-8")
    groups = root / "_groups"
    (groups / "A.md").rename(groups / "tmp.md")
    (groups / "B.md").rename(groups / "A.md")
    (groups / "tmp.md").rename(groups / "B.md")
    result = vault.sync_group_set(root)
    assert {group.id: group.name for group in result.groups} == {"a": "B", "b": "A"}
    assert top_folders(root) == ["A (old)", "B (old)", "Unsorted"]
    assert (root / "A (old)" / "a.md").read_text(encoding="utf-8") == "in A"
    assert (root / "B (old)" / "b.md").read_text(encoding="utf-8") == "in B"


# Folders that are never a group's: the dump folder and the tool's own


def test_init_keeps_the_dump_folder_in_its_plain_form(tmp_path):
    """`--dump lectures/` (as tab completion gives it) is the folder lectures."""
    (tmp_path / "lectures").mkdir()
    config = vault.init_vault(tmp_path, "lectures/")
    assert config.dump_dir == "lectures"
    assert Config.load(tmp_path).dump_dir == "lectures"
    assert vault.source_link("a.txt", config) == "[[lectures/a.txt]]"


@pytest.mark.parametrize(
    "dump", [".", "./", "..", "", "/tmp/files", "a\\b", "Unsorted", "_groups/x"]
)
def test_init_refuses_a_dump_folder_that_is_not_a_plain_folder_inside_the_vault(tmp_path, dump):
    """The vault itself, a path out of it and the tool's own folders: nothing is set up."""
    (tmp_path / "files").mkdir()
    with pytest.raises(ValueError, match="the dump folder"):
        vault.init_vault(tmp_path, dump)
    assert [path.name for path in tmp_path.iterdir()] == ["files"]


@pytest.mark.parametrize(
    ("dump", "names"),
    [
        ("lectures/", ["lectures", "Lectures", "LECTURES"]),
        ("course/files", ["course", "Course"]),
        ("Übungen", ["Übungen", "übungen", unicodedata.normalize("NFD", "Übungen")]),
    ],
)
def test_a_group_cannot_be_named_after_the_dump_folder(tmp_path, dump, names):
    """Its folder would be the one with the originals, or the folder they sit in."""
    (tmp_path / dump).mkdir(parents=True)
    vault.init_vault(tmp_path, dump)
    (tmp_path / dump / "lecture01.txt").write_text("an original", encoding="utf-8")
    vault.add_group(tmp_path, "HW", "homework")
    before = originals(tmp_path, dump)
    for name in names:
        with pytest.raises(ValueError, match="cannot be a group name: that folder holds"):
            vault.add_group(tmp_path, name, "lecture notes")
        with pytest.raises(ValueError, match="cannot be a group name: that folder holds"):
            vault.rename_group(tmp_path, "HW", name)
    assert originals(tmp_path, dump) == before
    assert [group.name for group in vault.read_groups(tmp_path)] == ["HW"]


@pytest.mark.parametrize("name", ["_dump", "_Dump", "Unsorted", "unsorted", ".decide", "HW (old)"])
def test_a_group_note_made_by_hand_with_a_name_that_is_not_allowed_is_refused(root, name):
    """The check of `p0 group add` holds for notes made in Obsidian too. The note is named."""
    (root / "_dump" / "a.txt").write_text("an original", encoding="utf-8")
    note = root / "_groups" / f"{name}.md"
    note.write_text("---\ncovers: anything\n---\n", encoding="utf-8")
    before = originals(root)
    for read in (vault.read_groups, vault.sync_group_set):
        with pytest.raises(ValueError, match="cannot be a group name") as error:
            read(root)
        assert str(note) in str(error.value)
    # Renaming the bad note with the command is refused too, and moves nothing.
    with pytest.raises(ValueError, match="cannot be a group name"):
        vault.rename_group(root, name, "Raw")
    assert note.exists()
    assert not (root / "Raw").exists()
    assert note.read_text(encoding="utf-8") == "---\ncovers: anything\n---\n"  # no id written
    assert vault.latest_set(root) is None
    assert originals(root) == before
    # Deleting the note is the way out, and it leaves the folder alone.
    vault.delete_group(root, name)
    assert originals(root) == before
    assert (root / "Unsorted").is_dir()
    assert (root / ".decide" / "config.yaml").is_file()


def test_a_set_on_record_never_makes_the_tool_move_the_dump_folder(root):
    """A vault from before the check has a group named after the dump folder in its sets.

    Its note is then renamed or deleted by hand. The folder of the old name is the dump
    folder, and it stays where it is, with every original in it.
    """
    for file_id in ("a.txt", "week 1/b.txt"):
        path = root / "_dump" / file_id
        path.parent.mkdir(exist_ok=True)
        path.write_text("an original", encoding="utf-8")
    (root / "_dump" / ".DS_Store").write_bytes(b"Finder")
    on_record = {"version": 1, "groups": [{"id": "dump", "name": "_dump", "covers": "x"}]}
    (root / ".decide" / "sets" / "v1.yaml").write_text(yaml.safe_dump(on_record), encoding="utf-8")
    before = originals(root)

    (root / "_groups" / "Raw.md").write_text("---\nid: dump\ncovers: x\n---\n", encoding="utf-8")
    assert vault.sync_group_set(root).version == 2  # read as a rename of the group by hand
    assert originals(root) == before
    assert top_folders(root) == ["Unsorted"]

    (root / "_groups" / "Raw.md").unlink()  # and now as a delete by hand
    assert vault.sync_group_set(root).version == 3
    assert originals(root) == before
    assert (root / "_dump" / ".DS_Store").exists()


def test_an_empty_dump_folder_is_not_removed_as_a_leftover(tmp_path):
    """The dump folder is named like a set-aside folder and has nothing in it yet: it stays."""
    (tmp_path / "Scans (old)").mkdir()
    vault.init_vault(tmp_path, "Scans (old)")
    vault.add_group(tmp_path, "Scans", "scanned pages")
    (tmp_path / "Scans" / "mine.md").write_text("my own note", encoding="utf-8")
    vault.delete_group(tmp_path, "Scans")  # set aside under the next free name
    assert top_folders(tmp_path) == ["Scans (old)", "Scans (old) 2", "Unsorted"]
    assert list((tmp_path / "Scans (old)").iterdir()) == []
    vault.remove_empty_set_asides(tmp_path)
    assert (tmp_path / "Scans (old)").is_dir()


def test_two_group_notes_whose_names_are_one_folder_are_refused(root, monkeypatch):
    """Where HW.md and hw.md are two files, both notes can exist. Their groups would share a folder.

    A Mac cannot hold two such notes. So that the check runs on every file system, this
    test makes any two names that start with the same letter count as one folder.
    """
    vault.add_group(root, "Homework", "homework")
    vault.add_group(root, "Exams", "midterms")
    monkeypatch.setattr(vault, "folder_key", lambda name: name[:1].casefold())
    (root / "_groups" / "HW.md").write_text("---\ncovers: more\n---\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'HW' and 'Homework' would share one folder"):
        vault.sync_group_set(root)
    assert vault.latest_set(root).version == 2
    (root / "_groups" / "HW.md").unlink()
    # The commands refuse such a name before they write anything.
    with pytest.raises(FileExistsError, match="group 'Homework' already exists"):
        vault.add_group(root, "Hw2", "more homework")
    with pytest.raises(FileExistsError, match="group 'Homework' already exists"):
        vault.rename_group(root, "Exams", "History")
    assert sorted(path.name for path in (root / "_groups").iterdir()) == [
        "Exams.md",
        "Homework.md",
    ]
    assert vault.sync_group_set(root).version == 2


# Deleted groups and folders left behind


def test_a_deleted_groups_folder_is_set_aside_when_it_still_holds_notes(root):
    """The name is free at once, so a group that takes it later gets none of the old notes."""
    vault.add_group(root, "Homework", "homework")
    vault.add_group(root, "Exams", "midterms")
    card_note(root, "hw1.txt", "homework")
    card_note(root, "exam1.txt", "exams")
    vault.delete_group(root, "Exams")
    assert top_folders(root) == ["Exams (old)", "Homework", "Unsorted"]
    vault.rename_group(root, "Homework", "Exams")
    assert top_folders(root) == ["Exams", "Exams (old)", "Unsorted"]
    assert [path.name for path in (root / "Exams").iterdir()] == ["hw1.txt.md"]
    assert [path.name for path in (root / "Exams (old)").iterdir()] == ["exam1.txt.md"]


def test_a_group_deleted_by_hand_is_set_aside_before_a_rename_takes_its_name(root):
    """The note is deleted in Obsidian, then `p0 group rename` reuses the name right away."""
    vault.add_group(root, "Homework", "homework")
    vault.add_group(root, "Exams", "midterms")
    card_note(root, "hw1.txt", "homework")
    card_note(root, "exam1.txt", "exams")
    (root / "_groups" / "Exams.md").unlink()
    vault.rename_group(root, "Homework", "Exams")
    assert [path.name for path in (root / "Exams").iterdir()] == ["hw1.txt.md"]
    assert [path.name for path in (root / "Exams (old)").iterdir()] == ["exam1.txt.md"]


def test_adding_a_group_respells_a_folder_that_is_the_same_one(root):
    """A folder hw is already there. The new group HW takes it, under the group's spelling."""
    (root / "hw").mkdir()
    (root / "hw" / "mine.md").write_text("my own note", encoding="utf-8")
    vault.add_group(root, "HW", "homework")
    assert top_folders(root) == ["HW", "Unsorted"]
    assert (root / "HW" / "mine.md").read_text(encoding="utf-8") == "my own note"


def test_a_group_added_again_under_another_spelling_gets_its_own_spelling(root):
    """Delete HW, add Hw: the folder on disk is Hw, and the notes of HW were set aside."""
    vault.add_group(root, "HW", "homework")
    card_note(root, "hw1.txt", "hw")
    (root / "Exams").mkdir()
    vault.delete_group(root, "HW")
    vault.add_group(root, "Hw", "homework again")
    assert top_folders(root) == ["Exams", "HW (old)", "Hw", "Unsorted"]
    assert vault.folder_for(root, "hw") == root / "Hw"
    assert vault.group_id_of_folder(root, "HW") == "hw"  # any spelling of the folder


# Group notes that cannot be read


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ("covers: syllabus, schedule, deadlines: for homework", "is not valid YAML"),
        ("covers: [unclosed", "is not valid YAML"),
        ("- syllabus\n- schedule", "must be `key: value` lines"),
        ("just a sentence", "must be `key: value` lines"),
        ("covers: [syllabus, schedule]", "the covers of a group must be text"),
        ("covers: 2024", "the covers of a group must be text"),
        ("covers: 2026-11-15", "the covers of a group must be text"),
        ("covers: true", "the covers of a group must be text"),
        ("covers: syllabus\nexcludes: [projects]", "the excludes of a group must be text"),
        ("covers: syllabus\nexcludes: 7", "the excludes of a group must be text"),
    ],
)
def test_a_group_note_that_cannot_be_read_is_a_value_error_that_names_it(
    root, frontmatter, message
):
    """A slip made by hand in a group note is reported with the note's path, never as a crash."""
    vault.add_group(root, "HW", "homework")
    note = root / "_groups" / "Logistics.md"
    note.write_text(f"---\n{frontmatter}\n---\nMy notes.\n", encoding="utf-8")
    for read in (vault.read_groups, vault.sync_group_set):
        with pytest.raises(ValueError, match=message) as error:
            read(root)
        assert str(note) in str(error.value)
    # Editing the other field leaves the slip in place, and reports it the same way.
    other = {"covers": "syllabus"} if "excludes" in message else {"excludes": "projects"}
    with pytest.raises(ValueError, match=message):
        vault.edit_group(root, "Logistics", **other)
    assert vault.latest_set(root).version == 1


def test_read_note_names_the_file_and_the_line(tmp_path):
    """The line is the one in the file, counted from the opening dashes."""
    note = tmp_path / "note.md"
    note.write_text("---\nid: x\ncovers: a: b\n---\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        vault.read_note(note)
    assert str(error.value).startswith(f"the frontmatter of {note} is not valid YAML: ")
    assert "line 3" in str(error.value)
    note.write_bytes("---\ncovers: caf\xe9\n---\n".encode("latin-1"))
    with pytest.raises(ValueError, match="note.md is not UTF-8 text"):
        vault.read_note(note)
    note.write_text("---\n\n---\nNo frontmatter keys at all.\n", encoding="utf-8")
    assert vault.read_note(note) == ({}, "No frontmatter keys at all.\n")


def test_two_group_notes_with_one_id_are_refused(root):
    """A group note copied in Obsidian carries the same id: the id is named, nothing is recorded."""
    vault.add_group(root, "HW", "homework")
    vault.add_group(root, "Logistics", "syllabus")
    copy = root / "_groups" / "HW copy.md"
    copy.write_bytes((root / "_groups" / "HW.md").read_bytes())
    with pytest.raises(ValueError, match="two group notes share an id: hw"):
        vault.sync_group_set(root)
    with pytest.raises(ValueError, match="two group notes share an id: hw"):
        vault.folder_for(root, "hw")
    assert vault.latest_set(root).version == 2
    assert sorted(path.name for path in (root / ".decide" / "sets").iterdir()) == [
        "v1.yaml",
        "v2.yaml",
    ]
    copy.unlink()
    assert vault.sync_group_set(root).version == 2


# Group ids


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("HW", "hw"),
        ("Lecture slides", "lecture-slides"),
        ("  Exams & quizzes (2026)  ", "exams-quizzes-2026"),
        ("my_group", "my-group"),
        ("a.b/c", "a-b-c"),
        ("---", "group"),
        ("", "group"),
        ("강의 자료", "강의-자료"),
        ("과제", "과제"),
        ("Übungen", "übungen"),
        (unicodedata.normalize("NFD", "Übungen"), "übungen"),
        ("हिन्दी नोट्स", "हिन्दी-नोट्स"),
        ("Week 3 講義", "week-3-講義"),
    ],
)
def test_slugify_keeps_letters_and_digits_of_any_script(name, slug):
    """An ASCII name gives the id it always gave. Other scripts are kept, not dropped."""
    assert vault.slugify(name) == slug


def test_groups_named_in_another_script_get_ids_of_their_own(root):
    """Two Korean names used to share the id "group", by command and by hand."""
    assert vault.add_group(root, "강의 자료", "lecture material").version == 1
    assert vault.add_group(root, "과제", "homework").version == 2
    for name in ("시험", "프로젝트"):
        (root / "_groups" / f"{name}.md").write_text("---\ncovers: more\n---\n", encoding="utf-8")
    result = vault.sync_group_set(root)
    assert sorted(group.id for group in result.groups) == ["강의-자료", "과제", "시험", "프로젝트"]
    assert vault.folder_for(root, "과제") == root / "과제"


# The dump listing


def test_dump_files_does_not_follow_linked_folders(root, tmp_path_factory):
    """A link in the dump to a folder outside it, or back to the dump, is not walked into."""
    outside = tmp_path_factory.mktemp("outside")
    (outside / "private.txt").write_text("not for sorting", encoding="utf-8")
    dump = root / "_dump"
    (dump / "a.txt").write_text("an original", encoding="utf-8")
    (dump / "week 1").mkdir()
    (dump / "week 1" / "b.txt").write_text("an original", encoding="utf-8")
    try:
        os.symlink(outside, dump / "linked", target_is_directory=True)
        os.symlink(dump, dump / "week 1" / "back", target_is_directory=True)
    except OSError:
        pytest.skip("links cannot be made on this file system")
    listed = vault.dump_files(root, Config.load(root))
    assert [file_id for file_id, _ in listed] == ["a.txt", "week 1/b.txt"]


def test_a_folder_that_holds_only_empty_folders_counts_as_empty(root):
    """What is left of HW/week 3 once its notes are gone does not keep a folder alive."""
    vault.add_group(root, "HW", "homework")
    vault.add_group(root, "Exams", "midterms")
    (root / "HW" / "week 3" / "drafts").mkdir(parents=True)
    (root / "HW" / "week 3" / ".DS_Store").write_bytes(b"Finder")
    (root / "Exams" / "2025").mkdir()
    (root / "Exams" / "2025" / "mine.md").write_text("my own note", encoding="utf-8")
    vault.delete_group(root, "HW")
    vault.delete_group(root, "Exams")
    assert top_folders(root) == ["Exams (old)", "Unsorted"]
    assert (root / "Exams (old)" / "2025" / "mine.md").read_text(encoding="utf-8") == "my own note"
    vault.remove_empty_set_asides(root)
    assert top_folders(root) == ["Exams (old)", "Unsorted"]
    (root / "Exams (old)" / "2025" / "mine.md").unlink()
    vault.remove_empty_set_asides(root)
    assert top_folders(root) == ["Unsorted"]


def test_a_folder_is_never_set_aside_under_the_name_of_the_dump_folder(tmp_path):
    """The dump folder is named Scans (old) and is missing right now. Its name stays free."""
    (tmp_path / "Scans (old)").mkdir()
    vault.init_vault(tmp_path, "Scans (old)")
    vault.add_group(tmp_path, "Scans", "scanned pages")
    (tmp_path / "Scans" / "mine.md").write_text("my own note", encoding="utf-8")
    (tmp_path / "Scans (old)").rmdir()  # removed by hand, for now
    vault.delete_group(tmp_path, "Scans")
    assert top_folders(tmp_path) == ["Scans (old) 2", "Unsorted"]
    assert vault.dump_files(tmp_path, Config.load(tmp_path)) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("_dump", "Raw"),
        ("_DUMP", "Raw"),
        ("Unsorted", "Misc"),
        ("_groups", "Mine"),
        ("HW", "_Dump"),
    ],
)
def test_the_folder_move_itself_refuses_the_dump_folder_and_the_tools_own(root, old, new):
    """The last line of defense: whatever asks for it, these folders are never renamed."""
    (root / "_dump" / "a.txt").write_text("an original", encoding="utf-8")
    vault.add_group(root, "HW", "homework")
    before = originals(root)
    for move in (vault._check_folder_move, vault._move_folder):
        with pytest.raises(ValueError, match="is never renamed"):
            move(root, old, new)
    assert originals(root) == before
    assert top_folders(root) == ["HW", "Unsorted"]
    assert [path.name for path in (root / "_groups").iterdir()] == ["HW.md"]


def test_a_group_id_typed_by_hand_cannot_be_unsorted(root):
    """Unsorted names the lane for files that fit no group, so no group may have it as its id."""
    (root / "_groups" / "Misc.md").write_text(
        "---\nid: unsorted\ncovers: odds and ends\n---\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="a group's id cannot be 'unsorted'"):
        vault.read_groups(root)
