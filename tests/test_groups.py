import pytest
import yaml

from decide_already import vault
from decide_already.config import Config


@pytest.fixture
def root(tmp_path):
    vault.init_vault(tmp_path)
    return tmp_path


def test_init_creates_layout_and_default_config(tmp_path):
    config = vault.init_vault(tmp_path)
    for folder in ("_dump", "_groups", "Unsorted", ".decide/sets"):
        assert (tmp_path / folder).is_dir()
    assert Config.load(tmp_path) == config
    assert config.model == "jev-1.14.0"
    assert config.dump_dir == "_dump"


def test_init_with_an_existing_dump_folder(tmp_path):
    (tmp_path / "mock_data").mkdir()
    config = vault.init_vault(tmp_path, "mock_data")
    assert config.dump_dir == "mock_data"
    assert not (tmp_path / "_dump").exists()


def test_init_refuses_a_missing_custom_dump_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        vault.init_vault(tmp_path, "nope")


def test_init_twice_keeps_the_existing_config(tmp_path):
    vault.init_vault(tmp_path)
    Config(model="jev-1.13.0").save(tmp_path)
    assert vault.init_vault(tmp_path).model == "jev-1.13.0"


def test_add_writes_note_folder_and_first_set(root):
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
    assert vault.add_group(root, "HW", "homework").version == 1
    assert vault.add_group(root, "Logistics", "syllabus, deadlines").version == 2
    assert vault.edit_group(root, "Logistics", covers="syllabus, deadlines, policies").version == 3
    assert vault.rename_group(root, "Logistics", "Course logistics").version == 4
    assert vault.delete_group(root, "HW").version == 5
    assert sorted(p.name for p in (root / ".decide" / "sets").iterdir()) == [
        f"v{i}.yaml" for i in range(1, 6)
    ]


def test_no_change_means_no_bump(root):
    vault.add_group(root, "HW", "homework")
    assert vault.sync_group_set(root).version == 1
    assert vault.edit_group(root, "HW", covers="homework").version == 1


def test_rename_keeps_id_and_moves_note_and_folder(root):
    vault.add_group(root, "Logistics", "syllabus")
    (root / "Logistics" / "syllabus.pdf.md").write_text("a card note")
    result = vault.rename_group(root, "Logistics", "Course logistics")
    assert [g.id for g in result.groups] == ["logistics"]
    assert result.groups[0].name == "Course logistics"
    assert not (root / "_groups" / "Logistics.md").exists()
    assert (root / "_groups" / "Course logistics.md").exists()
    assert (root / "Course logistics" / "syllabus.pdf.md").exists()


def test_add_requires_a_covers_description(root):
    with pytest.raises(ValueError):
        vault.add_group(root, "HW", "   ")


def test_duplicate_names_and_ids_are_rejected(root):
    vault.add_group(root, "HW", "homework")
    with pytest.raises((FileExistsError, ValueError)):
        vault.add_group(root, "HW", "again")
    with pytest.raises((FileExistsError, ValueError)):
        vault.add_group(root, "hw", "same slug")


@pytest.mark.parametrize("name", ["Unsorted", "_groups", "_dump", ".decide", "a/b", " HW", ""])
def test_reserved_and_unsafe_names_are_rejected(root, name):
    with pytest.raises(ValueError):
        vault.add_group(root, name, "anything")


def test_hand_edited_notes_are_picked_up_and_given_an_id(root):
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
    (root / "_groups" / "Empty.md").write_text("---\nid: empty\n---\n")
    with pytest.raises(ValueError):
        vault.sync_group_set(root)


def test_find_vault_walks_up(root, tmp_path_factory):
    (root / "HW").mkdir()
    assert vault.find_vault(root / "HW") == root.resolve()
    with pytest.raises(FileNotFoundError):
        vault.find_vault(tmp_path_factory.mktemp("elsewhere"))


def test_an_empty_vault_has_no_set_version_yet(root):
    result = vault.sync_group_set(root)
    assert result.version == 0
    assert result.groups == []
    assert list((root / ".decide" / "sets").iterdir()) == []
    assert vault.add_group(root, "HW", "homework").version == 1


def test_a_group_note_renamed_by_hand_takes_its_folder_along(root):
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "hw1.pdf.md").write_text("a card note", encoding="utf-8")
    (root / "_groups" / "HW.md").rename(root / "_groups" / "Homework.md")
    result = vault.sync_group_set(root)
    assert (result.version, result.groups[0].id, result.groups[0].name) == (2, "hw", "Homework")
    assert (root / "Homework" / "hw1.pdf.md").read_text(encoding="utf-8") == "a card note"
    assert not (root / "HW").exists()


def test_a_rename_onto_a_leftover_folder_moves_the_files_into_it(root):
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "hw1.pdf.md").write_text("a card note", encoding="utf-8")
    (root / "Old").mkdir()  # as a deleted group leaves it
    (root / "Old" / "mine.md").write_text("my own note", encoding="utf-8")
    vault.rename_group(root, "HW", "Old")
    assert sorted(path.name for path in (root / "Old").iterdir()) == ["hw1.pdf.md", "mine.md"]
    assert not (root / "HW").exists()


def test_a_rename_never_writes_over_a_file_in_the_other_folder(root):
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "notes.md").write_text("from HW", encoding="utf-8")
    (root / "Old").mkdir()
    (root / "Old" / "notes.md").write_text("already here", encoding="utf-8")
    with pytest.raises(FileExistsError, match="notes.md"):
        vault.rename_group(root, "HW", "Old")
    assert (root / "HW" / "notes.md").read_text(encoding="utf-8") == "from HW"
    assert (root / "Old" / "notes.md").read_text(encoding="utf-8") == "already here"


def test_a_rename_that_only_changes_the_case(root):
    vault.add_group(root, "HW", "homework")
    (root / "HW" / "hw1.pdf.md").write_text("a card note", encoding="utf-8")
    result = vault.rename_group(root, "HW", "Hw")
    assert [(group.id, group.name) for group in result.groups] == [("hw", "Hw")]
    assert [path.name for path in (root / "_groups").iterdir()] == ["Hw.md"]
    assert "Hw" in [path.name for path in root.iterdir()]
    assert "HW" not in [path.name for path in root.iterdir()]
    assert (root / "Hw" / "hw1.pdf.md").exists()


def test_two_groups_that_swap_names_by_hand_keep_their_folders(root):
    vault.add_group(root, "A", "first")
    vault.add_group(root, "B", "second")
    (root / "A" / "a.md").write_text("in A", encoding="utf-8")
    groups = root / "_groups"
    (groups / "A.md").rename(groups / "tmp.md")
    (groups / "B.md").rename(groups / "A.md")
    (groups / "tmp.md").rename(groups / "B.md")
    result = vault.sync_group_set(root)
    assert {group.id: group.name for group in result.groups} == {"a": "B", "b": "A"}
    assert (root / "A" / "a.md").exists()
