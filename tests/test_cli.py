"""The p0 commands, run through Typer's test runner. Sorting gets a fake backend."""

import pytest
from typer.testing import CliRunner

from decide_already.backends.base import BackendError
from decide_already.card import hash_file
from decide_already.cli import app
from decide_already.config import JEV_LOG_FILE, LABELS_FILE, Config, decide_path
from decide_already.log import append_jsonl, read_jsonl
from decide_already.sort import NO_NOTE
from decide_already.types import Placement
from decide_already.vault import add_group, init_vault, read_card_notes, write_card_note
from fakes import FakeBackend

runner = CliRunner()


def test_init_then_group_commands(tmp_path):
    """init sets up the vault, and every group command reports the new set version."""
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "dump: _dump" in result.output

    vault = ["--vault", str(tmp_path)]
    result = runner.invoke(app, ["group", "add", "HW", "--covers", "homework", *vault])
    assert result.exit_code == 0, result.output
    assert "Set version 1" in result.output

    result = runner.invoke(
        app, ["group", "add", "Logistics", "--covers", "syllabus", "--excludes", "projects", *vault]
    )
    assert "Set version 2" in result.output

    result = runner.invoke(
        app, ["group", "edit", "Logistics", "--covers", "syllabus and deadlines", *vault]
    )
    assert "Set version 3" in result.output

    result = runner.invoke(app, ["group", "rename", "HW", "Homework", *vault])
    assert "Set version 4" in result.output

    result = runner.invoke(app, ["group", "list", *vault])
    assert result.exit_code == 0, result.output
    assert "Set version 4" in result.output
    assert "Homework (hw): covers homework" in result.output
    assert "excludes projects" in result.output

    result = runner.invoke(app, ["group", "delete", "Homework", *vault])
    assert "Set version 5" in result.output


def test_errors_exit_nonzero(tmp_path):
    """A known failure prints one error line and exits with status 1. It is never a crash."""
    runner.invoke(app, ["init", str(tmp_path)])
    for name in ("HW", "Logistics"):
        runner.invoke(app, ["group", "add", name, "--covers", "x", "--vault", str(tmp_path)])
    cases = {
        ("group", "add", "Exams", "--covers", "  "): "every group needs a covers description",
        ("group", "add", "HW", "--covers", "again"): "group 'HW' already exists",
        ("group", "add", "Unsorted", "--covers", "x"): (
            "'Unsorted' cannot be a group name: that folder holds the originals or is one of "
            "the tool's own"
        ),
        ("group", "edit", "Nope", "--covers", "x"): "no group named 'Nope'",
        ("group", "edit", "HW"): "pass --covers and/or --excludes",
        ("group", "edit", "HW", "--covers", " "): "every group needs a covers description",
        ("group", "rename", "Nope", "Exams"): "no group named 'Nope'",
        ("group", "rename", "HW", "Logistics"): "group 'Logistics' already exists",
        ("group", "rename", "HW", "a/b"): "'a/b' cannot be a group name",
        ("group", "delete", "Nope"): "no group named 'Nope'",
    }
    for command, message in cases.items():
        result = runner.invoke(app, [*command, "--vault", str(tmp_path)])
        assert isinstance(result.exception, SystemExit), command  # not an uncaught error
        assert result.exit_code == 1, command
        assert result.output == f"error: {message}\n", command
    listed = runner.invoke(app, ["group", "list", "--vault", str(tmp_path)])
    assert listed.output.splitlines() == [
        "Set version 2",
        "  HW (hw): covers x",
        "  Logistics (logistics): covers x",
    ]


def test_init_with_missing_dump_fails(tmp_path):
    """A custom dump folder that does not exist is an error."""
    result = runner.invoke(app, ["init", str(tmp_path), "--dump", "missing"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


# p0 sort and p0 board

TO_HW = ({"hw": 0.914, "logistics": 0.05, "project": 0.036}, 0.95)
TO_LOGISTICS = ({"hw": 0.05, "logistics": 0.9, "project": 0.05}, 0.95)
NO_FIT = ({"hw": 0.41, "logistics": 0.3, "project": 0.29}, 0.2)


@pytest.fixture
def root(tmp_path):
    """A vault with three groups (HW, Logistics, Project), at set version 3, and an empty dump."""
    init_vault(tmp_path)
    add_group(tmp_path, "HW", "homework specs, starter code and solutions")
    add_group(tmp_path, "Logistics", "syllabus, schedule and deadlines")
    add_group(tmp_path, "Project", "project specs and examples")
    return tmp_path


def drop(root, file_id, text="Homework 3: implement the kernel functions.\n"):
    """Put a file in the dump folder."""
    path = root / "_dump" / file_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def use_backend(monkeypatch, backend):
    """Have the CLI build this backend in place of Jev. Returns what it was built with."""
    built = []

    def make_backend(root, config):
        """Stand in for backends.make_backend: no key is looked up and no client is made."""
        built.append((root, config))
        return backend

    monkeypatch.setattr("decide_already.cli.make_backend", make_backend)
    return built


def write_note(root, file_id, group_id, **changes):
    """Write a card note as the sorter would, with any placement fields changed."""
    fields = {
        "source": f"[[_dump/{file_id}]]",
        "file_id": file_id,
        "content_hash": "0" * 64,
        "file_type": "Text",
        "group_id": group_id,
        "placed_by": "jev",
        "set_version": 3,
        "probs": {"hw": 0.75, "logistics": 0.2, "project": 0.05},
        "fits_any": 0.9,
        "model": "fake-0",
    }
    return write_card_note(root, Placement(**{**fields, **changes}), "a card\n")


def test_sort_then_board(root, monkeypatch):
    """p0 sort prints one summary line; p0 board then shows where every file went."""
    drop(root, "hw3/hw3_solution.py", "def kernel():\n    pass\n")
    drop(root, "syllabus.md", "# Syllabus\n\nGrading and deadlines.\n")
    drop(root, "checklist.txt", "Boxes, tape, movers.\n")
    drop(root, "scan.txt", "")
    backend = FakeBackend(
        answers={
            "hw3/hw3_solution.py": TO_HW,
            "syllabus.md": TO_LOGISTICS,
            "checklist.txt": NO_FIT,
        }
    )
    built = use_backend(monkeypatch, backend)

    result = runner.invoke(app, ["sort", "--vault", str(root)])
    assert result.exit_code == 0, result.output
    assert result.output == (
        "Asked 3, placed 2, Unsorted 2, close calls 0, frozen skipped 0. "
        "Set version 3. Spent so far: nothing.\n"
    )
    assert built == [(root.resolve(), Config.load(root))]
    assert [request.file_id for request in backend.calls] == [
        "checklist.txt",
        "hw3/hw3_solution.py",
        "syllabus.md",
    ]

    drop(root, "hw4/new.py")
    result = runner.invoke(app, ["board", "--vault", str(root)])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "Set version 3",
        "HW (1)",
        "  hw3_solution.py  0.91",
        "Logistics (1)",
        "  syllabus.md  0.90",
        "Project (0)",
        "Unsorted (2)",
        "  checklist.txt  0.41  no group fits",
        "  scan.txt  no text",
        "New files not sorted yet: 1",
    ]


def test_sort_shows_what_the_vault_has_spent_so_far(root, monkeypatch):
    """The spend is the total in the vault's Jev log, to four decimals."""
    drop(root, "a.txt")
    append_jsonl(decide_path(root, JEV_LOG_FILE), {"file_id": "a.txt", "input_tokens": 1_500_000})
    use_backend(monkeypatch, FakeBackend(default=TO_HW))
    result = runner.invoke(app, ["sort", "--vault", str(root)])
    assert result.exit_code == 0, result.output
    assert result.output.endswith("Set version 3. Spent so far: $0.0630.\n")


def test_sort_prints_one_line_per_error_and_exits_1(root, monkeypatch):
    """Files that could not be sorted are listed after the summary, and the exit status says so."""
    for file_id in ("a.txt", "b.txt", "hw3/c.txt"):
        drop(root, file_id)
    use_backend(monkeypatch, FakeBackend(default=TO_HW, fail={"b.txt", "hw3/c.txt"}))
    result = runner.invoke(app, ["sort", "--vault", str(root)])
    assert result.exit_code == 1
    summary, *errors = result.output.splitlines()
    assert summary == (
        "Asked 3, placed 1, Unsorted 0, close calls 0, frozen skipped 0. "
        "Set version 3. Spent so far: nothing."
    )
    assert errors == [
        "error: b.txt: the fake backend was told to fail for b.txt",
        "error: hw3/c.txt: the fake backend was told to fail for hw3/c.txt",
    ]


def test_sort_counts_frozen_files_and_close_calls(root, monkeypatch):
    """A rerun: the frozen file is skipped, and a near tie is placed and counted."""
    Config.load(root).model_copy(update={"unsorted_top_below": 0.4}).save(root)
    drop(root, "a.txt")
    drop(root, "b.txt")
    write_note(root, "a.txt", "hw", frozen=True, content_hash=hash_file(root / "_dump" / "a.txt"))
    close = ({"hw": 0.5, "logistics": 0.4, "project": 0.1}, 0.9)
    backend = FakeBackend(default=close)
    use_backend(monkeypatch, backend)
    result = runner.invoke(app, ["sort", "--vault", str(root)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith(
        "Asked 1, placed 1, Unsorted 0, close calls 1, frozen skipped 1. Set version 3."
    )
    assert [request.file_id for request in backend.calls] == ["b.txt"]


def test_sort_exits_1_when_the_backend_cannot_be_built(root, monkeypatch):
    """No API key: one error line, no summary, and nothing is written."""
    drop(root, "a.txt")

    def make_backend(root, config):
        """Fail the way JevBackend does when it is given no key."""
        raise BackendError("no TypeSafe API key found: set TYPESAFE_API_KEY")

    monkeypatch.setattr("decide_already.cli.make_backend", make_backend)
    result = runner.invoke(app, ["sort", "--vault", str(root)])
    assert result.exit_code == 1
    assert result.output == "error: no TypeSafe API key found: set TYPESAFE_API_KEY\n"
    assert read_card_notes(root) == []


def test_sort_and_board_print_an_error_line_when_a_file_is_in_the_way_of_a_card_note(
    root, monkeypatch
):
    """A file of the user's where sync has to put a card note: one error line, no traceback."""
    drop(root, "a.txt")
    write_note(root, "a.txt", "hw", content_hash=hash_file(root / "_dump" / "a.txt"))
    (root / "Unsorted" / "a.txt.md").write_text("my own note\n", encoding="utf-8")
    runner.invoke(app, ["group", "delete", "HW", "--vault", str(root)])
    use_backend(monkeypatch, FakeBackend())
    result = runner.invoke(app, ["sort", "--vault", str(root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.output.startswith("error: ")
    assert "a.txt.md exists and is not the card note of a.txt" in result.output
    assert (root / "Unsorted" / "a.txt.md").read_text(encoding="utf-8") == "my own note\n"


def test_sort_exits_1_when_there_are_no_groups(tmp_path, monkeypatch):
    """With no groups there is nothing to sort into, and the backend is never asked."""
    init_vault(tmp_path)
    drop(tmp_path, "a.txt")
    backend = FakeBackend()
    use_backend(monkeypatch, backend)
    result = runner.invoke(app, ["sort", "--vault", str(tmp_path)])
    assert result.exit_code == 1
    assert result.output.startswith("error: there are no groups yet")
    assert backend.calls == []


def test_sort_exits_1_for_an_unknown_variant_in_the_config(root, monkeypatch):
    """A typo in the config's variant is reported before anything is asked."""
    Config.load(root).model_copy(update={"variant": "cards"}).save(root)
    drop(root, "a.txt")
    backend = FakeBackend()
    use_backend(monkeypatch, backend)
    result = runner.invoke(app, ["sort", "--vault", str(root)])
    assert result.exit_code == 1
    assert result.output.startswith("error: unknown variant 'cards'")
    assert backend.calls == []


def test_board_shows_marks_reasons_and_new_files(root):
    """Close calls first in each group, each file with its marks, Unsorted with the reason."""
    add_group(root, "Exams", "midterms and finals")  # set version 4: version 3 is older
    current = {"set_version": 4}
    write_note(root, "hw3/hw3_starter.py", "hw", probs={"hw": 0.966, "project": 0.034})
    write_note(root, "hw3/hw3_solution.py", "hw", frozen=True, **current)
    write_note(root, "hw3/hw3_spec.pdf", "hw", close_call=True, **current)
    write_note(root, "syllabus.md", "logistics", placed_by="you", frozen=True, **current)
    write_note(root, "everything.txt", "project", close_call=True, frozen=True, placed_by="you")
    write_note(root, "checklist.txt", None, reason="no_group_fits", **current)
    write_note(root, "mixed.txt", None, reason="weak_top", probs={"hw": 0.344}, **current)
    write_note(root, "moved_out.txt", None, placed_by="you", frozen=True, **current)
    # Dragged from Unsorted into HW and not synced yet: its old reason no longer applies.
    dragged = write_note(root, "dragged.txt", None, reason="weak_top", **current)
    dragged.rename(root / "HW" / dragged.name)
    write_note(
        root,
        "scan.pdf",
        None,
        placed_by="code",
        reason="no_text",
        probs=None,
        fits_any=None,
        model=None,
        **current,
    )
    drop(root, "new1.txt")
    drop(root, "hw4/new2.txt")

    result = runner.invoke(app, ["board", "--vault", str(root)])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "Set version 4",
        "Exams (0)",
        "HW (4)",
        "  hw3_spec.pdf  0.75  close call",
        "  dragged.txt  0.75",
        "  hw3_solution.py  0.75  frozen",
        "  hw3_starter.py  0.97  older set",
        "Logistics (1)",
        "  syllabus.md  0.75  frozen, moved by you",
        "Project (1)",
        "  everything.txt  0.75  close call, frozen, moved by you, older set",
        "Unsorted (4)",
        "  checklist.txt  0.75  no group fits",
        "  mixed.txt  0.34  weak top pick",
        "  moved_out.txt  0.75  frozen, moved by you",
        "  scan.pdf  no text",
        "New files not sorted yet: 2",
    ]


def test_board_of_a_vault_with_no_groups(tmp_path):
    """Before any group exists the board is short, and it is not an error."""
    init_vault(tmp_path)
    drop(tmp_path, "a.txt")
    result = runner.invoke(app, ["board", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "Set version 0",
        "Unsorted (0)",
        "New files not sorted yet: 1",
    ]


def test_sort_and_board_exit_1_outside_a_vault(tmp_path, monkeypatch):
    """A folder that was never set up as a vault is an error, and nothing is created in it."""
    use_backend(monkeypatch, FakeBackend())
    for command in ("sort", "board"):
        result = runner.invoke(app, [command, "--vault", str(tmp_path)])
        assert result.exit_code == 1
        assert result.output.startswith("error: no vault at")
    assert list(tmp_path.iterdir()) == []


def test_board_reports_a_broken_group_note(root):
    """A group note without covers is the user's to fix: one error line, not a traceback."""
    (root / "_groups" / "Empty.md").write_text("---\nid: empty\n---\n", encoding="utf-8")
    result = runner.invoke(app, ["board", "--vault", str(root)])
    assert result.exit_code == 1
    assert result.output.startswith("error: group 'Empty' has no covers description")


# p0 move, freeze, unfreeze, suggest and sync


def place(root, file_id, group_id, **changes):
    """Put a file in the dump and write its card note, as if a sort had placed it."""
    drop(root, file_id)
    content_hash = hash_file(root / "_dump" / file_id)
    return write_note(root, file_id, group_id, content_hash=content_hash, **changes)


def p0(root, *args, exit_code=0):
    """Run one p0 command on a vault, check its exit status and return what it printed."""
    result = runner.invoke(app, [*args, "--vault", str(root)])
    assert result.exit_code == exit_code, result.output
    return result.output


def labels(root):
    """Every line of the labels file as (kind, file id, from group, to group)."""
    lines = read_jsonl(decide_path(root, LABELS_FILE))
    return [(line["kind"], line["file_id"], line["from_group"], line["to_group"]) for line in lines]


def folders(root):
    """The folder each file's card note sits in."""
    return {note.placement.file_id: note.folder for note in read_card_notes(root)}


def test_move_puts_a_file_in_a_group_and_freezes_it(root):
    """One line says where the file is now. The move is a label, and the file is frozen."""
    place(root, "project/sheet.md", "project")
    assert p0(root, "move", "sheet.md", "Logistics") == (
        "project/sheet.md is now in Logistics, frozen.\n"
    )
    (note,) = read_card_notes(root)
    assert note.path == root / "Logistics" / "project__sheet.md.md"
    assert (note.placement.placed_by, note.placement.frozen) == ("you", True)
    # By its full id and the group's id. Unsorted is a place to move a file to as well.
    assert p0(root, "move", "project/sheet.md", "hw") == "project/sheet.md is now in HW, frozen.\n"
    assert p0(root, "move", "sheet.md", "Unsorted") == (
        "project/sheet.md is now in Unsorted, frozen.\n"
    )
    # Already there: nothing more to log.
    assert p0(root, "move", "sheet.md", "Unsorted") == (
        "project/sheet.md is now in Unsorted, frozen.\n"
    )
    assert labels(root) == [
        ("moved", "project/sheet.md", "project", "logistics"),
        ("moved", "project/sheet.md", "logistics", "hw"),
        ("moved", "project/sheet.md", "hw", None),
    ]


def test_freeze_and_unfreeze_print_each_file_and_a_total(root):
    """A group or a file: one line per file that changed, then how many."""
    place(root, "a.txt", "hw")
    place(root, "hw3/b.py", "hw")
    place(root, "c.txt", "logistics")
    assert p0(root, "freeze", "HW") == "frozen: a.txt\nfrozen: hw3/b.py\nFroze 2 files.\n"
    assert p0(root, "freeze", "c.txt") == "frozen: c.txt\nFroze 1 file.\n"
    assert p0(root, "freeze", "hw") == "Nothing to freeze.\n"
    assert p0(root, "freeze", "Unsorted") == "Nothing to freeze.\n"
    assert all(note.placement.frozen for note in read_card_notes(root))

    assert p0(root, "unfreeze", "b.py") == "unfrozen: hw3/b.py\nUnfroze 1 file.\n"
    assert p0(root, "unfreeze", "Logistics") == "unfrozen: c.txt\nUnfroze 1 file.\n"
    assert p0(root, "unfreeze", "Project") == "Nothing to unfreeze.\n"
    assert p0(root, "unfreeze", "c.txt") == "Nothing to unfreeze.\n"
    assert labels(root) == [
        ("confirmed", "a.txt", "hw", "hw"),
        ("confirmed", "hw3/b.py", "hw", "hw"),
        ("confirmed", "c.txt", "logistics", "logistics"),
        ("unfrozen", "hw3/b.py", "hw", "hw"),
        ("unfrozen", "c.txt", "logistics", "logistics"),
    ]


def test_a_reference_that_names_nothing_exits_1(root):
    """An unknown file or group, or a name that several files share: one error line each."""
    place(root, "a/notes.txt", "hw")
    place(root, "b/notes.txt", "hw")
    shared = "2 files are named 'notes.txt': a/notes.txt, b/notes.txt; use the full id of one"
    hint = "(a new file gets its card note from `p0 sort`)"
    cases = {
        ("move", "nope.txt", "HW"): f"no sorted file with the id or name 'nope.txt' {hint}",
        ("move", "notes.txt", "HW"): shared,
        ("move", "a/notes.txt", "Exams"): (
            "no group with the id or name 'Exams'; pick one of: HW, Logistics, Project, Unsorted"
        ),
        ("freeze", "nope"): f"no group or sorted file with the id or name 'nope' {hint}",
        ("freeze", "notes.txt"): shared,
        ("unfreeze", "nope"): f"no group or sorted file with the id or name 'nope' {hint}",
        ("unfreeze", "notes.txt"): shared,
    }
    for args, message in cases.items():
        assert p0(root, *args, exit_code=1) == f"error: {message}\n"
    assert labels(root) == []
    assert not any(note.placement.frozen for note in read_card_notes(root))


def test_move_reports_a_note_of_the_users_that_is_in_the_way(root):
    """The file's note cannot go where a file of the user's sits: an error line, no traceback."""
    place(root, "a.txt", "hw")
    (root / "Logistics" / "a.txt.md").write_text("My own note.\n", encoding="utf-8")
    output = p0(root, "move", "a.txt", "Logistics", exit_code=1)
    assert output == "error: Logistics/a.txt.md exists and is not the card note of a.txt\n"
    assert labels(root) == []


def test_suggest_lists_the_frozen_files_the_model_would_place_elsewhere(root, monkeypatch):
    """One line per difference and a total. Nothing moves."""
    place(root, "a.txt", "hw")
    place(root, "b.txt", "hw")
    place(root, "c.txt", "logistics")
    place(root, "d.txt", None, reason="weak_top")
    place(root, "not_frozen.txt", "project")
    for group in ("HW", "Logistics", "Unsorted"):
        p0(root, "freeze", group)
    backend = FakeBackend(
        answers={"a.txt": TO_LOGISTICS, "b.txt": TO_HW, "c.txt": NO_FIT, "d.txt": TO_HW}
    )
    built = use_backend(monkeypatch, backend)
    before = folders(root)
    assert p0(root, "suggest").splitlines() == [
        "a.txt: in HW, would now go to Logistics",
        "c.txt: in Logistics, would now go to Unsorted",
        "d.txt: in Unsorted, would now go to HW",
        "3 suggestions. Nothing was moved.",
    ]
    assert built == [(root.resolve(), Config.load(root))]
    assert [request.file_id for request in backend.calls] == ["a.txt", "b.txt", "c.txt", "d.txt"]
    assert folders(root) == before
    assert len(labels(root)) == 4  # the four confirmations from the freezes, and nothing new

    agrees = FakeBackend(
        answers={"a.txt": TO_HW, "b.txt": TO_HW, "c.txt": TO_LOGISTICS, "d.txt": NO_FIT}
    )
    use_backend(monkeypatch, agrees)
    assert p0(root, "suggest") == "No suggestions.\n"
    one = FakeBackend(
        answers={"a.txt": TO_HW, "b.txt": TO_LOGISTICS, "c.txt": TO_LOGISTICS, "d.txt": NO_FIT}
    )
    use_backend(monkeypatch, one)
    assert p0(root, "suggest") == (
        "b.txt: in HW, would now go to Logistics\n1 suggestion. Nothing was moved.\n"
    )


def test_suggest_reports_the_files_the_model_could_not_be_asked_about(root, monkeypatch):
    """A failure is not agreement: it is listed after the suggestions, and the exit status is 1."""
    for file_id in ("a.txt", "b.txt", "c.txt"):
        place(root, file_id, "hw")
    p0(root, "freeze", "HW")
    use_backend(monkeypatch, FakeBackend(default=TO_LOGISTICS, fail={"b.txt"}))
    assert p0(root, "suggest", exit_code=1).splitlines() == [
        "a.txt: in HW, would now go to Logistics",
        "c.txt: in HW, would now go to Logistics",
        "2 suggestions. Nothing was moved.",
        "error: b.txt: the fake backend was told to fail for b.txt",
    ]
    use_backend(monkeypatch, FakeBackend(default=TO_HW, fail={"a.txt", "b.txt", "c.txt"}))
    assert p0(root, "suggest", exit_code=1).splitlines() == [
        "No suggestions.",
        "error: a.txt: the fake backend was told to fail for a.txt",
        "error: b.txt: the fake backend was told to fail for b.txt",
        "error: c.txt: the fake backend was told to fail for c.txt",
    ]


def test_suggest_exits_1_when_the_backend_cannot_be_built(root, monkeypatch):
    """No API key: one error line, and nothing is asked or written."""
    place(root, "a.txt", "hw", frozen=True)

    def make_backend(root, config):
        """Fail the way JevBackend does when it is given no key."""
        raise BackendError("no TypeSafe API key found: set TYPESAFE_API_KEY")

    monkeypatch.setattr("decide_already.cli.make_backend", make_backend)
    assert p0(root, "suggest", exit_code=1) == (
        "error: no TypeSafe API key found: set TYPESAFE_API_KEY\n"
    )
    assert labels(root) == []


def test_sync_prints_what_it_picked_up(root):
    """One line per file for each kind of edit made by hand, the new set, then the totals."""
    for file_id in ("dragged.txt", "frozen.txt", "thawed.txt", "gone.txt", "stray.txt"):
        place(root, file_id, "hw")
    place(root, "orphan.txt", "project")
    p0(root, "freeze", "thawed.txt")
    assert p0(root, "sync") == "Nothing to pick up.\n"

    dragged = root / "HW" / "dragged.txt.md"
    dragged.rename(root / "Logistics" / dragged.name)
    (root / "Old stuff").mkdir()  # a folder that is no group: a note dragged there is put back
    (root / "HW" / "stray.txt.md").rename(root / "Old stuff" / "stray.txt.md")
    for name, old, new in (("frozen.txt", "false", "true"), ("thawed.txt", "true", "false")):
        note = root / "HW" / f"{name}.md"
        text = note.read_text(encoding="utf-8")
        assert f"\nfrozen: {old}\n" in text
        note.write_text(text.replace(f"\nfrozen: {old}\n", f"\nfrozen: {new}\n"), encoding="utf-8")
    (root / "_dump" / "gone.txt").unlink()
    (root / "_groups" / "Project.md").unlink()  # the group deleted by hand, its folder left behind

    assert p0(root, "sync").splitlines() == [
        "moved: dragged.txt",
        "confirmed: frozen.txt",
        "unfrozen: thawed.txt",
        "sent to Unsorted, group deleted: orphan.txt",
        "put back in its group's folder: stray.txt",
        "original missing: gone.txt",
        "Group notes changed: set version 4.",
        "Moved 1, confirmed 1, unfrozen 1, sent to Unsorted 1, put back 1, originals missing 1.",
    ]
    assert folders(root) == {
        "dragged.txt": "Logistics",
        "frozen.txt": "HW",
        "gone.txt": "HW",
        "orphan.txt": "Unsorted",
        "stray.txt": "HW",
        "thawed.txt": "HW",
    }
    assert labels(root) == [
        ("confirmed", "thawed.txt", "hw", "hw"),
        ("moved", "dragged.txt", "hw", "logistics"),
        ("confirmed", "frozen.txt", "hw", "hw"),
        ("unfrozen", "thawed.txt", "hw", "hw"),
    ]
    # Again: all of it is recorded, so only the missing original is still listed.
    assert p0(root, "sync").splitlines() == [
        "original missing: gone.txt",
        "Moved 0, confirmed 0, unfrozen 0, sent to Unsorted 0, put back 0, originals missing 1.",
    ]
    assert len(labels(root)) == 4


def test_sync_reports_a_group_note_edited_by_hand(root):
    """An edit to a group note is a new set version, even when no card note was touched."""
    note = root / "_groups" / "Logistics.md"
    text = note.read_text(encoding="utf-8")
    note.write_text(text.replace("syllabus, schedule", "policies, schedule"), encoding="utf-8")
    assert p0(root, "sync") == "Group notes changed: set version 4.\n"
    assert p0(root, "sync") == "Nothing to pick up.\n"


def test_sync_reports_a_broken_group_note(root):
    """A group note without covers is the user's to fix: one error line, not a traceback."""
    (root / "_groups" / "Empty.md").write_text("---\nid: empty\n---\n", encoding="utf-8")
    for command in (("sync",), ("freeze", "HW"), ("unfreeze", "HW"), ("move", "a.txt", "HW")):
        output = p0(root, *command, exit_code=1)
        assert output.startswith("error: group 'Empty' has no covers description")


def test_the_correction_commands_exit_1_outside_a_vault(tmp_path, monkeypatch):
    """A folder that was never set up as a vault is an error, and nothing is created in it."""
    use_backend(monkeypatch, FakeBackend())
    commands = (
        ("move", "a.txt", "HW"),
        ("freeze", "HW"),
        ("unfreeze", "HW"),
        ("suggest",),
        ("sync",),
    )
    for command in commands:
        assert p0(tmp_path, *command, exit_code=1).startswith("error: no vault at")
    assert list(tmp_path.iterdir()) == []


def test_the_walkthrough_through_the_commands(tmp_path, monkeypatch):
    """docs/phase0.md end to end with p0 commands, the fake backend and one drag by hand."""
    root = tmp_path
    runner.invoke(app, ["init", str(root)])
    p0(root, "group", "add", "HW", "--covers", "homework specs, starter code, solutions")
    p0(root, "group", "add", "Logistics", "--covers", "syllabus, schedule, deadlines")
    p0(root, "group", "add", "Project", "--covers", "project specs and examples")
    answers = {
        "hw3_skeleton.py": ({"hw": 0.91, "logistics": 0.03, "project": 0.06}, 0.96),
        "hw3_solution.py": ({"hw": 0.93, "logistics": 0.02, "project": 0.05}, 0.97),
        "lecture_slides.md": ({"hw": 0.12, "logistics": 0.7, "project": 0.18}, 0.62),
        "project_examples.md": ({"hw": 0.1, "logistics": 0.04, "project": 0.86}, 0.94),
        "project_logistics_sheet.md": ({"hw": 0.03, "logistics": 0.3, "project": 0.67}, 0.9),
        "syllabus.md": ({"hw": 0.02, "logistics": 0.95, "project": 0.03}, 0.98),
    }
    for file_id in answers:
        drop(root, file_id, f"Text of {file_id}.\n")
    backend = FakeBackend(answers=answers)
    use_backend(monkeypatch, backend)
    counts = "Unsorted 0, close calls 0"

    assert p0(root, "sort").startswith(f"Asked 6, placed 6, {counts}, frozen skipped 0.")
    p0(root, "group", "add", "Lecture slides and notes", "--covers", "lecture slides, notes")
    assert p0(root, "freeze", "HW") == (
        "frozen: hw3_skeleton.py\nfrozen: hw3_solution.py\nFroze 2 files.\n"
    )
    assert p0(root, "freeze", "Project").endswith("Froze 2 files.\n")
    assert p0(root, "freeze", "syllabus.md") == "frozen: syllabus.md\nFroze 1 file.\n"

    slides = {"hw": 0.03, "lecture-slides-and-notes": 0.9, "logistics": 0.05, "project": 0.02}
    backend.answers["lecture_slides.md"] = (slides, 0.97)
    assert p0(root, "sort").startswith(f"Asked 1, placed 1, {counts}, frozen skipped 5.")
    assert [request.file_id for request in backend.calls[6:]] == ["lecture_slides.md"]

    # The sheet is dragged to Logistics in Obsidian, and `p0 sync` picks the move up.
    sheet = root / "Project" / "project_logistics_sheet.md.md"
    sheet.rename(root / "Logistics" / sheet.name)
    assert p0(root, "sync").splitlines() == [
        "moved: project_logistics_sheet.md",
        "Moved 1, confirmed 0, unfrozen 0, sent to Unsorted 0, put back 0, originals missing 0.",
    ]
    assert p0(root, "freeze", "Lecture slides and notes") == (
        "frozen: lecture_slides.md\nFroze 1 file.\n"
    )

    # Everything is frozen: a rerun asks about nothing, and the board shows the final groups.
    assert p0(root, "sort").startswith(f"Asked 0, placed 0, {counts}, frozen skipped 6.")
    assert len(backend.calls) == 7
    assert p0(root, "board").splitlines() == [
        "Set version 4",
        "HW (2)",
        "  hw3_skeleton.py  0.91  frozen, older set",
        "  hw3_solution.py  0.93  frozen, older set",
        "Lecture slides and notes (1)",
        "  lecture_slides.md  0.90  frozen",
        "Logistics (2)",
        "  project_logistics_sheet.md  0.67  frozen, moved by you, older set",
        "  syllabus.md  0.95  frozen, older set",
        "Project (1)",
        "  project_examples.md  0.86  frozen, older set",
        "Unsorted (0)",
        "New files not sorted yet: 0",
    ]
    lectures = "lecture-slides-and-notes"
    assert labels(root) == [
        ("confirmed", "hw3_skeleton.py", "hw", "hw"),
        ("confirmed", "hw3_solution.py", "hw", "hw"),
        ("confirmed", "project_examples.md", "project", "project"),
        ("confirmed", "project_logistics_sheet.md", "project", "project"),
        ("confirmed", "syllabus.md", "logistics", "logistics"),
        ("moved", "project_logistics_sheet.md", "project", "logistics"),
        ("confirmed", "lecture_slides.md", lectures, lectures),
    ]


# Running p0 inside the vault, with no --vault


def test_the_commands_find_the_vault_from_a_folder_inside_it(root, monkeypatch):
    """The usual way to run p0: from the vault or a folder in it, with no --vault."""
    drop(root, "a.txt")
    use_backend(monkeypatch, FakeBackend(default=TO_HW))
    monkeypatch.chdir(root / "HW")
    result = runner.invoke(app, ["sort"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Asked 1, placed 1, Unsorted 0")
    result = runner.invoke(app, ["board"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[:3] == ["Set version 3", "HW (1)", "  a.txt  0.91"]
    monkeypatch.chdir(root)
    result = runner.invoke(app, ["group", "list"])
    assert result.output.startswith("Set version 3\n  HW (hw): covers homework specs")
    assert runner.invoke(app, ["freeze", "a.txt"]).output == "frozen: a.txt\nFroze 1 file.\n"


def test_the_commands_exit_1_when_no_vault_is_at_or_above_the_folder(tmp_path, monkeypatch):
    """Outside any vault and with no --vault: one error line that says to run p0 init."""
    for above in [tmp_path.resolve(), *tmp_path.resolve().parents]:
        if Config.path(above).exists():
            pytest.skip(f"{above} is a vault, so no temp folder under it is outside one")
    use_backend(monkeypatch, FakeBackend())
    monkeypatch.chdir(tmp_path)
    commands = (
        ("board",),
        ("sort",),
        ("sync",),
        ("move", "a.txt", "HW"),
        ("group", "list"),
        ("group", "add", "HW", "--covers", "homework"),
    )
    for command in commands:
        result = runner.invoke(app, list(command))
        assert isinstance(result.exception, SystemExit), command
        assert result.exit_code == 1, command
        assert result.output.startswith("error: no vault at or above"), command
        assert result.output.endswith("run `p0 init` first\n"), command
    assert list(tmp_path.iterdir()) == []


def test_the_group_commands_exit_1_in_a_folder_that_is_not_a_vault(tmp_path):
    """--vault names a plain folder: an error line, and nothing is created in it."""
    commands = (
        ("group", "add", "HW", "--covers", "homework"),
        ("group", "edit", "HW", "--covers", "homework"),
        ("group", "rename", "HW", "Homework"),
        ("group", "delete", "HW"),
        ("group", "list"),
    )
    for command in commands:
        assert p0(tmp_path, *command, exit_code=1) == (
            f"error: no vault at {tmp_path.resolve()}; run `p0 init` first\n"
        )
    assert list(tmp_path.iterdir()) == []


# Slips made by hand are error lines, never tracebacks

EVERY_COMMAND = (
    ("group", "list"),
    ("group", "add", "Exams", "--covers", "midterms"),
    ("group", "edit", "HW", "--covers", "problem sets"),
    ("group", "rename", "HW", "Homework"),
    ("group", "delete", "Project"),
    ("board",),
    ("sync",),
    ("sort",),
    ("suggest",),
    ("move", "a.txt", "HW"),
    ("freeze", "HW"),
    ("unfreeze", "HW"),
)


def fails_with_a_line(root, command):
    """Run a command that must fail, check that it failed cleanly, and return its output."""
    result = runner.invoke(app, [*command, "--vault", str(root)])
    assert isinstance(result.exception, SystemExit), (command, result.exception)
    assert result.exit_code == 1, command
    assert result.output.startswith("error: "), command
    assert "Traceback" not in result.output
    return result.output


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ("covers: syllabus, schedule, deadlines: for homework and projects", "not valid YAML"),
        ("covers: [syllabus, schedule]", "the covers of a group must be text"),
        ("covers: 2024", "the covers of a group must be text"),
        ("covers: 2026-11-15", "the covers of a group must be text"),
        ("- syllabus\n- schedule", "must be `key: value` lines"),
    ],
)
def test_a_slip_in_a_group_note_is_an_error_line_from_every_command(
    root, monkeypatch, frontmatter, message
):
    """A colon in a description, or a list where text belongs: each command names the note."""
    place(root, "a.txt", "logistics")
    note = root / "_groups" / "Logistics.md"
    note.write_text(f"---\n{frontmatter}\n---\n", encoding="utf-8")
    backend = FakeBackend()
    use_backend(monkeypatch, backend)
    for command in EVERY_COMMAND:
        output = fails_with_a_line(root, command)
        assert message in output, command
        assert str(note) in output, command
    assert backend.calls == []
    # The rename was stopped before anything was renamed.
    assert (root / "_groups" / "HW.md").exists()
    assert (root / "HW").is_dir()
    assert not (root / "Homework").exists()
    # The delete was carried out, and its error line said so.
    assert not (root / "_groups" / "Project.md").exists()
    # Once the slip is fixed, everything works again.
    note.write_text("---\nid: logistics\ncovers: syllabus\n---\n", encoding="utf-8")
    assert p0(root, "group", "list").startswith("Set version 4\n")


def test_a_delete_that_another_broken_note_follows_says_that_it_was_carried_out(root):
    """The note is gone, as asked. The line says so before it names the note that is broken."""
    draft = root / "_groups" / "Draft.md"
    draft.write_text("---\nid: draft\n---\n", encoding="utf-8")
    assert p0(root, "group", "delete", "HW", exit_code=1) == (
        "error: the note of 'HW' was deleted, but the groups could not be recorded: "
        f"group 'Draft' has no covers description: {draft}\n"
    )
    assert not (root / "_groups" / "HW.md").exists()
    # Deleting the broken note is the way out.
    assert p0(root, "group", "delete", "Draft") == "Deleted 'Draft'. Set version 4 (2 groups).\n"


def test_a_slip_in_the_config_is_an_error_line_from_every_command(root, monkeypatch):
    """A YAML slip in .decide/config.yaml: each command names the file and changes nothing."""
    place(root, "a.txt", "hw")
    config = Config.path(root)
    text = config.read_text(encoding="utf-8")
    config.write_text(text.replace("max_spend_usd: 5.0", "max_spend_usd: : 3"), encoding="utf-8")
    built = use_backend(monkeypatch, FakeBackend())
    for command in EVERY_COMMAND:
        output = fails_with_a_line(root, command)
        assert f"{config} is not valid YAML" in output, command
    result = runner.invoke(app, ["init", str(root)])
    assert (result.exit_code, isinstance(result.exception, SystemExit)) == (1, True)
    assert result.output.startswith(f"error: {config} is not valid YAML")
    assert built == []
    assert sorted(path.name for path in (root / "_groups").iterdir()) == [
        "HW.md",
        "Logistics.md",
        "Project.md",
    ]


def test_a_key_typed_into_the_config_is_never_printed(root, monkeypatch):
    """`api_key: <key>` in the config by mistake: the setting is named, the key is not shown."""
    key = "ts_live_MADEUP_0123456789abcdefghijklmnopqrstuvwxyz"
    config = Config.path(root)
    with config.open("a", encoding="utf-8") as handle:
        handle.write(f"api_key: {key}\n")
    built = use_backend(monkeypatch, FakeBackend())
    outputs = [fails_with_a_line(root, command) for command in EVERY_COMMAND]
    outputs.append(runner.invoke(app, ["init", str(root)]).output)
    for output in outputs:
        assert "api_key: Extra inputs are not permitted" in output
        assert str(config) in output
        assert key not in output
        assert key[:10] not in output
    assert built == []


# The model pin


@pytest.mark.parametrize(
    "model", ["jev-latest", "jev-preview", "JEV-LATEST", "jev", "jev-1.14", ""]
)
def test_sort_exits_1_before_any_backend_is_built_when_the_model_is_not_an_exact_version(
    root, monkeypatch, model
):
    """A config that names a moving alias never reaches the model: no client is even made."""
    drop(root, "a.txt")
    Config.load(root).model_copy(update={"model": model}).save(root)
    built = use_backend(monkeypatch, FakeBackend(default=TO_HW))
    for command in ("sort", "suggest"):
        output = p0(root, command, exit_code=1)
        assert output.startswith(f"error: {Config.path(root)} is not a valid config: ")
        assert "model: an exact model version is needed, such as jev-1.14.0" in output
    assert built == []
    assert list((root / "HW").iterdir()) == []


# The dump folder is never a group's


@pytest.mark.parametrize("dump", [".", "./", "..", "_groups", "Unsorted/files", "/tmp/files"])
def test_init_refuses_a_dump_folder_that_is_not_a_plain_folder_inside_the_vault(tmp_path, dump):
    """`--dump .` made every note an original. It is refused, and nothing is set up."""
    result = runner.invoke(app, ["init", str(tmp_path), "--dump", dump])
    assert isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    assert result.output.startswith("error: the dump folder ")
    assert list(tmp_path.iterdir()) == []


def test_no_group_command_moves_the_dump_folder(tmp_path, monkeypatch):
    """`p0 init --dump lectures/`, then a group named after that folder: refused every way."""
    originals = tmp_path / "lectures"
    originals.mkdir()
    (originals / "lecture01.txt").write_text("Lecture 1: what is learning?\n", encoding="utf-8")
    result = runner.invoke(app, ["init", str(tmp_path), "--dump", "lectures/"])
    assert result.exit_code == 0, result.output
    assert "(dump: lectures, model: jev-1.14.0)" in result.output
    p0(tmp_path, "group", "add", "Notes", "--covers", "lecture notes")
    refused = "cannot be a group name: that folder holds the originals or is one of the tool's own"
    for name in ("lectures", "Lectures", "LECTURES"):
        added = p0(tmp_path, "group", "add", name, "--covers", "lecture notes", exit_code=1)
        renamed = p0(tmp_path, "group", "rename", "Notes", name, exit_code=1)
        assert added == renamed == f"error: {name!r} {refused}\n"
    # A group note made by hand under that name stops every command, and moves nothing.
    by_hand = tmp_path / "_groups" / "Lectures.md"
    by_hand.write_text("---\ncovers: lecture notes\n---\n", encoding="utf-8")
    use_backend(monkeypatch, FakeBackend())
    for command in (("sort",), ("board",), ("group", "rename", "Lectures", "Slides")):
        output = p0(tmp_path, *command, exit_code=1)
        assert output == f"error: 'Lectures' {refused}: {by_hand}\n"
    p0(tmp_path, "group", "delete", "Lectures")

    backend = FakeBackend()
    use_backend(monkeypatch, backend)
    for _ in range(2):
        assert p0(tmp_path, "sort").startswith("Asked 1, placed 1")
    assert [request.file_id for request in backend.calls] == ["lecture01.txt"] * 2
    assert [path.name for path in originals.iterdir()] == ["lecture01.txt"]
    assert (originals / "lecture01.txt").read_text(encoding="utf-8").startswith("Lecture 1")
    assert folders(tmp_path) == {"lecture01.txt": "Notes"}


# Group notes that share an id, and labeled files whose note is gone


def test_a_copied_group_note_is_an_error_line(root, monkeypatch):
    """A group note duplicated in Obsidian carries the same id: the id is named, nothing runs."""
    drop(root, "a.txt")
    copy = root / "_groups" / "HW copy.md"
    copy.write_bytes((root / "_groups" / "HW.md").read_bytes())
    backend = FakeBackend()
    use_backend(monkeypatch, backend)
    for command in (("group", "list"), ("board",), ("sort",), ("sync",)):
        assert p0(root, *command, exit_code=1) == "error: two group notes share an id: hw\n"
    assert backend.calls == []
    copy.unlink()
    assert p0(root, "group", "list").startswith("Set version 3\n")


def test_a_labeled_file_whose_note_is_gone_is_reported_and_unfreeze_is_the_way_out(
    root, monkeypatch
):
    """The sort names the file and what to do. `p0 unfreeze` with its id starts it over."""
    place(root, "week1/sheet.txt", "logistics")
    p0(root, "move", "sheet.txt", "HW")
    (root / "HW" / "week1__sheet.txt.md").unlink()
    backend = FakeBackend(default=TO_LOGISTICS)
    use_backend(monkeypatch, backend)
    summary, error = p0(root, "sort", exit_code=1).splitlines()
    assert summary.startswith("Asked 0, placed 0, Unsorted 0, close calls 0, frozen skipped 0.")
    assert error == f"error: week1/sheet.txt: {NO_NOTE}"
    assert backend.calls == []

    assert p0(root, "unfreeze", "week1/sheet.txt") == (
        "unfrozen: week1/sheet.txt\nUnfroze 1 file.\n"
    )
    assert p0(root, "sort").startswith("Asked 1, placed 1")
    assert folders(root) == {"week1/sheet.txt": "Logistics"}
    assert labels(root) == [
        ("moved", "week1/sheet.txt", "logistics", "hw"),
        ("unfrozen", "week1/sheet.txt", "hw", "hw"),
    ]


def test_remove_asks_first_and_deletes_only_on_yes(root):
    """p0 remove deletes an original for good, so it asks, unless --yes is given."""
    drop(root, "a.txt")
    vault = ["--vault", str(root)]
    result = runner.invoke(app, ["remove", "a.txt", *vault], input="n\n")
    assert result.exit_code == 1
    assert (root / "_dump" / "a.txt").exists()
    result = runner.invoke(app, ["remove", "a.txt", *vault], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Deleted a.txt." in result.output
    assert not (root / "_dump" / "a.txt").exists()
    result = runner.invoke(app, ["remove", "a.txt", "--yes", *vault])
    assert result.exit_code == 1
    assert "error: no file with the id 'a.txt'" in result.output


def test_a_broken_set_file_is_an_error_line_from_every_command(root, monkeypatch):
    """A set file cut off half way: each command names it, with no traceback."""
    place(root, "a.txt", "hw")
    broken = root / ".decide" / "sets" / "v4.yaml"
    broken.write_text("version: 4\ngroups:\n- id: hw\n  name: [", encoding="utf-8")
    use_backend(monkeypatch, FakeBackend())
    for command in EVERY_COMMAND:
        output = fails_with_a_line(root, command)
        assert f"{broken} is not valid YAML" in output, command
