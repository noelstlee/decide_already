import json
import os

import pytest

from decide_already.log import append_jsonl, check_appendable, read_jsonl, total_input_tokens


def test_append_then_read_keeps_the_records_in_order(tmp_path):
    path = tmp_path / "calls.jsonl"
    records = [{"file_id": "a.txt", "n": 1}, {"file_id": "b.txt", "n": 2, "nested": {"x": [1, 2]}}]
    for record in records:
        append_jsonl(path, record)
    assert read_jsonl(path) == records


def test_each_record_is_one_line_of_json(tmp_path):
    path = tmp_path / "calls.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"b": 2})
    text = path.read_text()
    assert text.endswith("\n")
    assert [json.loads(line) for line in text.splitlines()] == [{"a": 1}, {"b": 2}]


def test_text_with_line_breaks_and_unicode_stays_on_one_line(tmp_path):
    path = tmp_path / "calls.jsonl"
    record = {"opening": "first line\nsecond line\r\nthird", "title": "Übung 3   notes"}
    append_jsonl(path, record)
    assert path.read_bytes().count(b"\n") == 1
    assert read_jsonl(path) == [record]


def test_append_creates_the_folder(tmp_path):
    path = tmp_path / ".decide" / "jev.jsonl"
    append_jsonl(path, {"a": 1})
    assert read_jsonl(path) == [{"a": 1}]


def test_check_appendable_creates_the_folder_and_an_empty_file(tmp_path):
    """A log that does not exist yet is created empty, so a later append has somewhere to go."""
    path = tmp_path / "new" / ".decide" / "jev.jsonl"
    check_appendable(path)
    assert path.read_bytes() == b""
    assert read_jsonl(path) == []


def test_check_appendable_writes_nothing_to_an_existing_file(tmp_path):
    """Checking twice, before and after an append, leaves exactly the appended line."""
    path = tmp_path / "calls.jsonl"
    check_appendable(path)
    append_jsonl(path, {"a": 1})
    before = path.read_bytes()
    check_appendable(path)
    assert path.read_bytes() == before
    assert read_jsonl(path) == [{"a": 1}]


def test_check_appendable_raises_when_the_folder_cannot_be_made(tmp_path):
    """A plain file sits where the folder should be."""
    (tmp_path / ".decide").write_text("not a folder")
    with pytest.raises(FileExistsError):
        check_appendable(tmp_path / ".decide" / "jev.jsonl")
    assert (tmp_path / ".decide").read_text() == "not a folder"


def test_check_appendable_raises_when_the_file_cannot_be_opened_for_append(tmp_path):
    """A folder sits where the file should be, then a file that may only be read."""
    folder = tmp_path / "folder.jsonl"
    folder.mkdir()
    with pytest.raises(IsADirectoryError):
        check_appendable(folder)
    read_only = tmp_path / "read_only.jsonl"
    read_only.write_text('{"a": 1}\n')
    read_only.chmod(0o444)
    if os.access(read_only, os.W_OK):
        pytest.skip("this user may write to a read-only file")
    with pytest.raises(PermissionError):
        check_appendable(read_only)
    assert read_jsonl(read_only) == [{"a": 1}]


def test_a_missing_file_reads_as_empty(tmp_path):
    path = tmp_path / "nothing.jsonl"
    assert read_jsonl(path) == []
    assert total_input_tokens(path) == 0
    assert not path.exists()


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"b": 2}\n')
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_a_broken_line_is_an_error_that_names_the_line(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"a": 1}\n{"b": \n')
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        read_jsonl(path)
    with pytest.raises(ValueError, match="line 2"):
        total_input_tokens(path)


def test_a_line_that_is_not_an_object_is_an_error(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="line 1 is not a JSON object"):
        read_jsonl(path)


def test_a_record_that_is_not_json_writes_nothing(tmp_path):
    path = tmp_path / "calls.jsonl"
    append_jsonl(path, {"a": 1})
    with pytest.raises(TypeError):
        append_jsonl(path, {"when": object()})
    assert read_jsonl(path) == [{"a": 1}]


def test_a_line_that_only_partly_fits_is_cut_off_again(tmp_path):
    """A failed append leaves the file as it was: half a line would break every later read.

    A limit on how large this process may make a file stands in for a full disk, so the file
    opens and the write fails part way through the line.
    """
    resource = pytest.importorskip("resource")
    path = tmp_path / "calls.jsonl"
    append_jsonl(path, {"n": 1})
    before = path.read_bytes()
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (len(before) + 100, hard))
    try:
        with pytest.raises(OSError):
            append_jsonl(path, {"opening": "kernel methods " * 2000})
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    assert path.read_bytes() == before
    append_jsonl(path, {"n": 2})
    assert read_jsonl(path) == [{"n": 1}, {"n": 2}]


def test_total_input_tokens_counts_every_line_that_has_tokens(tmp_path):
    path = tmp_path / "jev.jsonl"
    append_jsonl(path, {"status": "ok", "input_tokens": 2710})
    append_jsonl(path, {"status": "error", "input_tokens": None})
    append_jsonl(path, {"status": "error", "input_tokens": 900})
    append_jsonl(path, {"status": "ok", "input_tokens": 290})
    append_jsonl(path, {"status": "ok", "input_tokens": None})
    append_jsonl(path, {"status": "ok"})
    assert total_input_tokens(path) == 3900
