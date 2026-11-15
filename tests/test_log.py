import json

import pytest

from decide_already.log import append_jsonl, read_jsonl, total_input_tokens


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


def test_total_input_tokens_counts_every_line_that_has_tokens(tmp_path):
    path = tmp_path / "jev.jsonl"
    append_jsonl(path, {"status": "ok", "input_tokens": 2710})
    append_jsonl(path, {"status": "error", "input_tokens": None})
    append_jsonl(path, {"status": "error", "input_tokens": 900})
    append_jsonl(path, {"status": "ok", "input_tokens": 290})
    append_jsonl(path, {"status": "ok", "input_tokens": None})
    append_jsonl(path, {"status": "ok"})
    assert total_input_tokens(path) == 3900
