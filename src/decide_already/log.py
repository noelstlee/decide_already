"""JSON Lines files: one JSON object per line, only ever added to. The Jev call log is one."""

import json
from collections.abc import Iterator
from pathlib import Path


def append_jsonl(path: Path, record: dict) -> None:
    """Add one record to the end of a JSONL file as a single line, creating the file if needed.

    The line is built before the file is opened, so a record that cannot be written as JSON
    leaves the file untouched.
    """
    line = json.dumps(record) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _records(path: Path) -> Iterator[dict]:
    """Yield a JSONL file's records one by one. A missing file has none; a bad line is an error."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {number} is not valid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                # Bad file content, so a ValueError like any other bad line (the CLI reports those).
                raise ValueError(f"{path} line {number} is not a JSON object")  # noqa: TRY004
            yield record


def read_jsonl(path: Path) -> list[dict]:
    """Read every record of a JSONL file, in order. A missing file gives an empty list."""
    return list(_records(path))


def total_input_tokens(path: Path) -> int:
    """Add up input_tokens over every call in a call log, whatever its status.

    Spend is computed from this number. A call that was answered but could not be used was
    still billed, so it counts. The log is read one line at a time and nothing is kept,
    because lines are large and this runs every time a backend starts.
    """
    return sum(record.get("input_tokens") or 0 for record in _records(path))
