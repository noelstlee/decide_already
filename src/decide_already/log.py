"""JSON Lines files: one JSON object per line, only ever added to. The Jev call log is one."""

import json
import os
from collections.abc import Iterator
from pathlib import Path


def check_appendable(path: Path) -> None:
    """Make sure a JSONL file can be added to, by opening it the way append_jsonl does.

    The folder and the file are created if needed, and nothing is written. Raises OSError when
    the file cannot be opened. A full disk only shows at the write, so it passes this check.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8"):
        pass


def append_jsonl(path: Path, record: dict) -> None:
    """Add one record to the end of a JSONL file as a single line, creating the file if needed.

    The line is built before the file is opened, so a record that cannot be written as JSON
    leaves the file untouched. So does a line that only partly fits (a full disk): what was
    written of it is cut off again before the error is passed on, because half a line would
    make every later read of the file fail.
    """
    line = json.dumps(record) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    size = None
    try:
        with path.open("a", encoding="utf-8") as handle:
            size = os.fstat(handle.fileno()).st_size
            handle.write(line)
    except OSError:
        if size is not None:
            # Back to the size before this call. That removes only this call's bytes, because
            # appends to one file come one at a time (the Jev backend holds its lock for them).
            os.truncate(path, size)
        raise


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
