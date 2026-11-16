"""What every extractor returns, and the small helpers they share."""

import re
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from pydantic import Field

from ..types import StrictModel

MAX_OUTLINE = 40  # outline entries kept per file
MAX_OUTLINE_CHARS = 120  # longest outline entry
MAX_TITLE_CHARS = 200  # longest title
BINARY_PROBE_BYTES = 8192  # a NUL byte this early in a file marks it as binary
# shortcut: the card of a text file over 8 MB is built from its first 8 MB (excerpts included).
# A whole 1 GB dataset in memory can kill a sort; raise this if a real file needs more.
MAX_TEXT_BYTES = 8_000_000

# Control characters say nothing to a reader, and PDF math fonts leave many of them in the text.
# They are dropped. Tab and newline stay; the other line breaks (vertical tab, form feed,
# carriage return) become newlines. Half of a surrogate pair (a broken "\ud83d" escape in a
# notebook's JSON) cannot be written as UTF-8, so it becomes the replacement character, the
# same as a byte that is not UTF-8.
_CONTROLS = {
    **dict.fromkeys([*range(0x09), *range(0x0E, 0x20), *range(0x7F, 0xA0)]),
    **dict.fromkeys([0x0B, 0x0C, 0x0D], "\n"),
    **dict.fromkeys(range(0xD800, 0xE000), "\ufffd"),
}
# A file written out as base64 inside the text: a picture pasted into a notebook or a note.
_INLINE_DATA = re.compile(r"data:[\w.+/;=-]*;base64,[A-Za-z0-9+/=]+")
_WORDY_LINE = re.compile(r"^.*?[^\W\d_]{3}.*$", re.MULTILINE)  # a line with three letters in a row


class Extracted(StrictModel):
    """What the extractors pull out of one file. The card is built from this."""

    file_type: str  # human label, e.g. "Python code" or "PDF, 12 pages"
    title: str | None = None
    outline: list[str] = Field(default_factory=list)  # headings, slide titles, or names in code
    text: str = ""  # all extracted text; "" when nothing could be extracted


def read_text(path: Path, whole: bool = False) -> str | None:
    """Read a file as UTF-8 text, cleaned as `clean` does. Bytes that are not UTF-8 are replaced.

    Returns None for a binary file (a NUL byte in its first 8 KB) or one that cannot be opened.
    Only the first 8 KB of a binary file are read, so a video in the dump costs nothing, and
    only the first MAX_TEXT_BYTES of a text file, unless `whole` (a notebook is JSON, and
    half of it cannot be read).
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(BINARY_PROBE_BYTES)
            if b"\x00" in data:
                return None
            data += handle.read() if whole else handle.read(MAX_TEXT_BYTES - len(data))
    except OSError:
        return None
    return clean(data.decode("utf-8-sig", errors="replace"))


def clean(text: str) -> str:
    """Drop control characters, pictures written out as base64, and the white space around a text.

    Tabs and newlines stay, and every other kind of line break becomes a newline. Images are
    never read, and a picture pasted into a notebook cell is one more image: thousands of
    characters of base64 that would crowd the real text out of the card.
    """
    text = text.replace("\r\n", "\n").translate(_CONTROLS)
    return _INLINE_DATA.sub("", text).strip()


def one_line(text: str, limit: int) -> str:
    """Squeeze a string onto one line of at most `limit` characters; a cut is marked with dots."""
    line = " ".join(text.translate(_CONTROLS).split())
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."


def as_title(text: str | None) -> str | None:
    """Clean a title candidate: one line, cut to size, or None when nothing is left."""
    return one_line(text or "", MAX_TITLE_CHARS) or None


def first_line(text: str) -> str | None:
    """The first line that holds a word (three letters in a row), as a title.

    Lines of stray symbols or page numbers come first in many PDFs and are skipped this way.
    """
    match = _WORDY_LINE.search(text)
    return as_title(match.group()) if match else None


def as_outline(entries: Iterable[str]) -> list[str]:
    """Clean headings or names into an outline: one line each, empty ones dropped, at most 40."""
    lines = (one_line(entry, MAX_OUTLINE_CHARS) for entry in entries)
    return list(islice((line for line in lines if line), MAX_OUTLINE))


def count(number: int, noun: str) -> str:
    """A number with its noun, in the singular for one: "1 page", "12 pages"."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"
