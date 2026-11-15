"""Markdown notes: the headings give the title and the outline."""

import re
from pathlib import Path

from .base import Extracted, as_outline, first_line, read_text

_HEADING = re.compile(r" {0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*")  # "## Grading ##"
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")  # the line that opens or closes a code block
_BOILERPLATE = ("copyright", "license")  # how a heading over a license notice starts


def headings(text: str) -> list[str]:
    """The text of every # heading in markdown, in order. Also used for notebook cells.

    Lines inside fenced code blocks are skipped: there, "# install" is a comment, not a heading.
    """
    found: list[str] = []
    fence = None  # the marker that opened the code block we are inside, if any
    for line in text.splitlines():
        if marker := _FENCE.match(line):
            if fence is None:
                fence = marker.group(1)
            elif marker.group(1).startswith(fence):
                fence = None
        elif fence is None and (match := _HEADING.fullmatch(line)):
            found.append(match.group(1))
    return found


def title_heading(found: list[str]) -> str | None:
    """The heading that names a file: the first one that is not about copyright or a license.

    Many lab notebooks open with a "Copyright Information" cell, and that heading says nothing
    about the file. Headings that start with "Copyright" or "License" (in any case) are passed
    over. When there is no other heading, the first one is the title after all. Also used for
    notebooks.
    """
    for heading in found:
        if not heading.lower().startswith(_BOILERPLATE):
            return heading
    return found[0] if found else None


def extract_markdown(path: Path) -> Extracted:
    """Read a markdown file whole. The title is a heading (see title_heading), or the first line."""
    text = read_text(path) or ""
    outline = as_outline(headings(text))
    return Extracted(
        file_type="Markdown",
        title=title_heading(outline) or first_line(text),
        outline=outline,
        text=text,
    )
