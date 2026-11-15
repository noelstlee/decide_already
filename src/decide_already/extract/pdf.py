"""PDFs: the text of every page, read with PyMuPDF. A scan with no text layer gives no text."""

import re
from pathlib import Path

import pymupdf

from .base import Extracted, as_outline, as_title, clean, count, first_line

LABEL = "PDF"
_FILE_NAME = re.compile(r"\.[A-Za-z]\w{1,4}$")  # ends like a file name: "hw3.dvi", "Word - a.doc"
_TOOL_DEFAULTS = {"untitled", "untitled document", "powerpoint presentation", "slide 1"}

# MuPDF prints every flaw it finds in a damaged file to stderr, thousands of lines for one file.
pymupdf.TOOLS.mupdf_display_errors(False)


def _sensible(title: str) -> bool:
    """True when a PDF's stated title reads like a title, not a file name or a tool's default."""
    title = title.strip()
    return (
        len(title) >= 4
        and any(char.isalpha() for char in title)
        and not _FILE_NAME.search(title)
        and title.lower() not in _TOOL_DEFAULTS
    )


def extract_pdf(path: Path) -> Extracted:
    """Read a PDF: the text of its pages, its table of contents as the outline, and a title.

    The title is the one stated in the document's metadata when that is sensible, and the
    first line of the text otherwise. Images are not read, so a scanned PDF has no text.
    """
    try:
        with pymupdf.open(path) as document:
            pages = document.page_count
            text = clean("\n".join(page.get_text() for page in document))
            stated = (document.metadata or {}).get("title") or ""
            contents = [entry[1] for entry in document.get_toc()]
    # A damaged or locked file can fail inside PyMuPDF in many ways. None of them may stop a sort.
    except Exception:  # noqa: BLE001
        return Extracted(file_type=LABEL)
    return Extracted(
        file_type=f"{LABEL}, {count(pages, 'page')}",
        title=as_title(stated) if _sensible(stated) else first_line(text),
        outline=as_outline(contents),
        text=text,
    )
