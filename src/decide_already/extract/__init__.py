"""Text extraction: one file in; its type, title, outline and text out. No model is involved."""

from pathlib import Path

from .base import Extracted
from .code import LANGUAGES, extract_code
from .html import extract_html
from .markdown import extract_markdown
from .notebook import extract_notebook
from .pdf import extract_pdf
from .slides import extract_slides
from .text import extract_text

__all__ = ["Extracted", "extract"]

_READERS = {
    ".md": extract_markdown,
    ".markdown": extract_markdown,
    ".ipynb": extract_notebook,
    ".pdf": extract_pdf,
    ".pptx": extract_slides,
    ".html": extract_html,
    ".htm": extract_html,
    **dict.fromkeys(LANGUAGES, extract_code),
}


def extract(path: Path) -> Extracted:
    """Pull the type, title, outline and text out of one file, choosing the reader by extension.

    Every other extension (.txt, .rst, .csv, none at all) is read as plain text when the file
    is text, and comes back with text "" when it is binary. A damaged file never raises here:
    it comes back with text "" as well, and the sorter sends it to Unsorted.
    """
    reader = _READERS.get(path.suffix.lower(), extract_text)
    return reader(path)
