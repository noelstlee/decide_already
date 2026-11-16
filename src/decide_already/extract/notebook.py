"""Jupyter notebooks: the source of every cell, without the outputs."""

import json
from pathlib import Path

from .base import Extracted, as_outline, clean, count, read_text
from .code import python_names
from .markdown import headings, title_heading

LABEL = "Jupyter notebook"


def _source(cell: dict) -> str:
    """One cell's source as a single string. Notebooks store it as a string or a list of lines."""
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(part for part in source if isinstance(part, str))
    return source if isinstance(source, str) else ""


def extract_notebook(path: Path) -> Extracted:
    """Read a notebook's cells. The title is a markdown heading (see title_heading).

    The outline lists the markdown headings first, then the functions and classes in code cells.
    """
    try:
        cells = json.loads(read_text(path, whole=True) or "")["cells"]
        cells = [cell for cell in cells if isinstance(cell, dict)]
    except (ValueError, TypeError, KeyError, RecursionError):
        return Extracted(file_type=LABEL)  # not JSON, or not shaped like a notebook
    sources = [(cell.get("cell_type"), _source(cell)) for cell in cells]
    found = as_outline(
        heading for kind, source in sources if kind == "markdown" for heading in headings(source)
    )
    names = [name for kind, source in sources if kind == "code" for name in python_names(source)]
    return Extracted(
        file_type=f"{LABEL}, {count(len(cells), 'cell')}",
        title=title_heading(found),
        outline=as_outline([*found, *names]),
        text=clean("\n\n".join(source.strip() for _, source in sources if source.strip())),
    )
