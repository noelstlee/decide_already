"""Slide decks (.pptx): slide titles and the text on every slide, read with python-pptx."""

from collections.abc import Iterator
from pathlib import Path

from pptx import Presentation

from .base import Extracted, as_outline, as_title, clean

LABEL = "Slides"


def _texts(shapes) -> Iterator[str]:
    """Every piece of text among some shapes: text boxes, table rows, and shapes inside groups."""
    for shape in shapes:
        if hasattr(shape, "shapes"):  # a group: look at the shapes inside it
            yield from _texts(shape.shapes)
        elif shape.has_text_frame:
            yield shape.text_frame.text
        elif shape.has_table:
            for row in shape.table.rows:
                yield " ".join(cell.text for cell in row.cells)


def extract_slides(path: Path) -> Extracted:
    """Read a deck. The first slide's title is the title; all slide titles are the outline.

    Pictures and speaker notes are not read.
    """
    try:
        slides = list(Presentation(str(path)).slides)
        titles = [
            slide.shapes.title.text_frame.text if slide.shapes.title is not None else ""
            for slide in slides
        ]
        pages = [
            "\n".join(text for text in _texts(slide.shapes) if text.strip()) for slide in slides
        ]
    # A damaged deck can fail inside python-pptx in many ways. None of them may stop a sort.
    except Exception:  # noqa: BLE001
        return Extracted(file_type=LABEL)
    text = "\n\n".join(page for page in pages if page)
    return Extracted(
        file_type=f"{LABEL}, {len(slides)}",
        title=as_title(titles[0]) if titles else None,
        outline=as_outline(titles),
        text=clean(text),  # python-pptx marks a line break with a vertical tab: now a newline
    )
