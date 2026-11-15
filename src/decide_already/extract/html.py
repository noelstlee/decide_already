"""HTML pages: the visible text of the main content, read with the standard library's parser."""

import re
from html.parser import HTMLParser
from pathlib import Path

from .base import Extracted, as_outline, as_title, clean, read_text

LABEL = "HTML page"
_DROPPED = {"script", "style", "nav", "noscript", "template", "svg"}  # never part of the text
_MAIN = ("main", "article")  # a page that has one of these keeps only what is inside it
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3}
_BREAKS = {  # tags that start a new line of text
    *("address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption"),
    *("figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li"),
    *("main", "ol", "p", "pre", "section", "table", "tr", "ul"),
}
_CELLS = {"td", "th"}  # the cells of a table row stay on one line
_SPACES = re.compile(r"\s+")


class _Page(HTMLParser):
    """Collects a page's title, headings and visible text.

    Text and headings are kept three times: for the whole page, for what is inside <main>, and
    for what is inside <article>. The caller picks one once the whole page has been read.
    """

    def __init__(self) -> None:
        """Start with nothing collected and no element open."""
        super().__init__()
        self.title: list[str] = []
        self.text: dict[str, list[str]] = {part: [] for part in ("page", *_MAIN)}
        self.headings: dict[str, list[tuple[int, str]]] = {part: [] for part in ("page", *_MAIN)}
        self._dropped = 0  # how many dropped elements we are inside
        self._inside = dict.fromkeys(_MAIN, 0)  # how many <main> and <article> we are inside
        self._pre = 0  # how many <pre> we are inside: there, line breaks in the source count
        self._in_title = False
        self._heading: list[str] | None = None  # the pieces of the heading being read, if any

    def _parts(self) -> list[str]:
        """The parts of the page that anything read right now belongs to."""
        return ["page", *(tag for tag in _MAIN if self._inside[tag])]

    def _add(self, piece: str) -> None:
        """Add a piece of text to every part we are inside."""
        for part in self._parts():
            self.text[part].append(piece)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Note which element opens, and start a new line for block elements."""
        if tag in _DROPPED:
            self._dropped += 1
        if self._dropped:
            return
        if tag in _MAIN:
            self._inside[tag] += 1
        elif tag == "pre":
            self._pre += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _HEADINGS:
            self._heading = []
        if tag in _BREAKS:
            self._add("\n")
        elif tag in _CELLS:
            self._add(" ")

    def handle_endtag(self, tag: str) -> None:
        """Note which element closes, and keep a finished heading."""
        if tag in _DROPPED:
            self._dropped = max(self._dropped - 1, 0)
            return
        if self._dropped:
            return
        if tag in _BREAKS:
            self._add("\n")
        if tag in _MAIN:
            self._inside[tag] = max(self._inside[tag] - 1, 0)
        elif tag == "pre":
            self._pre = max(self._pre - 1, 0)
        elif tag == "title":
            self._in_title = False
        elif tag in _HEADINGS and self._heading is not None:
            for part in self._parts():
                self.headings[part].append((_HEADINGS[tag], "".join(self._heading)))
            self._heading = None

    def handle_data(self, data: str) -> None:
        """Keep text that a reader of the page would see."""
        if self._dropped:
            return
        if self._in_title:
            self.title.append(data)
            return
        if not self._pre:
            data = _SPACES.sub(" ", data)  # outside <pre>, a line break in the source is a space
        if self._heading is not None:
            self._heading.append(data)
        self._add(data)


def _tidy(text: str) -> str:
    """Squeeze the spaces inside each line and drop the empty lines."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def extract_html(path: Path) -> Extracted:
    """Read a web page: its visible text, with the h1 to h3 headings as the outline.

    When the page marks its content with <main> (or else <article>), only that part is kept,
    so the menus, headers and footers around it never reach the card. Scripts, styles and
    <nav> are dropped everywhere. The title is <title>, or else the first <h1>: the first one
    in the part that is kept, or the first one on the page when that part has none.
    """
    page = _Page()
    try:
        page.feed(read_text(path) or "")
        page.close()
    except AssertionError:
        pass  # older Python releases give up on some broken markup: keep what was read so far
    texts = {part: _tidy(clean("".join(pieces))) for part, pieces in page.text.items()}
    part = next((part for part in _MAIN if texts[part]), "page")
    found = page.headings[part]
    anywhere = [*found, *page.headings["page"]]
    first_h1 = next((text for level, text in anywhere if level == 1 and text.strip()), None)
    return Extracted(
        file_type=LABEL,
        title=as_title("".join(page.title)) or as_title(first_h1),
        outline=as_outline(text for _, text in found),
        text=texts[part],
    )
