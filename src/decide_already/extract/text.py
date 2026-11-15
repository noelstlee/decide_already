"""Plain text, and any file of an unknown type that still reads as text."""

from pathlib import Path

from .base import Extracted, first_line, read_text


def extract_text(path: Path) -> Extracted:
    """Read a text file whole. Its first line that says something is the title.

    This is also the fallback for every extension no other reader claims, so a binary file
    (an image, an archive, a video) ends here and comes back with no text.
    """
    text = read_text(path)
    if text is None:
        return Extracted(file_type="Binary file")
    return Extracted(file_type="Text", title=first_line(text), text=text)
