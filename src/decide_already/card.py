"""The card: the short view of one file that the model reads, and the experiment variants of it.

Code builds the card. No model is involved, and the card is the only part of a file that
leaves the machine.
"""

import hashlib
import json
import math
import re
from pathlib import Path

from .config import Config
from .extract import Extracted, extract
from .types import File, StrictModel

VARIANTS = ("card", "card_no_name", "full_text", "window_vote", "names_only")

CHARS_PER_TOKEN = 4  # the rule of thumb behind every token count here
OPENING_CHARS = 6_000  # about 1,500 tokens
EXCERPT_CHARS = 1_200  # about 300 tokens
EXCERPT_MARKS = (40, 70, 95)  # where the excerpts start, in percent of the text
FULL_TEXT_CHARS = 30_000 * CHARS_PER_TOKEN
WINDOW_CHARS = 2_000 * CHARS_PER_TOKEN
MAX_WINDOWS = 8


class Card(StrictModel):
    """What the model is shown of one file."""

    file: File
    title: str | None
    states: list[dict]  # one per request; several only for window_vote; [] for a file with no text


def estimate_tokens(text: str) -> int:
    """Estimate the size of a text in tokens: 4 characters per token, rounded up."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def hash_file(path: Path) -> str:
    """The sha256 of a file's bytes, as hex. The file is read in pieces, so it can be large."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _state_tokens(state: dict) -> int:
    """Estimate the size of one state in tokens, counted on the JSON that is sent."""
    return estimate_tokens(json.dumps(state, ensure_ascii=False))


def _cut(text: str, limit: int, start: int = 0) -> str:
    """Up to `limit` characters of a text from `start`, ending between words when it can.

    A text that goes on past the limit is cut at the last gap between words, unless that gap
    lies in the first half of the passage (one very long word): then it is cut at the limit.
    """
    passage = text[start : start + limit + 1]  # one character more shows whether the text goes on
    if len(passage) <= limit:
        return passage
    gap = max(passage.rfind(" "), passage.rfind("\n"), passage.rfind("\t"))
    return passage[: gap if gap > limit // 2 else limit].rstrip()


def _excerpts(text: str, opening_end: int) -> dict[str, str]:
    """Passages of up to 1,200 characters from later in a text, keyed by where they start.

    They start at 40, 70 and 95 percent of the text. One that would start inside the opening is
    left out, and each stops where the next begins, so no part of the text is sent twice.
    A text that fits in the opening gets no excerpts.
    """
    starts = [len(text) * mark // 100 for mark in EXCERPT_MARKS]
    ends = [*starts[1:], len(text)]
    excerpts = {}
    for mark, start, end in zip(EXCERPT_MARKS, starts, ends, strict=True):
        passage = _cut(text, min(EXCERPT_CHARS, end - start), start).strip()
        if start >= opening_end and passage:
            excerpts[f"at_{mark}_percent"] = passage
    return excerpts


def _filled(state: dict) -> dict:
    """A state without its empty fields. An empty list or string could read as a signal."""
    return {key: value for key, value in state.items() if value}


def _over(state: dict, budget: int) -> int:
    """How many characters a state is over the token budget by. Zero or less means it fits."""
    return (_state_tokens(_filled(state)) - budget) * CHARS_PER_TOKEN


def _fit(state: dict, budget: int) -> dict:
    """Shrink a card state until it fits the token budget, and leave out its empty fields.

    The outline is shortened first, from its end, and then the opening. Excerpts and the title
    only go when a very small budget leaves no other choice.
    """
    while state["outline"] and _over(state, budget) > 0:
        state["outline"].pop()
    if (over := _over(state, budget)) > 0:
        state["opening"] = _cut(state["opening"], max(len(state["opening"]) - over, 0))
    while state["excerpts"] and _over(state, budget) > 0:
        state["excerpts"].popitem()
    if _over(state, budget) > 0:
        state["title"] = None
    return _filled(state)


def _card_state(info: dict, extracted: Extracted, text: str, budget: int) -> dict:
    """The default card: file info, title, outline, the opening, and excerpts from later on."""
    opening = _cut(text, OPENING_CHARS)
    state = {
        "file": info,
        "title": extracted.title,
        "outline": list(extracted.outline),
        "opening": opening,
        "excerpts": _excerpts(text, len(opening)),
    }
    return _fit(state, budget)


def _windows(text: str) -> list[str]:
    """Split a text into equal windows of at most 2,000 tokens, for the window vote.

    Equal sizes keep the last window from being a short leftover that still gets a full vote.
    When there are more than 8 windows, 8 evenly spaced ones are kept, the first and the last
    among them.
    """
    total = math.ceil(len(text) / WINDOW_CHARS)
    size = math.ceil(len(text) / total)
    if total > MAX_WINDOWS:
        kept = [round(step * (total - 1) / (MAX_WINDOWS - 1)) for step in range(MAX_WINDOWS)]
    else:
        kept = list(range(total))
    windows = (text[index * size : (index + 1) * size].strip() for index in kept)
    return [window for window in windows if window]


def build_card(path: Path, file_id: str, config: Config, variant: str | None = None) -> Card:
    """Read one file and build what the model will see of it.

    `variant` picks the input format and defaults to the one in the config. A file with no
    text gets no states: the sorter sends it to Unsorted without asking the model.
    """
    variant = variant or config.variant
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; pick one of: {', '.join(VARIANTS)}")
    extracted = extract(path)
    text = extracted.text.strip()
    info = {"name": file_id.rsplit("/", 1)[-1], "path": file_id, "type": extracted.file_type}
    if not text:
        states = []
    elif variant == "full_text":
        states = [{"file": info, "text": _cut(text, FULL_TEXT_CHARS)}]
    elif variant == "window_vote":
        states = [{"file": dict(info), "text": window} for window in _windows(text)]
    else:
        states = [_card_state(info, extracted, text, config.card_token_budget)]
        if variant == "card_no_name":
            states[0]["file"] = {"type": extracted.file_type}
    file = File(
        id=file_id,
        content_hash=hash_file(path),
        file_type=extracted.file_type,
        card_tokens=sum(_state_tokens(state) for state in states),
    )
    return Card(file=file, title=extracted.title, states=states)


def _ticks(text: str, least: int) -> str:
    """A run of backticks longer than any run inside the text, and at least `least` long."""
    longest = max((len(run) for run in re.findall("`+", text)), default=0)
    return "`" * max(least, longest + 1)


def _code(text: str) -> str:
    """Wrap one line in a markdown code span, so that it is shown exactly as it is."""
    line = " ".join(text.split())
    ticks = _ticks(line, 1)
    pad = " " if line.startswith("`") or line.endswith("`") else ""
    return f"{ticks}{pad}{line}{pad}{ticks}"


def _fence(text: str) -> str:
    """Wrap text in a fenced block, so that it is shown exactly as it is.

    File text can hold links, tags, images and HTML. A card note should show them, not act on
    them: inside a fence, Obsidian renders nothing and fetches nothing.
    """
    ticks = _ticks(text, 3)
    return f"{ticks}text\n{text}\n{ticks}"


def _sections(states: list[dict]) -> list[tuple[str, str]]:
    """What the model saw, as (heading, text) pairs in reading order."""
    sections: list[tuple[str, str]] = []
    for number, state in enumerate(states, start=1):
        if "outline" in state:
            sections.append(("Outline", "\n".join(state["outline"])))
        if "opening" in state:
            sections.append(("Opening", state["opening"]))
        for key, passage in state.get("excerpts", {}).items():
            sections.append((f"Excerpt {key.replace('_', ' ')}", passage))
        if "text" in state:
            heading = "Text" if len(states) == 1 else f"Window {number} of {len(states)}"
            sections.append((heading, state["text"]))
    return sections


def render_card(card: Card) -> str:
    """The markdown body of a card note, for a person to read in Obsidian.

    A heading with the file name, then the type and title, then what the model saw: the
    outline and the text of each state. Everything taken from the file sits in code spans and
    fenced blocks (see _fence).
    """
    lines = [f"# {_code(card.file.id.rsplit('/', 1)[-1])}", "", f"- Type: {card.file.file_type}"]
    if card.title:
        lines.append(f"- Title: {_code(card.title)}")
    if not card.states:
        lines += ["", "No text could be extracted from this file."]
    for heading, text in _sections(card.states):
        lines += ["", f"## {heading}", "", _fence(text)]
    return "\n".join(lines) + "\n"
