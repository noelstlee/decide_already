"""On-disk records for Phase 0. Everything the tool writes is one of these, checked by Pydantic."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for every record: an unknown field is an error, so a typo in a hand-edited file fails loudly."""

    model_config = ConfigDict(extra="forbid")


class Group(StrictModel):
    """One group the user defines. Its name is the option Jev picks from."""

    id: str  # slug, stable across renames
    name: str  # the option name Jev sees; also the note file name and the folder name
    covers: str
    excludes: str | None = None


class GroupSet(StrictModel):
    """The full list of groups at one moment, plus a version number.

    Every add, edit, rename or delete writes a new version file.
    Jev's probabilities only compare fairly under the same options,
    so each result records which version it ran under.
    """

    version: int  # bumps on any add, rename, edit or delete
    groups: list[Group]


class File(StrictModel):
    """One file in the dump folder."""

    id: str  # path relative to the dump folder, with forward slashes
    content_hash: str  # sha256 of the file's bytes
    file_type: str  # human label, e.g. "Python code" or "PDF, 12 pages"
    card_tokens: int  # estimated size of what is sent to the model


UnsortedReason = Literal["no_group_fits", "weak_top", "no_text"]


class Placement(StrictModel):
    """The result of sorting one file. Stored as the frontmatter of its card note."""

    source: str  # Obsidian link to the original, e.g. "[[_dump/hw3/hw3_solution.py]]"
    file_id: str
    content_hash: str
    file_type: str
    group_id: str | None  # the group as of the last write or sync; None means Unsorted
    placed_by: Literal["jev", "you", "code"]  # code: no model was asked (no text, deleted group)
    frozen: bool = False
    close_call: bool = False
    reason: UnsortedReason | None = None  # why it is Unsorted, when the rules sent it there
    set_version: int  # the category set version the probabilities were computed under
    probs: dict[str, float] | None = None  # the model's distribution over group ids
    fits_any: float | None = None
    variant: str = "card"
    model: str | None = None  # None when no model was asked


class Label(StrictModel):
    """One line in .decide/labels.jsonl: something the user did to a file's group.

    A file's current label is its last line here, if that line is "moved" or "confirmed".
    "unfrozen" withdraws the label.
    """

    file_id: str
    kind: Literal["moved", "confirmed", "unfrozen"]
    from_group: str | None  # group id the tool had it in; None means Unsorted
    to_group: str | None  # group id it is in now; None means Unsorted
    jev_probs: dict[str, float] | None  # what the model said when it placed the file
    jev_set_version: int | None  # the set version jev_probs were computed under
    set_version: int  # the set version when the user acted
    at: datetime


class Suggestion(StrictModel):
    """One line in .decide/suggestions.jsonl: a frozen file the model would now place elsewhere."""

    file_id: str
    current_group: str | None
    suggested_group: str | None
    probs: dict[str, float]
    fits_any: float
    set_version: int
    model: str
    at: datetime
