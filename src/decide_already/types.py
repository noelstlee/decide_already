"""On-disk records for Phase 0. Everything the tool writes is one of these, checked by Pydantic."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Group(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # slug, stable across renames
    name: str  # the option name Jev sees; also the note file name and the folder name
    covers: str
    excludes: str | None = None


class GroupSet(BaseModel):
    """
    full list of groups at one moment, plus a verion number.
    Every add, edit, rename or delete writes a new version file.
    Jev's prob only compare fairly under the same optoins,
    so each result records which version it ran under.
    """
    model_config = ConfigDict(extra="forbid")

    version: int  # bumps on any add, rename, edit or delete
    groups: list[Group]


class File(BaseModel):
    id: str  # path relative to the dump folder
    content_hash: str
    file_type: str
    card_tokens: int


UnsortedReason = Literal["no_group_fits", "weak_top", "no_text"]


class Placement(BaseModel):
    """
    Stored in a card note's frontmatter.
    The result of sorting. 
    """

    file_id: str
    content_hash: str
    group_id: str | None  # None means Unsorted
    placed_by: Literal["jev", "you", "code"]
    frozen: bool = False
    close_call: bool = False
    set_version: int
    probs: dict[str, float] | None = None  # Jev's distribution over group ids
    fits_any: float | None = None
    reason: UnsortedReason | None = None
    variant: str = "card"
    model: str | None = None  # None when Jev was never asked (no_text)


class Label(BaseModel):
    """One line in .decide/labels.jsonl."""

    file_id: str
    kind: Literal["moved", "confirmed"]
    from_group: str | None
    to_group: str
    jev_probs: dict[str, float] | None
    set_version: int
    at: datetime
