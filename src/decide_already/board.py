"""The board: every sorted file under the group whose folder its card note sits in.

The CLI and the UI both show this. Building it changes no card note.
"""

from pathlib import Path

from . import vault
from .config import Config
from .types import Group, StrictModel, UnsortedReason


class BoardFile(StrictModel):
    """One file as the board shows it."""

    file_id: str
    name: str  # last path component
    file_type: str
    group_id: str | None  # the group of the folder its note sits in; None means Unsorted
    frozen: bool
    close_call: bool
    placed_by: str
    reason: UnsortedReason | None
    top_prob: float | None  # the model's highest probability; None when no model was asked
    probs: dict[str, float] | None
    fits_any: float | None
    stale: bool  # sorted under an older set version than the current one


class BoardGroup(StrictModel):
    """One group and the files in its folder."""

    group: Group
    files: list[BoardFile]


class Board(StrictModel):
    """The whole vault at a glance."""

    set_version: int
    groups: list[BoardGroup]
    unsorted: list[BoardFile]
    new_files: list[str]  # dump files that have no card note yet


def _board_file(note: vault.CardNote, group_id: str | None, set_version: int) -> BoardFile:
    """Turn one card note into a board entry, under the group its folder stands for."""
    placement = note.placement
    return BoardFile(
        file_id=placement.file_id,
        name=placement.file_id.rsplit("/", 1)[-1],
        file_type=placement.file_type,
        group_id=group_id,
        frozen=placement.frozen,
        close_call=placement.close_call,
        placed_by=placement.placed_by,
        reason=placement.reason,
        top_prob=max(placement.probs.values()) if placement.probs else None,
        probs=placement.probs,
        fits_any=placement.fits_any,
        stale=placement.set_version < set_version,
    )


def _board_order(file: BoardFile) -> tuple[bool, str, str]:
    """Sort key within a column: close calls first (they are the ones to check), then by name."""
    return (not file.close_call, file.name.lower(), file.file_id)


def build_board(root: Path) -> Board:
    """Read the vault into a board.

    Files are grouped by the folder their card note sits in, which is their current group even
    when a note was dragged there and not synced yet. A note in a folder that is no current
    group (its group was deleted) is listed under Unsorted, where sync will move it.
    The set version is the current one: group notes edited by hand are recorded as a new
    version first, as every command that reads the groups does.
    """
    group_set = vault.sync_group_set(root)
    columns = {group.name: BoardGroup(group=group, files=[]) for group in group_set.groups}
    unsorted: list[BoardFile] = []
    notes = vault.read_card_notes(root)
    for note in notes:
        column = columns.get(note.folder)  # None for Unsorted and for a folder that is no group
        if column is None:
            unsorted.append(_board_file(note, None, group_set.version))
        else:
            column.files.append(_board_file(note, column.group.id, group_set.version))
    for files in [unsorted, *(column.files for column in columns.values())]:
        files.sort(key=_board_order)
    carded = {note.placement.file_id for note in notes}
    dump = vault.dump_files(root, Config.load(root))
    return Board(
        set_version=group_set.version,
        groups=list(columns.values()),
        unsorted=unsorted,
        new_files=[file_id for file_id, _ in dump if file_id not in carded],
    )
