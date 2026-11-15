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
    when a note was dragged there and not synced yet. Folders are matched to groups by key,
    so a group's folder counts under any spelling the file system treats as the same. A note
    in a folder that is no group's is listed where sync will put it: under its own group
    when that group still exists (the group was renamed by hand), else under Unsorted (the
    group was deleted). The set version is the current one: group notes edited by hand are
    recorded as a new version first, as every command that reads the groups does.
    """
    group_set = vault.sync_group_set(root)
    columns = {group.id: BoardGroup(group=group, files=[]) for group in group_set.groups}
    folders = vault.folder_ids(group_set.groups)
    unsorted: list[BoardFile] = []
    notes = vault.read_card_notes(root)
    for note in notes:
        # A folder that is no group's says nothing: the note's own group decides, if it exists.
        group_id = folders.get(vault.folder_key(note.folder), note.placement.group_id)
        if group_id in columns:
            columns[group_id].files.append(_board_file(note, group_id, group_set.version))
        else:
            unsorted.append(_board_file(note, None, group_set.version))
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
