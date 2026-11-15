"""The p0 command line.

Each command is a thin wrapper over vault.py, sort.py, board.py or corrections.py.
"""

from pathlib import Path
from typing import Annotated

import typer

from . import corrections, vault
from .backends import make_backend
from .backends.base import BackendError
from .board import BoardFile, build_board
from .config import UNSORTED_DIR, Config
from .sort import sort_vault

# Why a file is in Unsorted, in the words the board shows.
REASONS = {"no_group_fits": "no group fits", "weak_top": "weak top pick", "no_text": "no text"}

app = typer.Typer(no_args_is_help=True, help="Phase 0: Jev sorts the dump into your groups.")
group_app = typer.Typer(no_args_is_help=True, help="Manage the groups (the category set).")
app.add_typer(group_app, name="group")

VaultOption = Annotated[
    Path | None,
    typer.Option(
        "--vault", help="Vault folder. Default: the nearest vault at or above the current folder."
    ),
]


def _root(vault_path: Path | None) -> Path:
    """The vault to work on: the --vault folder, or the nearest vault at or above this folder."""
    try:
        return vault_path.resolve() if vault_path else vault.find_vault(Path.cwd())
    except FileNotFoundError as error:
        _fail(str(error))


def _fail(message: str) -> None:
    """Print an error line and exit with status 1."""
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _report(action: str, result) -> None:
    """Print what a group command did, with the set version it left behind."""
    typer.echo(f"{action}. Set version {result.version} ({len(result.groups)} groups).")


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Folder to use as the vault.")] = Path("."),
    dump: Annotated[
        str,
        typer.Option(
            help="Folder with the files to sort, relative to the vault. Originals are never moved."
        ),
    ] = vault.DEFAULT_DUMP_DIR,
) -> None:
    """Create the vault layout: the dump folder, _groups/, Unsorted/ and .decide/."""
    try:
        config = vault.init_vault(path, dump)
    except FileNotFoundError as error:
        _fail(str(error))
    typer.echo(f"Vault ready at {path.resolve()} (dump: {config.dump_dir}, model: {config.model}).")


@group_app.command("add")
def group_add(
    name: str,
    covers: Annotated[str, typer.Option(help="What the group holds.")],
    excludes: Annotated[str | None, typer.Option(help="What it does not hold.")] = None,
    vault_path: VaultOption = None,
) -> None:
    """Add a group. Nothing moves until the next sort."""
    try:
        result = vault.add_group(_root(vault_path), name, covers, excludes)
    except (ValueError, FileExistsError) as error:
        _fail(str(error))
    _report(f"Added {name!r}", result)


@group_app.command("edit")
def group_edit(
    name: str,
    covers: Annotated[str | None, typer.Option(help="New covers text.")] = None,
    excludes: Annotated[
        str | None, typer.Option(help="New excludes text; pass '' to clear.")
    ] = None,
    vault_path: VaultOption = None,
) -> None:
    """Change a group's description."""
    if covers is None and excludes is None:
        _fail("pass --covers and/or --excludes")
    try:
        result = vault.edit_group(_root(vault_path), name, covers, excludes)
    except (ValueError, FileNotFoundError) as error:
        _fail(str(error))
    _report(f"Edited {name!r}", result)


@group_app.command("rename")
def group_rename(old: str, new: str, vault_path: VaultOption = None) -> None:
    """Rename a group; its id and folder contents stay."""
    try:
        result = vault.rename_group(_root(vault_path), old, new)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        _fail(str(error))
    _report(f"Renamed {old!r} to {new!r}", result)


@group_app.command("delete")
def group_delete(name: str, vault_path: VaultOption = None) -> None:
    """Delete a group. Its files go to Unsorted on the next sync."""
    try:
        result = vault.delete_group(_root(vault_path), name)
    except FileNotFoundError as error:
        _fail(str(error))
    _report(f"Deleted {name!r}", result)


@group_app.command("list")
def group_list(vault_path: VaultOption = None) -> None:
    """Show the current set version and its groups."""
    try:
        result = vault.sync_group_set(_root(vault_path))
    except (ValueError, OSError) as error:
        _fail(str(error))
    typer.echo(f"Set version {result.version}")
    for group in result.groups:
        line = f"  {group.name} ({group.id}): covers {group.covers}"
        if group.excludes:
            line += f"; excludes {group.excludes}"
        typer.echo(line)


def _vault_root(vault_path: Path | None) -> Path:
    """Like _root, but fails when the folder has not been set up as a vault."""
    root = _root(vault_path)
    if not Config.path(root).exists():
        _fail(f"no vault at {root}; run `p0 init` first")
    return root


@app.command("sort")
def sort_files(vault_path: VaultOption = None) -> None:
    """Sort every unfrozen file in the dump into your groups and write its card note."""
    root = _vault_root(vault_path)
    try:
        report = sort_vault(root, make_backend(root, Config.load(root)))
    except (BackendError, ValueError, OSError) as error:
        # OSError: a file of the user's sits where sync has to put a card note.
        _fail(str(error))
    spent = "nothing" if report.spent_usd is None else f"${report.spent_usd:.4f}"
    typer.echo(
        f"Asked {report.asked}, placed {report.placed}, Unsorted {report.unsorted}, "
        f"close calls {report.close_calls}, frozen skipped {report.frozen_skipped}. "
        f"Set version {report.set_version}. Spent so far: {spent}."
    )
    for file_id, message in report.errors.items():
        typer.echo(f"error: {file_id}: {message}", err=True)
    if report.errors:
        raise typer.Exit(code=1)


def _board_line(file: BoardFile) -> str:
    """One file's line on the board: its name, the model's top probability, then its marks."""
    parts = [file.name]
    if file.top_prob is not None:
        parts.append(f"{file.top_prob:.2f}")
    marks = []
    if file.group_id is None and file.reason:
        marks.append(REASONS[file.reason])  # only while the file is still in Unsorted
    if file.close_call:
        marks.append("close call")
    if file.frozen:
        marks.append("frozen")
    if file.placed_by == "you":
        marks.append("moved by you")
    if file.stale:
        marks.append("older set")
    if marks:
        parts.append(", ".join(marks))
    return "  " + "  ".join(parts)


@app.command("board")
def show_board(vault_path: VaultOption = None) -> None:
    """Show each group with its files, then Unsorted, then how many files are not sorted yet."""
    try:
        board = build_board(_vault_root(vault_path))
    except (ValueError, OSError) as error:
        _fail(str(error))
    typer.echo(f"Set version {board.set_version}")
    for column in board.groups:
        typer.echo(f"{column.group.name} ({len(column.files)})")
        for file in column.files:
            typer.echo(_board_line(file))
    typer.echo(f"Unsorted ({len(board.unsorted)})")
    for file in board.unsorted:
        typer.echo(_board_line(file))
    typer.echo(f"New files not sorted yet: {len(board.new_files)}")


FileOrGroup = Annotated[
    str,
    typer.Argument(
        metavar="FILE_OR_GROUP",
        help="A group (name or id, or Unsorted), or a file (id, or a name only one file has).",
    ),
]


def _group_names(root: Path) -> dict[str | None, str]:
    """The name to print for each group id. None stands for Unsorted."""
    names: dict[str | None, str] = {group.id: group.name for group in vault.read_groups(root)}
    names[None] = UNSORTED_DIR
    return names


def _count(number: int, noun: str) -> str:
    """A number with its noun, as in "1 file" and "3 files"."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


@app.command("move")
def move_file(
    file: Annotated[str, typer.Argument(help="File id, or a file name that only one file has.")],
    group: Annotated[str, typer.Argument(help="Group name or id, or Unsorted.")],
    vault_path: VaultOption = None,
) -> None:
    """Put a file in a group yourself. It is frozen there, and the move is logged as a label."""
    root = _vault_root(vault_path)
    try:
        placement = corrections.move_file(root, file, group)
        name = _group_names(root)[placement.group_id]
    except (ValueError, OSError) as error:
        _fail(str(error))
    typer.echo(f"{placement.file_id} is now in {name}, frozen.")


@app.command("freeze")
def freeze_files(ref: FileOrGroup, vault_path: VaultOption = None) -> None:
    """Freeze a file, or every file in a group. Reruns leave frozen files where they are."""
    root = _vault_root(vault_path)
    try:
        file_ids = corrections.freeze(root, ref)
    except (ValueError, OSError) as error:
        _fail(str(error))
    for file_id in file_ids:
        typer.echo(f"frozen: {file_id}")
    typer.echo(f"Froze {_count(len(file_ids), 'file')}." if file_ids else "Nothing to freeze.")


@app.command("unfreeze")
def unfreeze_files(ref: FileOrGroup, vault_path: VaultOption = None) -> None:
    """Unfreeze a file, or every file in a group. The next rerun sorts them again."""
    root = _vault_root(vault_path)
    try:
        file_ids = corrections.unfreeze(root, ref)
    except (ValueError, OSError) as error:
        _fail(str(error))
    for file_id in file_ids:
        typer.echo(f"unfrozen: {file_id}")
    typer.echo(f"Unfroze {_count(len(file_ids), 'file')}." if file_ids else "Nothing to unfreeze.")


@app.command("suggest")
def suggest_moves(vault_path: VaultOption = None) -> None:
    """List the frozen files the model would now place elsewhere. Nothing moves."""
    root = _vault_root(vault_path)
    failed: dict[str, str] = {}
    try:
        found = corrections.suggest(root, make_backend(root, Config.load(root)), failed)
        names = _group_names(root)
    except (BackendError, ValueError, OSError) as error:
        _fail(str(error))
    for item in found:
        current = names.get(item.current_group, item.current_group)
        suggested = names.get(item.suggested_group, item.suggested_group)
        typer.echo(f"{item.file_id}: in {current}, would now go to {suggested}")
    if found:
        typer.echo(f"{_count(len(found), 'suggestion')}. Nothing was moved.")
    else:
        typer.echo("No suggestions.")
    for file_id, message in failed.items():
        typer.echo(f"error: {file_id}: {message}", err=True)
    if failed:
        raise typer.Exit(code=1)


@app.command("sync")
def sync_edits(vault_path: VaultOption = None) -> None:
    """Pick up edits made by hand: notes dragged to another folder, frozen flags, group notes."""
    root = _vault_root(vault_path)
    try:
        before = vault.latest_set(root)
        report = corrections.sync(root)
    except (ValueError, OSError) as error:
        _fail(str(error))
    picked_up = {
        "moved": report.moved,
        "confirmed": report.confirmed,
        "unfrozen": report.unfrozen,
        "sent to Unsorted, group deleted": report.orphaned,
        "original missing": report.missing_sources,
    }
    for what, file_ids in picked_up.items():
        for file_id in file_ids:
            typer.echo(f"{what}: {file_id}")
    new_set = report.set_version != (before.version if before else 0)
    if new_set:
        typer.echo(f"Group notes changed: set version {report.set_version}.")
    if any(picked_up.values()):
        typer.echo(
            f"Moved {len(report.moved)}, confirmed {len(report.confirmed)}, "
            f"unfrozen {len(report.unfrozen)}, sent to Unsorted {len(report.orphaned)}, "
            f"originals missing {len(report.missing_sources)}."
        )
    elif not new_set:
        typer.echo("Nothing to pick up.")
