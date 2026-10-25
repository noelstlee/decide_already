"""The p0 command line. Each command is a thin wrapper over vault.py."""

from pathlib import Path
from typing import Annotated

import typer

from . import vault

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
    try:
        return vault_path.resolve() if vault_path else vault.find_vault(Path.cwd())
    except FileNotFoundError as error:
        _fail(str(error))


def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _report(action: str, result) -> None:
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
    except ValueError as error:
        _fail(str(error))
    typer.echo(f"Set version {result.version}")
    for group in result.groups:
        line = f"  {group.name} ({group.id}): covers {group.covers}"
        if group.excludes:
            line += f"; excludes {group.excludes}"
        typer.echo(line)
