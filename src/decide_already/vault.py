"""The vault on disk: layout, group notes, set versions, the dump listing and card notes."""

import os
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import Field

from .config import (
    DECIDE_DIR,
    DEFAULT_DUMP_DIR,
    GROUPS_DIR,
    SETS_DIR,
    TOOL_DIRS,
    UNSORTED_DIR,
    YAML_LOADER,
    Config,
    clean_dump_dir,
    folder_key,
    yaml_problem,
)
from .types import Group, GroupSet, Placement, StrictModel

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
_SET_FILE = re.compile(r"v(\d+)\.yaml")
# What a folder's name gets when it is set aside (see _free_old_folders): "HW (old)", and
# "HW (old) 2" when that is taken. No group can have such a name.
SET_ASIDE_SUFFIX = " (old)"
_SET_ASIDE = re.compile(r".* \(old\)( \d+)?", re.DOTALL)


def init_vault(root: Path, dump_dir: str = DEFAULT_DUMP_DIR) -> Config:
    """Set up a folder as a vault: create _groups/, Unsorted/ and .decide/, and write the config.

    If the folder is already a vault, return its existing config and change nothing.
    A custom dump folder must already exist; only the default _dump/ is created for you.
    Raises ValueError for a dump folder that is not a plain folder inside the vault.
    """
    root = root.resolve()
    if Config.path(root).exists():
        return Config.load(root)
    dump_dir = clean_dump_dir(dump_dir)
    dump = root / dump_dir
    if not dump.is_dir():
        if dump_dir != DEFAULT_DUMP_DIR:
            raise FileNotFoundError(f"dump folder {dump} does not exist; create it or pass another")
        dump.mkdir(parents=True)
    (root / DECIDE_DIR / SETS_DIR).mkdir(parents=True, exist_ok=True)
    (root / GROUPS_DIR).mkdir(exist_ok=True)
    (root / UNSORTED_DIR).mkdir(exist_ok=True)
    config = Config(dump_dir=dump_dir)
    config.save(root)
    return config


def find_vault(start: Path) -> Path:
    """Walk up from a folder until one with .decide/config.yaml is found, and return it."""
    for candidate in [start.resolve(), *start.resolve().parents]:
        if Config.path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"no vault at or above {start}; run `p0 init` first")


def read_note(path: Path) -> tuple[dict, str]:
    """Split a markdown note into its YAML frontmatter (as a dict) and the body text below it.

    Raises ValueError that names the file when the note is not UTF-8 text, when its
    frontmatter is not valid YAML, and when it is anything but `key: value` lines.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{path} is not UTF-8 text") from None
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    try:
        frontmatter = yaml.load(match.group(1), Loader=YAML_LOADER)
    except yaml.YAMLError as error:
        # The frontmatter starts on the file's second line, after the opening dashes.
        problem = yaml_problem(error, first_line=2)
        raise ValueError(f"the frontmatter of {path} is not valid YAML: {problem}") from None
    if frontmatter is None:
        return {}, match.group(2)
    if not isinstance(frontmatter, dict):
        # Bad file content, so a ValueError like any other broken note (the CLI reports those).
        raise ValueError(f"the frontmatter of {path} must be `key: value` lines")  # noqa: TRY004
    return frontmatter, match.group(2)


def write_note(path: Path, frontmatter: dict, body: str = "") -> None:
    """Write a markdown note: the frontmatter between --- lines, then the body."""
    front = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    path.write_text(f"---\n{front}---\n{body}", encoding="utf-8")


def slugify(name: str) -> str:
    """Turn a group name into its id: lowercase, with dashes. "Lecture slides" gives "lecture-slides".

    Letters and digits of any script are kept, with the marks that belong to them, so a name
    in Korean or Hindi gets an id of its own. Everything else becomes a dash.
    """
    text = unicodedata.normalize("NFC", name).lower()
    kept = (
        char if char.isalnum() or unicodedata.category(char).startswith("M") else "-"
        for char in text
    )
    return re.sub(r"-+", "-", "".join(kept)).strip("-") or "group"


def _group_note(root: Path, name: str) -> Path:
    """Path of a group's note: _groups/<name>.md."""
    return root / GROUPS_DIR / f"{name}.md"


def is_set_aside(name: str) -> bool:
    """True for a name the tool gives a folder it sets aside: "HW (old)" or "HW (old) 2"."""
    return _SET_ASIDE.fullmatch(folder_key(name)) is not None


def _protected_keys(config: Config) -> set[str]:
    """Keys of the folders that are never a group's: the tool's own, and the originals'.

    When the dump folder sits inside another folder (course/files), that top-level folder
    (course) is the protected one: renaming or removing it would move the originals.
    """
    return {folder_key(name) for name in (*TOOL_DIRS, config.dump_dir.split("/")[0])}


def _name_problem(name: str, config: Config) -> str | None:
    """Why a name cannot be a group's, or None when it can.

    A group's name is its folder's name too. So it cannot be empty, padded, hidden or a
    path, and its folder cannot be one that is kept for something else: a tool folder, the
    dump folder, or a folder set aside. Folders are compared by key, so `unsorted` and
    `_Dump` are refused like `Unsorted` and `_dump`.
    """
    if not name.strip() or name != name.strip():
        return "group name must not be empty or padded with spaces"
    if name.startswith(".") or "/" in name or "\\" in name:
        return f"{name!r} cannot be a group name"
    if folder_key(name) in _protected_keys(config):
        return (
            f"{name!r} cannot be a group name: that folder holds the originals or is one of "
            "the tool's own"
        )
    if is_set_aside(name):
        return (
            f"{name!r} cannot be a group name: a name that ends in {SET_ASIDE_SUFFIX!r} is "
            "kept for folders the tool sets aside"
        )
    return None


def _check_name(root: Path, name: str) -> None:
    """Reject a name that cannot be a group's (see _name_problem) with ValueError."""
    problem = _name_problem(name, Config.load(root))
    if problem:
        raise ValueError(problem)


def _clean(text: str | None) -> str | None:
    """Strip a string; return None if nothing is left."""
    text = (text or "").strip()
    return text or None


def _text(value: object, what: str, path: Path) -> str | None:
    """A covers or excludes value from a group note as stripped text, or None when empty.

    YAML reads `covers: 2024` as a number and `covers: [a, b]` as a list. Neither is a
    description, so anything but text is a ValueError that names the note.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        message = f"the {what} of a group must be text, not a list or a number: {path}"
        raise ValueError(message)  # noqa: TRY004 (bad file content, as in read_note)
    return value.strip() or None


def read_groups(root: Path) -> list[Group]:
    """Read every note in _groups/ into a Group, sorted by id.

    The file name is the group name. A hand-made note without an id gets one written back
    into it. Each of these is a ValueError that names the note: a name that cannot be a
    group's (see _name_problem), no covers, a covers or excludes that is not text, an id
    that another note has, and a name that is the same folder as another note's.
    """
    config = Config.load(root)
    groups: list[Group] = []
    for path in sorted((root / GROUPS_DIR).glob("*.md")):
        name = path.stem
        problem = _name_problem(name, config)
        if problem:
            raise ValueError(f"{problem}: {path}")
        frontmatter, body = read_note(path)
        covers = _text(frontmatter.get("covers"), "covers", path)
        if covers is None:
            raise ValueError(f"group {name!r} has no covers description: {path}")
        excludes = _text(frontmatter.get("excludes"), "excludes", path)
        if not frontmatter.get("id"):
            frontmatter = {"id": slugify(name), **frontmatter}
            write_note(path, frontmatter, body)
        group_id = str(frontmatter["id"])
        if folder_key(group_id) == folder_key(UNSORTED_DIR):
            # Commands and the page take "Unsorted" to mean the Unsorted lane, never a group.
            raise ValueError(f"a group's id cannot be {group_id!r}: {path}")
        groups.append(Group(id=group_id, name=name, covers=covers, excludes=excludes))
    ids = [g.id for g in groups]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"two group notes share an id: {', '.join(duplicates)}")
    names: dict[str, str] = {}
    for group in groups:
        other = names.setdefault(folder_key(group.name), group.name)
        if other != group.name:
            raise ValueError(
                f"the group notes {other!r} and {group.name!r} would share one folder "
                f"(their names differ only in capitals or accents); rename one of them: "
                f"{root / GROUPS_DIR}"
            )
    return sorted(groups, key=lambda g: g.id)


def _set_path(root: Path, version: int) -> Path:
    """Path of one set version file: .decide/sets/v<version>.yaml."""
    return root / DECIDE_DIR / SETS_DIR / f"v{version}.yaml"


def latest_set(root: Path) -> GroupSet | None:
    """Load the highest-numbered set version, or None if no set has been written yet."""
    versions = []
    for path in (root / DECIDE_DIR / SETS_DIR).glob("v*.yaml"):
        match = _SET_FILE.fullmatch(path.name)
        if match:
            versions.append(int(match.group(1)))
    if not versions:
        return None
    data = yaml.safe_load(_set_path(root, max(versions)).read_text(encoding="utf-8"))
    return GroupSet.model_validate(data)


def _in_the_way(path: Path) -> bool:
    """True when something is at a path, even a link that points nowhere."""
    return path.exists() or path.is_symlink()


def _is_empty(folder: Path) -> bool:
    """True when a folder holds nothing anyone put there.

    The Finder's .DS_Store does not count, and neither does a folder inside it that is
    empty in the same way (what is left of HW/week 3 once its notes are gone).
    """
    for entry in folder.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            if not _is_empty(entry):
                return False
        elif entry.name != ".DS_Store" or entry.is_symlink():
            return False
    return True


def _remove_folder(folder: Path) -> None:
    """Remove a folder that _is_empty.

    Only .DS_Store files and folders are ever removed here. If anything else turns out to
    be in the folder, the last step fails and that file is still there.
    """
    for entry in list(folder.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            _remove_folder(entry)
        elif entry.name == ".DS_Store":
            entry.unlink()
    folder.rmdir()


def _set_aside_path(root: Path, name: str, protected: set[str]) -> Path:
    """A free path to set a folder aside under: "<name> (old)", then "<name> (old) 2", ..."""
    number = 1
    while True:
        candidate = f"{name}{SET_ASIDE_SUFFIX}" + (f" {number}" if number > 1 else "")
        if not _in_the_way(root / candidate) and folder_key(candidate) not in protected:
            return root / candidate
        number += 1


def _free_old_folders(root: Path, before: list[Group], now: list[Group]) -> None:
    """Take the folders of groups that are gone or renamed out of the way of the new names.

    A group whose name only changed its spelling (HW to Hw) has its folder respelled. Any
    other old folder is removed when it is empty and set aside when it is not: renamed to
    "<old name> (old)". Nothing is merged, so a name that is used again never brings
    another group's card notes with it. The next sync sorts the set-aside notes out one by
    one: those of a group that still exists go to its folder, the others to Unsorted.

    The dump folder and the tool's own folders are never touched, whatever a set on record
    says. Safe to repeat: a folder that was dealt with is no longer there under its old name.
    """
    protected = _protected_keys(Config.load(root))
    names_now = {group.id: group.name for group in now}
    for group in before:
        name_now = names_now.get(group.id)
        folder = root / group.name
        if name_now == group.name or folder_key(group.name) in protected or not folder.is_dir():
            continue
        if name_now is not None and folder_key(name_now) == folder_key(group.name):
            target = root / name_now
            if not _in_the_way(target) or folder.samefile(target):
                folder.rename(target)
        elif _is_empty(folder):
            _remove_folder(folder)
        else:
            folder.rename(_set_aside_path(root, group.name, protected))


def remove_empty_set_asides(root: Path) -> None:
    """Remove every set-aside folder that has nothing left in it (see _free_old_folders)."""
    protected = _protected_keys(Config.load(root))
    for folder in root.iterdir():
        if (
            is_set_aside(folder.name)
            and folder_key(folder.name) not in protected
            and folder.is_dir()
            and not folder.is_symlink()
            and _is_empty(folder)
        ):
            _remove_folder(folder)


def sync_group_set(root: Path) -> GroupSet:
    """Read the group notes; if they differ from the latest set version, write the next one.

    Before the new version is written, the folder of every group that is gone or renamed
    since the last version is taken out of the way (see _free_old_folders). A group note
    renamed by hand keeps its id, so its card notes are still its own: the next sync puts
    them in the folder of the new name.
    """
    groups = read_groups(root)
    current = latest_set(root)
    if current is None and not groups:
        return GroupSet(version=0, groups=[])  # nothing defined yet, so nothing to record
    if current is not None and current.groups == groups:
        return current
    if current is not None:
        _free_old_folders(root, current.groups, groups)
    new = GroupSet(version=1 if current is None else current.version + 1, groups=groups)
    _set_path(root, new.version).write_text(
        yaml.safe_dump(new.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return new


def _record(root: Path, done: str) -> GroupSet:
    """Record the set after a change to one group note that is already on disk.

    Another group note that cannot be read stops the set from being recorded. The change
    itself stands, so the error says what was done before it says what is wrong.
    """
    try:
        return sync_group_set(root)
    except ValueError as error:
        raise ValueError(f"{done}, but the groups could not be recorded: {error}") from error


def _check_folder_free(root: Path, name: str) -> None:
    """Raise FileExistsError when something that is not a folder sits where a group's folder goes."""
    path = root / name
    if _in_the_way(path) and not path.is_dir():
        raise FileExistsError(
            f"{name!r} cannot be a group name: a file of that name is in the vault, where "
            "the group's folder has to go"
        )


def _check_other_names(root: Path, groups: list[Group], name: str, own: Path | None = None) -> None:
    """Raise FileExistsError when another group's name is the same folder as `name`.

    `own` is the note of the group that is taking the name, when it is one of the groups
    (a rename): its own name is never in its way.
    """
    for group in groups:
        if folder_key(group.name) != folder_key(name):
            continue
        if own is None or not _group_note(root, group.name).samefile(own):
            raise FileExistsError(f"group {group.name!r} already exists")


def _make_group_folder(root: Path, name: str) -> None:
    """Make a new group's folder.

    A folder that is the same one under another spelling (hw for a group named HW) gets
    the group's spelling, so that what the disk says and what the group is called agree.
    """
    target = root / name
    for folder in list(root.iterdir()):
        if folder.name != name and folder_key(folder.name) == folder_key(name) and folder.is_dir():
            if not _in_the_way(target) or folder.samefile(target):
                folder.rename(target)
            break
    target.mkdir(exist_ok=True)


def add_group(root: Path, name: str, covers: str, excludes: str | None = None) -> GroupSet:
    """Create a group: write its note, make its folder, and record a new set version."""
    _check_name(root, name)
    if _clean(covers) is None:
        raise ValueError("every group needs a covers description")
    note = _group_note(root, name)
    if note.exists():
        raise FileExistsError(f"group {name!r} already exists")
    groups = read_groups(root)
    _check_other_names(root, groups, name)
    slug = slugify(name)
    if any(g.id == slug for g in groups):
        raise ValueError(f"a group with id {slug!r} already exists; pick a more distinct name")
    _check_folder_free(root, name)
    frontmatter = {"id": slug, "covers": covers.strip()}
    if _clean(excludes):
        frontmatter["excludes"] = excludes.strip()
    write_note(note, frontmatter)
    # The set comes first: it sets aside what a deleted group of this name left behind.
    result = sync_group_set(root)
    _make_group_folder(root, name)
    return result


def edit_group(
    root: Path, name: str, covers: str | None = None, excludes: str | None = None
) -> GroupSet:
    """Change a group's covers and/or excludes text. An empty excludes removes it.

    Arguments left as None are not touched. Bumps the set version only if something changed.
    A group note that cannot be read (this one or another) is a ValueError. When it comes
    after the change was written, the message says so.
    """
    note = _group_note(root, name)
    if not note.exists():
        raise FileNotFoundError(f"no group named {name!r}")
    frontmatter, body = read_note(note)
    if covers is not None:
        if _clean(covers) is None:
            raise ValueError("every group needs a covers description")
        frontmatter["covers"] = covers.strip()
    if excludes is not None:
        if _clean(excludes):
            frontmatter["excludes"] = excludes.strip()
        else:
            frontmatter.pop("excludes", None)
    write_note(note, frontmatter, body)
    return _record(root, f"the change to {name!r} was saved")


def _markdown_files(folder: Path) -> Iterator[Path]:
    """Every markdown file in a folder and in the folders inside it. Hidden folders are skipped."""
    for parent, subfolders, names in os.walk(folder):
        # Changing the list in place tells os.walk not to enter the hidden folders.
        subfolders[:] = [name for name in subfolders if not name.startswith(".")]
        for name in names:
            if name.endswith(".md"):
                yield Path(parent, name)


def _check_folder_move(root: Path, old: str, new: str) -> None:
    """Raise when a group's folder could not follow a rename from `old` to `new`.

    Called before anything is renamed. ValueError for a folder that is never renamed (the
    dump folder, a tool folder). FileExistsError when a file sits where the folder has to
    go, when a folder of the new name already holds card notes (they are another group's,
    and a merge would take them over), and when the two folders hold a file of the same
    name. A hidden file such as .DS_Store is never in the way.
    """
    protected = _protected_keys(Config.load(root))
    for name in (old, new):
        if folder_key(name) in protected:
            raise ValueError(f"the folder {name!r} is never renamed: it is not a group's")
    _check_folder_free(root, new)
    source, target = root / old, root / new
    if not source.is_dir() or not target.is_dir() or source.samefile(target):
        return
    if any(_card_note(path, new) is not None for path in _markdown_files(target)):
        raise FileExistsError(
            f"cannot rename {old!r} to {new!r}: the folder {new!r} already holds card notes; "
            "run `p0 sync` first, which puts them where they belong"
        )
    clashes = sorted(
        entry.name
        for entry in source.iterdir()
        if _in_the_way(target / entry.name) and not _hidden_file(entry)
    )
    if clashes:
        raise FileExistsError(
            f"cannot move the files of {old!r} into the folder {new!r}: "
            f"{', '.join(clashes)} already there"
        )


def _hidden_file(path: Path) -> bool:
    """True for a file whose name starts with a dot, such as the Finder's .DS_Store."""
    return path.name.startswith(".") and path.is_file() and not path.is_symlink()


def _move_folder(root: Path, old: str, new: str) -> None:
    """Make a group's folder follow the group's new name.

    When a folder with the new name is already there, the old folder's files are moved
    into it one by one. Nothing is ever written over: _check_folder_move refuses the move
    first when a file of the same name is already there. A hidden file that both folders
    have (.DS_Store) is dropped from the old one.
    """
    _check_folder_move(root, old, new)
    source, target = root / old, root / new
    if not source.is_dir():
        return
    # samefile: on a file system that ignores case, HW and Hw are one folder, and renaming
    # it is how its spelling changes.
    if not target.exists() or source.samefile(target):
        source.rename(target)
        return
    for entry in list(source.iterdir()):
        if not _in_the_way(target / entry.name):
            entry.rename(target / entry.name)
        elif _hidden_file(entry):
            entry.unlink()  # both folders have it: the old folder's copy is the one dropped
    source.rmdir()


def rename_group(root: Path, old: str, new: str) -> GroupSet:
    """Rename a group's note and folder. The id stays the same, so the group keeps its history.

    Everything that can fail is checked before the note is renamed, and when the folder
    still cannot follow, the note gets its old name back: a rename that fails leaves
    nothing half done.
    """
    _check_name(root, new)
    note = _group_note(root, old)
    if not note.exists():
        raise FileNotFoundError(f"no group named {old!r}")
    target = _group_note(root, new)
    # samefile: a rename that only changes the case finds its own note on a Mac.
    if target.exists() and not target.samefile(note):
        raise FileExistsError(f"group {new!r} already exists")
    # Group notes edited by hand are recorded first. A broken one stops the rename here,
    # and a group deleted by hand has its folder set aside before anything can join it.
    groups = sync_group_set(root).groups
    _check_other_names(root, groups, new, own=note)
    _check_folder_move(root, old, new)
    note.rename(target)
    try:
        _move_folder(root, old, new)
    except (OSError, ValueError):
        target.rename(note)
        raise
    return sync_group_set(root)


def delete_group(root: Path, name: str) -> GroupSet:
    """Delete a group's note, and its folder if empty, then record a new set version.

    A folder that still holds card notes is set aside by the new set version (see
    _free_old_folders), and the next sync sends those notes to Unsorted. The note is
    deleted even when another group note cannot be read: deleting is how a broken note is
    got rid of. The ValueError that follows says that the delete was carried out.
    """
    note = _group_note(root, name)
    if not note.exists():
        raise FileNotFoundError(f"no group named {name!r}")
    protected = _protected_keys(Config.load(root))
    note.unlink()
    folder = root / name
    if folder_key(name) not in protected and folder.is_dir() and _is_empty(folder):
        _remove_folder(folder)
    return _record(root, f"the note of {name!r} was deleted")


def dump_files(root: Path, config: Config) -> list[tuple[str, Path]]:
    """List the files in the dump folder as (file id, path), sorted by id.

    The file id is the path relative to the dump folder, with forward slashes. Folders are
    walked all the way down. Anything whose name starts with a dot (.DS_Store, .git/) is
    skipped, together with everything inside it.
    """
    dump = root / config.dump_dir
    files = []
    for folder, subfolders, names in os.walk(dump):
        # Changing the list in place tells os.walk not to enter the hidden folders.
        subfolders[:] = [name for name in subfolders if not name.startswith(".")]
        for name in names:
            path = Path(folder, name)
            if not name.startswith(".") and path.is_file():
                files.append((path.relative_to(dump).as_posix(), path))
    return sorted(files)


def delete_original(root: Path, file_id: str) -> int:
    """Delete one file from the dump folder for good, with its card note. The one place the
    tool removes an original, and only when the user asks for that file by its id.

    The file is matched by its exact id in the dump listing, never joined to a path, so
    nothing outside the dump folder can be named. A link in the dump is removed as a link;
    what it points to stays. Folders inside the dump that are left with nothing in them go
    too (the dump folder itself never does). Labels in .decide/labels.jsonl are not touched
    here (corrections.delete_file withdraws an active one). Returns how many card notes were removed.
    Raises FileNotFoundError for an id that is not a file in the dump folder.
    """
    config = Config.load(root)
    paths = dict(dump_files(root, config))
    if file_id not in paths:
        raise FileNotFoundError(f"no file with the id {file_id!r} in the dump folder")
    path = paths[file_id]
    notes = [note for note in read_card_notes(root) if note.placement.file_id == file_id]
    path.unlink()  # first: when this fails, nothing has changed
    for note in notes:
        note.path.unlink(missing_ok=True)
    dump = root / config.dump_dir
    folder = path.parent
    while folder != dump and dump in folder.parents and _is_empty(folder):
        _remove_folder(folder)
        folder = folder.parent
    return len(notes)


def note_name(file_id: str) -> str:
    """The file name of a file's card note: the file id with "/" written as "__", plus ".md"."""
    return file_id.replace("/", "__") + ".md"


def source_link(file_id: str, config: Config) -> str:
    """The Obsidian link from a card note back to its original in the dump folder."""
    return f"[[{config.dump_dir}/{file_id}]]"


class CardNote(StrictModel):
    """One card note as it sits on disk."""

    path: Path
    folder: str  # name of the top-level folder the note is under: its group's, or Unsorted
    placement: Placement  # the placement fields of the note's frontmatter
    body: str  # the card as markdown, below the frontmatter
    # Any other frontmatter keys (properties added in Obsidian). Kept on every rewrite.
    extras: dict = Field(default_factory=dict)
    stamp: tuple[int, int] | None = None  # the file's size and time of change when it was read


def _stamp(path: Path) -> tuple[int, int] | None:
    """A file's size and time of change, or None when the file is not there."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def unchanged(note: CardNote) -> bool:
    """True when a card note is still on disk as it was read: same place, size and time."""
    return note.stamp is not None and _stamp(note.path) == note.stamp


def restamp(note: CardNote) -> CardNote:
    """The same card note with the stamp of its file as it is now, after the tool rewrote it."""
    return note.model_copy(update={"stamp": _stamp(note.path)})


def _card_note(path: Path, folder: str) -> CardNote | None:
    """Read one markdown file as a card note. None when its frontmatter holds no Placement.

    Keys that are not placement fields (a tag or any other property added in Obsidian) do
    not make a card note the user's own note: they are kept apart, as the note's extras.
    """
    stamp = _stamp(path)  # taken before the read, so a change during the read shows later
    try:
        frontmatter, body = read_note(path)
        fields = {key: frontmatter[key] for key in frontmatter if key in Placement.model_fields}
        placement = Placement.model_validate(fields)
    except (OSError, ValueError):
        # ValueError covers frontmatter that is not YAML or holds no Placement, and text
        # that is not UTF-8.
        return None
    extras = {key: frontmatter[key] for key in frontmatter if key not in Placement.model_fields}
    return CardNote(
        path=path, folder=folder, placement=placement, body=body, extras=extras, stamp=stamp
    )


def read_card_notes(root: Path) -> list[CardNote]:
    """Read every card note in the vault, sorted by file id.

    Card notes sit in the top-level folders (the group folders and Unsorted) or in a folder
    inside one: a note filed into HW/week 3 is still in HW. The dump folder, _groups,
    .decide and hidden folders are not looked in. A markdown file whose frontmatter holds
    no Placement is one of the user's own notes and is left out.
    """
    config = Config.load(root)
    skipped = _protected_keys(config) - {folder_key(UNSORTED_DIR)}
    notes = []
    for folder in root.iterdir():
        if (
            folder.is_dir()
            and folder_key(folder.name) not in skipped
            and not folder.name.startswith(".")
        ):
            for path in _markdown_files(folder):
                note = _card_note(path, folder.name)
                if note is not None:
                    notes.append(note)
    return sorted(notes, key=lambda note: (note.placement.file_id, note.path))


def folder_for(root: Path, group_id: str | None) -> Path:
    """The folder a card note belongs in: the one named after its group, or Unsorted for None.

    Raises KeyError for an id that is no current group.
    """
    if group_id is None:
        return root / UNSORTED_DIR
    for group in read_groups(root):
        if group.id == group_id:
            return root / group.name
    raise KeyError(f"no group with id {group_id!r}")


def folder_ids(groups: list[Group]) -> dict[str, str | None]:
    """The group each folder stands for, by folder key: each group's id, and None for Unsorted.

    Folders are looked up by key, so a group's folder is found under any spelling the file
    system treats as the same name (hw for HW).
    """
    ids: dict[str, str | None] = {folder_key(group.name): group.id for group in groups}
    ids[folder_key(UNSORTED_DIR)] = None
    return ids


def group_id_of_folder(root: Path, folder: str) -> str | None:
    """The group a folder stands for: its group's id, or None for Unsorted.

    Raises KeyError for a folder that is no current group, such as one a deleted group left.
    """
    ids = folder_ids(read_groups(root))
    if folder_key(folder) not in ids:
        raise KeyError(f"{folder!r} is not the folder of a current group")
    return ids[folder_key(folder)]


def card_frontmatter(placement: Placement, extras: dict) -> dict:
    """A card note's frontmatter: the placement field by field, then the keys the user added."""
    return {**placement.model_dump(mode="json"), **extras}


def write_card_note(
    root: Path, placement: Placement, body: str, notes: list[CardNote] | None = None
) -> Path:
    """Write a file's card note into the folder of its group and return the note's path.

    The frontmatter is the placement, field by field, then whatever other keys the note it
    replaces had (a property added in Obsidian stays). The body is the card. Any other card
    note for the same file is removed, so a file has one note, at the top of its group's
    folder. A file already at that path that is not this file's card note is never
    overwritten: that raises FileExistsError.

    Finding the file's other notes means reading every card note in the vault. A caller that
    writes many notes in a row (a sort) can pass what read_card_notes just returned as
    `notes`, so the vault is read once and not once per file.
    """
    path = folder_for(root, placement.group_id) / note_name(placement.file_id)
    if notes is None:
        notes = read_card_notes(root)
    previous = [
        note for note in notes if note.placement.file_id == placement.file_id and note.path.exists()
    ]
    # samefile, not ==: on a file system that ignores case, HW/x.md and hw/x.md are one file.
    if path.exists() and not any(old.path.samefile(path) for old in previous):
        raise FileExistsError(
            f"{path.relative_to(root)} exists and is not the card note of {placement.file_id}"
        )
    path.parent.mkdir(exist_ok=True)
    # With several notes for one file (a copy made by hand), the last one counts, as elsewhere.
    extras = previous[-1].extras if previous else {}
    write_note(path, card_frontmatter(placement, extras), body)
    for old in previous:
        if not old.path.samefile(path):
            old.path.unlink()
    return path
