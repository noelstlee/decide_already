"""The vault on disk: layout, group notes, set versions, the dump listing and card notes."""

import os
import re
from pathlib import Path

import yaml

from .config import DECIDE_DIR, DEFAULT_DUMP_DIR, GROUPS_DIR, SETS_DIR, UNSORTED_DIR, Config
from .types import Group, GroupSet, Placement, StrictModel

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
_SET_FILE = re.compile(r"v(\d+)\.yaml")
# The C loader reads the same YAML several times faster, and every command reads every card
# note. PyYAML falls back to its Python loader when it was built without libyaml.
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def init_vault(root: Path, dump_dir: str = DEFAULT_DUMP_DIR) -> Config:
    """Set up a folder as a vault: create _groups/, Unsorted/ and .decide/, and write the config.

    If the folder is already a vault, return its existing config and change nothing.
    A custom dump folder must already exist; only the default _dump/ is created for you.
    """
    root = root.resolve()
    if Config.path(root).exists():
        return Config.load(root)
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
    """Split a markdown note into its YAML frontmatter (as a dict) and the body text below it."""
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    frontmatter = yaml.load(match.group(1), Loader=_YAML_LOADER) or {}
    return frontmatter, match.group(2)


def write_note(path: Path, frontmatter: dict, body: str = "") -> None:
    """Write a markdown note: the frontmatter between --- lines, then the body."""
    front = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    path.write_text(f"---\n{front}---\n{body}", encoding="utf-8")


def slugify(name: str) -> str:
    """Turn a group name into its id: lowercase, with dashes. "Lecture slides" gives "lecture-slides"."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "group"


def _group_note(root: Path, name: str) -> Path:
    """Path of a group's note: _groups/<name>.md."""
    return root / GROUPS_DIR / f"{name}.md"


def _check_name(root: Path, name: str) -> None:
    """Reject group names that are empty, padded, hidden, contain slashes, or clash with a tool folder."""
    config = Config.load(root)
    reserved = {GROUPS_DIR, UNSORTED_DIR, DECIDE_DIR, config.dump_dir}
    if not name.strip() or name != name.strip():
        raise ValueError("group name must not be empty or padded with spaces")
    if name in reserved or name.startswith(".") or "/" in name or "\\" in name:
        raise ValueError(f"{name!r} cannot be a group name")


def _clean(text: str | None) -> str | None:
    """Strip a string; return None if nothing is left."""
    text = (text or "").strip()
    return text or None


def read_groups(root: Path) -> list[Group]:
    """Read every note in _groups/ into a Group, sorted by id.

    The file name is the group name. A note without covers is an error.
    A hand-made note without an id gets one written back into it.
    """
    groups: list[Group] = []
    for path in sorted((root / GROUPS_DIR).glob("*.md")):
        frontmatter, body = read_note(path)
        name = path.stem
        covers = _clean(frontmatter.get("covers"))
        if covers is None:
            raise ValueError(f"group {name!r} has no covers description: {path}")
        if not frontmatter.get("id"):
            frontmatter = {"id": slugify(name), **frontmatter}
            write_note(path, frontmatter, body)
        groups.append(
            Group(
                id=str(frontmatter["id"]),
                name=name,
                covers=covers,
                excludes=_clean(frontmatter.get("excludes")),
            )
        )
    ids = [g.id for g in groups]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"two group notes share an id: {', '.join(duplicates)}")
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


def _move_folder(root: Path, old: str, new: str) -> None:
    """Make a group's folder follow the group's new name.

    When a folder with the new name is already there (a deleted group can leave one), the
    old folder's files are moved into it one by one. Nothing is ever written over: when a
    file of the same name is already there, FileExistsError is raised and nothing has moved.
    """
    source, target = root / old, root / new
    if not source.is_dir():
        return
    # samefile: on a file system that ignores case, HW and Hw are one folder, and renaming
    # it is how its spelling changes.
    if not target.exists() or source.samefile(target):
        source.rename(target)
        return
    clashes = sorted(entry.name for entry in source.iterdir() if (target / entry.name).exists())
    if clashes:
        raise FileExistsError(
            f"cannot move the files of {old!r} into the folder {new!r}: "
            f"{', '.join(clashes)} already there"
        )
    for entry in list(source.iterdir()):
        entry.rename(target / entry.name)
    source.rmdir()


def sync_group_set(root: Path) -> GroupSet:
    """Read the group notes; if they differ from the latest set version, write the next one.

    A group whose note was renamed by hand keeps its id, so its folder is renamed to match,
    as `rename_group` does. Left under the old name, the folder would be no group's, and
    sync would send its card notes to Unsorted and withdraw their labels.
    """
    groups = read_groups(root)
    current = latest_set(root)
    if current is None and not groups:
        return GroupSet(version=0, groups=[])  # nothing defined yet, so nothing to record
    if current is not None and current.groups == groups:
        return current
    if current is not None:
        old_names = {group.id: group.name for group in current.groups}
        new_names = {group.name for group in groups}
        for group in groups:
            old = old_names.get(group.id)
            # A folder that another group now goes by stays put (two groups swapped names).
            if old is not None and old != group.name and old not in new_names:
                _move_folder(root, old, group.name)
    new = GroupSet(version=1 if current is None else current.version + 1, groups=groups)
    _set_path(root, new.version).write_text(
        yaml.safe_dump(new.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return new


def add_group(root: Path, name: str, covers: str, excludes: str | None = None) -> GroupSet:
    """Create a group: write its note, make its folder, and record a new set version."""
    _check_name(root, name)
    if _clean(covers) is None:
        raise ValueError("every group needs a covers description")
    note = _group_note(root, name)
    if note.exists():
        raise FileExistsError(f"group {name!r} already exists")
    slug = slugify(name)
    if any(g.id == slug for g in read_groups(root)):
        raise ValueError(f"a group with id {slug!r} already exists; pick a more distinct name")
    frontmatter = {"id": slug, "covers": covers.strip()}
    if _clean(excludes):
        frontmatter["excludes"] = excludes.strip()
    write_note(note, frontmatter)
    (root / name).mkdir(exist_ok=True)
    return sync_group_set(root)


def edit_group(
    root: Path, name: str, covers: str | None = None, excludes: str | None = None
) -> GroupSet:
    """Change a group's covers and/or excludes text. An empty excludes removes it.

    Arguments left as None are not touched. Bumps the set version only if something changed.
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
    return sync_group_set(root)


def rename_group(root: Path, old: str, new: str) -> GroupSet:
    """Rename a group's note and folder. The id stays the same, so the group keeps its history."""
    _check_name(root, new)
    note = _group_note(root, old)
    if not note.exists():
        raise FileNotFoundError(f"no group named {old!r}")
    target = _group_note(root, new)
    # samefile: a rename that only changes the case finds its own note on a Mac.
    if target.exists() and not target.samefile(note):
        raise FileExistsError(f"group {new!r} already exists")
    note.rename(target)
    _move_folder(root, old, new)
    return sync_group_set(root)


def delete_group(root: Path, name: str) -> GroupSet:
    """Delete a group's note, and its folder if empty, then record a new set version."""
    note = _group_note(root, name)
    if not note.exists():
        raise FileNotFoundError(f"no group named {name!r}")
    note.unlink()
    folder = root / name
    # Card notes left in the folder are picked up by sync (step 5), which sends them to Unsorted.
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
    return sync_group_set(root)


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


def note_name(file_id: str) -> str:
    """The file name of a file's card note: the file id with "/" written as "__", plus ".md"."""
    return file_id.replace("/", "__") + ".md"


def source_link(file_id: str, config: Config) -> str:
    """The Obsidian link from a card note back to its original in the dump folder."""
    return f"[[{config.dump_dir}/{file_id}]]"


class CardNote(StrictModel):
    """One card note as it sits on disk."""

    path: Path
    folder: str  # name of the folder the note sits in: the file's current group, or Unsorted
    placement: Placement  # the note's frontmatter
    body: str  # the card as markdown, below the frontmatter


def _card_note(path: Path) -> CardNote | None:
    """Read one markdown file as a card note. None when its frontmatter is not a Placement."""
    try:
        frontmatter, body = read_note(path)
        placement = Placement.model_validate(frontmatter)
    except (OSError, ValueError, yaml.YAMLError):
        # ValueError covers a frontmatter that is not a Placement and text that is not UTF-8.
        return None
    return CardNote(path=path, folder=path.parent.name, placement=placement, body=body)


def read_card_notes(root: Path) -> list[CardNote]:
    """Read every card note in the vault, sorted by file id.

    Card notes sit directly in the top-level folders: the group folders and Unsorted. The dump
    folder, _groups, .decide and hidden folders are not looked in. A markdown file whose
    frontmatter is not a Placement is one of the user's own notes and is left out.
    """
    skipped = {Config.load(root).dump_dir, GROUPS_DIR, DECIDE_DIR}
    notes = []
    for folder in root.iterdir():
        if folder.is_dir() and folder.name not in skipped and not folder.name.startswith("."):
            for path in folder.glob("*.md"):
                note = _card_note(path)
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


def group_id_of_folder(root: Path, folder: str) -> str | None:
    """The group a folder stands for: its group's id, or None for Unsorted.

    Raises KeyError for a folder that is no current group, such as one a deleted group left.
    """
    if folder == UNSORTED_DIR:
        return None
    for group in read_groups(root):
        if group.name == folder:
            return group.id
    raise KeyError(f"{folder!r} is not the folder of a current group")


def write_card_note(
    root: Path, placement: Placement, body: str, notes: list[CardNote] | None = None
) -> Path:
    """Write a file's card note into the folder of its group and return the note's path.

    The frontmatter is the placement, field by field, and the body is the card. Any other
    card note for the same file is removed, so a file has one note, in its group's folder.
    A file already at that path that is not this file's card note is never overwritten:
    that raises FileExistsError.

    Finding the file's other notes means reading every card note in the vault. A caller that
    writes many notes in a row (a sort) can pass what read_card_notes just returned as
    `notes`, so the vault is read once and not once per file.
    """
    path = folder_for(root, placement.group_id) / note_name(placement.file_id)
    if notes is None:
        notes = read_card_notes(root)
    previous = [
        note.path
        for note in notes
        if note.placement.file_id == placement.file_id and note.path.exists()
    ]
    # samefile, not ==: on a file system that ignores case, HW/x.md and hw/x.md are one file.
    if path.exists() and not any(old.samefile(path) for old in previous):
        raise FileExistsError(
            f"{path.relative_to(root)} exists and is not the card note of {placement.file_id}"
        )
    path.parent.mkdir(exist_ok=True)
    write_note(path, placement.model_dump(mode="json"), body)
    for old in previous:
        if not old.samefile(path):
            old.unlink()
    return path
