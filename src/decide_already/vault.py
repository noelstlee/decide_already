"""The vault on disk: layout, group notes and set versions. Card notes and sync come later."""

import re
import shutil
from pathlib import Path

import yaml

from .config import DECIDE_DIR, DEFAULT_DUMP_DIR, GROUPS_DIR, SETS_DIR, UNSORTED_DIR, Config
from .types import Group, GroupSet

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
_SET_FILE = re.compile(r"v(\d+)\.yaml")


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
    frontmatter = yaml.safe_load(match.group(1)) or {}
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


def sync_group_set(root: Path) -> GroupSet:
    """Read the group notes; if they differ from the latest set version, write the next one."""
    groups = read_groups(root)
    current = latest_set(root)
    if current is None and not groups:
        return GroupSet(version=0, groups=[])  # nothing defined yet, so nothing to record
    if current is not None and current.groups == groups:
        return current
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
    if target.exists():
        raise FileExistsError(f"group {new!r} already exists")
    note.rename(target)
    if (root / old).is_dir():
        shutil.move(str(root / old), str(root / new))
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
