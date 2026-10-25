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
    for candidate in [start.resolve(), *start.resolve().parents]:
        if Config.path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"no vault at or above {start}; run `p0 init` first")


def read_note(path: Path) -> tuple[dict, str]:
    text = path.read_text()
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    frontmatter = yaml.safe_load(match.group(1)) or {}
    return frontmatter, match.group(2)


def write_note(path: Path, frontmatter: dict, body: str = "") -> None:
    front = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    path.write_text(f"---\n{front}---\n{body}")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "group"


def _group_note(root: Path, name: str) -> Path:
    return root / GROUPS_DIR / f"{name}.md"


def _check_name(root: Path, name: str) -> None:
    config = Config.load(root)
    reserved = {GROUPS_DIR, UNSORTED_DIR, DECIDE_DIR, config.dump_dir}
    if not name.strip() or name != name.strip():
        raise ValueError("group name must not be empty or padded with spaces")
    if name in reserved or name.startswith(".") or "/" in name or "\\" in name:
        raise ValueError(f"{name!r} cannot be a group name")


def _clean(text: str | None) -> str | None:
    text = (text or "").strip()
    return text or None


def read_groups(root: Path) -> list[Group]:
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
    return root / DECIDE_DIR / SETS_DIR / f"v{version}.yaml"


def latest_set(root: Path) -> GroupSet | None:
    versions = []
    for path in (root / DECIDE_DIR / SETS_DIR).glob("v*.yaml"):
        match = _SET_FILE.fullmatch(path.name)
        if match:
            versions.append(int(match.group(1)))
    if not versions:
        return None
    data = yaml.safe_load(_set_path(root, max(versions)).read_text())
    return GroupSet.model_validate(data)


def sync_group_set(root: Path) -> GroupSet:
    """Read the group notes; if they differ from the latest set version, write the next one."""
    groups = read_groups(root)
    current = latest_set(root)
    if current is not None and current.groups == groups:
        return current
    new = GroupSet(version=1 if current is None else current.version + 1, groups=groups)
    _set_path(root, new.version).write_text(yaml.safe_dump(new.model_dump(), sort_keys=False))
    return new


def add_group(root: Path, name: str, covers: str, excludes: str | None = None) -> GroupSet:
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
    note = _group_note(root, name)
    if not note.exists():
        raise FileNotFoundError(f"no group named {name!r}")
    note.unlink()
    folder = root / name
    # Card notes left in the folder are picked up by sync (step 5), which sends them to Unsorted.
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
    return sync_group_set(root)
