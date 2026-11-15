"""Tool settings, stored in <vault>/.decide/config.yaml. The API key is never stored here."""

import os
import re
import unicodedata
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError, field_validator

from .types import StrictModel

DECIDE_DIR = ".decide"
SETS_DIR = "sets"
GROUPS_DIR = "_groups"
UNSORTED_DIR = "Unsorted"
DEFAULT_DUMP_DIR = "_dump"
LABELS_FILE = "labels.jsonl"
JEV_LOG_FILE = "jev.jsonl"
SUGGESTIONS_FILE = "suggestions.jsonl"
# The folders the tool keeps for itself. No group and no dump folder can go by these names.
TOOL_DIRS = (GROUPS_DIR, UNSORTED_DIR, DECIDE_DIR)
# An exact model version: a lower-case name, then a full dotted version, as in jev-1.14.0.
_EXACT_VERSION = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+)*-\d+\.\d+\.\d+")
# The C loader reads the same YAML several times faster, and every command reads the config
# and every card note. PyYAML falls back to its Python loader when built without libyaml.
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def folder_key(name: str) -> str:
    """A folder name as the file system may compare it: one Unicode form, capitals ignored.

    On a Mac, HW and hw are one folder. Two names with the same key are treated as the same
    folder everywhere, so the tool behaves the same on every file system.
    """
    return unicodedata.normalize("NFC", name).casefold()


def yaml_problem(error: yaml.YAMLError, first_line: int = 1) -> str:
    """What is wrong with a piece of YAML and where, without quoting the text itself.

    `first_line` is the line of the file the YAML starts on. PyYAML's own message can show
    the line it stopped at, and a line of the config may hold a key pasted in by mistake, so
    only its fixed wording is passed on: a part that quotes anything is left out.
    """
    problem = getattr(error, "problem", None) or ""
    mark = getattr(error, "problem_mark", None)
    where = "" if mark is None else f"line {mark.line + first_line}, column {mark.column + 1}"
    if "'" in problem or '"' in problem:
        problem = ""
    return ", ".join(part for part in (problem, where) if part) or "it cannot be read"


def clean_dump_dir(value: str) -> str:
    """The dump folder as the config keeps it: a path inside the vault, relative to it.

    Spaces around it, slashes after it and a leading "./" are dropped, so "course/" and
    "./course" both read as "course". Refused with ValueError: nothing at all, the vault
    itself ("."), an absolute path, a backslash, a ".." part, and a path that is or starts in
    one of the tool's own folders. The messages never repeat what was typed.
    """
    text = value.strip()
    if "\\" in text:
        raise ValueError("the dump folder is written with forward slashes, not backslashes")
    if text.startswith("/"):
        raise ValueError("the dump folder must be given relative to the vault, not as a full path")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(
            "the dump folder must be a folder inside the vault, not the vault itself or nothing"
        )
    if ".." in parts:
        raise ValueError("the dump folder must be inside the vault: no '..' in its path")
    if folder_key(parts[0]) in {folder_key(name) for name in TOOL_DIRS}:
        raise ValueError(
            f"the dump folder cannot be one of the tool's own folders ({', '.join(TOOL_DIRS)}), "
            "or sit inside one"
        )
    return "/".join(parts)


def _problems(error: ValidationError) -> str:
    """A validation error as "setting: what is wrong" parts, without the values typed in."""
    parts = []
    for problem in error.errors(include_url=False, include_context=False, include_input=False):
        setting = ".".join(str(step) for step in problem["loc"])
        message = problem["msg"].removeprefix("Value error, ")
        parts.append(f"{setting}: {message}" if setting else message)
    return "; ".join(parts)


class Config(StrictModel):
    """Settings for one vault. The values here are the defaults `p0 init` writes."""

    model: str = "jev-1.14.0"  # always an exact version, never an alias
    backend: Literal["jev"] = "jev"
    dump_dir: str = DEFAULT_DUMP_DIR  # relative to the vault; see clean_dump_dir
    variant: str = "card"  # which input format the model sees
    card_token_budget: int = 3000
    unsorted_fits_any_below: float = 0.5  # fits_any under this: Unsorted
    unsorted_top_below: float = 0.6  # top group's probability under this: Unsorted
    close_call_margin: float = 0.15  # top two groups closer than this: placed, but flagged
    max_spend_usd: float = 5.0  # stop calling the model once this vault's log shows this much spent
    usd_per_million_tokens: float = 0.042

    @field_validator("model")
    @classmethod
    def _exact_version(cls, value: str) -> str:
        """Accept only an exact model version: results must be tied to one version.

        A moving alias such as jev-latest, a bare name, a partial version and a padded or
        upper-case spelling are all refused. The message does not repeat the value.
        """
        if not _EXACT_VERSION.fullmatch(value):
            raise ValueError(
                "an exact model version is needed, such as jev-1.14.0 (a lower-case name, then "
                "a full dotted version); a moving alias such as jev-latest is never used"
            )
        return value

    @field_validator("dump_dir")
    @classmethod
    def _inside_the_vault(cls, value: str) -> str:
        """Keep the dump folder in one plain form, and refuse one that is not safe to use."""
        return clean_dump_dir(value)

    @staticmethod
    def path(vault: Path) -> Path:
        """Where a vault's config file lives."""
        return vault / DECIDE_DIR / "config.yaml"

    @classmethod
    def load(cls, vault: Path) -> "Config":
        """Read and validate a vault's config file.

        Raises ValueError that names the file when it is not valid YAML or holds a setting
        that is not allowed. The message gives the setting's name and what is wrong with it,
        never the value that was typed: a key pasted into the config by mistake must not be
        shown again.
        """
        path = cls.path(vault)
        try:
            data = yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}
        except UnicodeDecodeError:
            raise ValueError(f"{path} is not UTF-8 text") from None
        except yaml.YAMLError as error:
            raise ValueError(f"{path} is not valid YAML: {yaml_problem(error)}") from None
        try:
            return cls.model_validate(data)
        except ValidationError as error:
            # from None: the original error quotes the values, and must not be printed either.
            raise ValueError(f"{path} is not a valid config: {_problems(error)}") from None

    def save(self, vault: Path) -> None:
        """Write this config to the vault, creating .decide/ if needed."""
        self.path(vault).parent.mkdir(parents=True, exist_ok=True)
        self.path(vault).write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False), encoding="utf-8"
        )


def decide_path(vault: Path, name: str) -> Path:
    """Path of a tool state file inside .decide/, e.g. labels.jsonl."""
    return vault / DECIDE_DIR / name


def env_value(name: str, start: Path | None = None) -> str | None:
    """Look up a secret: the environment first, then a .env file in `start` or any folder above it.

    Secrets never live in the vault's own files; .env is gitignored and sits outside tool state.
    """
    value = os.environ.get(name, "").strip()
    if value:
        return value
    here = (start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        env_file = folder / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, sep, raw = line.partition("=")
                if sep and key.strip() == name:
                    found = raw.strip().strip("'\"")
                    if found:
                        return found
    return None
