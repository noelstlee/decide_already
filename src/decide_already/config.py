"""Tool settings, stored in <vault>/.decide/config.yaml. The API key is never stored here."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import field_validator

from .types import StrictModel

DECIDE_DIR = ".decide"
SETS_DIR = "sets"
GROUPS_DIR = "_groups"
UNSORTED_DIR = "Unsorted"
DEFAULT_DUMP_DIR = "_dump"
LABELS_FILE = "labels.jsonl"
JEV_LOG_FILE = "jev.jsonl"
SUGGESTIONS_FILE = "suggestions.jsonl"


class Config(StrictModel):
    """Settings for one vault. The values here are the defaults `p0 init` writes."""

    model: str = "jev-1.14.0"  # always an exact version, never an alias
    backend: Literal["jev"] = "jev"
    dump_dir: str = DEFAULT_DUMP_DIR
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
        """Reject moving aliases such as jev-latest: results must be tied to one version."""
        if value.endswith(("-latest", "-preview")):
            raise ValueError(
                f"{value!r} is a moving alias; pin an exact version such as jev-1.14.0"
            )
        return value

    @staticmethod
    def path(vault: Path) -> Path:
        """Where a vault's config file lives."""
        return vault / DECIDE_DIR / "config.yaml"

    @classmethod
    def load(cls, vault: Path) -> "Config":
        """Read and validate a vault's config file."""
        data = yaml.safe_load(cls.path(vault).read_text()) or {}
        return cls.model_validate(data)

    def save(self, vault: Path) -> None:
        """Write this config to the vault, creating .decide/ if needed."""
        self.path(vault).parent.mkdir(parents=True, exist_ok=True)
        self.path(vault).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))


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
            for line in env_file.read_text().splitlines():
                key, sep, raw = line.partition("=")
                if sep and key.strip() == name:
                    found = raw.strip().strip("'\"")
                    if found:
                        return found
    return None
