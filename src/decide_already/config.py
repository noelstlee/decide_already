"""Tool settings, stored in <vault>/.decide/config.yaml. The API key is never stored here."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

DECIDE_DIR = ".decide"
SETS_DIR = "sets"
GROUPS_DIR = "_groups"
UNSORTED_DIR = "Unsorted"
DEFAULT_DUMP_DIR = "_dump"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "jev-1.14.0"
    backend: Literal["jev", "fake"] = "jev"
    dump_dir: str = DEFAULT_DUMP_DIR
    variant: str = "card"
    card_token_budget: int = 3000
    unsorted_fits_any_below: float = 0.5
    unsorted_top_below: float = 0.6
    close_call_margin: float = 0.15

    @staticmethod
    def path(vault: Path) -> Path:
        return vault / DECIDE_DIR / "config.yaml"

    @classmethod
    def load(cls, vault: Path) -> "Config":
        data = yaml.safe_load(cls.path(vault).read_text()) or {}
        return cls.model_validate(data)

    def save(self, vault: Path) -> None:
        self.path(vault).parent.mkdir(parents=True, exist_ok=True)
        self.path(vault).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
