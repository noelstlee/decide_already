"""Decision-model backends. Everything outside this package talks to the Backend interface only."""

from pathlib import Path

from ..config import JEV_LOG_FILE, Config, decide_path, env_value
from .base import Backend


def make_backend(root: Path, config: Config) -> Backend:
    """Build the backend for a vault: Jev, logging to .decide/jev.jsonl.

    The API key comes from the environment, or from a .env file in the vault or a folder above it.
    """
    # Imported here so that only code about to call the model loads the TypeSafe SDK.
    from .jev import API_KEY_ENV, JevBackend

    return JevBackend(config, decide_path(root, JEV_LOG_FILE), api_key=env_value(API_KEY_ENV, root))
