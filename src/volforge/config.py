"""Project configuration helpers.

VolForge keeps local secrets in a repository-root ``.env`` file. Real process
environment variables take precedence so CI, containers, and production jobs
can override local development values without changing files.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

__all__ = ["load_project_env", "get_env", "require_env"]


def _candidate_env_files() -> list[Path]:
    """Return likely repo-local .env locations without searching parent trees."""
    candidates = [Path.cwd() / ".env"]
    try:
        repo_root = Path(__file__).resolve().parents[2]
    except IndexError:  # pragma: no cover - defensive for unusual packaging layouts
        repo_root = None
    if repo_root is not None:
        candidate = repo_root / ".env"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def load_project_env(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load VolForge's local ``.env`` file if present.

    Parameters
    ----------
    path:
        Optional explicit dotenv path. When omitted, VolForge first checks the
        current working directory and then the source-tree repository root.
    override:
        Passed to :func:`dotenv.load_dotenv`. The default ``False`` preserves
        already-defined process environment variables.

    Returns
    -------
    Path | None
        The loaded dotenv path, or ``None`` when no file exists.
    """
    candidates = [Path(path).expanduser()] if path is not None else _candidate_env_files()
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=override)
            return candidate
    return None


def get_env(name: str, default: str | None = None) -> str | None:
    """Load project dotenv once, then read an environment value."""
    load_project_env()
    return os.getenv(name, default)


def require_env(name: str, *, hint: str | None = None) -> str:
    """Return a required configuration value with a clear setup error."""
    value = get_env(name)
    if value:
        return value
    message = f"Missing required environment variable {name}."
    if hint:
        message += f" {hint}"
    raise RuntimeError(message)
