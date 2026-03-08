"""Lightweight I/O and filesystem helpers shared by all scripts."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    """Create *path* and any missing parents (no-op if it already exists).

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def utc_timestamp() -> str:
    """Return the current UTC time as a ``YYYYMMDD_HHMMSS`` string.

    Used to generate unique, chronologically sortable run-directory
    names under ``runs/``.

    Returns
    -------
    str
        Timestamp in ``YYYYMMDD_HHMMSS`` format, e.g.
        ``"20260307_111257"``.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_json(path: str, obj: Dict[str, Any]) -> None:
    """Write *obj* to *path* as pretty-printed, key-sorted JSON.

    Parameters
    ----------
    path : str
        Destination file path (created or overwritten).
    obj : dict
        JSON-serialisable dictionary.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
