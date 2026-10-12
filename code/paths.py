"""Locations of stored inputs and generated outputs, relative to the archive root.

Stored inputs (read-only) live in ``data/``; everything a script computes is written below
``outputs/`` (figures, tables/logs, generated data).  All paths are resolved from the location
of this file, so the scripts can be started from any working directory.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
GENERATED = OUTPUTS / "data"

for _directory in (FIGURES, TABLES, GENERATED):
    _directory.mkdir(parents=True, exist_ok=True)


def rel(path) -> str:
    """Path relative to the archive root (for log messages)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name
