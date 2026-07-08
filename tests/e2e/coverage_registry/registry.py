"""Load and validate the registry: the denominator, built in one shot from the YAMLs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from .schema import CELL_ADAPTER, Cell

REGISTRY_DIR = Path(__file__).resolve().parent


def load_registry(registry_dir: Path = REGISTRY_DIR) -> tuple[Cell, ...]:
    """Every cell across every `*.yaml`, validated. Raises on a schema violation or
    a duplicate id, since either would corrupt the coverage denominator."""
    cells = tuple(
        CELL_ADAPTER.validate_python(row)
        for path in sorted(registry_dir.glob("*.yaml"))
        for row in (yaml.safe_load(path.read_text()) or ())
    )
    duplicates = sorted(cid for cid, n in Counter(c.id for c in cells).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate cell ids in registry: {duplicates}")
    return cells
