"""Load and validate the registry: the denominator, built in one shot from the YAMLs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from .schema import Cell

REGISTRY_DIR = Path(__file__).resolve().parent

_CELLS_ADAPTER: TypeAdapter[tuple[Cell, ...]] = TypeAdapter(tuple[Cell, ...])


def _load_cells(path: Path) -> tuple[Cell, ...]:
    return _CELLS_ADAPTER.validate_python(yaml.safe_load(path.read_text()) or ())


def load_registry(registry_dir: Path = REGISTRY_DIR) -> tuple[Cell, ...]:
    """Every cell across every `*.yaml`, validated. Raises on a schema violation or
    a duplicate id, since either would corrupt the coverage denominator."""
    cells = tuple(cell for path in sorted(registry_dir.glob("*.yaml")) for cell in _load_cells(path))
    duplicates = sorted(cid for cid, n in Counter(c.id for c in cells).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate cell ids in registry: {duplicates}")
    return cells
