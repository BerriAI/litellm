"""Load the curated human overlay: the per-cell fields a person decides, keyed by id.

The overlay never defines whether a behavior exists; the generated product surface does
that. It only annotates a cell with judgement (tier, source, rationale, fail-before-fix,
support) and enumerates the ids generation does not yet produce. Ordinary test PRs do not
touch it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import TypeAdapter

from .schema import OverlayRow

OVERLAY_PATH = Path(__file__).resolve().parent / "overlay.yaml"

_OVERLAY_ADAPTER: TypeAdapter[dict[str, OverlayRow]] = TypeAdapter(dict[str, OverlayRow])


def load_overlay(path: Path = OVERLAY_PATH) -> dict[str, OverlayRow]:
    return _OVERLAY_ADAPTER.validate_python(yaml.safe_load(path.read_text()) or {})
