"""Build the coverage denominator: the generated product surface unioned with the ids
the overlay still enumerates, each annotated with its curated human fields."""

from __future__ import annotations

from pathlib import Path

from .overlay import OVERLAY_PATH, load_overlay
from .product_surface import generate_llm_cell_ids
from .schema import Cell, OverlayRow, Tier, parse_module

_DEFAULT_ROW = OverlayRow(tier=Tier.P2)


def _cell(cell_id: str, row: OverlayRow) -> Cell:
    return Cell(
        id=cell_id,
        module=parse_module(cell_id),
        tier=row.tier,
        source=row.source,
        rationale=row.rationale,
        fail_before_fix=row.fail_before_fix,
        supported=row.supported,
    )


def load_registry(overlay_path: Path = OVERLAY_PATH) -> tuple[Cell, ...]:
    """Every denominator cell, validated. The id set is the generated LLM surface
    unioned with the overlay's ids; each cell carries its overlay row when one exists
    and otherwise a default P2 row, so a newly generated surface shows up as an
    uncovered gap rather than silently vanishing. Raises on a bad module prefix or a
    malformed overlay row, since either would corrupt the denominator."""
    overlay = load_overlay(overlay_path)
    cell_ids = generate_llm_cell_ids() | frozenset(overlay)
    return tuple(_cell(cid, overlay.get(cid, _DEFAULT_ROW)) for cid in sorted(cell_ids))
