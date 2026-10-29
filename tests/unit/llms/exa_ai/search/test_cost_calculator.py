import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.llms.exa_ai.search.cost_calculator import EXA_AI_LEGACY_TIERS, ExaAITier

REPO_ROOT: Final = Path(__file__).resolve().parents[5]
SHIPPED_COST_MAPS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)


def _search_entries(path: str) -> dict[str, dict[str, object]]:
    return {
        key: entry
        for key, entry in json.loads((REPO_ROOT / path).read_text()).items()
        if isinstance(entry, dict) and entry.get("mode") == "search"
    }


@pytest.mark.parametrize("path", SHIPPED_COST_MAPS)
def test_litellm_ships_exa_search_with_only_the_legacy_tiers_the_cost_calculator_treats_as_its_own(path: str) -> None:
    exa_entry: Final = _search_entries(path)["exa_ai/search"]

    assert TypeAdapter(tuple[ExaAITier, ...]).validate_python(exa_entry["tiered_pricing"]) == EXA_AI_LEGACY_TIERS
    assert exa_entry.get("input_cost_per_query") is None
    assert [key for key in _search_entries(path) if key.startswith("exa_ai/search-")] == []


@pytest.mark.parametrize("path", SHIPPED_COST_MAPS)
def test_litellm_ships_no_search_entry_with_both_a_flat_price_and_tiers(path: str) -> None:
    assert [
        key
        for key, entry in _search_entries(path).items()
        if entry.get("input_cost_per_query") is not None and entry.get("tiered_pricing") is not None
    ] == []
