import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.llms.exa_ai.search.cost_calculator import (
    EXA_AI_ADDITIONAL_RESULT_COST,
    EXA_AI_CATALOG_PRICES,
    EXA_AI_INCLUDED_RESULTS,
    EXA_AI_LEGACY_TIERS,
    EXA_AI_SUMMARY_COST_PER_PAGE,
    ExaAITier,
    exa_ai_list_price,
)

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
def test_shipped_exa_prices_agree_with_the_receipt_override_baseline(path: str) -> None:
    entries: Final = _search_entries(path)
    assert {
        key: entry["input_cost_per_query"] for key, entry in entries.items() if key.startswith("exa_ai/")
    } == EXA_AI_CATALOG_PRICES
    assert (
        TypeAdapter(tuple[ExaAITier, ...]).validate_python(entries["exa_ai/search"]["tiered_pricing"])
        == EXA_AI_LEGACY_TIERS
    )


@pytest.mark.parametrize(
    ("params", "extra_results", "summaries"),
    (
        ({}, 0, 0),
        ({"max_results": None}, 0, 0),
        ({"max_results": True, "numResults": EXA_AI_INCLUDED_RESULTS + 3}, 3, 0),
        ({"numResults": EXA_AI_INCLUDED_RESULTS + 3}, 3, 0),
        ({"max_results": EXA_AI_INCLUDED_RESULTS + 2, "numResults": 100}, 2, 0),
        ({"max_results": 3, "contents": {"summary": {}}}, 0, 3),
        ({"max_results": 3, "contents": {"summary": False}}, 0, 0),
        ({"type": "lite", "max_results": 3, "contents": {"summary": {}}}, 0, 0),
    ),
)
def test_fallback_adds_only_requested_results_and_summaries(
    params: Mapping[str, object], extra_results: int, summaries: int
) -> None:
    base_price: Final = 0.123
    assert exa_ai_list_price(base_price, params) == pytest.approx(
        base_price + extra_results * EXA_AI_ADDITIONAL_RESULT_COST + summaries * EXA_AI_SUMMARY_COST_PER_PAGE
    )
