import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

REPO_ROOT: Final = Path(__file__).parents[2]

CostMap = dict[str, dict[str, object]]
COST_MAP_ADAPTER: Final = TypeAdapter(CostMap)


@pytest.fixture(scope="module")
def cost_map() -> CostMap:
    with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
        return COST_MAP_ADAPTER.validate_python(json.load(f))


def test_together_chat_entries_never_carry_context_length_as_output_ceiling(cost_map: CostMap):
    inflated = sorted(
        model
        for model, info in cost_map.items()
        if info.get("litellm_provider") == "together_ai"
        and info.get("mode") == "chat"
        and "max_output_tokens" in info
        and info["max_output_tokens"] == info.get("max_input_tokens")
    )
    assert inflated == []


def _successor(info: dict[str, object]) -> str | None:
    metadata = info.get("metadata")
    if not isinstance(metadata, dict):
        return None
    successor = metadata.get("successor")
    return successor if isinstance(successor, str) else None


def test_together_successor_metadata_points_at_known_models(cost_map: CostMap):
    successors = {
        model: successor
        for model, info in cost_map.items()
        if model.startswith("together_ai/") and (successor := _successor(info)) is not None
    }
    for model, successor in successors.items():
        assert successor in cost_map, f"{model} names successor {successor} that is not in the map"


def test_together_backup_cost_map_in_sync(cost_map: CostMap):
    with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup = COST_MAP_ADAPTER.validate_python(json.load(f))
    together_main = {k: v for k, v in cost_map.items() if k.startswith("together_ai/")}
    together_backup = {k: v for k, v in backup.items() if k.startswith("together_ai/")}
    assert together_backup == together_main


def test_together_prompt_caching_flag_implies_cache_read_rate(cost_map: CostMap):
    for model, info in cost_map.items():
        if model.startswith("together_ai/") and info.get("supports_prompt_caching"):
            assert "cache_read_input_token_cost" in info, f"{model} flags caching without a cache read rate"
