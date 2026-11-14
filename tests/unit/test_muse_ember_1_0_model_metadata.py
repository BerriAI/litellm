import json
from pathlib import Path

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts

MUSE_EMBER = "meta/muse-ember-1.0"
LONG_CONTEXT_THRESHOLD = 200_000


def _load_cost_map(filename: str = "model_prices_and_context_window.json") -> dict:
    with open(Path(__file__).parents[2] / filename) as f:
        return json.load(f)


def test_muse_ember_routes_to_meta_model_api():
    routed_model, provider, _, api_base = get_llm_provider(model=MUSE_EMBER, api_key="sk-test")

    assert routed_model == "muse-ember-1.0"
    assert provider == "meta"
    assert api_base == "https://api.meta.ai/v1"


def test_muse_ember_backup_matches_main():
    main_cost = _load_cost_map()
    backup_cost = _load_cost_map("litellm/model_prices_and_context_window_backup.json")

    assert main_cost[MUSE_EMBER] == backup_cost.get(MUSE_EMBER)


def test_muse_ember_reasoning_efforts_include_max_and_exclude_none():
    entry = dict(_load_cost_map()[MUSE_EMBER])
    entry["key"] = MUSE_EMBER

    resolved = resolve_supported_reasoning_efforts(entry, deployment_is_mapped=True)

    assert resolved is not None
    assert "max" in resolved
    assert "xhigh" in resolved
    assert "none" not in resolved


def test_muse_ember_long_context_tier_costs_more_than_standard():
    row = _load_cost_map()[MUSE_EMBER]

    for field in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost"):
        assert row[f"{field}_above_200k_tokens"] > row[field], f"{field} long-context rate should exceed the base rate"


def test_muse_ember_bills_long_context_rates_above_200k_input_tokens():
    row = _load_cost_map()[MUSE_EMBER]
    completion_tokens = 1_000

    standard_prompt = LONG_CONTEXT_THRESHOLD
    standard_cost = sum(
        litellm.cost_per_token(model=MUSE_EMBER, prompt_tokens=standard_prompt, completion_tokens=completion_tokens)
    )
    assert standard_cost == pytest.approx(
        standard_prompt * row["input_cost_per_token"] + completion_tokens * row["output_cost_per_token"]
    )

    long_prompt = LONG_CONTEXT_THRESHOLD + 50_000
    long_cost = sum(
        litellm.cost_per_token(model=MUSE_EMBER, prompt_tokens=long_prompt, completion_tokens=completion_tokens)
    )
    assert long_cost == pytest.approx(
        long_prompt * row["input_cost_per_token_above_200k_tokens"]
        + completion_tokens * row["output_cost_per_token_above_200k_tokens"]
    )
    assert long_cost > 3 * standard_cost
