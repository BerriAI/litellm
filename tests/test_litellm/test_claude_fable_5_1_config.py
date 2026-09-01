"""
Validate the Claude Fable 5.1 (Anthropic API) model configuration entry.

Fable 5.1 keeps Fable 5's $10/$50 per MTok pricing and adaptive-only API
surface, but prices cache reads at 0.025x base input ($0.25/MTok) instead of
0.1x, and rejects forced tool use (``tool_choice`` type ``any``/``tool``) with
a 400 because thinking is always on and a forced call would skip it.
"""

import json
import os

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


def test_fable_5_1_pricing_and_capabilities():
    info = _load_root_cost_map()["claude-fable-5-1"]

    assert info["litellm_provider"] == "anthropic"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000
    assert info["deprecation_date"] == "2027-09-01"

    assert info["input_cost_per_token"] == 1e-05
    assert info["output_cost_per_token"] == 5e-05
    assert info["cache_creation_input_token_cost"] == 1.25e-05
    assert info["cache_creation_input_token_cost_above_1hr"] == 2e-05
    assert info["cache_read_input_token_cost"] == 2.5e-07
    assert "input_cost_per_token_above_200k_tokens" not in info
    assert "output_cost_per_token_above_200k_tokens" not in info

    assert info["supports_adaptive_thinking"] is True
    assert info["thinking_always_on"] is True
    assert info["supports_mid_conversation_system"] is True
    assert info["supports_assistant_prefill"] is False
    assert info["supports_sampling_params"] is False
    assert info["supports_forced_tool_use"] is False
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_native_structured_output"] is True
    assert info["supports_prompt_caching"] is True
    assert info["prompt_cache_min_tokens"] == 512
    assert info["supports_reasoning"] is True
    assert info["supports_vision"] is True
    assert info["supports_xhigh_reasoning_effort"] is True
    assert info["supports_max_reasoning_effort"] is True
    assert info["provider_specific_entry"] == {"us": 1.1}
    assert "supports_speed" not in info


def test_fable_5_1_cache_read_is_a_quarter_of_fable_5():
    cost_map = _load_root_cost_map()
    fable_5 = cost_map["claude-fable-5"]
    fable_5_1 = cost_map["claude-fable-5-1"]
    assert fable_5_1["cache_read_input_token_cost"] == fable_5["cache_read_input_token_cost"] / 4
    for key in (
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
    ):
        assert fable_5_1[key] == fable_5[key], key


def test_fable_5_1_present_in_bundled_backup():
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    assert backup["claude-fable-5-1"] == root["claude-fable-5-1"]


def test_fable_5_1_cost_uses_cheaper_cache_read(local_model_cost_map):
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="claude-fable-5-1",
        custom_llm_provider="anthropic",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
    )
    assert round(prompt_cost, 6) == 0.25
    assert completion_cost == 0


def test_fable_5_1_provider_resolves_via_model_info(local_model_cost_map):
    info = litellm.get_model_info(model="claude-fable-5-1")
    assert info["litellm_provider"] == "anthropic"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000
    assert info["supports_adaptive_thinking"] is True
    assert info["thinking_always_on"] is True
