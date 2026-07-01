"""
Validate Claude Sonnet 5 model configuration entries.

Sonnet 5 launched 2026-06-29 with introductory pricing ($2/$10 per MTok)
through August 31, 2026, after which it moves to standard pricing ($3/$15).
The cost-map entries use the introductory rates so spend tracking is accurate
during the promotional period.
"""

import json
import os

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_sonnet_5_model_pricing_and_capabilities():
    model_data = _load_root_cost_map()

    expected_models = [
        ("claude-sonnet-5", "anthropic"),
        ("anthropic.claude-sonnet-5", "bedrock_converse"),
        ("vertex_ai/claude-sonnet-5", "vertex_ai-anthropic_models"),
        ("azure_ai/claude-sonnet-5", "azure_ai"),
    ]

    for model_name, provider in expected_models:
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]

        assert info["litellm_provider"] == provider
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        assert info["max_tokens"] == 128000

        assert info["input_cost_per_token"] == 2e-06
        assert info["output_cost_per_token"] == 1e-05
        assert info["cache_creation_input_token_cost"] == 2.5e-06
        assert info["cache_creation_input_token_cost_above_1hr"] == 4e-06
        assert info["cache_read_input_token_cost"] == 2e-07

        assert "input_cost_per_token_above_200k_tokens" not in info
        assert "output_cost_per_token_above_200k_tokens" not in info

        assert info["supports_adaptive_thinking"] is True
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is True


def test_sonnet_5_bedrock_regional_model_pricing():
    model_data = _load_root_cost_map()

    expected_models = {
        "global.anthropic.claude-sonnet-5": {
            "input_cost_per_token": 2e-06,
            "output_cost_per_token": 1e-05,
            "cache_creation_input_token_cost": 2.5e-06,
            "cache_read_input_token_cost": 2e-07,
        },
        "us.anthropic.claude-sonnet-5": {
            "input_cost_per_token": 2.2e-06,
            "output_cost_per_token": 1.1e-05,
            "cache_creation_input_token_cost": 2.75e-06,
            "cache_read_input_token_cost": 2.2e-07,
        },
        "eu.anthropic.claude-sonnet-5": {
            "input_cost_per_token": 2.2e-06,
            "output_cost_per_token": 1.1e-05,
            "cache_creation_input_token_cost": 2.75e-06,
            "cache_read_input_token_cost": 2.2e-07,
        },
        "au.anthropic.claude-sonnet-5": {
            "input_cost_per_token": 2.2e-06,
            "output_cost_per_token": 1.1e-05,
            "cache_creation_input_token_cost": 2.75e-06,
            "cache_read_input_token_cost": 2.2e-07,
        },
        "jp.anthropic.claude-sonnet-5": {
            "input_cost_per_token": 2.2e-06,
            "output_cost_per_token": 1.1e-05,
            "cache_creation_input_token_cost": 2.75e-06,
            "cache_read_input_token_cost": 2.2e-07,
        },
    }

    for model_name, expected in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]
        assert info["litellm_provider"] == "bedrock_converse"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        for key, value in expected.items():
            assert info[key] == value, f"{model_name}.{key}: expected {value}, got {info[key]}"


def test_sonnet_5_present_in_bundled_backup():
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    for model_name in (
        "claude-sonnet-5",
        "anthropic.claude-sonnet-5",
        "global.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
        "eu.anthropic.claude-sonnet-5",
        "au.anthropic.claude-sonnet-5",
        "jp.anthropic.claude-sonnet-5",
        "vertex_ai/claude-sonnet-5",
        "vertex_ai/claude-sonnet-5@default",
        "azure_ai/claude-sonnet-5",
        "snowflake/claude-sonnet-5",
    ):
        assert model_name in backup, f"Missing from backup cost map: {model_name}"
        assert backup[model_name] == root[model_name], model_name


def test_sonnet_5_provider_resolves_via_model_info(local_model_cost_map):
    info = litellm.get_model_info(model="claude-sonnet-5")
    assert info["litellm_provider"] == "anthropic"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000
    assert info["input_cost_per_token"] == 2e-06
    assert info["output_cost_per_token"] == 1e-05


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_sonnet_5_all_variants_carry_adaptive_thinking_flag(cost_map):
    variants = [k for k in cost_map if "claude-sonnet-5" in k and "sonnet-4-5" not in k]
    assert variants, "no claude-sonnet-5 entries found in cost map"
    missing = [
        k for k in variants if cost_map[k].get("supports_adaptive_thinking") is not True
    ]
    assert not missing, f"missing supports_adaptive_thinking: {missing}"
