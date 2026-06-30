"""
Validate Claude Sonnet 5 model configuration entries.

Sonnet 5 ships with the gen-5 adaptive-thinking profile (adaptive thinking
always on, no extended thinking, ``effort`` defaults to ``high``), so it must
mirror the sampling-param and prefill restrictions that Fable 5 / Opus 4.8 carry
rather than the older Sonnet 4.6 behavior. The cost-map entries are also what
populate ``litellm.anthropic_models`` at import, which is what lets a bare
``claude-sonnet-5`` name resolve to the ``anthropic`` provider (and match an
``anthropic/*`` wildcard deployment).
"""

import json
import os

import pytest

import litellm
from litellm.constants import BEDROCK_CONVERSE_MODELS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")

ALL_SONNET_5_VARIANTS = (
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
)


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled backup cost map so assertions don't depend on the
    network-fetched ``main`` copy (which lags this branch until merge)."""
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_sonnet_5_pricing_and_capabilities():
    model_data = _load_root_cost_map()

    expected_providers = {
        "claude-sonnet-5": "anthropic",
        "anthropic.claude-sonnet-5": "bedrock_converse",
        "vertex_ai/claude-sonnet-5": "vertex_ai-anthropic_models",
        "azure_ai/claude-sonnet-5": "azure_ai",
    }

    for model_name, provider in expected_providers.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]

        assert info["litellm_provider"] == provider
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        assert info["max_tokens"] == 128000

        # Standard Sonnet pricing: $3 / $15 per MTok, with the 1.25x cache-write
        # and 0.1x cache-read multipliers.
        assert info["input_cost_per_token"] == 3e-06
        assert info["output_cost_per_token"] == 1.5e-05
        assert info["cache_creation_input_token_cost"] == 3.75e-06
        assert info["cache_read_input_token_cost"] == 3e-07

        # gen-5 adaptive-thinking profile: effort-driven, no sampling params, no
        # assistant prefill.
        assert info["supports_adaptive_thinking"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_sampling_params"] is False
        assert info["supports_assistant_prefill"] is False

        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is True


def test_sonnet_5_bedrock_regional_pricing():
    """Global/base endpoints use base pricing; the us./eu./au./jp. regional
    cross-region inference profiles carry a 10% premium."""
    model_data = _load_root_cost_map()

    base_pricing = {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_creation_input_token_cost": 3.75e-06,
        "cache_read_input_token_cost": 3e-07,
    }
    regional_pricing = {
        "input_cost_per_token": 3.3e-06,
        "output_cost_per_token": 1.65e-05,
        "cache_creation_input_token_cost": 4.125e-06,
        "cache_read_input_token_cost": 3.3e-07,
    }

    expected = {
        "anthropic.claude-sonnet-5": base_pricing,
        "global.anthropic.claude-sonnet-5": base_pricing,
        "us.anthropic.claude-sonnet-5": regional_pricing,
        "eu.anthropic.claude-sonnet-5": regional_pricing,
        "au.anthropic.claude-sonnet-5": regional_pricing,
        "jp.anthropic.claude-sonnet-5": regional_pricing,
    }

    for model_name, pricing in expected.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]
        assert info["litellm_provider"] == "bedrock_converse"
        assert info["bedrock_output_config_effort_ceiling"] == "xhigh"
        for key, value in pricing.items():
            assert info[key] == value, f"{model_name}.{key} = {info[key]}, want {value}"


def test_sonnet_5_present_in_bundled_backup():
    """The bundled backup is the runtime fallback (and what tests load with
    ``LITELLM_LOCAL_MODEL_COST_MAP=True``); it must carry the same entries as the
    root cost map, otherwise the model resolves on one path but not the other."""
    backup = GetModelCostMap.load_local_model_cost_map()
    for model_name in ALL_SONNET_5_VARIANTS:
        assert model_name in backup, f"Missing from backup cost map: {model_name}"


def test_sonnet_5_registered_for_bedrock_converse():
    assert "anthropic.claude-sonnet-5" in BEDROCK_CONVERSE_MODELS


def test_sonnet_5_provider_resolves_via_model_info(local_model_cost_map):
    """Regression: ``claude-sonnet-5`` must resolve to provider ``anthropic``.

    Before the cost-map entry existed, the model was unknown to LiteLLM, so it
    could not be tied to the ``anthropic`` provider and an ``anthropic/*``
    wildcard deployment would not match it."""
    info = litellm.get_model_info(model="claude-sonnet-5")
    assert info["litellm_provider"] == "anthropic"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_sonnet_5_all_variants_carry_adaptive_thinking_flag(cost_map):
    """Every Sonnet 5 entry must advertise ``supports_adaptive_thinking``.

    Adaptive-thinking detection is cost-map driven, so a single variant missing
    the flag silently sends the legacy ``thinking.type='enabled'`` shape and the
    provider 400s. This guards against a future variant being added without it."""
    variants = [k for k in cost_map if "claude-sonnet-5" in k]
    assert variants, "no claude-sonnet-5 entries found in cost map"
    missing = [k for k in variants if cost_map[k].get("supports_adaptive_thinking") is not True]
    assert not missing, f"missing supports_adaptive_thinking: {missing}"
