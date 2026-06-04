"""
Validate Claude Opus 4.8 model configuration entries.

Regression coverage for the wildcard-routing failure where a bare model name
(``claude-opus-4-8``) could not match an ``anthropic/*`` deployment because
LiteLLM could not infer its provider — the model was simply missing from the
model cost map, so ``get_llm_provider`` raised and the router returned
"no healthy deployments for this model". The fix is the cost-map entries added
for Anthropic, Bedrock, Vertex AI, and Azure AI; those entries are what populate
``litellm.anthropic_models`` at import time, which is what the bare-name lookup
in ``get_llm_provider`` consumes.
"""

import json
import os

import pytest

import litellm
from litellm.constants import BEDROCK_CONVERSE_MODELS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


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


def test_opus_4_8_model_pricing_and_capabilities():
    model_data = _load_root_cost_map()

    expected_models = {
        "claude-opus-4-8": {
            "provider": "anthropic",
            "max_input_tokens": 1000000,
        },
        "anthropic.claude-opus-4-8": {
            "provider": "bedrock_converse",
            "max_input_tokens": 1000000,
        },
        "vertex_ai/claude-opus-4-8": {
            "provider": "vertex_ai-anthropic_models",
            "max_input_tokens": 1000000,
        },
        # Microsoft Foundry / Azure caps Opus 4.8 at a 200k context window.
        "azure_ai/claude-opus-4-8": {
            "provider": "azure_ai",
            "max_input_tokens": 200000,
        },
    }

    for model_name, config in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]

        assert info["litellm_provider"] == config["provider"]
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == config["max_input_tokens"]
        assert info["max_output_tokens"] == 128000
        assert info["max_tokens"] == 128000

        # Base pricing matches Opus 4.7: $5 / $25 per MTok, with the standard
        # 1.25x cache-write and 0.1x cache-read multipliers.
        assert info["input_cost_per_token"] == 5e-06
        assert info["output_cost_per_token"] == 2.5e-05
        assert info["cache_creation_input_token_cost"] == 6.25e-06
        assert info["cache_read_input_token_cost"] == 5e-07

        # Opus 4.x flagships are flat-rate across the full context window.
        assert "input_cost_per_token_above_200k_tokens" not in info
        assert "output_cost_per_token_above_200k_tokens" not in info

        assert info["supports_assistant_prefill"] is False
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is True


def test_opus_4_8_bedrock_regional_model_pricing():
    model_data = _load_root_cost_map()

    # Global endpoints use base pricing; regional endpoints carry a 10% premium.
    expected_models = {
        "global.anthropic.claude-opus-4-8": {
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 2.5e-05,
            "cache_creation_input_token_cost": 6.25e-06,
            "cache_read_input_token_cost": 5e-07,
        },
        "us.anthropic.claude-opus-4-8": {
            "input_cost_per_token": 5.5e-06,
            "output_cost_per_token": 2.75e-05,
            "cache_creation_input_token_cost": 6.875e-06,
            "cache_read_input_token_cost": 5.5e-07,
        },
        "eu.anthropic.claude-opus-4-8": {
            "input_cost_per_token": 5.5e-06,
            "output_cost_per_token": 2.75e-05,
            "cache_creation_input_token_cost": 6.875e-06,
            "cache_read_input_token_cost": 5.5e-07,
        },
        "au.anthropic.claude-opus-4-8": {
            "input_cost_per_token": 5.5e-06,
            "output_cost_per_token": 2.75e-05,
            "cache_creation_input_token_cost": 6.875e-06,
            "cache_read_input_token_cost": 5.5e-07,
        },
    }

    for model_name, expected in expected_models.items():
        assert model_name in model_data, f"Missing model entry: {model_name}"
        info = model_data[model_name]
        assert info["litellm_provider"] == "bedrock_converse"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        assert info["bedrock_output_config_effort_ceiling"] == "xhigh"
        for key, value in expected.items():
            assert info[key] == value


def test_opus_4_8_fast_mode_multiplier():
    """Opus 4.8 dropped fast-mode pricing to 2x base ($10/$50 per MTok);
    Opus 4.7 was 6x ($30/$150)."""
    model_data = _load_root_cost_map()
    entry = model_data["claude-opus-4-8"]["provider_specific_entry"]
    assert entry["us"] == 1.1
    assert entry["fast"] == 2.0


def test_opus_4_8_present_in_bundled_backup():
    """The bundled backup is the runtime fallback (and what tests load with
    ``LITELLM_LOCAL_MODEL_COST_MAP=True``) — it must carry the same entries as
    the root cost map, otherwise the model resolves on one path but not the
    other."""
    backup = GetModelCostMap.load_local_model_cost_map()
    for model_name in (
        "claude-opus-4-8",
        "anthropic.claude-opus-4-8",
        "global.anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-8",
        "eu.anthropic.claude-opus-4-8",
        "au.anthropic.claude-opus-4-8",
        "vertex_ai/claude-opus-4-8",
        "vertex_ai/claude-opus-4-8@default",
        "azure_ai/claude-opus-4-8",
    ):
        assert model_name in backup, f"Missing from backup cost map: {model_name}"


def test_opus_4_8_registered_for_bedrock_converse():
    assert "anthropic.claude-opus-4-8" in BEDROCK_CONVERSE_MODELS


def test_opus_4_8_provider_resolves_via_model_info(local_model_cost_map):
    """Regression: ``claude-opus-4-8`` must resolve to provider ``anthropic``.

    Before the cost-map entry existed, the model was unknown to LiteLLM, so it
    could not be tied to the ``anthropic`` provider and an ``anthropic/*``
    wildcard deployment would not match it.
    """
    info = litellm.get_model_info(model="claude-opus-4-8")
    assert info["litellm_provider"] == "anthropic"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000
