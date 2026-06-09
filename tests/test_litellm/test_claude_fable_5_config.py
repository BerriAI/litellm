"""
Validate Claude Fable 5 model configuration entries.

Regression coverage for adding ``claude-fable-5`` to the model cost map so
that LiteLLM can resolve its provider and pricing for both the Anthropic and
Vertex AI providers.
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


def test_fable_5_model_pricing_and_capabilities():
    model_data = _load_root_cost_map()

    expected_models = {
        "claude-fable-5": {
            "provider": "anthropic",
            "max_input_tokens": 1000000,
        },
        "vertex_ai/claude-fable-5": {
            "provider": "vertex_ai-anthropic_models",
            "max_input_tokens": 1000000,
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

        assert info["input_cost_per_token"] == 1e-05
        assert info["output_cost_per_token"] == 5e-05
        assert info["cache_creation_input_token_cost"] == 1.25e-05
        assert info["cache_read_input_token_cost"] == 1e-06

        assert info["supports_computer_use"] is True
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_tool_choice"] is True
        assert info["supports_vision"] is True


def test_fable_5_vertex_batch_pricing():
    model_data = _load_root_cost_map()

    for model_name in ("vertex_ai/claude-fable-5", "vertex_ai/claude-fable-5@default"):
        info = model_data[model_name]
        assert info["input_cost_per_token_batches"] == 5e-06
        assert info["output_cost_per_token_batches"] == 2.5e-05


def test_fable_5_present_in_bundled_backup():
    """The bundled backup is the runtime fallback — it must carry the same
    entries as the root cost map."""
    backup = GetModelCostMap.load_local_model_cost_map()
    for model_name in (
        "claude-fable-5",
        "vertex_ai/claude-fable-5",
        "vertex_ai/claude-fable-5@default",
    ):
        assert model_name in backup, f"Missing from backup cost map: {model_name}"


def test_fable_5_provider_resolves_via_model_info(local_model_cost_map):
    """Regression: ``claude-fable-5`` must resolve to provider ``anthropic``."""
    info = litellm.get_model_info(model="claude-fable-5")
    assert info["litellm_provider"] == "anthropic"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000
