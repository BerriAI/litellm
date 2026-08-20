"""
Validates the sference catalog entries in model_prices_and_context_window.json
against the sference /v1/models source data, and that the bundled backup stays
in sync.
"""

import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


@pytest.fixture(autouse=True)
def clear_sference_env(monkeypatch):
    """Keep routing assertions hermetic when the host exports sference env vars."""
    monkeypatch.delenv("SFERENCE_API_BASE", raising=False)
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)


SFERENCE_SOURCE = "https://api.sference.com/v1/models"

SFERENCE_MODELS = {
    "sference/Qwen/Qwen3.6-35B-A3B": {
        "input_cost_per_token": 2e-07,
        "output_cost_per_token": 1.25e-06,
        "cache_read_input_token_cost": 5e-08,
        "max_input_tokens": 262144,
        "supports_reasoning": True,
        "supports_vision": False,
        "supported_modalities": ["text"],
    },
    "sference/Qwen/Qwen3-VL-30B-A3B-Instruct": {
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
        "cache_read_input_token_cost": 1e-07,
        "max_input_tokens": 262144,
        "supports_reasoning": False,
        "supports_vision": True,
        "supported_modalities": ["text", "image"],
    },
    "sference/bottlecapai/ThinkingCap-Qwen3.6-27B": {
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2.6e-06,
        "cache_read_input_token_cost": 5e-08,
        "max_input_tokens": 262144,
        "supports_reasoning": True,
        "supports_vision": False,
        "supported_modalities": ["text"],
    },
    "sference/deepseek-ai/DeepSeek-V4-Flash": {
        "input_cost_per_token": 1.4e-07,
        "output_cost_per_token": 2.8e-07,
        "cache_read_input_token_cost": 7e-08,
        "max_input_tokens": 1048576,
        "supports_reasoning": True,
        "supports_vision": False,
        "supported_modalities": ["text"],
    },
    "sference/moonshotai/Kimi-K3": {
        "input_cost_per_token": 2.25e-06,
        "output_cost_per_token": 1.125e-05,
        "cache_read_input_token_cost": 2.25e-07,
        "max_input_tokens": 1048576,
        "supports_reasoning": True,
        "supports_vision": False,
        "supported_modalities": ["text"],
    },
    "sference/zai-org/GLM-5.2": {
        "input_cost_per_token": 1.2e-06,
        "output_cost_per_token": 4.2e-06,
        "cache_read_input_token_cost": 2.6e-07,
        "max_input_tokens": 1048576,
        "supports_reasoning": True,
        "supports_vision": False,
        "supported_modalities": ["text"],
    },
}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", sorted(SFERENCE_MODELS))
def test_sference_catalog_entry(model: str):
    model_cost = _load(Path(__file__).parents[2] / "model_prices_and_context_window.json")

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    expected = SFERENCE_MODELS[model]
    assert info["litellm_provider"] == "sference"
    assert info["mode"] == "chat"
    assert info["source"] == SFERENCE_SOURCE

    assert info["input_cost_per_token"] == expected["input_cost_per_token"]
    assert info["output_cost_per_token"] == expected["output_cost_per_token"]
    assert info["cache_read_input_token_cost"] == expected["cache_read_input_token_cost"]
    assert info["max_input_tokens"] == expected["max_input_tokens"]
    assert info["max_output_tokens"] == expected["max_input_tokens"]
    assert info["max_tokens"] == expected["max_input_tokens"]

    assert info["supports_reasoning"] is expected["supports_reasoning"]
    assert info["supports_vision"] is expected["supports_vision"]
    assert info["supported_modalities"] == expected["supported_modalities"]
    assert info["supported_output_modalities"] == ["text"]
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_system_messages"] is True
    assert info["supports_native_streaming"] is True
    assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/messages"]


@pytest.mark.parametrize("model", sorted(SFERENCE_MODELS))
def test_sference_backup_matches_main(model: str):
    repo_root = Path(__file__).parents[2]
    main_cost = _load(repo_root / "model_prices_and_context_window.json")
    backup_cost = _load(repo_root / "litellm" / "model_prices_and_context_window_backup.json")

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"


@pytest.mark.parametrize("model", sorted(SFERENCE_MODELS))
def test_sference_model_routing(model: str):
    routed_model, provider, _, api_base = get_llm_provider(model=model, api_key="sk-test")

    assert routed_model == model.split("/", 1)[1]
    assert provider == "sference"
    assert api_base == "https://api.sference.com/v1"
