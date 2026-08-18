"""
Regression tests for Azure AI Foundry Fireworks (FW-*) model cost map entries.

Prices for Data Zone pay-per-token meters come from the Azure retail prices API
(product "Azure Fireworks Models"). Kimi K3 rates come from the Microsoft Foundry
announcement. Models without dedicated Azure meters use published Fireworks
serverless rates.
"""

import json
from importlib.resources import files

import pytest

FW_MODELS = {
    "azure_ai/FW-Kimi-K2.5": {
        "input_cost_per_token": 6.6e-07,
        "output_cost_per_token": 3.3e-06,
        "cache_read_input_token_cost": 1.1e-07,
        "max_input_tokens": 262144,
        "max_output_tokens": 262144,
        "supports_vision": True,
    },
    "azure_ai/FW-Kimi-K2.6": {
        "input_cost_per_token": 1.045e-06,
        "output_cost_per_token": 4.4e-06,
        "cache_read_input_token_cost": 1.76e-07,
        "max_input_tokens": 262144,
        "max_output_tokens": 262144,
        "supports_vision": True,
    },
    "azure_ai/FW-Kimi-K2.7-Code": {
        "input_cost_per_token": 1.05e-06,
        "output_cost_per_token": 4.4e-06,
        "cache_read_input_token_cost": 2.1e-07,
        "max_input_tokens": 262144,
        "max_output_tokens": 262144,
        "supports_vision": True,
    },
    "azure_ai/FW-Kimi-K3": {
        "input_cost_per_token": 3.3e-06,
        "output_cost_per_token": 1.65e-05,
        "cache_read_input_token_cost": 3.3e-07,
        "max_input_tokens": 1048576,
        "max_output_tokens": 131072,
        "supports_vision": True,
    },
    "azure_ai/FW-Inkling": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 4.05e-06,
        "cache_read_input_token_cost": 1.7e-07,
        "max_input_tokens": 1048576,
        "max_output_tokens": 1048576,
    },
    "azure_ai/FW-DeepSeek-V3.2": {
        "input_cost_per_token": 6.2e-07,
        "output_cost_per_token": 1.85e-06,
        "cache_read_input_token_cost": 3.1e-07,
        "max_input_tokens": 163840,
        "max_output_tokens": 163840,
    },
    "azure_ai/FW-DeepSeek-V4-Pro": {
        "input_cost_per_token": 1.925e-06,
        "output_cost_per_token": 3.828e-06,
        "cache_read_input_token_cost": 1.65e-07,
        "max_input_tokens": 1000000,
        "max_output_tokens": 384000,
    },
    "azure_ai/FW-MiniMax-M3": {
        "input_cost_per_token": 3.3e-07,
        "output_cost_per_token": 1.32e-06,
        "cache_read_input_token_cost": 6.6e-08,
        "max_input_tokens": 512000,
        "max_output_tokens": 512000,
        "supports_vision": True,
    },
    "azure_ai/FW-MiniMax-M2.5": {
        "input_cost_per_token": 3.3e-07,
        "output_cost_per_token": 1.32e-06,
        "cache_read_input_token_cost": 3.3e-08,
        "max_input_tokens": 1000000,
        "max_output_tokens": 1000000,
    },
    "azure_ai/FW-Nemotron-3-Ultra-NVFP4": {
        "input_cost_per_token": 6e-07,
        "output_cost_per_token": 2.4e-06,
        "cache_read_input_token_cost": 1.19e-07,
        "max_input_tokens": 262144,
        "max_output_tokens": 262144,
    },
    "azure_ai/FW-GLM-5.2-Fast": {
        "input_cost_per_token": 2.1e-06,
        "output_cost_per_token": 6.6e-06,
        "cache_read_input_token_cost": 2.1e-07,
        "max_input_tokens": 1048576,
        "max_output_tokens": 131072,
    },
    "azure_ai/FW-GLM-5.2": {
        "input_cost_per_token": 1.54e-06,
        "output_cost_per_token": 4.84e-06,
        "cache_read_input_token_cost": 1.5e-07,
        "max_input_tokens": 1048576,
        "max_output_tokens": 131072,
    },
    "azure_ai/FW-GLM-5.1": {
        "input_cost_per_token": 1.54e-06,
        "output_cost_per_token": 4.84e-06,
        "cache_read_input_token_cost": 2.86e-07,
        "max_input_tokens": 202800,
        "max_output_tokens": 131072,
    },
    "azure_ai/FW-GLM-5": {
        "input_cost_per_token": 1.1e-06,
        "output_cost_per_token": 3.52e-06,
        "cache_read_input_token_cost": 2.2e-07,
        "max_input_tokens": 200000,
        "max_output_tokens": 128000,
    },
}


@pytest.fixture(scope="module")
def use_local_model_cost_map():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    import litellm
    from litellm.utils import _invalidate_model_cost_lowercase_map

    original_model_cost = litellm.model_cost
    litellm.model_cost = json.loads(
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        monkeypatch.undo()


@pytest.mark.parametrize("model_key,expected", list(FW_MODELS.items()))
def test_azure_ai_fw_model_info(use_local_model_cost_map, model_key, expected):
    model_info = use_local_model_cost_map.get_model_info(model=model_key)

    assert model_info["litellm_provider"] == "azure_ai"
    assert model_info["mode"] == "chat"
    assert model_info["input_cost_per_token"] == pytest.approx(expected["input_cost_per_token"])
    assert model_info["output_cost_per_token"] == pytest.approx(expected["output_cost_per_token"])
    assert model_info["cache_read_input_token_cost"] == pytest.approx(
        expected["cache_read_input_token_cost"]
    )
    assert model_info["max_input_tokens"] == expected["max_input_tokens"]
    assert model_info["max_output_tokens"] == expected["max_output_tokens"]
    assert model_info["max_tokens"] == expected["max_output_tokens"]
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_tool_choice"] is True
    assert model_info["supports_prompt_caching"] is True
    if expected.get("supports_vision"):
        assert model_info["supports_vision"] is True


@pytest.mark.parametrize(
    "model_name,expected_prompt,expected_completion",
    [
        ("FW-Kimi-K2.6", 1.045, 4.4),
        ("FW-DeepSeek-V4-Pro", 1.925, 3.828),
        ("FW-GLM-5.2", 1.54, 4.84),
        ("FW-Kimi-K3", 3.3, 16.5),
        ("FW-MiniMax-M2.5", 0.33, 1.32),
        ("FW-Inkling", 1.0, 4.05),
        ("FW-Nemotron-3-Ultra-NVFP4", 0.6, 2.4),
    ],
)
def test_azure_ai_fw_cost_per_token(
    use_local_model_cost_map, model_name, expected_prompt, expected_completion
):
    from litellm.llms.azure_ai.cost_calculator import cost_per_token
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    prompt_cost, completion_cost = cost_per_token(model=model_name, usage=usage)

    assert prompt_cost == pytest.approx(expected_prompt)
    assert completion_cost == pytest.approx(expected_completion)


def test_azure_ai_fw_kimi_k26_case_insensitive_lookup(use_local_model_cost_map):
    upper = use_local_model_cost_map.get_model_info(model="azure_ai/FW-Kimi-K2.6")
    lower = use_local_model_cost_map.get_model_info(model="azure_ai/fw-kimi-k2.6")

    assert upper["input_cost_per_token"] == pytest.approx(lower["input_cost_per_token"])
    assert upper["output_cost_per_token"] == pytest.approx(lower["output_cost_per_token"])
