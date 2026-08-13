"""
Unit tests for Azure AI Foundry / Fireworks Kimi request shaping.

Mirrors the Moonshot Kimi multi-turn tool + fixed-sampling contract for
azure_ai/FW-Kimi-* and azure_ai/kimi-* deployments.
"""

import json
from importlib.resources import files

import pytest

from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig


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


class TestAzureAIKimiShaping:
    def test_detects_fw_kimi_k3(self):
        config = AzureAIStudioConfig()
        assert config._is_kimi_reasoning_model("FW-Kimi-K3") is True
        assert config._is_kimi_k3_model("FW-Kimi-K3") is True
        assert config._is_kimi_k3_model("azure_ai/FW-Kimi-K3") is True

    def test_detects_native_kimi_k26(self):
        config = AzureAIStudioConfig()
        assert config._is_kimi_reasoning_model("kimi-k2.6") is True
        assert config._is_kimi_k3_model("kimi-k2.6") is False

    def test_map_openai_params_drops_k3_fixed_sampling(self):
        config = AzureAIStudioConfig()
        result = config.map_openai_params(
            non_default_params={
                "temperature": 0.2,
                "top_p": 0.5,
                "n": 2,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1,
                "max_tokens": 64,
            },
            optional_params={},
            model="FW-Kimi-K3",
            drop_params=True,
        )
        assert "temperature" not in result
        assert "top_p" not in result
        assert "n" not in result
        assert "presence_penalty" not in result
        assert "frequency_penalty" not in result
        assert result.get("max_tokens") == 64

    def test_map_openai_params_drops_invalid_reasoning_effort(self):
        config = AzureAIStudioConfig()
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "medium", "max_tokens": 16},
            optional_params={},
            model="FW-Kimi-K3",
            drop_params=True,
        )
        assert "reasoning_effort" not in result
        assert result.get("max_tokens") == 16

    def test_map_openai_params_keeps_valid_reasoning_effort(self):
        config = AzureAIStudioConfig()
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "high", "max_tokens": 16},
            optional_params={},
            model="FW-Kimi-K3",
            drop_params=True,
        )
        assert result.get("reasoning_effort") == "high"

    def test_map_openai_params_k26_only_drops_temperature(self):
        config = AzureAIStudioConfig()
        result = config.map_openai_params(
            non_default_params={"temperature": 0.2, "top_p": 0.5, "max_tokens": 16},
            optional_params={},
            model="kimi-k2.6",
            drop_params=True,
        )
        assert "temperature" not in result
        assert result.get("top_p") == 0.5

    def test_fill_reasoning_content_injects_placeholder(self):
        config = AzureAIStudioConfig()
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        result = config.fill_reasoning_content(messages)
        assert result[1].get("reasoning_content") == " "

    def test_transform_request_fills_reasoning_for_fw_kimi(self):
        config = AzureAIStudioConfig()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            }
        ]
        result = config.transform_request(
            model="FW-Kimi-K3",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert result["messages"][0].get("reasoning_content") == " "

    def test_fw_kimi_k3_cost_map_entry(self, use_local_model_cost_map):
        model_info = use_local_model_cost_map.get_model_info(model="azure_ai/FW-Kimi-K3")
        assert model_info["supports_reasoning"] is True
        assert model_info["max_input_tokens"] == 1048576
        assert model_info["input_cost_per_token"] == pytest.approx(0.0000033)
        assert model_info["output_cost_per_token"] == pytest.approx(0.0000165)
