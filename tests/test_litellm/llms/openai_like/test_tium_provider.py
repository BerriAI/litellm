"""
Tests for Tium provider configuration and integration.
"""

import json

import httpx
import pytest
import respx

import litellm
from litellm import completion

TIUM_URL = "https://api.tium.ai/v1/chat/completions"

CHAT_PAYLOAD = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "glm-5.3",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}

TOOL_PAYLOAD = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 1,
    "model": "glm-5.3",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class TestTiumProviderConfig:
    def test_tium_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "TIUM")
        assert LlmProviders.TIUM.value == "tium"
        assert "tium" in litellm.provider_list

    def test_tium_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("tium")

        tium = JSONProviderRegistry.get("tium")
        assert tium is not None
        assert tium.base_url == "https://api.tium.ai/v1"
        assert tium.api_key_env == "TIUM_API_KEY"
        assert tium.api_base_env == "TIUM_API_BASE"
        assert tium.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_tium_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "tium" in openai_compatible_providers

    def test_tium_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="tium/glm-5.3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "glm-5.3"
        assert provider == "tium"
        assert api_base == "https://api.tium.ai/v1"

    def test_tium_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="tium/glm-5.3",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "tium"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_tium_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="glm-5.3",
            custom_llm_provider=None,
            api_base="https://api.tium.ai/v1",
            api_key=None,
        )

        assert provider == "tium"
        assert api_base == "https://api.tium.ai/v1"

    def test_tium_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "tium-chat",
                    "litellm_params": {
                        "model": "tium/glm-5.3",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "tium-chat"


@pytest.mark.usefixtures("local_model_cost_map")
class TestTiumCompletion:
    @respx.mock
    def test_tium_completion_hits_tium_url_with_bearer_auth(self):
        route = respx.post(TIUM_URL).mock(
            return_value=httpx.Response(200, json=CHAT_PAYLOAD)
        )

        response = completion(
            model="tium/glm-5.3",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="sk-tium-test",
        )

        assert route.called
        request = route.calls.last.request
        assert str(request.url) == TIUM_URL
        assert request.headers["authorization"] == "Bearer sk-tium-test"

        body = json.loads(request.content)
        assert body["model"] == "glm-5.3"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

        assert response.choices[0].message.content == "ok"

    @respx.mock
    def test_tium_completion_honours_api_base_override(self):
        route = respx.post("https://custom.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=CHAT_PAYLOAD)
        )

        completion(
            model="tium/glm-5.3",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="sk-tium-test",
            api_base="https://custom.example.com/v1",
        )

        assert route.called

    @respx.mock
    def test_tium_completion_sends_tools_and_returns_tool_calls(self):
        route = respx.post(TIUM_URL).mock(
            return_value=httpx.Response(200, json=TOOL_PAYLOAD)
        )

        response = completion(
            model="tium/glm-5.3",
            messages=[{"role": "user", "content": "Weather in Berlin?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            api_key="sk-tium-test",
        )

        assert route.called
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [WEATHER_TOOL]
        assert body["tool_choice"] == "auto"

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None
        assert tool_calls[0].function.name == "get_weather"

    @respx.mock
    def test_tium_completion_maps_max_completion_tokens_on_the_wire(self):
        route = respx.post(TIUM_URL).mock(
            return_value=httpx.Response(200, json=CHAT_PAYLOAD)
        )

        completion(
            model="tium/glm-5.3",
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=256,
            api_key="sk-tium-test",
        )

        assert route.called
        body = json.loads(route.calls.last.request.content)
        assert body["max_tokens"] == 256
        assert "max_completion_tokens" not in body


class TestTiumModelMetadata:
    TIUM_MODELS = (
        "tium/glm-5.3-flash",
        "tium/deepseek-v4-flash",
        "tium/deepseek-v4-pro",
        "tium/glm-5.3",
        "tium/kimi-k3",
    )
    VISION_MODELS = ("tium/glm-5.3-flash", "tium/kimi-k3")
    NO_RESPONSE_SCHEMA = ("tium/deepseek-v4-flash", "tium/deepseek-v4-pro")

    @staticmethod
    def _load(path_parts):
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_tium_models_registered_with_correct_metadata(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        for model in self.TIUM_MODELS:
            info = model_cost.get(model)
            assert info is not None, f"{model} missing from model_prices_and_context_window.json"
            assert info["litellm_provider"] == "tium"
            assert info["mode"] == "chat"
            assert info["input_cost_per_token"] > 0
            assert info["output_cost_per_token"] > 0

            assert info["supports_function_calling"] is True
            assert info["supports_tool_choice"] is True
            assert info["supports_reasoning"] is True

            assert info["supports_response_schema"] is (model not in self.NO_RESPONSE_SCHEMA)
            assert info.get("supports_vision", False) is (model in self.VISION_MODELS)

            assert info["supports_prompt_caching"] is True
            assert 0 < info["cache_read_input_token_cost"] < info["input_cost_per_token"]

            assert info["max_input_tokens"] == 128000
            assert info["max_output_tokens"] == 32768
            assert info["max_tokens"] == info["max_output_tokens"]

    def test_tium_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        for model in self.TIUM_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"

    def test_tium_supported_endpoints_matrix(self):
        from pathlib import Path

        import litellm as _litellm

        backup_path = (
            Path(_litellm.__file__).parent / "provider_endpoints_support_backup.json"
        )
        matrix = json.loads(backup_path.read_text())

        assert "tium" in matrix["providers"]
        endpoints = matrix["providers"]["tium"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["responses"] is False
        assert endpoints["embeddings"] is False

    def test_tium_listed_in_add_model_form(self):
        entries = self._load(
            ("litellm", "proxy", "public_endpoints", "provider_create_fields.json")
        )
        tium = [e for e in entries if e["litellm_provider"] == "tium"]
        assert len(tium) == 1

        entry = tium[0]
        assert entry["provider"] == "TIUM"
        assert entry["provider_display_name"] == "Tium"
        assert entry["default_model_placeholder"].startswith("tium/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
