"""
Unit tests for Xiaomi MiMo configuration.

These tests validate the XiaomiMiMoChatConfig class which extends OpenAIGPTConfig.
Xiaomi MiMo is an OpenAI-compatible provider that authenticates with the standard
OpenAI-style `Authorization: Bearer` header.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import pytest

import litellm
from litellm import completion
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.llms.xiaomi_mimo.chat.transformation import XiaomiMiMoChatConfig

DEFAULT_API_BASE = "https://api.xiaomimimo.com/v1"


class TestXiaomiMiMoConfig:
    """Test class for Xiaomi MiMo functionality"""

    def test_validate_environment_sets_bearer_header(self):
        """MiMo is OpenAI-compatible: auth via `Authorization: Bearer` (base config)."""
        config = XiaomiMiMoChatConfig()
        api_key = "sk-fake-mimo-key"

        result = config.validate_environment(
            headers={},
            model="xiaomi_mimo/mimo-v2.5-pro",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_preserves_existing_content_type(self):
        config = XiaomiMiMoChatConfig()
        result = config.validate_environment(
            headers={"Content-Type": "application/json; charset=utf-8"},
            model="xiaomi_mimo/mimo-v2.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-x",
            api_base=None,
        )
        assert result["Content-Type"] == "application/json; charset=utf-8"

    def test_get_openai_compatible_provider_info_defaults(self):
        """No api_base / api_key -> default base URL, key resolved from args."""
        config = XiaomiMiMoChatConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(api_base=None, api_key="sk-explicit")
        assert api_base == DEFAULT_API_BASE
        assert api_key == "sk-explicit"

    def test_get_openai_compatible_provider_info_reads_env(self, monkeypatch):
        monkeypatch.setenv("XIAOMI_MIMO_API_BASE", "https://custom.mimo.example/v1")
        monkeypatch.setenv("XIAOMI_MIMO_API_KEY", "sk-env-key")
        config = XiaomiMiMoChatConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(api_base=None, api_key=None)
        assert api_base == "https://custom.mimo.example/v1"
        assert api_key == "sk-env-key"

    def test_get_openai_compatible_provider_info_explicit_over_env(self, monkeypatch):
        monkeypatch.setenv("XIAOMI_MIMO_API_BASE", "https://custom.mimo.example/v1")
        config = XiaomiMiMoChatConfig()
        api_base, _ = config._get_openai_compatible_provider_info(
            api_base="https://override.example/v1", api_key="sk-x"
        )
        assert api_base == "https://override.example/v1"

    def test_get_complete_url_default(self):
        config = XiaomiMiMoChatConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="mimo-v2.5-pro",
            optional_params={},
            litellm_params={},
        )
        assert url == f"{DEFAULT_API_BASE}/chat/completions"

    def test_get_complete_url_appends_path(self):
        config = XiaomiMiMoChatConfig()
        url = config.get_complete_url(
            api_base="https://api.xiaomimimo.com/v1",
            api_key=None,
            model="mimo-v2.5-pro",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.xiaomimimo.com/v1/chat/completions"

    def test_get_complete_url_no_double_suffix(self):
        config = XiaomiMiMoChatConfig()
        already = "https://api.xiaomimimo.com/v1/chat/completions"
        url = config.get_complete_url(
            api_base=already,
            api_key=None,
            model="mimo-v2.5-pro",
            optional_params={},
            litellm_params={},
        )
        assert url == already

    def test_supported_params_include_openai_core(self):
        config = XiaomiMiMoChatConfig()
        params = config.get_supported_openai_params(model="xiaomi_mimo/mimo-v2.5-pro")
        for expected in ("tools", "tool_choice", "max_tokens", "max_completion_tokens", "stream"):
            assert expected in params

    def test_map_openai_params_translates_max_completion_tokens(self):
        """MiMo expects `max_tokens`; the OpenAI `max_completion_tokens` alias is translated."""
        config = XiaomiMiMoChatConfig()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 100},
            optional_params={},
            model="xiaomi_mimo/mimo-v2.5-pro",
            drop_params=False,
        )
        assert result.get("max_tokens") == 100
        assert "max_completion_tokens" not in result


class TestXiaomiMiMoReasoningParams:
    """MiMo's reasoning contract (live-probed): `thinking={"type": ...}` is honored natively;
    `reasoning_effort` accepts exactly low|medium|high ('none' -> 400, 'minimal' -> 500),
    so unsupported values must be translated rather than forwarded."""

    def _map(self, non_default_params, optional_params=None):
        config = XiaomiMiMoChatConfig()
        return config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params if optional_params is not None else {},
            model="xiaomi_mimo/mimo-v2.5-pro",
            drop_params=False,
        )

    def test_supported_params_include_reasoning(self):
        params = XiaomiMiMoChatConfig().get_supported_openai_params(model="xiaomi_mimo/mimo-v2.5-pro")
        assert "reasoning_effort" in params
        assert "thinking" in params

    def test_thinking_disabled_forwarded_via_extra_body(self):
        result = self._map({"thinking": {"type": "disabled"}})
        assert result["extra_body"]["thinking"] == {"type": "disabled"}
        assert "thinking" not in result  # not top-level: extra_body is the SDK transport

    def test_thinking_enabled_forwarded_verbatim(self):
        result = self._map({"thinking": {"type": "enabled"}})
        assert result["extra_body"]["thinking"] == {"type": "enabled"}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_native_reasoning_effort_forwarded(self, effort):
        result = self._map({"reasoning_effort": effort})
        assert result["reasoning_effort"] == effort
        assert "extra_body" not in result

    def test_reasoning_effort_none_becomes_thinking_disabled(self):
        """MiMo 400s on reasoning_effort='none'; the native disable is thinking={"type": "disabled"}."""
        result = self._map({"reasoning_effort": "none"})
        assert "reasoning_effort" not in result
        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_reasoning_effort_minimal_clamped_to_low(self):
        """MiMo 500s on 'minimal'; clamp to the closest accepted literal."""
        result = self._map({"reasoning_effort": "minimal"})
        assert result["reasoning_effort"] == "low"

    def test_reasoning_effort_out_of_set_clamped_to_high(self):
        """Out-of-set values (e.g. 'xhigh') hard-fail at MiMo; clamp instead of erroring."""
        result = self._map({"reasoning_effort": "xhigh"})
        assert result["reasoning_effort"] == "high"

    def test_thinking_wins_over_reasoning_effort(self):
        """Coexistence is undocumented at MiMo; forward only `thinking` when both are sent."""
        result = self._map({"thinking": {"type": "enabled"}, "reasoning_effort": "none"})
        assert result["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in result

    def test_extra_body_merge_not_clobber(self):
        """Caller-supplied extra_body content (e.g. chat_template_kwargs) must survive."""
        result = self._map(
            {"thinking": {"type": "disabled"}},
            optional_params={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        )
        assert result["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_caller_explicit_thinking_in_extra_body_wins(self):
        result = self._map(
            {"reasoning_effort": "none"},
            optional_params={"extra_body": {"thinking": {"type": "enabled"}}},
        )
        assert result["extra_body"]["thinking"] == {"type": "enabled"}

    def test_no_signal_adds_nothing(self):
        result = self._map({"max_tokens": 64})
        assert "extra_body" not in result
        assert "reasoning_effort" not in result

    def test_non_dict_thinking_ignored_without_crashing(self):
        """Only the dict shape is documented; other shapes are ignored (with a warning log)."""
        result = self._map({"thinking": "off"})
        assert "extra_body" not in result
        assert "thinking" not in result

    def test_non_dict_thinking_does_not_swallow_reasoning_effort(self):
        """An ignored non-dict `thinking` must not suppress a valid reasoning_effort."""
        result = self._map({"thinking": "off", "reasoning_effort": "low"})
        assert result["reasoning_effort"] == "low"


class TestXiaomiMiMoProviderRegistration:
    """xiaomi_mimo is a first-class provider (migrated off the openai_like JSON registry)."""

    def test_xiaomi_mimo_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "XIAOMI_MIMO")
        assert LlmProviders.XIAOMI_MIMO.value == "xiaomi_mimo"
        assert "xiaomi_mimo" in litellm.provider_list

    def test_xiaomi_mimo_not_in_json_registry(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert not JSONProviderRegistry.exists("xiaomi_mimo")

    def test_xiaomi_mimo_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "mimo-v2.5-pro",
                    "litellm_params": {
                        "model": "xiaomi_mimo/mimo-v2.5-pro",
                        "api_key": "test-key",
                    },
                }
            ]
        )
        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "mimo-v2.5-pro"


class TestXiaomiMiMoRouting:
    """get_llm_provider routing for Xiaomi MiMo."""

    def test_get_llm_provider_prefix(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, _ = get_llm_provider("xiaomi_mimo/mimo-v2.5-pro")
        assert model == "mimo-v2.5-pro"
        assert provider == "xiaomi_mimo"

    def test_get_llm_provider_by_api_base(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider("mimo-v2.5-pro", api_base="https://api.xiaomimimo.com/v1")
        assert model == "mimo-v2.5-pro"
        assert provider == "xiaomi_mimo"
        assert api_base == "https://api.xiaomimimo.com/v1"

    def test_provider_config_manager_returns_xiaomi_config(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="mimo-v2.5-pro", provider=LlmProviders.XIAOMI_MIMO
        )
        assert isinstance(config, XiaomiMiMoChatConfig)

    def test_validate_environment_util(self, monkeypatch):
        from litellm import validate_environment

        monkeypatch.delenv("XIAOMI_MIMO_API_KEY", raising=False)
        result = validate_environment(model="xiaomi_mimo/mimo-v2.5-pro")
        assert "XIAOMI_MIMO_API_KEY" in result["missing_keys"]

        monkeypatch.setenv("XIAOMI_MIMO_API_KEY", "sk-env")
        result = validate_environment(model="xiaomi_mimo/mimo-v2.5-pro")
        assert result["keys_in_environment"] is True


class TestXiaomiMiMoModelCostMap:
    """Assert the model_prices_and_context_window entries for Xiaomi MiMo.

    Loaded directly from the bundled local map so the test does not depend on the
    remote model-cost fetch (which may not yet carry the not-yet-merged entries).
    """

    @pytest.fixture(autouse=True)
    def model_cost_map(self):
        return GetModelCostMap.load_local_model_cost_map()

    def test_models_registered(self, model_cost_map):
        assert "xiaomi_mimo/mimo-v2.5-pro" in model_cost_map
        assert "xiaomi_mimo/mimo-v2.5" in model_cost_map

    def test_pro_metadata(self, model_cost_map):
        entry = model_cost_map["xiaomi_mimo/mimo-v2.5-pro"]
        assert entry["litellm_provider"] == "xiaomi_mimo"
        assert entry["mode"] == "chat"
        assert entry["max_input_tokens"] == 1048576
        assert entry["max_output_tokens"] == 131072
        assert entry["max_tokens"] == entry["max_output_tokens"]
        assert entry["input_cost_per_token"] == 4.35e-07
        assert entry["output_cost_per_token"] == 8.7e-07
        assert entry["cache_read_input_token_cost"] == 3.6e-09
        assert entry["supports_function_calling"] is True
        assert entry["supports_tool_choice"] is True
        assert entry["supports_reasoning"] is True

    def test_v25_is_multimodal(self, model_cost_map):
        entry = model_cost_map["xiaomi_mimo/mimo-v2.5"]
        assert entry["supports_vision"] is True
        assert entry["supports_audio_input"] is True


class TestXiaomiMiMoCompletionMock:
    @pytest.mark.respx()
    def test_completion_mock_sends_bearer_header(self, respx_mock):
        """End-to-end mocked completion; assert the request carried Bearer auth."""
        litellm.disable_aiohttp_transport = True

        api_key = "sk-fake-mimo-key"
        api_base = "https://api.xiaomimimo.com/v1"
        model = "xiaomi_mimo/mimo-v2.5-pro"
        model_name = "mimo-v2.5-pro"

        route = respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hey from LiteLLM!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        response = completion(
            model=model,
            messages=[{"role": "user", "content": "say hey"}],
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert response.choices[0].message.content == "Hey from LiteLLM!"

        sent = route.calls.last.request
        assert sent.headers["Authorization"] == f"Bearer {api_key}"
