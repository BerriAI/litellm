"""
Tests for the sference provider configuration and integration.
"""

import json
import os
from pathlib import Path

import pytest

import litellm


@pytest.fixture(scope="module", autouse=True)
def local_model_cost_map():
    """Force the bundled cost map so catalog-dependent assertions see this branch's
    sference entries, then restore the prior global state."""
    original_model_cost = litellm.model_cost
    previous_env = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        if previous_env is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = previous_env


@pytest.fixture(autouse=True)
def clear_sference_env(monkeypatch):
    """Keep tests hermetic when the host exports sference env vars."""
    monkeypatch.delenv("SFERENCE_API_BASE", raising=False)
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)


class TestSferenceProviderConfig:
    def test_sference_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "SFERENCE")
        assert LlmProviders.SFERENCE.value == "sference"
        assert "sference" in litellm.provider_list

    def test_sference_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("sference")

        sference = JSONProviderRegistry.get("sference")
        assert sference is not None
        assert sference.base_url == "https://api.sference.com/v1"
        assert sference.api_key_env == "SFERENCE_API_KEY"
        assert sference.api_base_env == "SFERENCE_API_BASE"

    def test_sference_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "sference" in openai_compatible_providers

    def test_sference_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="sference/Qwen/Qwen3.6-35B-A3B",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "Qwen/Qwen3.6-35B-A3B"
        assert provider == "sference"
        assert api_base == "https://api.sference.com/v1"
        assert api_key == "sk-test"

    def test_sference_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="sference/Qwen/Qwen3.6-35B-A3B",
            custom_llm_provider=None,
            api_base="https://custom.sference.example/v1",
            api_key="sk-test",
        )

        assert provider == "sference"
        assert api_base == "https://custom.sference.example/v1"
        assert api_key == "sk-test"

    def test_sference_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="Qwen/Qwen3.6-35B-A3B",
            custom_llm_provider=None,
            api_base="https://api.sference.com/v1",
            api_key=None,
        )
        assert provider == "sference"
        assert api_base == "https://api.sference.com/v1"

    def test_explicit_api_key_wins_over_env(self, monkeypatch):
        """In the api_base autodetection branch an explicit per-request key must
        beat the host-exported SFERENCE_API_KEY, otherwise requests bill the wrong account."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("SFERENCE_API_KEY", "sk-env-key")
        model, provider, api_key, api_base = get_llm_provider(
            model="Qwen/Qwen3.6-35B-A3B",
            custom_llm_provider=None,
            api_base="https://api.sference.com/v1",
            api_key="sk-explicit",
        )
        assert provider == "sference"
        assert api_key == "sk-explicit"

    def test_sference_api_key_resolved_from_env(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("SFERENCE_API_KEY", "sk-env-key")
        model, provider, api_key, api_base = get_llm_provider(
            model="sference/Qwen/Qwen3.6-35B-A3B",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert provider == "sference"
        assert api_key == "sk-env-key"

    def test_sference_complete_url(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("sference")
        assert provider is not None
        cfg = create_config_class(provider)()

        url = cfg.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="Qwen/Qwen3.6-35B-A3B",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.sference.com/v1/chat/completions"


class TestSferenceSupportedParams:
    def test_reasoning_model_supports_reasoning_effort(self):
        params = litellm.get_supported_openai_params(model="Qwen/Qwen3.6-35B-A3B", custom_llm_provider="sference")
        assert params is not None
        assert "reasoning_effort" in params

    def test_non_reasoning_model_excludes_reasoning_effort(self):
        """Qwen3-VL is listed with thinking unsupported in the sference catalog,
        so it must not advertise reasoning_effort."""
        params = litellm.get_supported_openai_params(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct", custom_llm_provider="sference"
        )
        assert params is not None
        assert "reasoning_effort" not in params

    def test_tool_params_supported(self):
        params = litellm.get_supported_openai_params(model="Qwen/Qwen3.6-35B-A3B", custom_llm_provider="sference")
        assert params is not None
        assert "tools" in params
        assert "tool_choice" in params

    def test_service_tier_and_prompt_cache_key_supported(self):
        params = litellm.get_supported_openai_params(model="Qwen/Qwen3.6-35B-A3B", custom_llm_provider="sference")
        assert params is not None
        assert "service_tier" in params
        assert "prompt_cache_key" in params

    def test_reasoning_effort_and_service_tier_mapped_through(self):
        cfg = litellm.ProviderConfigManager.get_provider_chat_config(
            model="Qwen/Qwen3.6-35B-A3B", provider=litellm.LlmProviders.SFERENCE
        )
        assert cfg is not None
        mapped = cfg.map_openai_params(
            non_default_params={"reasoning_effort": "high", "service_tier": "flex"},
            optional_params={},
            model="Qwen/Qwen3.6-35B-A3B",
            drop_params=False,
        )
        assert mapped["reasoning_effort"] == "high"
        assert mapped["service_tier"] == "flex"


class TestSferenceCompletion:
    @pytest.mark.respx()
    def test_completion_request_routing(self, respx_mock, monkeypatch):
        """A completion call must hit the sference chat completions endpoint with the
        catalog model id (provider prefix stripped), bearer auth, and pass-through params."""
        monkeypatch.setenv("SFERENCE_API_KEY", "sk-env-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

        respx_mock.post("https://api.sference.com/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1786206392,
                "model": "Qwen/Qwen3.6-35B-A3B",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
            status_code=200,
        )

        response = litellm.completion(
            model="sference/Qwen/Qwen3.6-35B-A3B",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
            service_tier="flex",
        )

        assert len(respx_mock.calls) == 1
        request = respx_mock.calls[0].request
        assert request.headers["Authorization"] == "Bearer sk-env-key"

        body = json.loads(request.content)
        assert body["model"] == "Qwen/Qwen3.6-35B-A3B"
        assert body["reasoning_effort"] == "high"
        assert body["service_tier"] == "flex"
        assert response.choices[0].message.content == "Hello!"


class TestSferenceValidateEnvironment:
    def test_missing_api_key_reported(self):
        result = litellm.validate_environment(model="sference/Qwen/Qwen3.6-35B-A3B")
        assert result["keys_in_environment"] is False
        assert "SFERENCE_API_KEY" in result["missing_keys"]

    def test_present_api_key_accepted(self, monkeypatch):
        monkeypatch.setenv("SFERENCE_API_KEY", "sk-env-key")
        result = litellm.validate_environment(model="sference/Qwen/Qwen3.6-35B-A3B")
        assert result["keys_in_environment"] is True
        assert result["missing_keys"] == []


class TestSferenceModelsByProvider:
    def test_models_by_provider_contains_catalog(self):
        """Wildcard sference/* proxy deployments expand via models_by_provider."""
        litellm.add_known_models(litellm.model_cost)

        sference_models = litellm.models_by_provider.get("sference")
        assert sference_models is not None
        assert len(sference_models) == 6
        assert "sference/moonshotai/Kimi-K3" in sference_models


class TestSferenceRuntimeEndpointsMatrix:
    def test_sference_in_runtime_supported_endpoints_matrix(self):
        """The proxy serves litellm/provider_endpoints_support_backup.json at
        GET /public/supported_endpoints, so sference must be listed there too."""
        matrix = json.loads((Path(litellm.__file__).parent / "provider_endpoints_support_backup.json").read_text())

        assert "sference" in matrix["providers"]
        endpoints = matrix["providers"]["sference"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True


class TestSferenceUncatalogedModels:
    def test_uncataloged_model_keeps_tool_params(self):
        """BYOM/custom sference models absent from the cost map fall back to the
        provider-level default capability instead of losing tool calling."""
        params = litellm.get_supported_openai_params(model="my-org/my-fine-tune", custom_llm_provider="sference")
        assert params is not None
        assert "tools" in params
        assert "tool_choice" in params

    def test_uncataloged_model_has_no_reasoning_effort(self):
        params = litellm.get_supported_openai_params(model="my-org/my-fine-tune", custom_llm_provider="sference")
        assert params is not None
        assert "reasoning_effort" not in params


class TestSferenceAnthropicMessages:
    def test_messages_config_resolves(self):
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        cfg = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
            model="Qwen/Qwen3.6-35B-A3B", provider=litellm.LlmProviders.SFERENCE
        )
        assert isinstance(cfg, JSONProviderAnthropicMessagesConfig)

    def test_messages_complete_url(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        provider = JSONProviderRegistry.get("sference")
        assert provider is not None
        cfg = JSONProviderAnthropicMessagesConfig(provider)

        url = cfg.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="Qwen/Qwen3.6-35B-A3B",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.sference.com/v1/messages"

    def test_messages_api_key_resolved_from_env(self, monkeypatch):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        monkeypatch.setenv("SFERENCE_API_KEY", "sk-env-key")
        provider = JSONProviderRegistry.get("sference")
        assert provider is not None
        cfg = JSONProviderAnthropicMessagesConfig(provider)

        headers, _ = cfg.validate_anthropic_messages_environment(
            headers={},
            model="Qwen/Qwen3.6-35B-A3B",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )
        assert headers["authorization"] == "Bearer sk-env-key"
        assert headers["anthropic-version"] == "2023-06-01"


class TestSferenceModelInfoAndCost:
    def test_model_info_from_catalog(self):
        """get_model_info resolves the catalog entry; the values themselves are
        pinned in test_sference_model_metadata.py."""
        info = litellm.get_model_info("sference/deepseek-ai/DeepSeek-V4-Flash")

        assert info["litellm_provider"] == "sference"
        assert info["max_input_tokens"] == info["max_output_tokens"]
        assert info["supports_reasoning"] is True
        assert info["supports_function_calling"] is True
        assert info["supports_prompt_caching"] is True
        assert info["supports_vision"] is False

    def test_get_max_tokens_from_catalog(self):
        """get_max_tokens must not raise for cataloged sference models; the pinned
        value is the published context ceiling."""
        assert litellm.get_max_tokens("sference/moonshotai/Kimi-K3") == 1048576
        assert litellm.get_max_tokens("sference/Qwen/Qwen3.6-35B-A3B") == 262144

    def test_vision_model_info_from_catalog(self):
        info = litellm.get_model_info("sference/Qwen/Qwen3-VL-30B-A3B-Instruct")

        assert info["litellm_provider"] == "sference"
        assert info["supports_vision"] is True
        assert info["supports_reasoning"] is False

    def test_cost_calculation_with_cached_tokens(self):
        """Cost math must combine uncached input at input price, cached input at
        cache-read price, and output at output price, all from the catalog entry."""
        from litellm import completion_cost
        from litellm.types.utils import ModelResponse, PromptTokensDetails, Usage

        entry = litellm.model_cost["sference/deepseek-ai/DeepSeek-V4-Flash"]
        response = ModelResponse(
            model="deepseek-ai/DeepSeek-V4-Flash",
            usage=Usage(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=200),
            ),
        )
        cost = completion_cost(
            completion_response=response,
            model="sference/deepseek-ai/DeepSeek-V4-Flash",
            custom_llm_provider="sference",
        )
        expected = (
            800 * entry["input_cost_per_token"]
            + 200 * entry["cache_read_input_token_cost"]
            + 500 * entry["output_cost_per_token"]
        )
        assert abs(cost - expected) < 1e-12
