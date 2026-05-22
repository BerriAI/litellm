"""
Unit tests for Wafer AI configuration.

These tests validate the WaferConfig class which extends OpenAIGPTConfig.
Wafer is an OpenAI-compatible inference gateway, so the transformation is a
thin passthrough.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.wafer.chat.transformation import WaferConfig


class TestWaferConfig:
    """Test class for WaferConfig functionality"""

    def test_validate_environment(self):
        """validate_environment sets bearer auth + json content-type."""
        config = WaferConfig()
        api_key = "fake-wafer-key"

        result = config.validate_environment(
            headers={},
            model="wafer/GLM-5.1",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://api.wafer.ai/v1",
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_preserves_extra_headers(self):
        """validate_environment does not clobber user-supplied headers."""
        config = WaferConfig()
        result = config.validate_environment(
            headers={"X-Trace-Id": "abc"},
            model="wafer/GLM-5.1",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key="k",
            api_base=None,
        )
        assert result["X-Trace-Id"] == "abc"
        assert result["Authorization"] == "Bearer k"

    def test_missing_api_key(self):
        """validate_environment raises if no api_key is provided."""
        config = WaferConfig()

        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="wafer/GLM-5.1",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://api.wafer.ai/v1",
            )

        assert "Missing Wafer API Key" in str(excinfo.value)

    def test_inheritance(self):
        """WaferConfig inherits from OpenAIGPTConfig."""
        config = WaferConfig()
        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params")

    def test_get_supported_openai_params(self):
        """Wafer advertises the standard OpenAI chat-completions params."""
        config = WaferConfig()
        params = config.get_supported_openai_params(model="wafer/GLM-5.1")

        for required in [
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "response_format",
            "stop",
        ]:
            assert required in params, f"missing supported param: {required}"

    def test_map_openai_params_passthrough(self):
        """Standard params flow through untouched."""
        config = WaferConfig()
        non_default = {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 256,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather"},
                }
            ],
            "tool_choice": "auto",
        }
        optional_params: dict = {}
        result = config.map_openai_params(
            non_default_params=non_default,
            optional_params=optional_params,
            model="wafer/GLM-5.1",
            drop_params=False,
        )

        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 256
        assert result["tool_choice"] == "auto"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_map_openai_params_max_completion_tokens_aliased(self):
        """max_completion_tokens is mapped to max_tokens for the upstream API."""
        config = WaferConfig()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 128},
            optional_params={},
            model="wafer/Kimi-K2.6",
            drop_params=False,
        )
        assert result["max_tokens"] == 128
        assert "max_completion_tokens" not in result

    def test_map_openai_params_drops_none_values(self):
        """None-valued params are not forwarded."""
        config = WaferConfig()
        result = config.map_openai_params(
            non_default_params={"temperature": None, "top_p": 0.5},
            optional_params={},
            model="wafer/GLM-5.1",
            drop_params=False,
        )
        assert "temperature" not in result
        assert result["top_p"] == 0.5

    def test_get_openai_compatible_provider_info_defaults(self, monkeypatch):
        """Default base URL is api.wafer.ai/v1; api_key comes from env."""
        monkeypatch.delenv("WAFER_API_KEY", raising=False)
        monkeypatch.delenv("WAFER_API_BASE", raising=False)
        monkeypatch.setenv("WAFER_API_KEY", "env-key")

        config = WaferConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )
        assert api_base == "https://api.wafer.ai/v1"
        assert api_key == "env-key"

    def test_get_openai_compatible_provider_info_explicit_overrides_env(
        self, monkeypatch
    ):
        """Explicit api_base / api_key win over env vars."""
        monkeypatch.setenv("WAFER_API_KEY", "env-key")
        monkeypatch.setenv("WAFER_API_BASE", "https://env.wafer.example/v1")

        config = WaferConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base="https://explicit.wafer.example/v1",
            api_key="explicit-key",
        )
        assert api_base == "https://explicit.wafer.example/v1"
        assert api_key == "explicit-key"


class TestWaferProviderRegistration:
    """Tests that wafer is wired into the provider registry."""

    def test_llm_providers_enum(self):
        from litellm.types.utils import LlmProviders

        assert LlmProviders.WAFER.value == "wafer"

    def test_provider_in_openai_compatible_lists(self):
        from litellm import constants

        assert "wafer" in constants.openai_compatible_providers
        assert "wafer" in constants.LITELLM_CHAT_PROVIDERS
        assert "api.wafer.ai/v1" in constants.openai_compatible_endpoints

    def test_get_llm_provider_strips_wafer_prefix(self):
        """`wafer/<model>` resolves to provider=wafer with the model intact."""
        import litellm

        model, provider, _api_key, _api_base = litellm.get_llm_provider(
            model="wafer/GLM-5.1",
            api_key="test-key",
        )
        assert provider == "wafer"
        assert model == "GLM-5.1"

    def test_wafer_models_in_price_map(self):
        """Wafer models are registered in the cost map."""
        import litellm

        for model_id in [
            "wafer/GLM-5.1",
            "wafer/Kimi-K2.6",
            "wafer/Qwen3.5-397B-A17B",
        ]:
            assert (
                model_id in litellm.model_cost
            ), f"{model_id} missing from model_cost map"
            entry = litellm.model_cost[model_id]
            assert entry["litellm_provider"] == "wafer"
            assert entry["mode"] == "chat"
            assert entry["input_cost_per_token"] > 0
            assert entry["output_cost_per_token"] > 0
