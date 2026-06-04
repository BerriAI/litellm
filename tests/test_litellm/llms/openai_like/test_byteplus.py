"""
Tests for the BytePlus provider configuration and integration.

BytePlus (ByteDance Ark, https://ark.ap-southeast.bytepluses.com/api/v3) is an
OpenAI-compatible provider wired through the JSON provider registry
(litellm/llms/openai_like/providers.json).
"""

import os
import sys

import pytest

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestBytePlusProviderConfig:
    """Configuration / registration tests (no network)."""

    def test_byteplus_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "BYTEPLUS")
        assert LlmProviders.BYTEPLUS.value == "byteplus"
        assert "byteplus" in litellm.provider_list
        assert "byteplus" in litellm.openai_compatible_providers

    def test_byteplus_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("byteplus")

        cfg = JSONProviderRegistry.get("byteplus")
        assert cfg is not None
        assert cfg.base_url == "https://ark.ap-southeast.bytepluses.com/api/v3"
        assert cfg.api_key_env == "BYTEPLUS_API_KEY"
        assert cfg.api_base_env == "BYTEPLUS_API_BASE"
        assert cfg.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_byteplus_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="byteplus/deepseek-v4-flash-260425",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "deepseek-v4-flash-260425"
        assert provider == "byteplus"
        assert api_base == "https://ark.ap-southeast.bytepluses.com/api/v3"

    def test_byteplus_api_base_env_override(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("BYTEPLUS_API_BASE", "https://ark.example.com/api/v3")
        monkeypatch.setenv("BYTEPLUS_API_KEY", "env-key")

        model, provider, api_key, api_base = get_llm_provider(
            model="byteplus/deepseek-v4-flash-260425",
        )

        assert api_base == "https://ark.example.com/api/v3"
        assert api_key == "env-key"

    def test_byteplus_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "deepseek-v4-flash",
                    "litellm_params": {
                        "model": "byteplus/deepseek-v4-flash-260425",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "deepseek-v4-flash"


class TestBytePlusCostMap:
    """The seed model must be registered in the local cost map."""

    def test_model_registered_in_cost_map(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")

        entry = litellm.model_cost["byteplus/deepseek-v4-flash-260425"]
        assert entry["litellm_provider"] == "byteplus"
        assert entry["mode"] == "chat"
        assert entry["supports_function_calling"] is True
        assert entry["supports_reasoning"] is True


class TestBytePlusRequestBuilding:
    """Config-layer tests for URL building and param mapping."""

    def _config(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        return create_config_class(JSONProviderRegistry.get("byteplus"))()

    def test_complete_url(self):
        url = self._config().get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="deepseek-v4-flash-260425",
            optional_params={},
            litellm_params={},
        )
        assert url == (
            "https://ark.ap-southeast.bytepluses.com/api/v3/chat/completions"
        )

    def test_max_completion_tokens_mapped_to_max_tokens(self):
        mapped = self._config().map_openai_params(
            non_default_params={"max_completion_tokens": 50},
            optional_params={},
            model="deepseek-v4-flash-260425",
            drop_params=False,
        )
        # Ark expects max_tokens, not max_completion_tokens
        assert mapped["max_tokens"] == 50
        assert "max_completion_tokens" not in mapped


class TestBytePlusCompletion:
    """Optional live smoke test."""

    @pytest.mark.skipif(
        not os.environ.get("BYTEPLUS_API_KEY"),
        reason="BYTEPLUS_API_KEY not set",
    )
    def test_completion_live(self):
        resp = litellm.completion(
            model="byteplus/deepseek-v4-flash-260425",
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=50,
        )
        assert resp.choices[0].message.content is not None
        assert len(resp.choices[0].message.content) > 0
