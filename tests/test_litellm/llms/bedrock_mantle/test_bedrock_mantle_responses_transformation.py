"""
Unit tests for Amazon Bedrock Mantle Responses API configuration.

Mantle's gpt-5.5 / gpt-5.4 are served ONLY on the non-standard
`/openai/v1/responses` path. These tests lock the URL construction and
Bearer auth that make that routing work.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

import litellm
from litellm.llms.bedrock_mantle.responses.transformation import (
    BedrockMantleResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class TestBedrockMantleResponsesURL:
    def test_url_uses_region_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-east-2")
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_normalizes_v1_suffix(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        assert "/v1/openai/v1/responses" not in url
        url_trailing = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1/",
            litellm_params={},
        )
        assert (
            url_trailing
            == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        )

    def test_url_does_not_double_openai_v1(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_full_endpoint_base_not_doubled(self, monkeypatch):
        # AWS model card tells users to set OPENAI_BASE_URL to the full endpoint.
        # If copied into api_base, it must not be doubled.
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        assert url.count("/responses") == 1

    def test_url_region_fallback_to_aws_region(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses"

    def test_url_region_default_us_east_1(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"


class TestBedrockMantleResponsesAuth:
    def test_config_api_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-5.5",
            litellm_params=GenericLiteLLMParams(api_key="config-key"),
        )
        assert headers["Authorization"] == "Bearer config-key"

    def test_env_key_fallback(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_bedrock_bearer_token_fallback(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-key")
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer bearer-key"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        with pytest.raises(ValueError, match="Bedrock Mantle API key"):
            cfg.validate_environment(
                headers={},
                model="openai.gpt-5.5",
                litellm_params=GenericLiteLLMParams(),
            )

    def test_custom_llm_provider(self):
        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.custom_llm_provider == LlmProviders.BEDROCK_MANTLE

    def test_native_websocket_disabled(self):
        # Mantle Responses has no realtime/websocket transport, so the config
        # must opt out; otherwise realtime routing would try a socket Mantle
        # does not serve.
        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.supports_native_websocket() is False

    def test_file_search_routes_to_emulation(self):
        # Mantle cannot reach OpenAI's vector stores, so a native file_search
        # tool forwarded as-is gets a 400. The config must opt out of native
        # file_search so LiteLLM's emulation handles it instead of forwarding.
        from litellm.responses.file_search.emulated_handler import (
            should_use_emulated_file_search,
        )

        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.supports_native_file_search() is False
        assert (
            should_use_emulated_file_search(
                tools=[{"type": "file_search", "vector_store_ids": ["vs_1"]}],
                provider_config=cfg,
            )
            is True
        )


class TestBedrockMantleResponsesRegistry:
    def test_registry_returns_config_for_gpt_5_5(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-5.5",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)

    def test_registry_returns_config_for_gpt_5_4_enum(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.BEDROCK_MANTLE,
            model="openai.gpt-5.4",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)

    def test_registry_returns_none_for_gpt_oss(self):
        # Regression guard: gpt-oss must NOT get the native Responses config; it
        # keeps the chat-completions emulation path (responses/main.py ~line 1109).
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-120b",
        )
        assert cfg is None

    def test_registry_returns_none_for_gpt_oss_safeguard(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-safeguard-20b",
        )
        assert cfg is None

    def test_registry_returns_config_for_future_frontier_model(self):
        # Forward-compatibility: an unseen OpenAI gpt frontier model (e.g. gpt-6) must
        # get the native Responses config without a code change. The gate allow-lists
        # the openai.gpt- family (minus gpt-oss), so gpt-6 matches automatically.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-6",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)

    @pytest.mark.parametrize(
        "model",
        [
            "nvidia.nemotron-nano-9b-v2",
            "mistral.ministral-3-3b-instruct",
            "google.gemma-3-27b-it",
            "zai.glm-4.6",
        ],
    )
    def test_registry_returns_none_for_non_openai_models(self, model):
        # Regression for the chat-only families on Mantle. These models 400 on
        # /openai/v1/responses and are served on /v1/chat/completions, so the
        # registry must NOT hand them the Responses config; they fall through to
        # None and keep the chat-completions emulation.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=model,
        )
        assert cfg is None

    def test_registry_returns_none_when_model_is_none(self):
        # By-id operations (delete/get/cancel) call with model=None; keep returning
        # None so those paths are unchanged.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=None,
        )
        assert cfg is None


@pytest.fixture
def local_cost_map(monkeypatch):
    """Force the bundled backup cost map and re-derive the provider model sets.

    ``litellm.model_cost`` is populated once at import time (here, from the
    network-fetched ``main`` copy, which lags this branch). ``add_known_models``
    only re-buckets whatever is already in ``model_cost``, so the cost map must
    first be reloaded from the local backup before the new keys appear.
    """
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    litellm.add_known_models()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


class TestBedrockMantleResponsesPricing:
    def test_gpt_5_5_pricing_and_mode(self, local_cost_map):
        info = litellm.get_model_info("bedrock_mantle/openai.gpt-5.5")
        assert info["mode"] == "responses"
        assert info["input_cost_per_token"] == pytest.approx(5.5e-06)
        assert info["output_cost_per_token"] == pytest.approx(3.3e-05)
        assert info["cache_read_input_token_cost"] == pytest.approx(5.5e-07)
        assert info["max_input_tokens"] == 272000

    def test_gpt_5_4_pricing_and_mode(self, local_cost_map):
        info = litellm.get_model_info("bedrock_mantle/openai.gpt-5.4")
        assert info["mode"] == "responses"
        assert info["input_cost_per_token"] == pytest.approx(2.75e-06)
        assert info["output_cost_per_token"] == pytest.approx(1.65e-05)
        assert info["cache_read_input_token_cost"] == pytest.approx(2.75e-07)
        assert info["max_input_tokens"] == 272000

    def test_models_registered(self, local_cost_map):
        assert "bedrock_mantle/openai.gpt-5.5" in litellm.bedrock_mantle_models
        assert "bedrock_mantle/openai.gpt-5.4" in litellm.bedrock_mantle_models
