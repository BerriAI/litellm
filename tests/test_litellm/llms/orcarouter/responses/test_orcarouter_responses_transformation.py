"""Tests for OrcaRouter Responses API configuration."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.orcarouter.responses.transformation import (
    OrcaRouterResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class TestOrcaRouterResponsesAPIConfig:
    def test_custom_llm_provider(self):
        config = OrcaRouterResponsesAPIConfig()
        assert config.custom_llm_provider == LlmProviders.ORCAROUTER

    def test_get_complete_url_default(self):
        config = OrcaRouterResponsesAPIConfig()
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://api.orcarouter.ai/v1/responses"

    def test_get_complete_url_custom_base(self):
        config = OrcaRouterResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://custom.orcarouter.ai/v1",
            litellm_params={},
        )
        assert url == "https://custom.orcarouter.ai/v1/responses"

    def test_get_complete_url_strips_trailing_slash(self):
        config = OrcaRouterResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://api.orcarouter.ai/v1/",
            litellm_params={},
        )
        assert url == "https://api.orcarouter.ai/v1/responses"

    def test_validate_environment_sets_auth_header(self):
        config = OrcaRouterResponsesAPIConfig()
        params = GenericLiteLLMParams(api_key="sk-or-test-key")
        headers = config.validate_environment(
            headers={}, model="openai/o4-mini", litellm_params=params
        )
        assert headers["Authorization"] == "Bearer sk-or-test-key"

    def test_validate_environment_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
        config = OrcaRouterResponsesAPIConfig()
        import litellm

        original_key = litellm.api_key
        litellm.api_key = None
        try:
            with pytest.raises(ValueError, match="OrcaRouter API key is required"):
                config.validate_environment(
                    headers={},
                    model="openai/o4-mini",
                    litellm_params=GenericLiteLLMParams(),
                )
        finally:
            litellm.api_key = original_key

    def test_supports_native_websocket_returns_false(self):
        assert OrcaRouterResponsesAPIConfig().supports_native_websocket() is False
