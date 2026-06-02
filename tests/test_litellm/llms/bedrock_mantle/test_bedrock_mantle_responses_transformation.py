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
