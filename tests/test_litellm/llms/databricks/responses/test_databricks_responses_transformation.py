import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import patch

import litellm
from litellm.llms.databricks.responses.transformation import (
    DatabricksResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestDatabricksResponsesAPIConfig:
    """Tests for DatabricksResponsesAPIConfig"""

    def test_custom_llm_provider(self):
        config = DatabricksResponsesAPIConfig()
        assert config.custom_llm_provider == LlmProviders.DATABRICKS

    def test_get_complete_url(self):
        config = DatabricksResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://my-workspace.cloud.databricks.com/serving-endpoints",
            litellm_params={},
        )
        assert (
            url
            == "https://my-workspace.cloud.databricks.com/serving-endpoints/responses"
        )

    def test_get_complete_url_strips_trailing_slash(self):
        config = DatabricksResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://my-workspace.cloud.databricks.com/serving-endpoints/",
            litellm_params={},
        )
        assert (
            url
            == "https://my-workspace.cloud.databricks.com/serving-endpoints/responses"
        )

    def test_transform_request_strips_provider_prefix(self):
        config = DatabricksResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="databricks/databricks-gpt-5-nano",
            input="Hello!",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["model"] == "databricks-gpt-5-nano"

    def test_transform_request_no_prefix(self):
        config = DatabricksResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="databricks-gpt-5-nano",
            input="Hello!",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["model"] == "databricks-gpt-5-nano"

    def test_transform_request_preserves_text_param(self):
        """Verify that the text/format param (response schema) is passed through to the request."""
        config = DatabricksResponsesAPIConfig()
        text_param = {
            "format": {
                "type": "json_schema",
                "name": "Color",
                "schema": {
                    "type": "object",
                    "properties": {"color": {"type": "string"}},
                    "required": ["color"],
                    "additionalProperties": False,
                },
            }
        }
        request = config.transform_responses_api_request(
            model="databricks-gpt-5-nano",
            input="Hello!",
            response_api_optional_request_params={"text": text_param},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["text"] == text_param

    def test_validate_environment_with_api_key(self):
        config = DatabricksResponsesAPIConfig()
        headers = config.validate_environment(
            headers={},
            model="databricks-gpt-5-nano",
            litellm_params=GenericLiteLLMParams(
                api_key="dapi_test_key",
                api_base="https://my-workspace.cloud.databricks.com/serving-endpoints",
            ),
        )
        assert headers["Authorization"] == "Bearer dapi_test_key"
        assert headers["Content-Type"] == "application/json"


class TestProviderConfigManagerDatabricks:
    """Tests for Databricks registration in ProviderConfigManager"""

    def test_gpt_model_returns_responses_config(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-gpt-5-nano",
        )
        assert config is not None
        assert isinstance(config, DatabricksResponsesAPIConfig)

    def test_gpt_model_case_insensitive(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-GPT-5-nano",
        )
        assert config is not None
        assert isinstance(config, DatabricksResponsesAPIConfig)

    def test_claude_model_returns_none(self):
        """Claude models should fall back to completion transformation, not use Responses API."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-claude-3-5-sonnet",
        )
        assert config is None

    def test_llama_model_returns_none(self):
        """Non-GPT models should fall back to completion transformation."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-meta-llama-3-1-70b-instruct",
        )
        assert config is None

    def test_no_model_returns_none(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model=None,
        )
        assert config is None
