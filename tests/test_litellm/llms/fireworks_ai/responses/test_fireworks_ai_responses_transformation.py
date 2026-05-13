import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.fireworks_ai.responses.transformation import (
    FireworksAIResponsesConfig,
)
from litellm.types.router import GenericLiteLLMParams


@pytest.fixture
def config():
    return FireworksAIResponsesConfig()


class TestCustomLlmProvider:
    """Tests for the custom_llm_provider property."""

    def test_should_return_fireworks_ai(self, config):
        assert config.custom_llm_provider == "fireworks_ai"


class TestValidateEnvironment:
    """Tests for validate_environment."""

    def test_should_set_auth_header_from_litellm_params(self, config):
        params = GenericLiteLLMParams(api_key="param-key")
        headers = config.validate_environment(
            headers={},
            model="accounts/fireworks/models/llama-v3-70b",
            litellm_params=params,
        )
        assert headers["Authorization"] == "Bearer param-key"

    def test_should_set_auth_header_from_fireworks_api_key_env(self, config):
        with patch(
            "litellm.llms.fireworks_ai.responses.transformation.get_secret_str",
            side_effect=lambda key: "env-key" if key == "FIREWORKS_API_KEY" else None,
        ):
            headers = config.validate_environment(
                headers={},
                model="accounts/fireworks/models/llama-v3-70b",
            )
            assert headers["Authorization"] == "Bearer env-key"

    def test_should_set_auth_header_from_fireworks_ai_api_key_env(self, config):
        with patch(
            "litellm.llms.fireworks_ai.responses.transformation.get_secret_str",
            side_effect=lambda key: (
                "ai-env-key" if key == "FIREWORKS_AI_API_KEY" else None
            ),
        ):
            headers = config.validate_environment(
                headers={},
                model="accounts/fireworks/models/llama-v3-70b",
            )
            assert headers["Authorization"] == "Bearer ai-env-key"

    def test_should_not_set_auth_header_when_no_key(self, config):
        with patch(
            "litellm.llms.fireworks_ai.responses.transformation.get_secret_str",
            return_value=None,
        ):
            headers = config.validate_environment(
                headers={},
                model="accounts/fireworks/models/llama-v3-70b",
            )
            assert "Authorization" not in headers

    def test_should_use_default_litellm_params_when_none(self, config):
        with patch(
            "litellm.llms.fireworks_ai.responses.transformation.get_secret_str",
            side_effect=lambda key: "env-key" if key == "FIREWORKS_API_KEY" else None,
        ):
            headers = config.validate_environment(
                headers={},
                model="accounts/fireworks/models/llama-v3-70b",
                litellm_params=None,
            )
            assert headers["Authorization"] == "Bearer env-key"

    def test_should_override_existing_authorization_with_explicit_key(self, config):
        params = GenericLiteLLMParams(api_key="new-key")
        headers = config.validate_environment(
            headers={"Authorization": "Bearer pre-existing"},
            model="accounts/fireworks/models/llama-v3-70b",
            litellm_params=params,
        )
        assert headers["Authorization"] == "Bearer new-key"


class TestGetCompleteUrl:
    """Tests for get_complete_url."""

    def test_should_use_default_base_url(self, config):
        with patch(
            "litellm.llms.fireworks_ai.responses.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(api_base=None, litellm_params={})
            assert url == "https://api.fireworks.ai/inference/v1/responses"

    def test_should_append_responses_to_custom_base(self, config):
        url = config.get_complete_url(
            api_base="https://custom.example.com/v1",
            litellm_params={},
        )
        assert url == "https://custom.example.com/v1/responses"

    def test_should_append_v1_responses_to_bare_base(self, config):
        url = config.get_complete_url(
            api_base="https://custom.example.com/api",
            litellm_params={},
        )
        assert url == "https://custom.example.com/api/v1/responses"

    def test_should_not_duplicate_responses_suffix(self, config):
        url = config.get_complete_url(
            api_base="https://custom.example.com/v1/responses",
            litellm_params={},
        )
        assert url == "https://custom.example.com/v1/responses"

    def test_should_strip_trailing_slash(self, config):
        url = config.get_complete_url(
            api_base="https://api.fireworks.ai/inference/v1/",
            litellm_params={},
        )
        assert url == "https://api.fireworks.ai/inference/v1/responses"

    def test_should_use_fireworks_api_base_env(self, config):
        with patch(
            "litellm.llms.fireworks_ai.responses.transformation.get_secret_str",
            side_effect=lambda key: (
                "https://env-base.example.com/v1"
                if key == "FIREWORKS_API_BASE"
                else None
            ),
        ):
            url = config.get_complete_url(api_base=None, litellm_params={})
            assert url == "https://env-base.example.com/v1/responses"
