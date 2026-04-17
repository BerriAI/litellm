"""
Tests for Gemini Interactions API transformation.

Covers credential leak prevention changes:
- validate_environment sets x-goog-api-key header
- get_complete_url excludes API key from URL
- get/delete/cancel interaction request URLs exclude API key
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.llms.gemini.interactions.transformation import (
    GoogleAIStudioInteractionsConfig,
)
from litellm.types.router import GenericLiteLLMParams

_PATCH_GET_API_KEY = "litellm.llms.gemini.common_utils.GeminiModelInfo.get_api_key"


@pytest.fixture
def config():
    return GoogleAIStudioInteractionsConfig()


class TestValidateEnvironment:
    def test_sets_x_goog_api_key_header(self, config):
        litellm_params = GenericLiteLLMParams(api_key="test-api-key-123")

        headers = config.validate_environment(
            headers={},
            model="gemini-2.5-flash",
            litellm_params=litellm_params,
        )

        assert headers["x-goog-api-key"] == "test-api-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_no_api_key_skips_header(self, config):
        litellm_params = GenericLiteLLMParams(api_key=None)

        with patch(_PATCH_GET_API_KEY, return_value=None):
            headers = config.validate_environment(
                headers={},
                model="gemini-2.5-flash",
                litellm_params=litellm_params,
            )

        assert "x-goog-api-key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_no_litellm_params_skips_header(self, config):
        headers = config.validate_environment(
            headers={},
            model="gemini-2.5-flash",
            litellm_params=None,
        )

        assert "x-goog-api-key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_preserves_existing_headers(self, config):
        litellm_params = GenericLiteLLMParams(api_key="test-key")

        headers = config.validate_environment(
            headers={"X-Custom": "value"},
            model="gemini-2.5-flash",
            litellm_params=litellm_params,
        )

        assert headers["X-Custom"] == "value"
        assert headers["x-goog-api-key"] == "test-key"


class TestGetCompleteUrl:
    def test_url_excludes_api_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url = config.get_complete_url(
                api_base=None,
                model="gemini-2.5-flash",
                litellm_params={"api_key": "secret-key"},
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert url.endswith("/interactions")

    def test_stream_url_has_alt_sse_only(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url = config.get_complete_url(
                api_base=None,
                model="gemini-2.5-flash",
                litellm_params={"api_key": "secret-key"},
                stream=True,
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert "alt=sse" in url

    def test_raises_without_api_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value=None):
            with pytest.raises(ValueError, match="Google API key is required"):
                config.get_complete_url(
                    api_base=None,
                    model="gemini-2.5-flash",
                    litellm_params={"api_key": None},
                )


class TestInteractionOperationUrls:
    """Test that get/delete/cancel interaction URLs exclude API key."""

    @pytest.mark.parametrize(
        "method_name,interaction_id,expected_suffix",
        [
            ("transform_get_interaction_request", "interaction-123", "interaction-123"),
            ("transform_delete_interaction_request", "interaction-456", "interaction-456"),
            ("transform_cancel_interaction_request", "interaction-789", "interaction-789:cancel"),
        ],
    )
    def test_url_excludes_key(self, config, method_name, interaction_id, expected_suffix):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url, params = getattr(config, method_name)(
                interaction_id=interaction_id,
                api_base="https://generativelanguage.googleapis.com",
                litellm_params=GenericLiteLLMParams(api_key="secret-key"),
                headers={},
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert expected_suffix in url

    def test_get_interaction_raises_without_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value=None):
            with pytest.raises(ValueError, match="Google API key is required"):
                config.transform_get_interaction_request(
                    interaction_id="interaction-123",
                    api_base="https://generativelanguage.googleapis.com",
                    litellm_params=GenericLiteLLMParams(api_key=None),
                    headers={},
                )
