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


class TestTransformRequest:
    def test_passes_environment_to_request_body(self, config):
        request_body = config.transform_request(
            model=None,
            agent="my-custom-slides-agent",
            input=[{"type": "text", "text": "Create a 5-slide presentation about AI trends."}],
            optional_params={
                "environment": "remote",
                "stream": False,
            },
            litellm_params=GenericLiteLLMParams(api_key="test-api-key"),
            headers={},
        )

        assert request_body["agent"] == "my-custom-slides-agent"
        assert request_body["environment"] == "remote"
        assert request_body["stream"] is False
        assert request_body["input"] == [
            {"type": "text", "text": "Create a 5-slide presentation about AI trends."}
        ]

    def test_passes_environment_object_to_request_body(self, config):
        environment_config = {
            "type": "remote",
            "sources": [{"type": "gcs", "uri": "gs://bucket/skills.zip"}],
            "network": {"egress": "allow_all"},
        }
        request_body = config.transform_request(
            model=None,
            agent="waverunner",
            input="What is 2 + 2?",
            optional_params={"environment": environment_config},
            litellm_params=GenericLiteLLMParams(api_key="test-api-key"),
            headers={},
        )

        assert request_body["environment"] == environment_config

    def test_passes_existing_environment_id_to_request_body(self, config):
        env_id = "env-abc123"
        request_body = config.transform_request(
            model=None,
            agent="my-custom-slides-agent",
            input="Continue the presentation.",
            optional_params={"environment": env_id},
            litellm_params=GenericLiteLLMParams(api_key="test-api-key"),
            headers={},
        )

        assert request_body["environment"] == env_id


class TestInteractionOperationUrls:
    """Test that get/delete/cancel interaction URLs exclude API key."""

    @pytest.mark.parametrize(
        "method_name,interaction_id,expected_suffix",
        [
            ("transform_get_interaction_request", "interaction-123", "interaction-123"),
            (
                "transform_delete_interaction_request",
                "interaction-456",
                "interaction-456",
            ),
            (
                "transform_cancel_interaction_request",
                "interaction-789",
                "interaction-789:cancel",
            ),
        ],
    )
    def test_url_excludes_key(
        self, config, method_name, interaction_id, expected_suffix
    ):
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

    def test_interaction_id_is_encoded_as_one_path_segment(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url, params = config.transform_cancel_interaction_request(
                interaction_id="../../interactions/other?x=1#frag",
                api_base="https://generativelanguage.googleapis.com",
                litellm_params=GenericLiteLLMParams(api_key="secret-key"),
                headers={},
            )

        assert (
            url
            == "https://generativelanguage.googleapis.com/v1beta/interactions/..%2F..%2Finteractions%2Fother%3Fx%3D1%23frag:cancel"
        )
        assert params == {}

    def test_get_interaction_raises_without_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value=None):
            with pytest.raises(ValueError, match="Google API key is required"):
                config.transform_get_interaction_request(
                    interaction_id="interaction-123",
                    api_base="https://generativelanguage.googleapis.com",
                    litellm_params=GenericLiteLLMParams(api_key=None),
                    headers={},
                )
