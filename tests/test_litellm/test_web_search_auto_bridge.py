"""
Test automatic bridging of web_search_options to Responses API
when a model only supports /v1/responses.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.main import responses_api_bridge_check


def _make_model_info(
    mode="chat",
    supports_web_search=False,
    supported_endpoints=None,
):
    """Return a minimal model_info dict for mocking."""
    return {
        "key": "openai/gpt-5.4",
        "max_tokens": 100000,
        "max_input_tokens": 1000000,
        "max_output_tokens": 100000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "litellm_provider": "openai",
        "mode": mode,
        "supports_web_search": supports_web_search,
        "supported_endpoints": supported_endpoints,
    }


class TestWebSearchAutoBridge:
    """Test that web_search_options triggers auto-bridge to Responses API
    when the model supports web search only via /v1/responses."""

    @patch("litellm.main._get_model_info_helper")
    def test_auto_bridge_when_model_only_supports_responses(self, mock_get_model_info):
        """Model supports web search + /v1/responses but mode is chat.
        web_search_options should trigger bridge to responses mode."""
        mock_get_model_info.return_value = _make_model_info(
            mode="chat",
            supports_web_search=True,
            supported_endpoints=["/v1/chat/completions", "/v1/responses"],
        )

        model_info, updated_model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="openai",
            web_search_options={},
        )

        assert model_info.get("mode") == "responses"
        assert updated_model == "gpt-5.4"

    @patch("litellm.main._get_model_info_helper")
    def test_no_bridge_without_web_search_options(self, mock_get_model_info):
        """Without web_search_options, no auto-bridge even if model supports it."""
        mock_get_model_info.return_value = _make_model_info(
            mode="chat",
            supports_web_search=True,
            supported_endpoints=["/v1/chat/completions", "/v1/responses"],
        )

        model_info, updated_model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="openai",
            web_search_options=None,
        )

        assert model_info.get("mode") == "chat"
        assert updated_model == "gpt-5.4"

    @patch("litellm.main._get_model_info_helper")
    def test_no_bridge_when_no_responses_endpoint(self, mock_get_model_info):
        """Model supports web search but NOT /v1/responses -- no bridge."""
        mock_get_model_info.return_value = _make_model_info(
            mode="chat",
            supports_web_search=True,
            supported_endpoints=["/v1/chat/completions"],
        )

        model_info, updated_model = responses_api_bridge_check(
            model="gpt-4o",
            custom_llm_provider="openai",
            web_search_options={},
        )

        assert model_info.get("mode") == "chat"

    @patch("litellm.main._get_model_info_helper")
    def test_no_bridge_when_supports_web_search_false(self, mock_get_model_info):
        """Model has /v1/responses but supports_web_search is False -- no bridge."""
        mock_get_model_info.return_value = _make_model_info(
            mode="chat",
            supports_web_search=False,
            supported_endpoints=["/v1/chat/completions", "/v1/responses"],
        )

        model_info, updated_model = responses_api_bridge_check(
            model="some-model",
            custom_llm_provider="openai",
            web_search_options={},
        )

        assert model_info.get("mode") == "chat"

    @patch("litellm.main._get_model_info_helper")
    def test_no_bridge_when_already_responses_mode(self, mock_get_model_info):
        """Model is already in responses mode -- no double-bridge."""
        mock_get_model_info.return_value = _make_model_info(
            mode="responses",
            supports_web_search=True,
            supported_endpoints=["/v1/responses"],
        )

        model_info, updated_model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="openai",
            web_search_options={},
        )

        # Still responses, but did not change -- already was responses
        assert model_info.get("mode") == "responses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
