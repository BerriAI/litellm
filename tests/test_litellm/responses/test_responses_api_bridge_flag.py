"""
Tests for the `use_responses_api_bridge` flag that allows openai/ models
with custom api_base to opt-in to the /responses → /chat/completions bridge.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm


class TestUseResponsesApiBridgeFlag:
    """Test that use_responses_api_bridge forces the chat completions bridge."""

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_bridge_used_when_flag_is_true(self, mock_get_config, mock_bridge_handler):
        """When use_responses_api_bridge=True, the bridge handler should be called
        even though the provider (openai) has native responses API support."""
        # Setup: provider config returns a non-None config (native support exists)
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()

        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_responses_api_bridge=True,
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()

    @patch("litellm.responses.main.base_llm_http_handler.response_api_handler")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_native_forwarding_when_flag_absent(
        self, mock_get_config, mock_native_handler
    ):
        """When use_responses_api_bridge is not set, openai/ models should use
        native responses API forwarding (existing behavior)."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_native_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/gpt-4o",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_native_handler.assert_called_once()

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_flag_does_not_leak_into_kwargs(self, mock_get_config, mock_bridge_handler):
        """The use_responses_api_bridge flag should be popped from kwargs and not
        passed through to the bridge handler."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_responses_api_bridge=True,
            litellm_logging_obj=MagicMock(),
        )

        call_kwargs = mock_bridge_handler.call_args
        # The flag should not appear in the kwargs passed to the bridge handler
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "use_responses_api_bridge" not in all_kwargs

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_bridge_used_when_provider_config_none(
        self, mock_get_config, mock_bridge_handler
    ):
        """When the provider has no native responses API config (returns None),
        the bridge should be used regardless of the flag (existing behavior)."""
        mock_get_config.return_value = None
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="anthropic/claude-3-haiku",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()
