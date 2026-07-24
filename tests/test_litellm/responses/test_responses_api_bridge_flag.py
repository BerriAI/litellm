"""
Tests for forcing the /responses → /chat/completions bridge for `openai/` models
(via `use_chat_completions_api` or the `openai/chat_completions/<model>` model id).

Includes file_search emulation: the flag must be forwarded on inner aresponses
calls so routed requests do not hit a custom api_base /v1/responses endpoint.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


class TestUseResponsesApiBridgeFlag:
    """Test that bridge opt-in forces the chat completions path."""

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_bridge_used_when_use_chat_completions_api_true(
        self, mock_get_config, mock_bridge_handler
    ):
        """When use_chat_completions_api=True, the bridge handler should be called."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_bridge_used_when_model_uses_chat_completions_prefix(
        self, mock_get_config, mock_bridge_handler
    ):
        """`openai/chat_completions/<name>` normalizes to `openai/<name>` and uses the bridge."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/chat_completions/my-custom-model",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()
        # Model string is provider-normalized after resolution; prefix only forces the bridge.
        assert mock_bridge_handler.call_args.kwargs["model"].endswith("my-custom-model")

    @patch("litellm.responses.main.base_llm_http_handler.response_api_handler")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_native_forwarding_when_flag_absent(
        self, mock_get_config, mock_native_handler
    ):
        """When use_chat_completions_api is not set, openai/ models should use
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
        """use_chat_completions_api should be popped and not passed to the bridge handler."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        call_kwargs = mock_bridge_handler.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "use_chat_completions_api" not in all_kwargs

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

    @patch("litellm.responses.file_search.emulated_handler._call_aresponses")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    async def test_bridge_flag_forwarded_to_file_search_emulation(
        self, mock_get_config, mock_call_aresponses
    ):
        """When use_chat_completions_api=True and file_search tool is present,
        the flag should be forwarded to the inner aresponses call in the
        file_search emulation path."""
        # Setup: provider has native responses API support
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()

        # Mock the inner aresponses call to return a valid response
        mock_response = ResponsesAPIResponse(
            id="resp_123",
            model="openai/my-custom-model",
            created_at=1234567890,
            output=[
                {"type": "message", "content": [{"type": "text", "text": "Answer"}]}
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )
        mock_call_aresponses.return_value = mock_response

        await litellm.aresponses(
            model="openai/my-custom-model",
            input="Search for information",
            tools=[{"type": "file_search"}],
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        # Verify _call_aresponses was called with use_chat_completions_api=True
        mock_call_aresponses.assert_called_once()
        call_kwargs = mock_call_aresponses.call_args.kwargs
        assert (
            call_kwargs.get("use_chat_completions_api") is True
        ), "use_chat_completions_api should be forwarded to inner aresponses call"

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch("litellm.vector_stores.main.asearch")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    async def test_bridge_flag_prevents_native_responses_endpoint_call(
        self, mock_get_config, mock_asearch, mock_bridge_handler
    ):
        """
        Concrete failing scenario: native OpenAI responses config + bridge flag +
        file_search → emulation must still route inner calls through the bridge
        (chat completions), not POST to api_base /v1/responses.
        """
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_asearch.return_value = []

        first_response = ResponsesAPIResponse(
            id="resp_first",
            model="openai/my-local-model",
            created_at=1234567890,
            output=[
                {
                    "type": "function_call",
                    "name": "litellm_file_search",
                    "call_id": "call_123",
                    "arguments": '{"queries": ["test query"]}',
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )
        second_response = ResponsesAPIResponse(
            id="resp_second",
            model="openai/my-local-model",
            created_at=1234567891,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Final answer"}],
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=20, output_tokens=10, total_tokens=30
            ),
        )
        mock_bridge_handler.side_effect = [first_response, second_response]

        result = await litellm.aresponses(
            model="openai/my-local-model",
            input="Search for information",
            tools=[
                {
                    "type": "file_search",
                    "file_search": {"vector_store_ids": ["vs_123"]},
                }
            ],
            use_chat_completions_api=True,
            api_base="http://localhost:8080/v1",
            litellm_logging_obj=MagicMock(),
        )

        assert mock_bridge_handler.call_count == 2, (
            "Bridge handler should be called twice: initial function-tool call "
            "and follow-up with tool results"
        )
        for call in mock_bridge_handler.call_args_list:
            all_kwargs = call.kwargs if call.kwargs else {}
            assert "use_chat_completions_api" not in all_kwargs
        assert result is not None
        assert result.id is not None

    @patch("litellm.responses.main.base_llm_http_handler.response_api_handler")
    @patch("litellm.vector_stores.main.asearch")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    async def test_without_bridge_flag_uses_native_endpoint(
        self, mock_get_config, mock_asearch, mock_native_handler
    ):
        """Without the bridge flag, openai/ with native config uses the native handler."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_asearch.return_value = []
        mock_native_handler.return_value = ResponsesAPIResponse(
            id="resp_native",
            model="openai/gpt-4o",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Native response"}],
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )

        result = await litellm.aresponses(
            model="openai/gpt-4o",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_native_handler.assert_called_once()
        assert result is not None
