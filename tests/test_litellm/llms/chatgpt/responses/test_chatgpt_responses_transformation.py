"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig


class TestChatGPTResponsesAPITransformation:
    def test_chatgpt_provider_config_registration(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="chatgpt/gpt-5.2",
            provider=LlmProviders.CHATGPT,
        )

        assert config is not None
        assert isinstance(config, ChatGPTResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.CHATGPT

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_chatgpt_responses_endpoint_url(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_base.return_value = "https://chatgpt.example.com"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://chatgpt.example.com/responses"

        custom_url = config.get_complete_url(
            api_base="https://custom.chatgpt.com", litellm_params={}
        )
        assert custom_url == "https://custom.chatgpt.com/responses"

        url_with_slash = config.get_complete_url(
            api_base="https://chatgpt.example.com/", litellm_params={}
        )
        assert url_with_slash == "https://chatgpt.example.com/responses"

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_validate_environment_headers(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(litellm_session_id="session-123")
        headers = config.validate_environment(
            headers={"originator": "custom-origin"},
            model="gpt-5.2",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["originator"] == "custom-origin"
        assert headers["content-type"] == "application/json"
        assert headers["accept"] == "text/event-stream"
        assert headers["session_id"] == "session-123"

    def test_chatgpt_forces_streaming_and_reasoning_include(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.2-codex",
            input="hi",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True
        assert "reasoning.encrypted_content" in request["include"]
        assert request["instructions"].startswith(
            "You are Codex, based on GPT-5."
        )

    def test_chatgpt_drops_unsupported_responses_params(self):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.2-codex",
            input="hi",
            response_api_optional_request_params={
                # unsupported by ChatGPT Codex
                "user": "user_123",
                "temperature": 0.2,
                "top_p": 0.9,
                "context_management": [{"type": "compaction", "compact_threshold": 200000}],
                "metadata": {"foo": "bar"},
                "max_output_tokens": 123,
                "stream_options": {"include_usage": True},
                # supported and should be preserved
                "truncation": "auto",
                "previous_response_id": "resp_123",
                "reasoning": {"effort": "medium"},
                "tools": [{"type": "function", "function": {"name": "hello"}}],
                "tool_choice": {"type": "function", "function": {"name": "hello"}},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "user" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "context_management" not in request
        assert "metadata" not in request
        assert "max_output_tokens" not in request
        assert "stream_options" not in request

        assert request["truncation"] == "auto"
        assert request["previous_response_id"] == "resp_123"
        assert request["reasoning"] == {"effort": "medium"}
        assert request["tools"] == [{"type": "function", "function": {"name": "hello"}}]
        assert request["tool_choice"] == {"type": "function", "function": {"name": "hello"}}

    def test_chatgpt_non_stream_sse_response_parsing(self):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.2-codex",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.2-codex",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello!"

    def test_chatgpt_extracts_system_messages_from_input(self):
        """ChatGPT backend rejects role:'system' in input. They should be
        extracted and prepended to instructions."""
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.2-codex",
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": "You are helpful"},
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello"},
                    ],
                },
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        # System message removed from input
        for item in request["input"]:
            if isinstance(item, dict):
                assert item.get("role") != "system"

        # System text appears in instructions, along with Codex defaults
        assert "You are helpful" in request["instructions"]
        assert request["instructions"].startswith("You are Codex, based on GPT-5.")

    def test_chatgpt_extracts_developer_messages_from_input(self):
        """role:'developer' should be treated the same as 'system'."""
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.2-codex",
            input=[
                {"role": "developer", "content": "Be concise"},
                {"role": "user", "content": "Hi"},
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        for item in request["input"]:
            if isinstance(item, dict):
                assert item.get("role") not in ("system", "developer")

        assert "Be concise" in request["instructions"]

    def test_chatgpt_system_messages_merged_with_explicit_instructions(self):
        """System messages + explicit instructions should all appear."""
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.2-codex",
            input=[
                {"role": "system", "content": "System prompt here"},
                {"role": "user", "content": "Hello"},
            ],
            response_api_optional_request_params={
                "instructions": "User-provided instructions",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        instructions = request["instructions"]
        assert "System prompt here" in instructions
        assert "User-provided instructions" in instructions
        assert instructions.startswith("You are Codex, based on GPT-5.")

    def test_chatgpt_no_system_messages_unchanged(self):
        """When there are no system messages, input should pass through unchanged."""
        config = ChatGPTResponsesAPIConfig()
        original_input = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        request = config.transform_responses_api_request(
            model="chatgpt/gpt-5.2-codex",
            input=original_input,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert len(request["input"]) == 2
