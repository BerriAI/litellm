"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig


class TestChatGPTResponsesAPITransformation:
    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.4",
            "chatgpt/gpt-5.4-pro",
            "chatgpt/gpt-5.3-chat-latest",
            "chatgpt/gpt-5.3-instant",
            "chatgpt/gpt-5.3-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_provider_config_registration(self, model_name):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=model_name,
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

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex",
        ],
    )
    def test_chatgpt_forces_streaming_and_reasoning_include(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True
        assert "reasoning.encrypted_content" in request["include"]
        assert request["instructions"].startswith("You are Codex, based on GPT-5.")

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_drops_unsupported_responses_params(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={
                # unsupported by ChatGPT Codex
                "user": "user_123",
                "temperature": 0.2,
                "top_p": 0.9,
                "context_management": [
                    {"type": "compaction", "compact_threshold": 200000}
                ],
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
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "hello"},
        }

    def _transform_headers(
        self, input: list[dict[str, str]], litellm_params: GenericLiteLLMParams
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        ChatGPTResponsesAPIConfig().transform_responses_api_request(
            model="chatgpt/gpt-5.4",
            input=input,
            response_api_optional_request_params={},
            litellm_params=litellm_params,
            headers=headers,
        )
        return headers

    def test_derived_session_id_is_stable_as_conversation_grows(self):
        params = GenericLiteLLMParams(chatgpt_derive_session_id=True)
        first_turn = [{"role": "user", "content": "start of conversation"}]
        later_turn = first_turn + [
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "follow-up"},
        ]

        headers_one = self._transform_headers(first_turn, params)
        headers_two = self._transform_headers(later_turn, params)

        assert headers_one["session_id"].startswith("litellm-derived-")
        assert headers_one["session_id"] == headers_two["session_id"]

    def test_derived_session_id_differs_across_conversations(self):
        params = GenericLiteLLMParams(chatgpt_derive_session_id=True)

        headers_one = self._transform_headers(
            [{"role": "user", "content": "conversation a"}], params
        )
        headers_two = self._transform_headers(
            [{"role": "user", "content": "conversation b"}], params
        )

        assert headers_one["session_id"] != headers_two["session_id"]

    def test_explicit_session_id_wins_over_derivation(self):
        params = GenericLiteLLMParams(
            chatgpt_derive_session_id=True, litellm_session_id="explicit-1"
        )

        headers = self._transform_headers([{"role": "user", "content": "hi"}], params)

        assert "session_id" not in headers

    def test_proxy_trace_id_does_not_block_derivation(self):
        params = GenericLiteLLMParams(
            chatgpt_derive_session_id=True,
            litellm_trace_id="b6c2977a-0652-43de-a9a6-165822f439f6",
        )
        headers = {"session_id": params.litellm_trace_id}

        ChatGPTResponsesAPIConfig().transform_responses_api_request(
            model="chatgpt/gpt-5.4",
            input=[{"role": "user", "content": "hi"}],
            response_api_optional_request_params={},
            litellm_params=params,
            headers=headers,
        )

        assert headers["session_id"].startswith("litellm-derived-")

    def test_different_api_keys_derive_different_session_ids(self):
        input = [{"role": "user", "content": "shared prefix"}]
        params_a = GenericLiteLLMParams(
            chatgpt_derive_session_id=True,
            litellm_metadata={"user_api_key_hash": "key-a"},
        )
        params_b = GenericLiteLLMParams(
            chatgpt_derive_session_id=True,
            litellm_metadata={"user_api_key_hash": "key-b"},
        )

        headers_a = self._transform_headers(input, params_a)
        headers_b = self._transform_headers(input, params_b)

        assert headers_a["session_id"] != headers_b["session_id"]
        assert headers_a["session_id"] == self._transform_headers(input, params_a)["session_id"]

    def test_quoted_false_string_does_not_enable_derivation(self):
        params = GenericLiteLLMParams(chatgpt_derive_session_id="false")

        headers = self._transform_headers([{"role": "user", "content": "hi"}], params)

        assert "session_id" not in headers

    def test_quoted_true_string_enables_derivation(self):
        params = GenericLiteLLMParams(chatgpt_derive_session_id="true")

        headers = self._transform_headers([{"role": "user", "content": "hi"}], params)

        assert headers["session_id"].startswith("litellm-derived-")

    def test_no_derivation_without_flag(self):
        headers = self._transform_headers(
            [{"role": "user", "content": "hi"}], GenericLiteLLMParams()
        )

        assert "session_id" not in headers

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_parsing(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
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
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello!"

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_recovers_output_items(
        self, model_name: str, response_model: str
    ):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [],
        }
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello from stream!"}],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': streamed_output_item})}",
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
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello from stream!"

    def test_chatgpt_non_stream_sse_recovers_whitespace_padded_chunks(self):
        """Chunks with leading whitespace before `data:` must still parse.

        `_strip_sse_data_from_chunk` only matches the prefix at position 0,
        so without an outer `.strip()` such chunks would fail JSON parsing
        and silently drop the contained event.
        """
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
        }
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Recovered from padded"}],
        }
        sse_body = "\n".join(
            [
                f"   data:  {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': streamed_output_item})}   ",
                f"\tdata: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.4",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Recovered from padded"

    @pytest.mark.parametrize(
        "error_chunk",
        [
            {
                "type": "response.failed",
                "response": {"error": {"message": "ChatGPT upstream failed"}},
            },
            {
                "type": "error",
                "error": {"message": "ChatGPT upstream failed"},
            },
        ],
    )
    def test_chatgpt_non_stream_sse_response_raises_openai_error(self, error_chunk):
        config = ChatGPTResponsesAPIConfig()
        sse_body = "\n".join(
            [
                f"data: {json.dumps(error_chunk)}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(
            502, headers={"content-type": "text/event-stream"}, text=sse_body
        )
        logging_obj = MagicMock()

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_response_api_response(
                model="chatgpt/gpt-5.4",
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        assert "ChatGPT upstream failed" in str(exc_info.value)
        assert exc_info.value.status_code == 502
