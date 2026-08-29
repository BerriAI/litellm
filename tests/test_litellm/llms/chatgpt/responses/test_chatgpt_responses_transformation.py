"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


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

    def test_transform_streaming_response_recovers_output_from_live_chunks(self):
        """chatgpt.com always ships `response.completed.output: []` (see the
        non-streaming recovery tests above) -- but this provider also always
        forces `stream=True` upstream, so `transform_streaming_response`,
        not `transform_response_api_response`, is what actually runs on
        every real call (BerriAI/litellm#25429, #27175). Without accumulating
        `output_item.done`/`output_text.done` chunks here, the completed
        event's `output` stays empty and the Responses-to-Chat-Completions
        bridge raises "Unknown items in responses API response: []".
        """
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}

        streamed_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello from stream!"}],
        }
        item_done_event = config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": "response.output_item.done",
                "output_index": 0,
                "item": streamed_item,
            },
            logging_obj=logging_obj,
        )
        # Passthrough chunks are still forwarded to the base transform.
        assert item_done_event.type == "response.output_item.done"

        completed_event = config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": "response.completed",
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
            logging_obj=logging_obj,
        )

        assert completed_event.response.output == [streamed_item]

    def test_transform_streaming_response_leaves_populated_output_untouched(self):
        """A completed event that already carries real output (the standard
        OpenAI shape) must not be altered by the recovery logic."""
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}

        real_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Already there"}],
        }
        completed_event = config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": "response.completed",
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [real_output_item],
                },
            },
            logging_obj=logging_obj,
        )

        assert completed_event.response.output == [real_output_item]

    def test_transform_streaming_response_state_scoped_per_request(self):
        """Accumulator state must live on `logging_obj.model_call_details`
        (per-request), not on the shared, long-lived `ChatGPTResponsesAPIConfig`
        instance -- otherwise output from one concurrent call could leak into
        another call's completed event on a proxy serving multiple requests.
        """
        config = ChatGPTResponsesAPIConfig()

        logging_obj_a = MagicMock()
        logging_obj_a.model_call_details = {}
        item_a = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "From request A"}],
        }
        config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": "response.output_item.done",
                "output_index": 0,
                "item": item_a,
            },
            logging_obj=logging_obj_a,
        )

        # A second, independent request must not see request A's accumulated item.
        logging_obj_b = MagicMock()
        logging_obj_b.model_call_details = {}
        completed_event_b = config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": "response.completed",
                "response": {
                    "id": "resp_test_b",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
            logging_obj=logging_obj_b,
        )

        assert completed_event_b.response.output == []

    def test_transform_streaming_response_cleans_up_accumulator_state_on_completion(
        self,
    ):
        """`_chatgpt_streamed_output_items`/`_chatgpt_text_only_output_items`
        must be popped off `model_call_details` once the completed event is
        handled -- otherwise a logging callback that reads
        `kwargs`/`model_call_details` directly (rather than only the
        redacted typed response) could export the raw recovered response
        text even when `litellm.turn_off_message_logging` is set, since
        `redact_message_input_output_from_logging` only scrubs a fixed set
        of recognized fields and has no knowledge of these two custom keys.
        """
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}

        streamed_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Sensitive content"}],
        }
        config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": "response.output_item.done",
                "output_index": 0,
                "item": streamed_item,
            },
            logging_obj=logging_obj,
        )
        assert "_chatgpt_streamed_output_items" in logging_obj.model_call_details

        config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": "response.completed",
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
            logging_obj=logging_obj,
        )

        assert "_chatgpt_streamed_output_items" not in logging_obj.model_call_details
        assert "_chatgpt_text_only_output_items" not in logging_obj.model_call_details

    @pytest.mark.parametrize(
        "terminal_event_type",
        ["response.failed", "response.incomplete"],
    )
    def test_transform_streaming_response_cleans_up_accumulator_state_on_failure(
        self, terminal_event_type
    ):
        """Same cleanup must happen on failed/incomplete terminal events, not
        just on a successful completion -- a response that errors out
        mid-stream must not leave recovered content sitting un-redacted in
        `model_call_details` either."""
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}

        streamed_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Partial before failure"}],
        }
        config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": "response.output_item.done",
                "output_index": 0,
                "item": streamed_item,
            },
            logging_obj=logging_obj,
        )
        assert "_chatgpt_streamed_output_items" in logging_obj.model_call_details

        config.transform_streaming_response(
            model="chatgpt/gpt-5.4",
            parsed_chunk={
                "type": terminal_event_type,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "failed",
                    "model": "gpt-5.4",
                    "output": [],
                },
            },
            logging_obj=logging_obj,
        )

        assert "_chatgpt_streamed_output_items" not in logging_obj.model_call_details
        assert "_chatgpt_text_only_output_items" not in logging_obj.model_call_details
