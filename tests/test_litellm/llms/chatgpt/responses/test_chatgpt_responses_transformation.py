"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""

import json
import os
import sys
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.responses.streaming_iterator import SyncResponsesAPIStreamingIterator


def _output_text(output_item: object) -> str:
    if hasattr(output_item, "model_dump"):
        serialized = cast("dict[str, object]", output_item.model_dump())
    else:
        serialized = cast("dict[str, object]", output_item)
    content = cast("list[dict[str, object]]", serialized["content"])
    return cast(str, content[0]["text"])


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

    def test_chatgpt_streaming_response_completed_recovers_output_item_done(self):
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"litellm_params": {}}
        logging_obj.async_success_handler = AsyncMock()
        iterator = SyncResponsesAPIStreamingIterator(
            response=httpx.Response(200),
            model="gpt-5.5",
            responses_api_provider_config=config,
            logging_obj=logging_obj,
            custom_llm_provider=LlmProviders.CHATGPT,
        )
        streamed_output_item = {
            "id": "msg_from_item",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "OK my lord",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
        completed_response = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.5",
            "output": [],
        }

        iterator._process_chunk(
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": streamed_output_item,
                }
            )
        )
        completed_event = iterator._process_chunk(
            json.dumps(
                {
                    "type": "response.completed",
                    "response": completed_response,
                }
            )
        )

        assert completed_event is not None
        assert completed_event.type == "response.completed"
        assert _output_text(completed_event.response.output[0]) == "OK my lord"

    def test_chatgpt_streaming_response_completed_keeps_authoritative_output(self):
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"litellm_params": {}}
        logging_obj.async_success_handler = AsyncMock()
        iterator = SyncResponsesAPIStreamingIterator(
            response=httpx.Response(200),
            model="gpt-5.5",
            responses_api_provider_config=config,
            logging_obj=logging_obj,
            custom_llm_provider=LlmProviders.CHATGPT,
        )

        iterator._process_chunk(
            json.dumps(
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": "msg_from_stream",
                    "text": "Earlier stream text",
                }
            )
        )
        completed_event = iterator._process_chunk(
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_test",
                        "object": "response",
                        "created_at": 1700000000,
                        "status": "completed",
                        "model": "gpt-5.5",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "Authoritative completed text",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                    },
                }
            )
        )

        assert completed_event is not None
        assert (
            _output_text(completed_event.response.output[0])
            == "Authoritative completed text"
        )

    def test_chatgpt_streaming_recovered_output_items_are_copied(self):
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"litellm_params": {}}
        logging_obj.async_success_handler = AsyncMock()
        iterator = SyncResponsesAPIStreamingIterator(
            response=httpx.Response(200),
            model="gpt-5.5",
            responses_api_provider_config=config,
            logging_obj=logging_obj,
            custom_llm_provider=LlmProviders.CHATGPT,
        )
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Recovered"}],
        }

        iterator._process_chunk(
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": streamed_output_item,
                }
            )
        )
        recovered_output = iterator._recovered_streamed_output_items()

        assert recovered_output[0] == streamed_output_item
        assert recovered_output[0] is not streamed_output_item

    def test_chatgpt_streaming_response_created_resets_recovered_output_state(self):
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"litellm_params": {}}
        logging_obj.async_success_handler = AsyncMock()
        iterator = SyncResponsesAPIStreamingIterator(
            response=httpx.Response(200),
            model="gpt-5.5",
            responses_api_provider_config=config,
            logging_obj=logging_obj,
            custom_llm_provider=LlmProviders.CHATGPT,
        )
        iterator._streamed_output_items[0] = {"type": "message"}
        iterator._streamed_text_only_output_items[1] = {"type": "message"}

        iterator._record_streamed_output_chunk({"type": "response.created"})

        assert iterator._streamed_output_items == {}
        assert iterator._streamed_text_only_output_items == {}

    def test_chatgpt_streaming_completed_output_backfill_ignores_invalid_payloads(self):
        config = ChatGPTResponsesAPIConfig()
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"litellm_params": {}}
        logging_obj.async_success_handler = AsyncMock()
        iterator = SyncResponsesAPIStreamingIterator(
            response=httpx.Response(200),
            model="gpt-5.5",
            responses_api_provider_config=config,
            logging_obj=logging_obj,
            custom_llm_provider=LlmProviders.CHATGPT,
        )
        non_dict_response_chunk = {
            "type": "response.completed",
            "response": None,
        }
        empty_unrecoverable_chunk = {
            "type": "response.completed",
            "response": {"output": []},
        }

        assert (
            iterator._backfill_completed_response_output(non_dict_response_chunk)
            is non_dict_response_chunk
        )
        assert (
            iterator._backfill_completed_response_output(empty_unrecoverable_chunk)
            is empty_unrecoverable_chunk
        )

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
