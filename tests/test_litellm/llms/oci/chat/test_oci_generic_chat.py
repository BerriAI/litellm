"""
Unit tests for litellm/llms/oci/chat/generic.py — error paths and stream handling.
"""

import json
import pytest
from unittest.mock import MagicMock

import httpx

from litellm import ModelResponse
from litellm.llms.oci.chat.generic import (
    adapt_messages_to_generic_oci_standard,
    adapt_messages_to_generic_oci_standard_content_message,
    adapt_messages_to_generic_oci_standard_tool_call,
    handle_generic_response,
    handle_generic_stream_chunk,
)
from litellm.llms.oci.chat.transformation import OCIChatConfig, OCIStreamWrapper
from litellm.llms.oci.common_utils import OCIError


# ---------------------------------------------------------------------------
# adapt_messages_to_generic_oci_standard_content_message — error paths
# ---------------------------------------------------------------------------


class TestGenericContentMessageErrors:
    def test_non_dict_content_item_raises(self):
        with pytest.raises(OCIError, match="must be a dictionary"):
            adapt_messages_to_generic_oci_standard_content_message(
                "user", ["not a dict"]
            )

    def test_non_string_type_field_raises(self):
        with pytest.raises(OCIError, match="string `type` field"):
            adapt_messages_to_generic_oci_standard_content_message(
                "user", [{"type": 123, "text": "hi"}]
            )

    def test_unsupported_content_type_raises(self):
        with pytest.raises(OCIError, match="not supported by OCI"):
            adapt_messages_to_generic_oci_standard_content_message(
                "user", [{"type": "video_url", "url": "https://example.com/v.mp4"}]
            )

    def test_non_string_text_raises(self):
        with pytest.raises(OCIError, match="must have a string `text` field"):
            adapt_messages_to_generic_oci_standard_content_message(
                "user", [{"type": "text", "text": 42}]
            )

    def test_image_url_as_invalid_type_raises(self):
        with pytest.raises(OCIError, match="must be a string or an object"):
            adapt_messages_to_generic_oci_standard_content_message(
                "user", [{"type": "image_url", "image_url": 99}]
            )

    def test_image_url_as_string(self):
        msg = adapt_messages_to_generic_oci_standard_content_message(
            "user", [{"type": "image_url", "image_url": "https://example.com/img.png"}]
        )
        assert msg.content[0].imageUrl.url == "https://example.com/img.png"

    def test_image_url_as_dict(self):
        msg = adapt_messages_to_generic_oci_standard_content_message(
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                }
            ],
        )
        assert msg.content[0].imageUrl.url == "https://example.com/img.png"

    def test_text_content_string(self):
        msg = adapt_messages_to_generic_oci_standard_content_message("user", "hello")
        assert msg.content[0].text == "hello"


# ---------------------------------------------------------------------------
# adapt_messages_to_generic_oci_standard_tool_call — error paths
# ---------------------------------------------------------------------------


class TestGenericToolCallErrors:
    def test_non_dict_tool_call_raises(self):
        with pytest.raises(OCIError, match="must be a dictionary"):
            adapt_messages_to_generic_oci_standard_tool_call("assistant", ["bad"])

    def test_non_function_type_raises(self):
        with pytest.raises(OCIError, match="only supports function tool calls"):
            adapt_messages_to_generic_oci_standard_tool_call(
                "assistant",
                [
                    {
                        "type": "database",
                        "id": "x",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            )

    def test_non_string_id_raises(self):
        with pytest.raises(OCIError, match="id.*must be a string"):
            adapt_messages_to_generic_oci_standard_tool_call(
                "assistant",
                [
                    {
                        "type": "function",
                        "id": 123,
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            )

    def test_non_dict_function_raises(self):
        with pytest.raises(OCIError, match="`function` must be a dictionary"):
            adapt_messages_to_generic_oci_standard_tool_call(
                "assistant",
                [{"type": "function", "id": "c1", "function": "not_a_dict"}],
            )

    def test_non_string_function_name_raises(self):
        with pytest.raises(OCIError, match="function.name.*must be a string"):
            adapt_messages_to_generic_oci_standard_tool_call(
                "assistant",
                [
                    {
                        "type": "function",
                        "id": "c1",
                        "function": {"name": 5, "arguments": "{}"},
                    }
                ],
            )

    def test_non_string_arguments_raises(self):
        with pytest.raises(OCIError, match="arguments.*must be a JSON string"):
            adapt_messages_to_generic_oci_standard_tool_call(
                "assistant",
                [
                    {
                        "type": "function",
                        "id": "c1",
                        "function": {"name": "fn", "arguments": {"key": "val"}},
                    }
                ],
            )


# ---------------------------------------------------------------------------
# adapt_messages_to_generic_oci_standard — combined paths
# ---------------------------------------------------------------------------


class TestGenericMessageAdaptation:
    def test_tool_calls_not_list_raises(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": "not_a_list",
            }
        ]
        with pytest.raises(OCIError, match="`tool_calls` must be a list"):
            adapt_messages_to_generic_oci_standard(messages)

    def test_tool_result_non_string_tool_call_id_raises(self):
        messages = [{"role": "tool", "content": "result", "tool_call_id": 999}]
        with pytest.raises(OCIError, match="string `tool_call_id`"):
            adapt_messages_to_generic_oci_standard(messages)

    def test_tool_result_non_string_content_raises(self):
        messages = [
            {"role": "tool", "content": {"structured": "data"}, "tool_call_id": "c1"}
        ]
        with pytest.raises(OCIError, match="`content` must be a string"):
            adapt_messages_to_generic_oci_standard(messages)

    def test_non_string_non_list_content_raises(self):
        messages = [{"role": "user", "content": 42}]
        with pytest.raises(OCIError, match="`content` must be a string or list"):
            adapt_messages_to_generic_oci_standard(messages)


# ---------------------------------------------------------------------------
# handle_generic_response — error and None message paths
# ---------------------------------------------------------------------------


class TestHandleGenericResponse:
    def _make_response(self, body: dict, status: int = 200) -> httpx.Response:
        return httpx.Response(status_code=status, json=body)

    def _valid_body(self, message=None):
        return {
            "modelId": "xai.grok-4",
            "modelVersion": "1",
            "chatResponse": {
                "apiFormat": "GENERIC",
                "timeCreated": "2024-01-01T00:00:00Z",
                "choices": [
                    {"message": message, "finishReason": "COMPLETE", "index": 0}
                ],
                "usage": {"promptTokens": 5, "completionTokens": 5, "totalTokens": 10},
            },
        }

    def test_none_response_message(self):
        body = self._valid_body(message=None)
        raw = self._make_response(body)
        # Should not raise — None message means no content set
        result = handle_generic_response(body, "xai.grok-4", ModelResponse(), raw)
        assert result.model == "xai.grok-4"

    def test_response_with_text_content(self):
        body = self._valid_body(
            message={
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": "Hello!"}],
            }
        )
        raw = self._make_response(body)
        result = handle_generic_response(body, "xai.grok-4", ModelResponse(), raw)
        assert result.choices[0].message.content == "Hello!"

    def test_response_with_tool_calls(self):
        body = self._valid_body(
            message={
                "role": "ASSISTANT",
                "content": [],
                "toolCalls": [
                    {
                        "id": "call_abc",
                        "type": "FUNCTION",
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    }
                ],
            }
        )
        raw = self._make_response(body)
        result = handle_generic_response(body, "xai.grok-4", ModelResponse(), raw)
        assert result.choices[0].message.tool_calls is not None

    def _body_with_finish_reason(self, reason):
        body = self._valid_body(
            message={"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "hi"}]}
        )
        body["chatResponse"]["choices"][0]["finishReason"] = reason
        return body

    def test_complete_maps_to_stop(self):
        body = self._body_with_finish_reason("COMPLETE")
        result = handle_generic_response(body, "xai.grok-4", ModelResponse(), self._make_response(body))
        assert result.choices[0].finish_reason == "stop"

    def test_max_tokens_maps_to_length(self):
        body = self._body_with_finish_reason("MAX_TOKENS")
        result = handle_generic_response(body, "xai.grok-4", ModelResponse(), self._make_response(body))
        assert result.choices[0].finish_reason == "length"

    def test_tool_calls_plural_maps_to_tool_calls(self):
        body = self._body_with_finish_reason("TOOL_CALLS")
        result = handle_generic_response(body, "xai.grok-4", ModelResponse(), self._make_response(body))
        assert result.choices[0].finish_reason == "tool_calls"

    def test_tool_call_singular_maps_to_tool_calls(self):
        body = self._body_with_finish_reason("TOOL_CALL")
        result = handle_generic_response(body, "xai.grok-4", ModelResponse(), self._make_response(body))
        assert result.choices[0].finish_reason == "tool_calls"

    def test_unknown_finish_reason_passes_through(self):
        body = self._body_with_finish_reason("SOMETHING_ELSE")
        result = handle_generic_response(body, "xai.grok-4", ModelResponse(), self._make_response(body))
        assert result.choices[0].finish_reason == "SOMETHING_ELSE"


# ---------------------------------------------------------------------------
# handle_generic_stream_chunk — finish reasons and error paths
# ---------------------------------------------------------------------------


class TestHandleGenericStreamChunk:
    def test_max_tokens_finish_reason(self):
        chunk = {"apiFormat": "GENERIC", "index": 0, "finishReason": "MAX_TOKENS"}
        result = handle_generic_stream_chunk(chunk)
        assert result.choices[0].finish_reason == "length"

    def test_tool_calls_finish_reason(self):
        chunk = {"apiFormat": "GENERIC", "index": 0, "finishReason": "TOOL_CALLS"}
        result = handle_generic_stream_chunk(chunk)
        assert result.choices[0].finish_reason == "tool_calls"

    def test_unknown_finish_reason_does_not_raise(self):
        chunk = {"apiFormat": "GENERIC", "index": 0, "finishReason": "SOME_NEW_REASON"}
        result = handle_generic_stream_chunk(chunk)
        assert result.choices[0] is not None

    def test_null_index_defaults_to_zero(self):
        chunk = {"apiFormat": "GENERIC", "index": None, "finishReason": None}
        result = handle_generic_stream_chunk(chunk)
        assert result.choices[0].index == 0

    def test_image_content_in_stream_raises(self):
        from litellm.types.llms.oci import OCIImageContentPart, OCIImageUrl, OCIMessage

        chunk = {
            "apiFormat": "GENERIC",
            "index": 0,
            "message": {
                "role": "ASSISTANT",
                "content": [
                    {
                        "type": "IMAGE",
                        "imageUrl": {"url": "https://example.com/img.png"},
                    }
                ],
            },
        }
        with pytest.raises(OCIError, match="image content"):
            handle_generic_stream_chunk(chunk)

    def test_stream_chunk_with_tool_calls(self):
        chunk = {
            "apiFormat": "GENERIC",
            "index": 0,
            "message": {
                "role": "ASSISTANT",
                "content": [],
                "toolCalls": [
                    {
                        "id": "call_abc",
                        "type": "FUNCTION",
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    }
                ],
            },
        }
        result = handle_generic_stream_chunk(chunk)
        assert result.choices[0].delta.tool_calls is not None


# ---------------------------------------------------------------------------
# OCIStreamWrapper.chunk_creator — non-string chunk
# ---------------------------------------------------------------------------


class TestOCIStreamWrapperChunkCreator:
    def _wrapper(self):
        return OCIStreamWrapper(
            completion_stream=MagicMock(),
            model="xai.grok-4",
            logging_obj=MagicMock(),
        )

    def test_non_string_chunk_raises(self):
        w = self._wrapper()
        with pytest.raises(ValueError, match="not a string"):
            w.chunk_creator({"already": "parsed"})
