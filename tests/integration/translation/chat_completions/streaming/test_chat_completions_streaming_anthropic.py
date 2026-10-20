from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import SSEFrame, StreamingTranslationTestCase
from integration.translation.chat_completions.bases.anthropic import (
    CLAUDE_HAIKU_5_5_TEST_CASE,
    CLAUDE_SONNET_5_5_TEST_CASE,
)
from integration.translation.runner import assert_streaming_translation
from pydantic import JsonValue

CLAUDE_SONNET_5_5_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="streaming",
    litellm_endpoint=CLAUDE_SONNET_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={
        "model": "anthropic/claude-sonnet-5-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
        "stream": True,
    },
    expected_provider_endpoint=CLAUDE_SONNET_5_5_TEST_CASE.expected_provider_endpoint,
    expected_provider_headers=CLAUDE_SONNET_5_5_TEST_CASE.expected_provider_headers,
    expected_provider_request={
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            event="message_start",
            data={
                "type": "message_start",
                "message": {
                    "model": "claude-sonnet-5-5",
                    "id": "msg_011CfyYUVrHK4BM53hfSH5Ld",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "container": None,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "stop_details": None,
                    "usage": {
                        "input_tokens": 23,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                        "output_tokens": 6,
                        "service_tier": "standard",
                        "inference_geo": "global",
                    },
                    "diagnostics": None,
                },
            },
        ),
        SSEFrame(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        SSEFrame(event="ping", data={"type": "ping"}),
        SSEFrame(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello."},
            },
        ),
        SSEFrame(event="content_block_stop", data={"type": "content_block_stop", "index": 0}),
        SSEFrame(
            event="message_delta",
            data={
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "stop_details": None,
                    "container": None,
                },
                "usage": {
                    "input_tokens": 23,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 6,
                    "output_tokens_details": {"thinking_tokens": 0},
                },
            },
        ),
        SSEFrame(event="message_stop", data={"type": "message_stop"}),
    ),
    expected_litellm_frames=(
        SSEFrame(
            data={
                "id": ANY,
                "object": "chat.completion.chunk",
                "created": ANY,
                "model": "anthropic/claude-sonnet-5-5",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello."}}],
            }
        ),
        SSEFrame(
            data={
                "id": ANY,
                "object": "chat.completion.chunk",
                "created": ANY,
                "model": "anthropic/claude-sonnet-5-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        SSEFrame(data="[DONE]"),
    ),
    expected_litellm_response={
        "id": ANY,
        "object": "chat.completion",
        "created": ANY,
        "model": "anthropic/claude-sonnet-5-5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}],
    },
)


PLACE_SCHEMA: Final = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": True,
}
PLACE_PROMPT: Final = "Which city is the capital of France? Answer as JSON with city and country."


def _place_delta(partial_json: str) -> SSEFrame:
    return SSEFrame(
        event="content_block_delta",
        data={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


def _place_chunk(delta: dict[str, JsonValue], *, finish_reason: str | None = None) -> SSEFrame:
    finish: Final = {} if finish_reason is None else {"finish_reason": finish_reason}
    return SSEFrame(
        data={
            "id": ANY,
            "object": "chat.completion.chunk",
            "created": ANY,
            "model": "anthropic/claude-haiku-5-5",
            "choices": [{"index": 0, "delta": delta, **finish}],
        }
    )


CLAUDE_HAIKU_5_5_JSON_SCHEMA_CUT_STREAM_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="json_schema_cut_stream",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={
        "model": "anthropic/claude-haiku-5-5",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": PLACE_PROMPT}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "place", "schema": PLACE_SCHEMA}},
        "cache": {"no-cache": True},
        "stream": True,
    },
    expected_provider_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_endpoint,
    expected_provider_headers=CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_headers,
    expected_provider_request={
        "model": "claude-haiku-5-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": PLACE_PROMPT}]}],
        "max_tokens": 128,
        "tool_choice": {"name": "json_tool_call", "type": "tool"},
        "tools": [{"name": "json_tool_call", "input_schema": PLACE_SCHEMA}],
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            event="message_start",
            data={
                "type": "message_start",
                "message": {
                    "model": "claude-haiku-5-5",
                    "id": "msg_011CgCoP5RkexwE6N7PRZEca",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "container": None,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "stop_details": None,
                    "usage": {
                        "input_tokens": 528,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                        "output_tokens": 16,
                        "service_tier": "standard",
                        "inference_geo": "global",
                        "speed": "standard",
                    },
                    "diagnostics": None,
                },
            },
        ),
        SSEFrame(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01B4QSMQtft8TA3Rn6XFziWe",
                    "name": "json_tool_call",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
        ),
        SSEFrame(event="ping", data={"type": "ping"}),
        _place_delta(""),
        _place_delta('{"city"'),
        _place_delta(': "Pari'),
        _place_delta('s"'),
        _place_delta(', "coun'),
        _place_delta('try": "'),
        _place_delta('France"}'),
    ),
    expected_litellm_frames=(
        _place_chunk({"role": "assistant", "content": '{"city": "Pari'}),
        _place_chunk({"content": 's"'}),
        _place_chunk({"content": ', "coun'}),
        _place_chunk({"content": 'try": "'}),
        _place_chunk({"content": 'France"'}),
        _place_chunk({"content": "}"}),
        _place_chunk({}, finish_reason="stop"),
        SSEFrame(data="[DONE]"),
    ),
    expected_litellm_response={
        "id": ANY,
        "object": "chat.completion",
        "created": ANY,
        "model": "anthropic/claude-haiku-5-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"city": "Paris", "country": "France"}'},
                "finish_reason": "stop",
            }
        ],
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_SONNET_5_5_STREAMING_TEST_CASE, CLAUDE_HAIKU_5_5_JSON_SCHEMA_CUT_STREAM_TEST_CASE],
    ids=lambda case: case.id,
)
def test_chat_completions_streaming_anthropic(
    case: StreamingTranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_streaming_translation(case, gateway, provider)
