from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import SSEFrame, StreamingTranslationTestCase
from integration.translation.chat_completions.bases.anthropic import CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.runner import assert_streaming_translation

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


@pytest.mark.parametrize("case", [CLAUDE_SONNET_5_5_STREAMING_TEST_CASE], ids=lambda case: case.id)
def test_chat_completions_streaming_anthropic(
    case: StreamingTranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_streaming_translation(case, gateway, provider)
