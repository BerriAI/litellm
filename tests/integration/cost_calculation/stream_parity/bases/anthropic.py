from dataclasses import replace
from typing import Final

from integration._support.provider import PROVIDER_URL
from integration.cost_calculation.stream_parity.case import MODEL, RESPONSE_ID, StreamParityTestCase

CLAUDE_OPUS_4_8_MESSAGES_TEST_CASE: Final = StreamParityTestCase(
    scenario="messages",
    litellm_endpoint="/v1/messages",
    deployment={
        "model": "anthropic/claude-opus-4-8",
        "api_key": "synthetic-anthropic-key",
        "api_base": PROVIDER_URL,
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    },
    litellm_request={
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    mock_provider_response={
        "id": RESPONSE_ID,
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 40},
    },
    mock_provider_stream=(
        {
            "type": "message_start",
            "message": {
                "id": RESPONSE_ID,
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 30, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello."}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 40},
        },
        {"type": "message_stop"},
    ),
    expected_spend_row={
        "model": "anthropic/claude-opus-4-8",
        "model_group": MODEL,
        "custom_llm_provider": "anthropic",
        "prompt_tokens": 30,
        "completion_tokens": 40,
        "total_tokens": 70,
        "spend": 0.11,
    },
)

CLAUDE_OPUS_4_8_CHAT_COMPLETIONS_TEST_CASE: Final = replace(
    CLAUDE_OPUS_4_8_MESSAGES_TEST_CASE,
    scenario="chat_completions",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    stream_parameters={"stream": True, "stream_options": {"include_usage": True}},
)

CLAUDE_OPUS_4_8_RESPONSES_TEST_CASE: Final = replace(
    CLAUDE_OPUS_4_8_MESSAGES_TEST_CASE,
    scenario="responses",
    litellm_endpoint="/v1/responses",
    litellm_request={"model": MODEL, "input": "Say hello.", "cache": {"no-cache": True}},
)
