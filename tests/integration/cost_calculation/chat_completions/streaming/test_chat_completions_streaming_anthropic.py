from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.chat_completions.bases.anthropic import CLAUDE_SONNET_5_TEST_CASE
from integration.cost_calculation.cost_tracking_case import SseResponse
from integration.cost_calculation.runner import assert_cost_tracking

CLAUDE_SONNET_5_STREAM_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_TEST_CASE,
    scenario="stream",
    litellm_request={
        **CLAUDE_SONNET_5_TEST_CASE.litellm_request,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "ca259a6916f2 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    mock_provider_response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_$REQUEST_ID", '
            '"type": "message", "role": "assistant", "model": "claude-sonnet-5", "content": [], '
            '"stop_reason": null, "usage": {"input_tokens": 1840}}}',
            'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": '
            '{"type": "text", "text": ""}}',
            'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": '
            '{"type": "text_delta", "text": "scripted answer ca259a6916f2"}}',
            'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}',
            'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
            '"usage": {"output_tokens": 412}}',
            'event: message_stop\ndata: {"type": "message_stop"}',
        ),
    ),
    expected_response_cost_header=None,
)

CLAUDE_SONNET_5_STREAM_NO_USAGE_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_TEST_CASE,
    scenario="stream_no_usage",
    litellm_request={
        **CLAUDE_SONNET_5_TEST_CASE.litellm_request,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "b14b060d38cc summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    mock_provider_response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_$REQUEST_ID", '
            '"type": "message", "role": "assistant", "model": "claude-sonnet-5", "content": [], '
            '"stop_reason": null}}',
            'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": '
            '{"type": "text", "text": ""}}',
            'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": '
            '{"type": "text_delta", "text": "scripted answer b14b060d38cc"}}',
            'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}',
            'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}',
            'event: message_stop\ndata: {"type": "message_stop"}',
        ),
    ),
    expected_response_cost_header=None,
    expected_spend_log={
        **CLAUDE_SONNET_5_TEST_CASE.expected_spend_log,
        "spend": 0.000294,
        "prompt_tokens": 48,
        "completion_tokens": 10,
        "input_cost": 0.000144,
        "output_cost": 0.00015,
        "total_cost": 0.000294,
        "original_cost": 0.000294,
    },
)

CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    CLAUDE_SONNET_5_STREAM_TEST_CASE,
    CLAUDE_SONNET_5_STREAM_NO_USAGE_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_streaming_anthropic(case: CostTrackingTestCase, gateway: Gateway) -> None:
    assert_cost_tracking(case, gateway)
