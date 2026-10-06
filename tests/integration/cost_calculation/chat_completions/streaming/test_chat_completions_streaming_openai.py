from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.chat_completions.bases.openai import GPT_5_6_TEST_CASE
from integration.cost_calculation.cost_tracking_case import SseResponse
from integration.cost_calculation.runner import assert_cost_tracking

GPT_5_6_STREAM_TEST_CASE: Final = replace(
    GPT_5_6_TEST_CASE,
    scenario="stream",
    litellm_request={
        **GPT_5_6_TEST_CASE.litellm_request,
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
                        "text": "0d4dc45197bd summarize the attached material in one line and name the city weather",
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
            'data: {"id": "chatcmpl-$REQUEST_ID", "object": "chat.completion.chunk", "created": 1789788262, '
            '"model": "gpt-5.6", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}], '
            '"usage": null}',
            'data: {"id": "chatcmpl-$REQUEST_ID", "object": "chat.completion.chunk", "created": 1789788262, '
            '"model": "gpt-5.6", "choices": [{"index": 0, "delta": {"role": "assistant", "content": '
            '"scripted answer 0d4dc45197bd"}, "finish_reason": null}], "usage": null}',
            'data: {"id": "chatcmpl-$REQUEST_ID", "object": "chat.completion.chunk", "created": 1789788262, '
            '"model": "gpt-5.6", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": null}',
            'data: {"id": "chatcmpl-$REQUEST_ID", "object": "chat.completion.chunk", "created": 1789788262, '
            '"model": "gpt-5.6", "choices": [], "usage": {"prompt_tokens": 1840, "completion_tokens": 412, '
            '"total_tokens": 2252}}',
            "data: [DONE]",
        ),
    ),
    expected_response_cost_header=None,
)

GPT_5_6_STREAM_NO_USAGE_TEST_CASE: Final = replace(
    GPT_5_6_TEST_CASE,
    scenario="stream_no_usage",
    litellm_request={
        **GPT_5_6_TEST_CASE.litellm_request,
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
                        "text": "d2437c6d35d6 summarize the attached material in one line and name the city weather",
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
            'data: {"id": "chatcmpl-$REQUEST_ID", "object": "chat.completion.chunk", "created": 1789788263, '
            '"model": "gpt-5.6", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}], '
            '"usage": null}',
            'data: {"id": "chatcmpl-$REQUEST_ID", "object": "chat.completion.chunk", "created": 1789788263, '
            '"model": "gpt-5.6", "choices": [{"index": 0, "delta": {"role": "assistant", "content": '
            '"scripted answer d2437c6d35d6"}, "finish_reason": null}], "usage": null}',
            'data: {"id": "chatcmpl-$REQUEST_ID", "object": "chat.completion.chunk", "created": 1789788263, '
            '"model": "gpt-5.6", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": null}',
            "data: [DONE]",
        ),
    ),
    expected_response_cost_header=None,
    expected_spend_log={
        **GPT_5_6_TEST_CASE.expected_spend_log,
        "spend": 0.00025375,
        "prompt_tokens": 49,
        "completion_tokens": 12,
        "input_cost": 0.00008575,
        "output_cost": 0.000168,
        "total_cost": 0.00025375,
        "original_cost": 0.00025375,
    },
)

CASES: Final[tuple[CostTrackingTestCase, ...]] = (GPT_5_6_STREAM_TEST_CASE, GPT_5_6_STREAM_NO_USAGE_TEST_CASE)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_streaming_openai(case: CostTrackingTestCase, gateway: Gateway) -> None:
    assert_cost_tracking(case, gateway)
