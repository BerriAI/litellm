from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.chat_completions.bases.anthropic import CLAUDE_SONNET_5_TEST_CASE
from integration.cost_calculation.cost_tracking_case import JsonResponse
from integration.cost_calculation.runner import assert_cost_tracking

CLAUDE_SONNET_5_CACHE_READ_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_TEST_CASE,
    scenario="cache_read",
    litellm_request={
        **CLAUDE_SONNET_5_TEST_CASE.litellm_request,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "68925ddd50c0 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
    },
    mock_provider_response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer 68925ddd50c0"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 640, "output_tokens": 380, "cache_read_input_tokens": 12288},
        },
    ),
    expected_response_cost_header=0.0113064,
    expected_spend_log={
        **CLAUDE_SONNET_5_TEST_CASE.expected_spend_log,
        "spend": 0.0113064,
        "prompt_tokens": 12928,
        "completion_tokens": 380,
        "input_cost": 0.0056064,
        "output_cost": 0.0057,
        "total_cost": 0.0113064,
        "cache_read_cost": 0.0036864,
        "original_cost": 0.0113064,
    },
)

CASES: Final[tuple[CostTrackingTestCase, ...]] = (CLAUDE_SONNET_5_CACHE_READ_TEST_CASE,)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_cache_read_anthropic(case: CostTrackingTestCase, gateway: Gateway) -> None:
    assert_cost_tracking(case, gateway)
