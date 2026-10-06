from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.chat_completions.bases.openai import GPT_5_6_TEST_CASE
from integration.cost_calculation.cost_tracking_case import JsonResponse
from integration.cost_calculation.runner import assert_cost_tracking

GPT_5_6_CACHE_READ_TEST_CASE: Final = replace(
    GPT_5_6_TEST_CASE,
    scenario="cache_read",
    mock_provider_response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788263,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer ed318a18ec07"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12928,
                "completion_tokens": 380,
                "total_tokens": 13308,
                "prompt_tokens_details": {"cached_tokens": 12288},
            },
        },
    ),
    expected_response_cost_header=0.0085904,
    expected_spend_log={
        **GPT_5_6_TEST_CASE.expected_spend_log,
        "spend": 0.0085904,
        "prompt_tokens": 12928,
        "completion_tokens": 380,
        "input_cost": 0.0032704,
        "output_cost": 0.00532,
        "total_cost": 0.0085904,
        "cache_read_cost": 0.0021504,
        "original_cost": 0.0085904,
    },
)

CASES: Final[tuple[CostTrackingTestCase, ...]] = (GPT_5_6_CACHE_READ_TEST_CASE,)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_cache_read_openai(case: CostTrackingTestCase, gateway: Gateway) -> None:
    assert_cost_tracking(case, gateway)
