from typing import Final

from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.cost_tracking_case import JsonResponse

GPT_5_6_TEST_CASE: Final = CostTrackingTestCase(
    scenario="basic",
    deployment={"model": "openai/gpt-5.6", "api_key": "sk-scripted-provider"},
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
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
                        "text": "ed318a18ec07 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    mock_provider_response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$UNIQUE_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer ed318a18ec07"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected_response_cost_header=0.008988,
    expected_spend_log={
        "spend": 0.008988,
        "prompt_tokens": 1840,
        "completion_tokens": 412,
        "input_cost": 0.00322,
        "output_cost": 0.005768,
        "total_cost": 0.008988,
        "service_tier": None,
        "original_cost": 0.008988,
        "data_residency": None,
        "margin_percent": 0.0,
        "discount_amount": 0.0,
        "tool_usage_cost": 0.0,
        "vertex_location": None,
        "discount_percent": 0.0,
        "margin_fixed_amount": 0.0,
        "margin_total_amount": 0.0,
    },
)
