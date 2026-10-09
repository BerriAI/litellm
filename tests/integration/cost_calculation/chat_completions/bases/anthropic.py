from typing import Final

from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.cost_tracking_case import JsonResponse

CLAUDE_SONNET_5_SYSTEM_CONTENT_BLOCK: Final = {
    "type": "text",
    "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
}
CLAUDE_SONNET_5_USER_MESSAGE: Final = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "e672859760ae summarize the attached material in one line and name the city weather",
        }
    ],
}

CLAUDE_SONNET_5_TEST_CASE: Final = CostTrackingTestCase(
    scenario="basic",
    deployment={"model": "anthropic/claude-sonnet-5", "api_key": "sk-scripted-provider"},
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "messages": [
            {"role": "system", "content": [CLAUDE_SONNET_5_SYSTEM_CONTENT_BLOCK]},
            CLAUDE_SONNET_5_USER_MESSAGE,
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    mock_provider_response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$UNIQUE_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer e672859760ae"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected_response_cost_header=0.0117,
    expected_spend_log={
        "spend": 0.0117,
        "prompt_tokens": 1840,
        "completion_tokens": 412,
        "input_cost": 0.00552,
        "output_cost": 0.00618,
        "total_cost": 0.0117,
        "service_tier": None,
        "original_cost": 0.0117,
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
