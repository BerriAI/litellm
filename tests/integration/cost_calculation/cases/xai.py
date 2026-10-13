"""xai cost tracking cases for grok-4.7 (OpenAI-compatible chat JSON and SSE, inclusive 200k tier)."""

from __future__ import annotations

from typing import Final

from integration.cost_calculation.cases.openai import with_usage
from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    ExactExpected,
    JsonResponse,
    SseResponse,
)
from integration.cost_calculation.stream_parity.case import StreamParityTestCase, sse_frames
from integration.cost_calculation.cases.misc import XAI_GROK_4_7_INPUT_TEXT

XAI_GROK_4_7_TOOL_CALL: Final = CostTrackingTestCase(
    name="xai-grok-4.7-tool_call",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="xai/grok-4.7",
    endpoint="/v1/chat/completions",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "cost tracking fixture"}],
        "max_tokens": 412,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather forecast for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                            "units": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "$REQUEST_ID",
            "object": "chat.completion",
            "created": 1791861753,
            "model": "grok-4.7",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Fetching a 7-day weather forecast for Berlin in metric units.",
                        "tool_calls": [
                            {
                                "id": "call-f9881b6d-f559-48ba-afed-83e5a96c7d52-0",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Berlin","days":7,"units":"metric"}',
                                },
                                "type": "function",
                            }
                        ],
                        "refusal": None,
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 1840,
                "completion_tokens": 412,
                "total_tokens": 2252,
                "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 96,
                    "rejected_prediction_tokens": 0,
                },
            },
            "system_fingerprint": "fp_c60958d81f66f847",
            "service_tier": "default",
        },
    ),
    expected=ExactExpected(
        spend=0.006152, input_cost=0.00368, output_cost=0.002472, prompt_tokens=1840, completion_tokens=412
    ),
)

XAI_GROK_4_7_REASONING: Final = CostTrackingTestCase(
    name="xai-grok-4.7-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="xai/grok-4.7",
    endpoint="/v1/chat/completions",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "cost tracking fixture"}],
        "max_tokens": 412,
        "reasoning_effort": "high",
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "$REQUEST_ID",
            "object": "chat.completion",
            "created": 1791861757,
            "model": "grok-4.7",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "757",
                        "reasoning_content": "I need to calculate 17 times 23 and add the number of days "
                        "in a leap year.",
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1840,
                "completion_tokens": 412,
                "total_tokens": 2252,
                "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 300,
                    "rejected_prediction_tokens": 0,
                },
            },
            "system_fingerprint": "fp_c60958d81f66f847",
            "service_tier": "default",
        },
    ),
    expected=ExactExpected(
        spend=0.006152, input_cost=0.00368, output_cost=0.002472, prompt_tokens=1840, completion_tokens=412
    ),
)

XAI_GROK_4_7_CACHE_READ: Final = with_usage(
    XAI_GROK_4_7_INPUT_TEXT,
    "xai-grok-4.7-cache_read",
    {
        "prompt_tokens": 1840,
        "completion_tokens": 412,
        "total_tokens": 2252,
        "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 640},
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    ExactExpected(spend=0.005192, input_cost=0.00272, output_cost=0.002472, prompt_tokens=1840, completion_tokens=412),
)

XAI_GROK_4_7_TIERED_INPUT_ABOVE_200K: Final = with_usage(
    XAI_GROK_4_7_INPUT_TEXT,
    "xai-grok-4.7-tiered_input_above_200k",
    {
        "prompt_tokens": 300000,
        "completion_tokens": 412,
        "total_tokens": 300412,
        "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    ExactExpected(spend=1.204944, input_cost=1.2, output_cost=0.004944, prompt_tokens=300000, completion_tokens=412),
)

XAI_GROK_4_7_TIERED_CACHE_READ_ABOVE_200K: Final = with_usage(
    XAI_GROK_4_7_INPUT_TEXT,
    "xai-grok-4.7-tiered_cache_read_above_200k",
    {
        "prompt_tokens": 300000,
        "completion_tokens": 412,
        "total_tokens": 300412,
        "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 200000},
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    ExactExpected(spend=0.604944, input_cost=0.6, output_cost=0.004944, prompt_tokens=300000, completion_tokens=412),
)

XAI_GROK_4_7_TIERED_AT_200K_BOUNDARY: Final = with_usage(
    XAI_GROK_4_7_INPUT_TEXT,
    "xai-grok-4.7-tiered_at_200k_boundary",
    {
        "prompt_tokens": 200000,
        "completion_tokens": 412,
        "total_tokens": 200412,
        "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    ExactExpected(spend=0.804944, input_cost=0.8, output_cost=0.004944, prompt_tokens=200000, completion_tokens=412),
)

CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    XAI_GROK_4_7_TOOL_CALL,
    XAI_GROK_4_7_REASONING,
    XAI_GROK_4_7_CACHE_READ,
    XAI_GROK_4_7_TIERED_INPUT_ABOVE_200K,
    XAI_GROK_4_7_TIERED_CACHE_READ_ABOVE_200K,
    XAI_GROK_4_7_TIERED_AT_200K_BOUNDARY,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = ()
