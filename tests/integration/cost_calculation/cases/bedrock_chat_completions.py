"""Bedrock runtime /openai/v1/chat/completions cost tracking cases (xAI grok-4.7 and OpenAI gpt-6-sol on Bedrock), the rows LiteLLM routes to the OpenAI-compatible Bedrock endpoint instead of Converse."""

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


BEDROCK_US_XAI_GROK_4_7_INPUT_TEXT: Final = CostTrackingTestCase(
    name="bedrock-us.xai.grok-4.7-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.xai.grok-4.7",
    deployment=Deployment(model="bedrock/us.xai.grok-4.7"),
    endpoint="/v1/chat/completions",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "cost tracking fixture"}], "max_tokens": 412},
    response=JsonResponse(
        content_type="application/json",
        body={
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "annotations": [],
                        "content": "scripted response",
                        "refusal": None,
                        "role": "assistant",
                    },
                }
            ],
            "created": 1791861646,
            "id": "chatcmpl-$REQUEST_ID",
            "model": "us.xai.grok-4.7",
            "object": "chat.completion",
            "service_tier": "default",
            "usage": {
                "prompt_tokens": 1840,
                "completion_tokens": 412,
                "total_tokens": 2252,
                "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
            "system_fingerprint": None,
        },
    ),
    expected=ExactExpected(
        spend=0.0069512, input_cost=0.004232, output_cost=0.0027192, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_XAI_GROK_4_7_STREAM: Final = CostTrackingTestCase(
    name="bedrock-us.xai.grok-4.7-stream",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.xai.grok-4.7",
    deployment=Deployment(model="bedrock/us.xai.grok-4.7"),
    endpoint="/v1/chat/completions",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "cost tracking fixture"}],
        "max_tokens": 412,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "choices": [
                    {"delta": {"content": "", "role": "assistant", "refusal": None}, "finish_reason": None, "index": 0}
                ],
                "created": 1791861761,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.xai.grok-4.7",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "obfuscation": "abcdefghijklmnopq",
            },
            {
                "choices": [{"delta": {"content": "Hello"}, "finish_reason": None, "index": 0}],
                "created": 1791861761,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.xai.grok-4.7",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "obfuscation": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
            },
            {
                "choices": [{"delta": {"content": "."}, "finish_reason": None, "index": 0}],
                "created": 1791861761,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.xai.grok-4.7",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "obfuscation": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "created": 1791861761,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.xai.grok-4.7",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "obfuscation": "abcdefghijklmnopqrstu",
            },
            {
                "choices": [],
                "created": 1791861761,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.xai.grok-4.7",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "usage": {
                    "prompt_tokens": 1840,
                    "completion_tokens": 412,
                    "total_tokens": 2252,
                    "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                },
                "system_fingerprint": None,
                "obfuscation": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01",
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0069512, input_cost=0.004232, output_cost=0.0027192, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_XAI_GROK_4_7_TOOL_CALL: Final = CostTrackingTestCase(
    name="bedrock-us.xai.grok-4.7-tool_call",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.xai.grok-4.7",
    deployment=Deployment(model="bedrock/us.xai.grok-4.7"),
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
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "annotations": [],
                        "content": None,
                        "refusal": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": '{"city":"Berlin","days":7,"units":"metric"}',
                                    "name": "get_weather",
                                },
                                "id": "call_0",
                                "type": "function",
                            }
                        ],
                    },
                }
            ],
            "created": 1791861763,
            "id": "chatcmpl-$REQUEST_ID",
            "model": "us.xai.grok-4.7",
            "object": "chat.completion",
            "service_tier": "default",
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
            "system_fingerprint": None,
        },
    ),
    expected=ExactExpected(
        spend=0.0069512, input_cost=0.004232, output_cost=0.0027192, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_XAI_GROK_4_7_REASONING: Final = CostTrackingTestCase(
    name="bedrock-us.xai.grok-4.7-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.xai.grok-4.7",
    deployment=Deployment(model="bedrock/us.xai.grok-4.7"),
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
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"annotations": [], "content": "757", "refusal": None, "role": "assistant"},
                }
            ],
            "created": 1791861765,
            "id": "chatcmpl-$REQUEST_ID",
            "model": "us.xai.grok-4.7",
            "object": "chat.completion",
            "service_tier": "default",
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
            "system_fingerprint": None,
        },
    ),
    expected=ExactExpected(
        spend=0.0069512, input_cost=0.004232, output_cost=0.0027192, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_XAI_GROK_4_7_CACHE_READ: Final = with_usage(
    BEDROCK_US_XAI_GROK_4_7_INPUT_TEXT,
    "bedrock-us.xai.grok-4.7-cache_read",
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
    ExactExpected(
        spend=0.0058312, input_cost=0.003112, output_cost=0.0027192, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_OPENAI_GPT_6_SOL_INPUT_TEXT: Final = CostTrackingTestCase(
    name="bedrock-us.openai.gpt-6-sol-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.openai.gpt-6-sol",
    deployment=Deployment(model="bedrock/us.openai.gpt-6-sol"),
    endpoint="/v1/chat/completions",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "cost tracking fixture"}], "max_tokens": 412},
    response=JsonResponse(
        content_type="application/json",
        body={
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"annotations": [], "content": "Hello.", "refusal": None, "role": "assistant"},
                }
            ],
            "created": 1791861651,
            "id": "chatcmpl-$REQUEST_ID",
            "model": "us.openai.gpt-6-sol",
            "object": "chat.completion",
            "service_tier": "default",
            "usage": {
                "prompt_tokens": 1840,
                "completion_tokens": 412,
                "total_tokens": 2252,
                "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
            "system_fingerprint": None,
        },
    ),
    expected=ExactExpected(
        spend=0.00858, input_cost=0.004048, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_OPENAI_GPT_6_SOL_STREAM: Final = CostTrackingTestCase(
    name="bedrock-us.openai.gpt-6-sol-stream",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.openai.gpt-6-sol",
    deployment=Deployment(model="bedrock/us.openai.gpt-6-sol"),
    endpoint="/v1/chat/completions",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "cost tracking fixture"}],
        "max_tokens": 412,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "choices": [
                    {"delta": {"content": "", "role": "assistant", "refusal": None}, "finish_reason": None, "index": 0}
                ],
                "created": 1791861768,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.openai.gpt-6-sol",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "obfuscation": "abcdefghijklmnopqrstuvwxyzAB",
            },
            {
                "choices": [{"delta": {"content": "Hello"}, "finish_reason": None, "index": 0}],
                "created": 1791861768,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.openai.gpt-6-sol",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "obfuscation": "abcd",
            },
            {
                "choices": [{"delta": {"content": "."}, "finish_reason": None, "index": 0}],
                "created": 1791861768,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.openai.gpt-6-sol",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "obfuscation": "abcdefghijklmnopqr",
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "created": 1791861768,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.openai.gpt-6-sol",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "obfuscation": "abcdefghijklmnopqrstuvwxyzABCDEFGHI",
            },
            {
                "choices": [],
                "created": 1791861768,
                "id": "chatcmpl-$REQUEST_ID",
                "model": "us.openai.gpt-6-sol",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "usage": {
                    "prompt_tokens": 1840,
                    "completion_tokens": 412,
                    "total_tokens": 2252,
                    "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                },
                "system_fingerprint": None,
                "obfuscation": "abcdefghijk",
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.00858, input_cost=0.004048, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_OPENAI_GPT_6_SOL_REASONING: Final = CostTrackingTestCase(
    name="bedrock-us.openai.gpt-6-sol-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.openai.gpt-6-sol",
    deployment=Deployment(model="bedrock/us.openai.gpt-6-sol"),
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
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"annotations": [], "content": "757", "refusal": None, "role": "assistant"},
                }
            ],
            "created": 1791861770,
            "id": "chatcmpl-$REQUEST_ID",
            "model": "us.openai.gpt-6-sol",
            "object": "chat.completion",
            "service_tier": "default",
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
            "system_fingerprint": None,
        },
    ),
    expected=ExactExpected(
        spend=0.00858, input_cost=0.004048, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_OPENAI_GPT_6_SOL_CACHE_READ: Final = with_usage(
    BEDROCK_US_OPENAI_GPT_6_SOL_INPUT_TEXT,
    "bedrock-us.openai.gpt-6-sol-cache_read",
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
    ExactExpected(
        spend=0.0073128, input_cost=0.0027808, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_US_OPENAI_GPT_6_SOL_TIERED_INPUT_ABOVE_272K: Final = with_usage(
    BEDROCK_US_OPENAI_GPT_6_SOL_INPUT_TEXT,
    "bedrock-us.openai.gpt-6-sol-tiered_input_above_272k",
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
    ExactExpected(spend=1.326798, input_cost=1.32, output_cost=0.006798, prompt_tokens=300000, completion_tokens=412),
)

BEDROCK_US_OPENAI_GPT_6_SOL_TIERED_CACHE_READ_ABOVE_272K: Final = with_usage(
    BEDROCK_US_OPENAI_GPT_6_SOL_INPUT_TEXT,
    "bedrock-us.openai.gpt-6-sol-tiered_cache_read_above_272k",
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
    ExactExpected(spend=0.534798, input_cost=0.528, output_cost=0.006798, prompt_tokens=300000, completion_tokens=412),
)

BEDROCK_US_OPENAI_GPT_6_SOL_TIERED_AT_272K_BOUNDARY: Final = with_usage(
    BEDROCK_US_OPENAI_GPT_6_SOL_INPUT_TEXT,
    "bedrock-us.openai.gpt-6-sol-tiered_at_272k_boundary",
    {
        "prompt_tokens": 272000,
        "completion_tokens": 412,
        "total_tokens": 272412,
        "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    ExactExpected(spend=0.602932, input_cost=0.5984, output_cost=0.004532, prompt_tokens=272000, completion_tokens=412),
)

CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    BEDROCK_US_XAI_GROK_4_7_INPUT_TEXT,
    BEDROCK_US_XAI_GROK_4_7_STREAM,
    BEDROCK_US_XAI_GROK_4_7_TOOL_CALL,
    BEDROCK_US_XAI_GROK_4_7_REASONING,
    BEDROCK_US_XAI_GROK_4_7_CACHE_READ,
    BEDROCK_US_OPENAI_GPT_6_SOL_INPUT_TEXT,
    BEDROCK_US_OPENAI_GPT_6_SOL_STREAM,
    BEDROCK_US_OPENAI_GPT_6_SOL_REASONING,
    BEDROCK_US_OPENAI_GPT_6_SOL_CACHE_READ,
    BEDROCK_US_OPENAI_GPT_6_SOL_TIERED_INPUT_ABOVE_272K,
    BEDROCK_US_OPENAI_GPT_6_SOL_TIERED_CACHE_READ_ABOVE_272K,
    BEDROCK_US_OPENAI_GPT_6_SOL_TIERED_AT_272K_BOUNDARY,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(plain=BEDROCK_US_XAI_GROK_4_7_INPUT_TEXT, streamed=BEDROCK_US_XAI_GROK_4_7_STREAM),
    StreamParityTestCase(plain=BEDROCK_US_OPENAI_GPT_6_SOL_INPUT_TEXT, streamed=BEDROCK_US_OPENAI_GPT_6_SOL_STREAM),
)
