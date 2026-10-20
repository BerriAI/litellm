"""fireworks_ai cost tracking cases, one CostTrackingTestCase literal per request shape (moved from cost_tracking_cases.json).

PARITY pairs the streamed case with its plain twin where both bill the same row."""

from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    ExactExpected,
    JsonResponse,
    RecountExpected,
    RecountRates,
    SseResponse,
)
from integration.cost_calculation.stream_parity.case import StreamParityTestCase, sse_frames


# fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash
FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_INPUT_TEXT: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "10dc41a37bf4 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788233,
            "model": "accounts/fireworks/models/deepseek-v4p1-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 10dc41a37bf4"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0005232, input_cost=0.000276, output_cost=0.0002472, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_FALLBACK_CACHE_READ_AT_HALF_INPUT_RATE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-fallback_cache_read_at_half_input_rate",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "ee35d47aaab5 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788234,
            "model": "accounts/fireworks/models/deepseek-v4p1-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer ee35d47aaab5"},
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
    expected=ExactExpected(
        spend=0.0012456,
        input_cost=0.0010176,
        output_cost=0.000228,
        cache_read_cost=0.0009216,
        prompt_tokens=12928,
        completion_tokens=380,
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "f77cb314f5aa summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer f77cb314f5aa"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0005232, input_cost=0.000276, output_cost=0.0002472, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "c5fac079eac8 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer c5fac079eac8"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-07, output_cost_per_token=6e-07)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "eea156c013c8 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_$REQUEST_ID",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-07, output_cost_per_token=6e-07)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "124287c4bcaa summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,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",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788240,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788240,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 124287c4bcaa"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788240,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-07, output_cost_per_token=6e-07)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "4f7445b95bbd summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788242,
            "model": "accounts/fireworks/models/kimi-k3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 4f7445b95bbd"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.002134, input_cost=0.001104, output_cost=0.00103, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "8e888a093f6c summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788245,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788245,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 8e888a093f6c"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788245,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788245,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.002134, input_cost=0.001104, output_cost=0.00103, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "ef4a0046af51 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788246,
            "model": "accounts/fireworks/models/deepseek-v4p1-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_$REQUEST_ID",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0005232, input_cost=0.000276, output_cost=0.0002472, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "5ae7b84f3854 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788248,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788248,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_$REQUEST_ID",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788248,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788248,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788248,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788248,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788248,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0005232, input_cost=0.000276, output_cost=0.0002472, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-deepseek-v4p1-flash-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash",
    request={
        "model": "$MODEL",
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
                        "text": "19d356ecf08f summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788251,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788251,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 19d356ecf08f"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788251,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788251,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0005232, input_cost=0.000276, output_cost=0.0002472, prompt_tokens=1840, completion_tokens=412
    ),
)


# fireworks_ai/accounts/fireworks/models/kimi-k3
FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_INPUT_TEXT: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "a9341cd5b3ec summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788229,
            "model": "accounts/fireworks/models/kimi-k3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer a9341cd5b3ec"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.002134, input_cost=0.001104, output_cost=0.00103, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_CACHE_READ: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "f80f2a5e5bec summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788229,
            "model": "accounts/fireworks/models/kimi-k3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer f80f2a5e5bec"},
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
    expected=ExactExpected(
        spend=0.00207128, input_cost=0.00112128, output_cost=0.00095, prompt_tokens=12928, completion_tokens=380
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "a764db4a4844 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788231,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788231,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer a764db4a4844"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788231,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788231,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.002134, input_cost=0.001104, output_cost=0.00103, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "f168dea08a8c summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788232,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788232,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer f168dea08a8c"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788232,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=6e-07, output_cost_per_token=2.5e-06)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "5f4b9e007f6c summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788235,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788235,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_$REQUEST_ID",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788235,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788235,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788235,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788235,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=6e-07, output_cost_per_token=2.5e-06)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "c5d6768437a1 summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,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",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer c5d6768437a1"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788236,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=6e-07, output_cost_per_token=2.5e-06)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "e88240789ba9 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788236,
            "model": "accounts/fireworks/models/qwen3p8-max",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer e88240789ba9"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0031392, input_cost=0.001656, output_cost=0.0014832, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "454606d6e5ae summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788239,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788239,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 454606d6e5ae"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788239,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788239,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0031392, input_cost=0.001656, output_cost=0.0014832, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "6655aac8edcd summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788240,
            "model": "accounts/fireworks/models/kimi-k3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_$REQUEST_ID",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.002134, input_cost=0.001104, output_cost=0.00103, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "cbcf2fb047bb summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788242,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788242,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_$REQUEST_ID",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788242,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788242,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788242,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788242,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788242,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.002134, input_cost=0.001104, output_cost=0.00103, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-kimi-k3-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/kimi-k3",
    request={
        "model": "$MODEL",
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
                        "text": "d540b1082db1 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788244,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788244,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer d540b1082db1"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788244,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788244,
                "model": "accounts/fireworks/models/kimi-k3",
                "choices": [],
                "usage": {
                    "prompt_tokens": 7984,
                    "completion_tokens": 412,
                    "total_tokens": 8396,
                    "prompt_tokens_details": {"cached_tokens": 6144},
                },
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.00250264, input_cost=0.00147264, output_cost=0.00103, prompt_tokens=7984, completion_tokens=412
    ),
)


# fireworks_ai/accounts/fireworks/models/qwen3p8-max
FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_INPUT_TEXT: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "037102bc4f02 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788246,
            "model": "accounts/fireworks/models/qwen3p8-max",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 037102bc4f02"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0031392, input_cost=0.001656, output_cost=0.0014832, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_CACHE_READ: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "246ab713a447 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788248,
            "model": "accounts/fireworks/models/qwen3p8-max",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 246ab713a447"},
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
    expected=ExactExpected(
        spend=0.00304992, input_cost=0.00168192, output_cost=0.001368, prompt_tokens=12928, completion_tokens=380
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "7fa6702c872d summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788250,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788250,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 7fa6702c872d"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788250,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788250,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0031392, input_cost=0.001656, output_cost=0.0014832, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "a52571ae25d8 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788228,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788228,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer a52571ae25d8"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788228,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=9e-07, output_cost_per_token=3.6e-06)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "d621057b8000 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788229,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788229,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_$REQUEST_ID",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788229,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788229,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788229,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788229,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=9e-07, output_cost_per_token=3.6e-06)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "e5bcee3da31d summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAMK0lEQVR4nAEgDN/zAM0HLNi+b59irEwJwoIG5+NVlKprNC9dCjpeSEL6tCj3YubiguXBZXx4w6lns2cR6zkGp8hgPXHUCeelTYe9wfcEQgJ6rx+pW3+GWJV430PkExZ66Nnc6zd2KDOBGnGnIwBzhiZIL2HGI3lifMEk1EYYPG5NnqGlpcz3LiFAwwS9/Goh5egSF1aINdSX+yErhrSeZWrPBkEWmgtZ9OYpQ58l2dRlT+yNSBn7QNa6ssjgEhxEGuYUp7jZKpIZrx7Oh1QAV1jeeB7zT4/0jccZhxGSWqLiJW9VRPJQuRZjnL/J8qO8F7vpSKdYNMGDc/cEMXKNBlAdehNaVHGe84Tdmm53hcOfr0ICjvEPuk0WzuaDIOumjneMSZ1+6qg8mAM8quAXAOyQPrjRNxDXsUwZZhYr07VMCinT0OP4yIEWDKvlaxGgReVKAJ1JpZx/Hlt+evT70yo3G94iWEhVavFwPnqJ87rKl0BT6+ohtOgz197MvB8QzMXpME/BwepPYkiRLpbBOAD37hU9bAWozdK3sPczg3okfSudzcFjAYtEIq5yw+1ZF9EYmBSexEP+IhnvUWC4BeDWZQiCjhF7/4syzu4C5kF9ETfrG+udK0232R+NA+uESn014bRJmPMfahYljDojL1UAbuaA0PuOGOzBBlCKXAkFNHIfvvZHQafMkl9qmqdFF4x3Em6WDuijSQXN6nFtM3UXbUGmmCF4Rcyl4YhifPspUXLdXZMIvPo9zAhTSlQFEi8m83sw5KxL0rWBzS9c4XAIAOaz3py4dTb7d9QaqDMLk0J87/15PZKvEaC6/hZ62sCtuFTywatjViHrBXTgOO5IJsiyYuzmaeQJKHmr11gnixR2rO7lqNEGs3shT+xK/lDUucFkigvA+a5C+itk/VV+1gDxc420UHpKhj31j0YnCZSFJebGz2/XSTyT6XfZhG4XNwZGIeUGCPKt0DX9lpB0RNN0yiPzQVJfa3LkZpRBN3RGgRpYc8WpHn5Q1wWpmnklpPHACv/N+kGz0Ki86g2GgvsAWloXy5pwfFtwZRYV9fMGU4Za35yfj4cddJuHfAyHSpYHVlGhNknUVQIBV9gOq7wwLZU3Pl5GJgSy4EK7f75iRVKD/B0ItpC0FBpwODhWP184ymnLgLGkK90WIVUY7hZtAEOu39BF2OsPDWrBGWN0esj6v3clX2z22gaLmrJHiwE4sXWUC8TLLtFp4uiS/VlbojLP9uiewb/vrTLBiFjYJ5rsFjuu/3XxEuiZ1QYyi9sfsFqPoiXjQjCA/jibX4aA1AAfp3GT1lykHtVMJmSxoW4Xvn3BXhXUdtWhIwP7NDO1HRP9UAlEUgKbd/iJBWS20DFBJQb2/UN/+CpSWi97Rta3+pe3H4kOr3qaV+g1Udgmupu6/cxmQqMP+DXd77Sy6a0AMxTVBR0DU4sVW/Vs2qPfnh3r+xlqt/3VIhyKQknN6xF5RIg4j7xsEpjsnKV/XhJNmN2sWekxom9lUCkuP3qgD25S7oCG6ZV3CrkUCm48s5hw+dUZANcGs4H6/PxIrSpkAC77CDPFF5hCvkfKWzerhufLBkq7AhZgeMKRnNboaP3m9qMh6+9Z3JEyaV8rfUacshIsMqwS/BI0uL9v9+fwcMQrbdwOstrkyWCPG61dXIAoEb9t2HnSdSlxy6FXa5+LhwCvCy1ANKkBHgVSx5hoE+LrBX47caskZapZ+MAsTFJhA3VxvHgKaWiuX4/vaG7TbOZdX7GRSzPz3yOeM4JeAuLqwOy6T0OBIKbnSm5b3A5+Y2j2cNYMiVmogx89QCIPRicA936Dj6rC2bDKBi8DmXw8dZvR170ELj4UjqD+VRopML3ww7ILyoVYi5P150ekt4QioS95PZdaHcPXSAD0GwVZe1t0K1qy2TGc7V2ySUwrZKwEnPRbLXAcl6praPJ8aVbkAJ1MPaI09ZDaZOT+npRQ4SFv1DK3halvT/cYVWNFxZy/HEwXatv4MtSF+5ymHEWqFC/kYwCJMTaYtzI7Ma9Q1rJ1WZtVBP36KFFdSj1G8Bw5b5ssozn/uHJxFO9g937ZtQBQvxvgA4h8rApfcpEHs+HfrYkWalhdEwiJ+fpmF/Um3ywbq7MF3kWROuUQa4H2rcVhq4WpdGuBte/A+QvHrGkq45kCcmvaWhAAslxCV5CWs7QlXig29URyTAgPh//Ji+IAJXC9fNNMdeirO7iPEEOemj9zZ8FMiAQAYKRF4o8E9glP+Jx+Vw9xU5oM40/363XW5T+Gd4xtwwwQoR3P38Wd0uURAeDvp2P53Wz6zxpyRmpc2iAwN5BujDYD2v+lpIn2AMUaEqKkrYP6sRheFZ0EY9Yt7r25gt9pITZVoPxxTMUDlHfdZgyMFfPLKLOtUk3gaiv88FD1Wd4AdI2pNnmYqAMKjqK35Ys3wVuBmgAbUPH6k4ae0ku7/qyRrkGHwRewnAAVZQgZavvBMFJ8cB4Ndb2bXEI2phv40TLtxqffTyNrTfbyrEdKKESwv/h/+pkLpi4bcaUZx0fBeRewm9oUozrsfuGNZcn0rKC8DdMUtqqXBWClU0bAHPTp2GD2c1FNwcwAFOXWyuK3ox0nFYLckD+pQW3ZJyKBKlCnanFXhVDRONHgupmm89QapXZ32FhAzncd/Xcy34ljiSee3UVHJtnq2YJJwQQVBNQKjuhoCv0YqzT/zVWuTbF5QtRm8I7cC5FQANj9TfJqwOxdaoY98K6RRZGzAezoRsF02Q3PwAowC6vYqUzHv85kEfb7BYqzoXGLm4fLxafznJZ+JhJdtsmV5qQiAxjnFQdrdTtL4KNZCXlvtdVYXxLGFN9os7WJoU2kLQB5RTU1rvAlYHfbTZZTTYFKXxRHAizqcSNldKkms7p4RYVAWlUXIGdWTb4kym3qAF4clO+3Njf9F3zxl1b73YfH8pXbnk9fIQnHRowK9vQFRcfD8ikeIUAmZrhe+3cOXZUAt7EeSjZgZFxWFhFll/ZC/Y93aYwMJjAhvbgcS/zZaR0ucmJ663gDu7xgXT3tgQrvb4daDFM10U5HZ/kt2r9m26me5oXdgom65hEkEgfTe03ZFzZnPz7n1fzuGVThCppMAAAyK2pIlNAb40xyGcWNko8VyJ3oghsue2leWHkJ6UpWCdZB13x+QcxkLOr5a8X0zx6N+VcXwNMfGGqlelK0sh1MrRizk4drf0prMW3Q264J3IVJTZ3tx1ULpBi8CnnmrwBeYvfkI5rZJ2W6cOv3bKMqoCpyoNo+gpCZcSVgB8hdzlfMfPl67dL9/IqNo2mBQBaVitoQ4y7KzqHnrBWSxLIyznP3v/VNlzgF7qcPaQiDYhyGAgLfihsZE5nQZLcedY4Apm3dhEOyMUpdzNNyViCP6ykfFn35Ss3JDUS7lfZV47bpxeatoe2dNzU5TGxY9GoVYvhKXmP0WjgOpwv91XPF/x1ty0JRgNGVMwnwxk4PpSpDgGRu/k4jMev9x1vlc3EdAIz+WBCWU23SZoIT3WfyVNpXi7azEQkSB7e1K+fS25uGiNScc4/Nn8BVsUqtwtGhM5nncB7tH0F/k0ngZeGmqEGNJw01v2UYpLQoxwqhCz7/X5it8liVJMxk2kQ3iCGYnwDwfHcl7gLJGiAqrjLBOlp17bPGYIGTlFw+TOCLsGS103TZvSbyptAazsca/TY3o5hZLqX7SBvnfL6s/SU7BVyQY4PpZwHLa6PI2w+uFmJZbBsaruCVPW+FL+zSfvPO560ABJLD6nKsReNBwoxShnNi1XPSmX+C7Ij/ik2nfzubk6IFlomrI0bs386ztTMDifWKNO0ZnC+Oau6ZgwE4PwF8AqKtGFd5yqG0aMFq9rvwPb8DZCXmUHN5Uc40cNG5HWNSAK1yFAkCcG2GPV/7JYEi3t9ZA8NBeh9Dmvsjv8jiaeknPCij14x6BmGOFzyVNqRcS8x517fM39S0cC6bzu8wbnh96fwQ9z/JzCEsqxV9KjuES+xv3rYkX+r3JxcK7p78EACPip85gRBoIU7cZgrwLepMArrpA9YvR8Icag3YILzWcgXVoVRTs6Xc2PsiKasZ98y5CBkmP9Ri3lwZ9acgvPJ1fw729yDzDl/0qEeBOf7ZYZ7dreaPeaGHIymfgKAwmwhnjvAGaqLWhgAAAABJRU5ErkJggg==",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788231,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788231,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer e5bcee3da31d"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788231,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=9e-07, output_cost_per_token=3.6e-06)),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "eeadd4cae922 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788233,
            "model": "accounts/fireworks/models/deepseek-v4p1-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer eeadd4cae922"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0005232, input_cost=0.000276, output_cost=0.0002472, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "8ed9cad57fc4 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788234,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788234,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 8ed9cad57fc4"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788234,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788234,
                "model": "accounts/fireworks/models/deepseek-v4p1-flash",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0005232, input_cost=0.000276, output_cost=0.0002472, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "96d301af3055 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788236,
            "model": "accounts/fireworks/models/qwen3p8-max",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_$REQUEST_ID",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0031392, input_cost=0.001656, output_cost=0.0014832, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "389fe82a3e30 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "allowed_openai_params": ["tool_choice"],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_$REQUEST_ID",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788238,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0031392, input_cost=0.001656, output_cost=0.0014832, prompt_tokens=1840, completion_tokens=412
    ),
)

FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="fireworks_ai-accounts-fireworks-models-qwen3p8-max-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="fireworks_ai/accounts/fireworks/models/qwen3p8-max",
    request={
        "model": "$MODEL",
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
                        "text": "d3a53c5889e6 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788241,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788241,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer d3a53c5889e6"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788241,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788241,
                "model": "accounts/fireworks/models/qwen3p8-max",
                "choices": [],
                "usage": {
                    "prompt_tokens": 7984,
                    "completion_tokens": 412,
                    "total_tokens": 8396,
                    "prompt_tokens_details": {"cached_tokens": 6144},
                },
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.00369216, input_cost=0.00220896, output_cost=0.0014832, prompt_tokens=7984, completion_tokens=412
    ),
)


# fireworks_ai/fireworks-embed-v1
FIREWORKS_EMBEDDINGS_V1: Final = CostTrackingTestCase(
    name="fireworks-embeddings-v1",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="fireworks_ai/fireworks-embed-v1",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": "fireworks embedding"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "fireworks-embed-v1",
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        },
    ),
    expected=ExactExpected(spend=7.7e-06, input_cost=7.7e-06, output_cost=0.0, prompt_tokens=7, completion_tokens=0),
)


CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_INPUT_TEXT,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_FALLBACK_CACHE_READ_AT_HALF_INPUT_RATE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_NO_USAGE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_NO_USAGE_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_NO_USAGE_IMAGE_INPUT,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_RESPONSE_MODEL_OVERRIDE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_RESPONSE_MODEL_OVERRIDE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_FULL_USAGE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_INPUT_TEXT,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_CACHE_READ,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_NO_USAGE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_NO_USAGE_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_NO_USAGE_IMAGE_INPUT,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_RESPONSE_MODEL_OVERRIDE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_RESPONSE_MODEL_OVERRIDE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_FULL_USAGE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_INPUT_TEXT,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_CACHE_READ,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_NO_USAGE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_NO_USAGE_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_NO_USAGE_IMAGE_INPUT,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_RESPONSE_MODEL_OVERRIDE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_RESPONSE_MODEL_OVERRIDE,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_TOOL_CALL,
    FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_FULL_USAGE,
    FIREWORKS_EMBEDDINGS_V1,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_INPUT_TEXT,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_RESPONSE_MODEL_OVERRIDE,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_RESPONSE_MODEL_OVERRIDE,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_TOOL_CALL,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_TOOL_CALL,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_INPUT_TEXT,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_DEEPSEEK_V4P1_FLASH_STREAM_FULL_USAGE,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_INPUT_TEXT,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_RESPONSE_MODEL_OVERRIDE,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_RESPONSE_MODEL_OVERRIDE,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_TOOL_CALL,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_KIMI_K3_STREAM_TOOL_CALL,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_INPUT_TEXT,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_RESPONSE_MODEL_OVERRIDE,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_RESPONSE_MODEL_OVERRIDE,
    ),
    StreamParityTestCase(
        plain=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_TOOL_CALL,
        streamed=FIREWORKS_AI_ACCOUNTS_FIREWORKS_MODELS_QWEN3P8_MAX_STREAM_TOOL_CALL,
    ),
)
