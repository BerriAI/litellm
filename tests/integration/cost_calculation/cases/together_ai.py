"""together_ai cost tracking cases, one CostTrackingTestCase literal per request shape (moved from cost_tracking_cases.json).

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


# together_ai/moonshotai/Kimi-K3
TOGETHER_AI_MOONSHOTAI_KIMI_K3_INPUT_TEXT: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "d6c7504381ab summarize the attached material in one line and name the city weather",
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
            "created": 1789788266,
            "model": "moonshotai/Kimi-K3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer d6c7504381ab"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0035374, input_cost=0.002116, output_cost=0.0014214, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "1fb11cd276fc summarize the attached material in one line and name the city weather",
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
                "created": 1789788266,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788266,
                "model": "moonshotai/Kimi-K3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 1fb11cd276fc"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788266,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788266,
                "model": "moonshotai/Kimi-K3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0035374, input_cost=0.002116, output_cost=0.0014214, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "316d5b71455c summarize the attached material in one line and name the city weather",
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
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 316d5b71455c"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.15e-06, output_cost_per_token=3.45e-06)),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "d522e5409f42 summarize the attached material in one line and name the city weather",
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
                "created": 1789788269,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788269,
                "model": "moonshotai/Kimi-K3",
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
                "created": 1789788269,
                "model": "moonshotai/Kimi-K3",
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
                "created": 1789788269,
                "model": "moonshotai/Kimi-K3",
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
                "created": 1789788269,
                "model": "moonshotai/Kimi-K3",
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
                "created": 1789788269,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.15e-06, output_cost_per_token=3.45e-06)),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "97f79b9004cf summarize the attached material in one line and name the city weather",
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
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 97f79b9004cf"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.15e-06, output_cost_per_token=3.45e-06)),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "10e55a5c4a81 summarize the attached material in one line and name the city weather",
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
            "created": 1789788268,
            "model": "zai-org/GLM-5.3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 10e55a5c4a81"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0019184, input_cost=0.001012, output_cost=0.0009064, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "598ed6ff4b9d summarize the attached material in one line and name the city weather",
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
                "created": 1789788269,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788269,
                "model": "zai-org/GLM-5.3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 598ed6ff4b9d"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788269,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788269,
                "model": "zai-org/GLM-5.3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0019184, input_cost=0.001012, output_cost=0.0009064, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_TOOL_CALL: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "761da386a9ae summarize the attached material in one line and name the city weather",
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
            "created": 1789788269,
            "model": "moonshotai/Kimi-K3",
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
        spend=0.0035374, input_cost=0.002116, output_cost=0.0014214, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "e33dced1d70c summarize the attached material in one line and name the city weather",
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
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
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
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
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
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
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
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
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
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0035374, input_cost=0.002116, output_cost=0.0014214, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="together_ai-moonshotai-Kimi-K3-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/moonshotai/Kimi-K3",
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
                        "text": "2e2f3465f331 summarize the attached material in one line and name the city weather",
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
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 2e2f3465f331"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788268,
                "model": "moonshotai/Kimi-K3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0035374, input_cost=0.002116, output_cost=0.0014214, prompt_tokens=1840, completion_tokens=412
    ),
)


# together_ai/zai-org/GLM-5.3
TOGETHER_AI_ZAI_ORG_GLM_5_3_INPUT_TEXT: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "5438abd6c548 summarize the attached material in one line and name the city weather",
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
            "created": 1789788271,
            "model": "zai-org/GLM-5.3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 5438abd6c548"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0019184, input_cost=0.001012, output_cost=0.0009064, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "25be31c2d005 summarize the attached material in one line and name the city weather",
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
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 25be31c2d005"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0019184, input_cost=0.001012, output_cost=0.0009064, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "3a906f4aa16d summarize the attached material in one line and name the city weather",
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
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 3a906f4aa16d"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5.5e-07, output_cost_per_token=2.2e-06)),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "1143ec257764 summarize the attached material in one line and name the city weather",
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
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
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
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
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
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
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
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
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
                "created": 1789788270,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5.5e-07, output_cost_per_token=2.2e-06)),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "79e942a4452d summarize the attached material in one line and name the city weather",
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
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 79e942a4452d"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5.5e-07, output_cost_per_token=2.2e-06)),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "29dffdfbd5fa summarize the attached material in one line and name the city weather",
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
            "created": 1789788271,
            "model": "moonshotai/Kimi-K3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 29dffdfbd5fa"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0035374, input_cost=0.002116, output_cost=0.0014214, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "5985ca98af28 summarize the attached material in one line and name the city weather",
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
                "created": 1789788270,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "moonshotai/Kimi-K3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 5985ca98af28"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "moonshotai/Kimi-K3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788270,
                "model": "moonshotai/Kimi-K3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0035374, input_cost=0.002116, output_cost=0.0014214, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_TOOL_CALL: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "60392e73043e summarize the attached material in one line and name the city weather",
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
            "created": 1789788270,
            "model": "zai-org/GLM-5.3",
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
        spend=0.0019184, input_cost=0.001012, output_cost=0.0009064, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "f2bebf9a77ac summarize the attached material in one line and name the city weather",
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
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
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
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
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
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
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
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
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
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788271,
                "model": "zai-org/GLM-5.3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0019184, input_cost=0.001012, output_cost=0.0009064, prompt_tokens=1840, completion_tokens=412
    ),
)

TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="together_ai-zai-org-GLM-5.3-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="together_ai/zai-org/GLM-5.3",
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
                        "text": "63a7c8ddf892 summarize the attached material in one line and name the city weather",
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
                "created": 1789788272,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788272,
                "model": "zai-org/GLM-5.3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 63a7c8ddf892"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788272,
                "model": "zai-org/GLM-5.3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788272,
                "model": "zai-org/GLM-5.3",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0019184, input_cost=0.001012, output_cost=0.0009064, prompt_tokens=1840, completion_tokens=412
    ),
)


# together_ai/together-embed-v1
TOGETHER_EMBEDDINGS_V1: Final = CostTrackingTestCase(
    name="together-embeddings-v1",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="together_ai/together-embed-v1",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": "together embedding"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "together-embed-v1",
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        },
    ),
    expected=ExactExpected(spend=7.63e-06, input_cost=7.63e-06, output_cost=0.0, prompt_tokens=7, completion_tokens=0),
)


# together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo
TOGETHER_COMPLETIONS_V1: Final = CostTrackingTestCase(
    name="together-completions-v1",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    endpoint="/v1/completions",
    request={"model": "$MODEL", "prompt": "together complete"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "cmpl-together-$REQUEST_ID",
            "object": "text_completion",
            "choices": [{"text": "done", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        },
    ),
    expected=ExactExpected(
        spend=1.908e-05, input_cost=1.0439999999999998e-05, output_cost=8.64e-06, prompt_tokens=9, completion_tokens=4
    ),
)


CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_INPUT_TEXT,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_NO_USAGE,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_NO_USAGE_TOOL_CALL,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_NO_USAGE_IMAGE_INPUT,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_RESPONSE_MODEL_OVERRIDE,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_RESPONSE_MODEL_OVERRIDE,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_TOOL_CALL,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_TOOL_CALL,
    TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_FULL_USAGE,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_INPUT_TEXT,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_NO_USAGE,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_NO_USAGE_TOOL_CALL,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_NO_USAGE_IMAGE_INPUT,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_RESPONSE_MODEL_OVERRIDE,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_RESPONSE_MODEL_OVERRIDE,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_TOOL_CALL,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_TOOL_CALL,
    TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_FULL_USAGE,
    TOGETHER_EMBEDDINGS_V1,
    TOGETHER_COMPLETIONS_V1,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(
        plain=TOGETHER_AI_MOONSHOTAI_KIMI_K3_INPUT_TEXT, streamed=TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM
    ),
    StreamParityTestCase(
        plain=TOGETHER_AI_MOONSHOTAI_KIMI_K3_RESPONSE_MODEL_OVERRIDE,
        streamed=TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_RESPONSE_MODEL_OVERRIDE,
    ),
    StreamParityTestCase(
        plain=TOGETHER_AI_MOONSHOTAI_KIMI_K3_TOOL_CALL, streamed=TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_TOOL_CALL
    ),
    StreamParityTestCase(
        plain=TOGETHER_AI_MOONSHOTAI_KIMI_K3_INPUT_TEXT, streamed=TOGETHER_AI_MOONSHOTAI_KIMI_K3_STREAM_FULL_USAGE
    ),
    StreamParityTestCase(plain=TOGETHER_AI_ZAI_ORG_GLM_5_3_INPUT_TEXT, streamed=TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM),
    StreamParityTestCase(
        plain=TOGETHER_AI_ZAI_ORG_GLM_5_3_RESPONSE_MODEL_OVERRIDE,
        streamed=TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_RESPONSE_MODEL_OVERRIDE,
    ),
    StreamParityTestCase(
        plain=TOGETHER_AI_ZAI_ORG_GLM_5_3_TOOL_CALL, streamed=TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_TOOL_CALL
    ),
    StreamParityTestCase(
        plain=TOGETHER_AI_ZAI_ORG_GLM_5_3_INPUT_TEXT, streamed=TOGETHER_AI_ZAI_ORG_GLM_5_3_STREAM_FULL_USAGE
    ),
)
