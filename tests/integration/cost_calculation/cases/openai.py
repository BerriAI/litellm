"""openai cost tracking cases, one CostTrackingTestCase literal per request shape (moved from cost_tracking_cases.json).

PARITY pairs the streamed case with its plain twin where both bill the same row."""

from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    BinaryResponse,
    CostTrackingTestCase,
    Deployment,
    ExactExpected,
    FailureDetails,
    FailureExpected,
    JsonResponse,
    PngUpload,
    RecountExpected,
    RecountRates,
    SseResponse,
    WavUpload,
)
from integration.cost_calculation.stream_parity.case import StreamParityTestCase, sse_frames


# gpt-5.3-codex
GPT_5_3_CODEX_INPUT_TEXT: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "0bb211ce54ec summarize the attached material in one line and name the city weather",
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
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 0bb211ce54ec", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.007704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_CACHE_READ: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "a1f465df7d59 summarize the attached material in one line and name the city weather",
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
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer a1f465df7d59", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 12928,
                "output_tokens": 380,
                "total_tokens": 13308,
                "input_tokens_details": {"cached_tokens": 12288},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.0073632, input_cost=0.0028032, output_cost=0.00456, prompt_tokens=12928, completion_tokens=380
    ),
)

GPT_5_3_CODEX_REASONING: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "d47bef2ddfda summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "reasoning_effort": "medium",
        "allowed_openai_params": ["reasoning_effort"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788254,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer d47bef2ddfda", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1240,
                "output_tokens": 4040,
                "total_tokens": 5280,
                "output_tokens_details": {"reasoning_tokens": 3480},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.05382, input_cost=0.00186, output_cost=0.05196, prompt_tokens=1240, completion_tokens=4040
    ),
)

GPT_5_3_CODEX_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "c6c8dc8b11ca summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "flex",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788255,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer c6c8dc8b11ca", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.003852, input_cost=0.00138, output_cost=0.002472, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "807b82ab682a summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "priority",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788256,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 807b82ab682a", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.015408, input_cost=0.00552, output_cost=0.009888, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "7c4724f24131 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "medium"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788256,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {"type": "web_search_call", "id": "ws_0", "status": "completed"},
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {"type": "web_search_call", "id": "ws_2", "status": "completed"},
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 7c4724f24131", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.045204, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_WEB_SEARCH_LOW: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-web_search_low",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "dfd08d79f164 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "low"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788257,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {"type": "web_search_call", "id": "ws_0", "status": "completed"},
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer dfd08d79f164", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.017704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_WEB_SEARCH_HIGH: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-web_search_high",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "401e61950557 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "high"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {"type": "web_search_call", "id": "ws_0", "status": "completed"},
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 401e61950557", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.022704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_FILE_SEARCH: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-file_search",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "8ebc31c05806 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_cost_calc_fixture"]}],
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "file_search_call",
                    "id": "fs_0",
                    "status": "completed",
                    "queries": ["query 0"],
                    "results": [],
                },
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 8ebc31c05806", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.010204,
        input_cost=0.00276,
        output_cost=0.004944,
        prompt_tokens=1840,
        completion_tokens=412,
        tool_usage_cost=0.0025,
    ),
)

GPT_5_3_CODEX_STREAM: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "75b82927ffe4 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788254,
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 75b82927ffe4", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 75b82927ffe4",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788254,
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 75b82927ffe4", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.007704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "430855aa14e3 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788255,
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 430855aa14e3", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 430855aa14e3",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788255,
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 430855aa14e3", "annotations": []}
                            ],
                        }
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-06, output_cost_per_token=1.2e-05)),
)

GPT_5_3_CODEX_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "5e8dac751b8d summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788255,
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_$REQUEST_ID",
                            "call_id": "call_$REQUEST_ID",
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                            "status": "completed",
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_$REQUEST_ID",
                    "call_id": "call_$REQUEST_ID",
                    "name": "get_weather",
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788255,
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_$REQUEST_ID",
                            "call_id": "call_$REQUEST_ID",
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                            "status": "completed",
                        }
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-06, output_cost_per_token=1.2e-05)),
)

GPT_5_3_CODEX_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "0245ffd5ae0f summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788256,
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 0245ffd5ae0f", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 0245ffd5ae0f",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788256,
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 0245ffd5ae0f", "annotations": []}
                            ],
                        }
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-06, output_cost_per_token=1.2e-05)),
)

GPT_5_3_CODEX_STREAM_INCOMPLETE: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_incomplete",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "15523b94e3fe summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788255,
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 15523b94e3fe", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 15523b94e3fe",
            },
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788255,
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 15523b94e3fe", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.007704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_STREAM_NO_USAGE_INCOMPLETE: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_no_usage_incomplete",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "6dcd71cdfaa5 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788255,
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 6dcd71cdfaa5", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 6dcd71cdfaa5",
            },
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788255,
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 6dcd71cdfaa5", "annotations": []}
                            ],
                        }
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-06, output_cost_per_token=1.2e-05)),
)

GPT_5_3_CODEX_STREAM_UNVALIDATED: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_unvalidated",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "2de88869bcff summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": "not-a-number",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {"type": "scripted_future_item", "id": "fut_$REQUEST_ID", "status": "completed"},
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 2de88869bcff", "annotations": []}
                            ],
                        },
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 1,
                "content_index": 0,
                "delta": "scripted answer 2de88869bcff",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": "not-a-number",
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {"type": "scripted_future_item", "id": "fut_$REQUEST_ID", "status": "completed"},
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 2de88869bcff", "annotations": []}
                            ],
                        },
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.007704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_STREAM_NO_USAGE_UNVALIDATED: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_no_usage_unvalidated",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "da22f5aa5869 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": "not-a-number",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {"type": "scripted_future_item", "id": "fut_$REQUEST_ID", "status": "completed"},
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer da22f5aa5869", "annotations": []}
                            ],
                        },
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 1,
                "content_index": 0,
                "delta": "scripted answer da22f5aa5869",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": "not-a-number",
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {"type": "scripted_future_item", "id": "fut_$REQUEST_ID", "status": "completed"},
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer da22f5aa5869", "annotations": []}
                            ],
                        },
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-06, output_cost_per_token=1.2e-05)),
)

GPT_5_3_CODEX_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "5210d175a94f summarize the attached material in one line and name the city weather",
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
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788258,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 5210d175a94f", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.07704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "b7edc51cdfbe summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788258,
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer b7edc51cdfbe", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer b7edc51cdfbe",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788258,
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer b7edc51cdfbe", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.07704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "6bf8aad8967c summarize the attached material in one line and name the city weather",
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
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788257,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_$REQUEST_ID",
                    "call_id": "call_$REQUEST_ID",
                    "name": "get_weather",
                    "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.007704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "01f141cd9d3b summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788258,
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_$REQUEST_ID",
                            "call_id": "call_$REQUEST_ID",
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                            "status": "completed",
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_$REQUEST_ID",
                    "call_id": "call_$REQUEST_ID",
                    "name": "get_weather",
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788258,
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_$REQUEST_ID",
                            "call_id": "call_$REQUEST_ID",
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.007704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_3_CODEX_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.3-codex",
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
                        "text": "8ef8970518fd summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788258,
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 8ef8970518fd", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 8ef8970518fd",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788258,
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 8ef8970518fd", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 7984,
                        "output_tokens": 1312,
                        "total_tokens": 9296,
                        "input_tokens_details": {"cached_tokens": 6144},
                        "output_tokens_details": {"reasoning_tokens": 900},
                    },
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.0203256, input_cost=0.0036816, output_cost=0.016644, prompt_tokens=7984, completion_tokens=1312
    ),
)

GPT_5_3_CODEX_RESPONSES_FILE_SEARCH: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-responses_file_search",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.3-codex",
    endpoint="/v1/responses",
    request={
        "model": "$MODEL",
        "input": "search files",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_scripted"]}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "file_search_call",
                    "id": "fs_$REQUEST_ID",
                    "status": "completed",
                    "queries": ["scripted query"],
                    "results": [],
                },
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 1840,
                "output_tokens": 412,
                "total_tokens": 2252,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.010204,
        input_cost=0.00276,
        output_cost=0.004944,
        prompt_tokens=1840,
        completion_tokens=412,
        tool_usage_cost=0.0025,
    ),
)


# gpt-5.4-mini
GPT_5_4_MINI_INPUT_TEXT: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "1157fc293d72 summarize the attached material in one line and name the city weather",
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
            "created": 1789788259,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 1157fc293d72"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0017976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_CACHE_READ: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "918d015fad34 summarize the attached material in one line and name the city weather",
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
            "created": 1789788260,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 918d015fad34"},
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
        spend=0.00171808, input_cost=0.00065408, output_cost=0.001064, prompt_tokens=12928, completion_tokens=380
    ),
)

GPT_5_4_MINI_AUDIO_INPUT: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-audio_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "b1238d45e42d summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "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",
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
        "stream": False,
        "modalities": ["text"],
        "allowed_openai_params": ["modalities"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788261,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer b1238d45e42d"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1546,
                "completion_tokens": 210,
                "total_tokens": 1756,
                "prompt_tokens_details": {"audio_tokens": 1450},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.0151216, input_cost=0.0145336, output_cost=0.000588, prompt_tokens=1546, completion_tokens=210
    ),
)

GPT_5_4_MINI_AUDIO_OUTPUT: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-audio_output",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "09073c011cb2 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "pcm16"},
        "allowed_openai_params": ["modalities", "audio"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788257,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 09073c011cb2"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 220,
                "completion_tokens": 1300,
                "total_tokens": 1520,
                "completion_tokens_details": {"audio_tokens": 1120},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.022981, input_cost=7.7e-05, output_cost=0.022904, prompt_tokens=220, completion_tokens=1300
    ),
)

GPT_5_4_MINI_REASONING: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "cfc4c1747119 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "reasoning_effort": "medium",
        "allowed_openai_params": ["reasoning_effort"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788258,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer cfc4c1747119"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1240,
                "completion_tokens": 4040,
                "total_tokens": 5280,
                "completion_tokens_details": {"reasoning_tokens": 3480},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.013138, input_cost=0.000434, output_cost=0.012704, prompt_tokens=1240, completion_tokens=4040
    ),
)

GPT_5_4_MINI_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "9bb4305a36a5 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "flex",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788259,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 9bb4305a36a5"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "flex",
        },
    ),
    expected=ExactExpected(
        spend=0.0008988, input_cost=0.000322, output_cost=0.0005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "4ebd6b6e27b7 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "priority",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788260,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 4ebd6b6e27b7"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "priority",
        },
    ),
    expected=ExactExpected(
        spend=0.0035952, input_cost=0.001288, output_cost=0.0023072, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "222c74ef3df5 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "medium"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788261,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 222c74ef3df5",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            },
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            },
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0142976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_WEB_SEARCH_LOW: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-web_search_low",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "75b4bbcdb164 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "low"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788258,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 75b4bbcdb164",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0117976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_WEB_SEARCH_HIGH: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-web_search_high",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "f2ded281685d summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "high"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788259,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer f2ded281685d",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0167976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_STREAM: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "85a0f6230523 summarize the attached material in one line and name the city weather",
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
                "created": 1789788260,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788260,
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 85a0f6230523"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788260,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788260,
                "model": "gpt-5.4-mini",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0017976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "8e85cbc8b78c summarize the attached material in one line and name the city weather",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788261,
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 8e85cbc8b78c"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788261,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.5e-07, output_cost_per_token=2.8e-06)),
)

GPT_5_4_MINI_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "24414e14870e summarize the attached material in one line and name the city weather",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788261,
                "model": "gpt-5.4-mini",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.5e-07, output_cost_per_token=2.8e-06)),
)

GPT_5_4_MINI_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "ddb683a1724a summarize the attached material in one line and name the city weather",
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
                "created": 1789788258,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788258,
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer ddb683a1724a"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788258,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.5e-07, output_cost_per_token=2.8e-06)),
)

GPT_5_4_MINI_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "dbcf34530ce5 summarize the attached material in one line and name the city weather",
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
            "created": 1789788258,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer dbcf34530ce5"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "ec3873b5f576 summarize the attached material in one line and name the city weather",
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
                "created": 1789788259,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788259,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer ec3873b5f576"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788259,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788259,
                "model": "gpt-5.6",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "454b9573dcf5 summarize the attached material in one line and name the city weather",
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
            "created": 1789788260,
            "model": "gpt-5.4-mini",
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
        spend=0.0017976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "c3e8188e02bf summarize the attached material in one line and name the city weather",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788261,
                "model": "gpt-5.4-mini",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
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
                "created": 1789788261,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788261,
                "model": "gpt-5.4-mini",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0017976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_4_MINI_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="gpt-5.4-mini-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.4-mini",
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
                        "text": "3efb75339951 summarize the attached material in one line and name the city weather",
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
                "created": 1789788258,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788258,
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 3efb75339951"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788258,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788258,
                "model": "gpt-5.4-mini",
                "choices": [],
                "usage": {
                    "prompt_tokens": 8314,
                    "completion_tokens": 1592,
                    "total_tokens": 9906,
                    "prompt_tokens_details": {"cached_tokens": 6144, "audio_tokens": 330},
                    "completion_tokens_details": {"reasoning_tokens": 900, "audio_tokens": 280},
                },
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.01379264, input_cost=0.00415904, output_cost=0.0096336, prompt_tokens=8314, completion_tokens=1592
    ),
)


# gpt-5.5-pro
GPT_5_5_PRO_INPUT_TEXT: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "eef4c5fe3dab summarize the attached material in one line and name the city weather",
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
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788259,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer eef4c5fe3dab", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.07704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_CACHE_READ: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "f6fd81220aad summarize the attached material in one line and name the city weather",
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
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788260,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer f6fd81220aad", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 12928,
                "output_tokens": 380,
                "total_tokens": 13308,
                "input_tokens_details": {"cached_tokens": 12288},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.073632, input_cost=0.028032, output_cost=0.0456, prompt_tokens=12928, completion_tokens=380
    ),
)

GPT_5_5_PRO_REASONING: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "09757dcdc501 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "reasoning_effort": "medium",
        "allowed_openai_params": ["reasoning_effort"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788260,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 09757dcdc501", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1240,
                "output_tokens": 4040,
                "total_tokens": 5280,
                "output_tokens_details": {"reasoning_tokens": 3480},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.5382, input_cost=0.0186, output_cost=0.5196, prompt_tokens=1240, completion_tokens=4040
    ),
)

GPT_5_5_PRO_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "e21acaffe79b summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "flex",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788258,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer e21acaffe79b", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.03852, input_cost=0.0138, output_cost=0.02472, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "7fee6f8e184f summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "priority",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788259,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 7fee6f8e184f", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.15408, input_cost=0.0552, output_cost=0.09888, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "d04d4797f3d0 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "medium"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788260,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {"type": "web_search_call", "id": "ws_0", "status": "completed"},
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {"type": "web_search_call", "id": "ws_2", "status": "completed"},
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer d04d4797f3d0", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.11454, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_WEB_SEARCH_LOW: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-web_search_low",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "3ce53f3d07ab summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "low"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788261,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {"type": "web_search_call", "id": "ws_0", "status": "completed"},
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 3ce53f3d07ab", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.08704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_WEB_SEARCH_HIGH: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-web_search_high",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "d084299afdbf summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "high"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788259,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {"type": "web_search_call", "id": "ws_0", "status": "completed"},
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer d084299afdbf", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.09204, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_FILE_SEARCH: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-file_search",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "0720f466abdc summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_cost_calc_fixture"]}],
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788260,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {
                    "type": "file_search_call",
                    "id": "fs_0",
                    "status": "completed",
                    "queries": ["query 0"],
                    "results": [],
                },
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer 0720f466abdc", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.07954, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_STREAM: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "1130d4d6e2dc summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788260,
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 1130d4d6e2dc", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 1130d4d6e2dc",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788260,
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 1130d4d6e2dc", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.07704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "62836f5d3fa3 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788261,
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 62836f5d3fa3", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 62836f5d3fa3",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788261,
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 62836f5d3fa3", "annotations": []}
                            ],
                        }
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-05, output_cost_per_token=0.00012)),
)

GPT_5_5_PRO_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "12478a1a276d summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788260,
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_$REQUEST_ID",
                            "call_id": "call_$REQUEST_ID",
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                            "status": "completed",
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_$REQUEST_ID",
                    "call_id": "call_$REQUEST_ID",
                    "name": "get_weather",
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788260,
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_$REQUEST_ID",
                            "call_id": "call_$REQUEST_ID",
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                            "status": "completed",
                        }
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-05, output_cost_per_token=0.00012)),
)

GPT_5_5_PRO_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "be189bbbfebe summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788261,
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer be189bbbfebe", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer be189bbbfebe",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788261,
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer be189bbbfebe", "annotations": []}
                            ],
                        }
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-05, output_cost_per_token=0.00012)),
)

GPT_5_5_PRO_STREAM_INCOMPLETE: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_incomplete",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "cad50498b33a summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788261,
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer cad50498b33a", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer cad50498b33a",
            },
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788261,
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer cad50498b33a", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.07704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_STREAM_NO_USAGE_INCOMPLETE: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_no_usage_incomplete",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "2f6f49c3d0f2 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788262,
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 2f6f49c3d0f2", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 2f6f49c3d0f2",
            },
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788262,
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 2f6f49c3d0f2", "annotations": []}
                            ],
                        }
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-05, output_cost_per_token=0.00012)),
)

GPT_5_5_PRO_STREAM_UNVALIDATED: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_unvalidated",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "9da2a01340b8 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": "not-a-number",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {"type": "scripted_future_item", "id": "fut_$REQUEST_ID", "status": "completed"},
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 9da2a01340b8", "annotations": []}
                            ],
                        },
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 1,
                "content_index": 0,
                "delta": "scripted answer 9da2a01340b8",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": "not-a-number",
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {"type": "scripted_future_item", "id": "fut_$REQUEST_ID", "status": "completed"},
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 9da2a01340b8", "annotations": []}
                            ],
                        },
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.07704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_STREAM_NO_USAGE_UNVALIDATED: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_no_usage_unvalidated",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "d162da290b52 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": "not-a-number",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {"type": "scripted_future_item", "id": "fut_$REQUEST_ID", "status": "completed"},
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer d162da290b52", "annotations": []}
                            ],
                        },
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 1,
                "content_index": 0,
                "delta": "scripted answer d162da290b52",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": "not-a-number",
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {"type": "scripted_future_item", "id": "fut_$REQUEST_ID", "status": "completed"},
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer d162da290b52", "annotations": []}
                            ],
                        },
                    ],
                },
            },
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.5e-05, output_cost_per_token=0.00012)),
)

GPT_5_5_PRO_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "d307e0210e1e summarize the attached material in one line and name the city weather",
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
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788262,
            "status": "completed",
            "model": "gpt-5.3-codex",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted answer d307e0210e1e", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.007704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "897338ee89fc summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788261,
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 897338ee89fc", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer 897338ee89fc",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788261,
                    "status": "completed",
                    "model": "gpt-5.3-codex",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer 897338ee89fc", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.007704, input_cost=0.00276, output_cost=0.004944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "17d97c0f8b6e summarize the attached material in one line and name the city weather",
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
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1789788262,
            "status": "completed",
            "model": "gpt-5.5-pro",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_$REQUEST_ID",
                    "call_id": "call_$REQUEST_ID",
                    "name": "get_weather",
                    "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.07704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "369677236d4b summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788263,
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_$REQUEST_ID",
                            "call_id": "call_$REQUEST_ID",
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                            "status": "completed",
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_$REQUEST_ID",
                    "call_id": "call_$REQUEST_ID",
                    "name": "get_weather",
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "delta": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_$REQUEST_ID",
                "output_index": 0,
                "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788263,
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_$REQUEST_ID",
                            "call_id": "call_$REQUEST_ID",
                            "name": "get_weather",
                            "arguments": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler "}',
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.07704, input_cost=0.0276, output_cost=0.04944, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_5_PRO_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="gpt-5.5-pro-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.5-pro",
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
                        "text": "c9d9cd92af28 summarize the attached material in one line and name the city weather",
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
                "type": "response.created",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788262,
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer c9d9cd92af28", "annotations": []}
                            ],
                        }
                    ],
                    "status": "in_progress",
                    "usage": None,
                },
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "content_index": 0,
                "delta": "scripted answer c9d9cd92af28",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "response",
                    "created_at": 1789788262,
                    "status": "completed",
                    "model": "gpt-5.5-pro",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_$REQUEST_ID",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "scripted answer c9d9cd92af28", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 7984,
                        "output_tokens": 1312,
                        "total_tokens": 9296,
                        "input_tokens_details": {"cached_tokens": 6144},
                        "output_tokens_details": {"reasoning_tokens": 900},
                    },
                },
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.203256, input_cost=0.036816, output_cost=0.16644, prompt_tokens=7984, completion_tokens=1312
    ),
)


# gpt-5.6
GPT_5_6_INPUT_TEXT: Final = CostTrackingTestCase(
    name="gpt-5.6-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "ed318a18ec07 summarize the attached material in one line and name the city weather",
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
    expected=ExactExpected(
        spend=0.008988,
        input_cost=0.00322,
        output_cost=0.005768,
        prompt_tokens=1840,
        completion_tokens=412,
        rollups=True,
    ),
)

GPT_5_6_CACHE_READ: Final = CostTrackingTestCase(
    name="gpt-5.6-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "2d376f5f39a0 summarize the attached material in one line and name the city weather",
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
            "created": 1789788263,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 2d376f5f39a0"},
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
        spend=0.0085904, input_cost=0.0032704, output_cost=0.00532, prompt_tokens=12928, completion_tokens=380
    ),
)

GPT_5_6_AUDIO_INPUT: Final = CostTrackingTestCase(
    name="gpt-5.6-audio_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "1eed63f65da0 summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "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",
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
        "stream": False,
        "modalities": ["text"],
        "allowed_openai_params": ["modalities"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 1eed63f65da0"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1546,
                "completion_tokens": 210,
                "total_tokens": 1756,
                "prompt_tokens_details": {"audio_tokens": 1450},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.061108, input_cost=0.058168, output_cost=0.00294, prompt_tokens=1546, completion_tokens=210
    ),
)

GPT_5_6_AUDIO_OUTPUT: Final = CostTrackingTestCase(
    name="gpt-5.6-audio_output",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "c2f69182025b summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "pcm16"},
        "allowed_openai_params": ["modalities", "audio"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer c2f69182025b"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 220,
                "completion_tokens": 1300,
                "total_tokens": 1520,
                "completion_tokens_details": {"audio_tokens": 1120},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.092505, input_cost=0.000385, output_cost=0.09212, prompt_tokens=220, completion_tokens=1300
    ),
)

GPT_5_6_REASONING: Final = CostTrackingTestCase(
    name="gpt-5.6-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "839418b0b1da summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "reasoning_effort": "medium",
        "allowed_openai_params": ["reasoning_effort"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788263,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 839418b0b1da"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1240,
                "completion_tokens": 4040,
                "total_tokens": 5280,
                "completion_tokens_details": {"reasoning_tokens": 3480},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.06569, input_cost=0.00217, output_cost=0.06352, prompt_tokens=1240, completion_tokens=4040
    ),
)

GPT_5_6_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="gpt-5.6-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "fa273468c07b summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "flex",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer fa273468c07b"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "flex",
        },
    ),
    expected=ExactExpected(
        spend=0.004494, input_cost=0.00161, output_cost=0.002884, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="gpt-5.6-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "dbb27812caea summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "priority",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer dbb27812caea"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "priority",
        },
    ),
    expected=ExactExpected(
        spend=0.017976, input_cost=0.00644, output_cost=0.011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="gpt-5.6-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "2fc2074db6f0 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "medium"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788263,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 2fc2074db6f0",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            },
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            },
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.021488, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_WEB_SEARCH_LOW: Final = CostTrackingTestCase(
    name="gpt-5.6-web_search_low",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "8a1c27e0ad34 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "low"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 8a1c27e0ad34",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.018988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_WEB_SEARCH_HIGH: Final = CostTrackingTestCase(
    name="gpt-5.6-web_search_high",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
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
                        "text": "14ebe654d39f summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "web_search_options": {"search_context_size": "high"},
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788263,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 14ebe654d39f",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "url": "https://scripted.example/source",
                                    "title": "scripted source",
                                    "start_index": 0,
                                    "end_index": 1,
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.023988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_STREAM: Final = CostTrackingTestCase(
    name="gpt-5.6-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "0d4dc45197bd summarize the attached material in one line and name the city weather",
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
                "created": 1789788262,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788262,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 0d4dc45197bd"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788262,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788262,
                "model": "gpt-5.6",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="gpt-5.6-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "d2437c6d35d6 summarize the attached material in one line and name the city weather",
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
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer d2437c6d35d6"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=1.75e-06, output_cost_per_token=1.4e-05),
        prompt_tokens=49,
        completion_tokens=12,
    ),
)

GPT_5_6_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.6-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "510682506548 summarize the attached material in one line and name the city weather",
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
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_fixture_0001",
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
                "created": 1789788263,
                "model": "gpt-5.6",
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
                "created": 1789788263,
                "model": "gpt-5.6",
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
                "created": 1789788263,
                "model": "gpt-5.6",
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
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=1.75e-06, output_cost_per_token=1.4e-05), min_completion_tokens=60
    ),
)

GPT_5_6_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="gpt-5.6-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "93ef594b4d91 summarize the attached material in one line and name the city weather",
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
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 93ef594b4d91"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=1.75e-06, output_cost_per_token=1.4e-05),
        prompt_tokens=302,
        completion_tokens=11,
    ),
)

GPT_5_6_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gpt-5.6-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "e6d045b77d68 summarize the attached material in one line and name the city weather",
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
            "created": 1789788263,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer e6d045b77d68"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0017976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gpt-5.6-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "49c74a898360 summarize the attached material in one line and name the city weather",
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
                "created": 1789788264,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788264,
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 49c74a898360"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788264,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788264,
                "model": "gpt-5.4-mini",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0017976, input_cost=0.000644, output_cost=0.0011536, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.6-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "8a209834c60c summarize the attached material in one line and name the city weather",
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
            "created": 1789788263,
            "model": "gpt-5.6",
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
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="gpt-5.6-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "77c2cb29e969 summarize the attached material in one line and name the city weather",
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
                "created": 1789788264,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788264,
                "model": "gpt-5.6",
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
                "created": 1789788264,
                "model": "gpt-5.6",
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
                "created": 1789788264,
                "model": "gpt-5.6",
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
                "created": 1789788264,
                "model": "gpt-5.6",
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
                "created": 1789788264,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788264,
                "model": "gpt-5.6",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="gpt-5.6-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
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
                        "text": "66e5a1e22691 summarize the attached material in one line and name the city weather",
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
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 66e5a1e22691"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788263,
                "model": "gpt-5.6",
                "choices": [],
                "usage": {
                    "prompt_tokens": 8314,
                    "completion_tokens": 1592,
                    "total_tokens": 9906,
                    "prompt_tokens_details": {"cached_tokens": 6144, "audio_tokens": 330},
                    "completion_tokens_details": {"reasoning_tokens": 900, "audio_tokens": 280},
                },
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0600632, input_cost=0.0174952, output_cost=0.042568, prompt_tokens=8314, completion_tokens=1592
    ),
)

GPT_5_6_RESPONSES_NATIVE_JSON: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_native_json",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "responses native fixture", "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "status": "completed",
            "created_at": 1700000000,
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.00011725, input_cost=1.925e-05, output_cost=9.8e-05, prompt_tokens=11, completion_tokens=7
    ),
)

GPT_5_6_UPSTREAM_500_ZERO_SPEND: Final = CostTrackingTestCase(
    name="gpt-5.6-upstream_500_zero_spend",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "scripted upstream failure 500"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={"error": {"message": "scripted upstream failure", "type": "server_error", "code": "500"}},
        status=500,
    ),
    expected=FailureExpected(failure=FailureDetails(status=500)),
)

GPT_5_6_UPSTREAM_429_ZERO_SPEND: Final = CostTrackingTestCase(
    name="gpt-5.6-upstream_429_zero_spend",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "scripted upstream failure 429"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={"error": {"message": "scripted upstream failure", "type": "rate_limit_error", "code": "429"}},
        status=429,
    ),
    expected=FailureExpected(failure=FailureDetails(status=429)),
)

GPT_5_6_RESPONSES_CACHE_READ: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "summarize this text"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 12928,
                "output_tokens": 380,
                "total_tokens": 13308,
                "input_tokens_details": {"cached_tokens": 12288},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.0085904,
        input_cost=0.0032704,
        output_cost=0.00532,
        prompt_tokens=12928,
        completion_tokens=380,
        cache_read_cost=0.0021504,
    ),
)

GPT_5_6_RESPONSES_REASONING: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "reason about this text"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {"type": "reasoning", "id": "rs_$REQUEST_ID", "status": "completed", "summary": []},
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 1240,
                "output_tokens": 4040,
                "total_tokens": 5280,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 3480},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.06569,
        input_cost=0.00217,
        output_cost=0.06352,
        prompt_tokens=1240,
        completion_tokens=4040,
        reasoning_cost=0.05568,
    ),
)

GPT_5_6_RESPONSES_STREAM: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "stream this text", "stream": True},
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_$REQUEST_ID","object":"response","created_at":1700000000,"status":"in_progress","model":"gpt-5.6","output":[{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}]}],"usage":null}}',
            'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_$REQUEST_ID","status":"in_progress","role":"assistant","content":[]}}',
            'event: response.content_part.added\ndata: {"type":"response.content_part.added","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"part":{"type":"content_part","text":""}}',
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"delta":"scripted "}',
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"delta":"response"}',
            'event: response.output_text.done\ndata: {"type":"response.output_text.done","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"text":"scripted response"}',
            'event: response.content_part.done\ndata: {"type":"response.content_part.done","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"part":{"type":"output_text","text":"scripted response","annotations":[]}}',
            'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}]}}',
            'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_$REQUEST_ID","object":"response","created_at":1700000000,"status":"completed","model":"gpt-5.6","output":[{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}]}],"usage":{"input_tokens":1840,"output_tokens":412,"total_tokens":2252,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}}',
        ),
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_RESPONSES_STREAM_CACHE_READ: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_stream_cache_read",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "stream cached text", "stream": True},
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_$REQUEST_ID","object":"response","created_at":1700000000,"status":"in_progress","model":"gpt-5.6","output":[{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}]}],"usage":null}}',
            'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_$REQUEST_ID","status":"in_progress","role":"assistant","content":[]}}',
            'event: response.content_part.added\ndata: {"type":"response.content_part.added","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"part":{"type":"content_part","text":""}}',
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"delta":"scripted "}',
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"delta":"response"}',
            'event: response.output_text.done\ndata: {"type":"response.output_text.done","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"text":"scripted response"}',
            'event: response.content_part.done\ndata: {"type":"response.content_part.done","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"part":{"type":"output_text","text":"scripted response","annotations":[]}}',
            'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}]}}',
            'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_$REQUEST_ID","object":"response","created_at":1700000000,"status":"completed","model":"gpt-5.6","output":[{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}]}],"usage":{"input_tokens":12928,"output_tokens":380,"total_tokens":13308,"input_tokens_details":{"cached_tokens":12288},"output_tokens_details":{"reasoning_tokens":0}}}}',
        ),
    ),
    expected=ExactExpected(
        spend=0.0085904,
        input_cost=0.0032704,
        output_cost=0.00532,
        prompt_tokens=12928,
        completion_tokens=380,
        cache_read_cost=0.0021504,
    ),
)

GPT_5_6_RESPONSES_INCOMPLETE: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_incomplete",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "truncate this text", "max_output_tokens": 100},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "incomplete",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1840,
                "output_tokens": 100,
                "total_tokens": 1940,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    ),
    expected=ExactExpected(
        spend=0.00462, input_cost=0.00322, output_cost=0.0014, prompt_tokens=1840, completion_tokens=100
    ),
)

GPT_5_6_RESPONSES_PREVIOUS_RESPONSE_ID: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_previous_response_id",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "continue this text", "previous_response_id": "$PRIOR_RESPONSE_ID"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1840,
                "output_tokens": 412,
                "total_tokens": 2252,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_RESPONSES_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={
        "model": "$MODEL",
        "input": "search this text",
        "tools": [{"type": "web_search_preview", "search_context_size": "medium"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_$REQUEST_ID",
                    "status": "completed",
                    "action": {"type": "search", "query": "scripted query"},
                },
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 1840,
                "output_tokens": 412,
                "total_tokens": 2252,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.021488,
        input_cost=0.00322,
        output_cost=0.005768,
        prompt_tokens=1840,
        completion_tokens=412,
        tool_usage_cost=0.0125,
    ),
)

GPT_5_6_RESPONSES_WEB_SEARCH_REPORTED_COUNT_MIXED_ACTIONS: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_web_search_reported_count_mixed_actions",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={
        "model": "$MODEL",
        "input": "search this text",
        "tools": [{"type": "web_search_preview", "search_context_size": "medium"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_0_$REQUEST_ID",
                    "status": "completed",
                    "action": {"type": "search", "query": "scripted query"},
                },
                {
                    "type": "web_search_call",
                    "id": "ws_1_$REQUEST_ID",
                    "status": "completed",
                    "action": {"type": "open_page", "url": "https://scripted.example/a"},
                },
                {
                    "type": "web_search_call",
                    "id": "ws_2_$REQUEST_ID",
                    "status": "completed",
                    "action": {"type": "open_page", "url": "https://scripted.example/b"},
                },
                {
                    "type": "web_search_call",
                    "id": "ws_3_$REQUEST_ID",
                    "status": "completed",
                    "action": {"type": "find_in_page", "pattern": "release", "url": "https://scripted.example/a"},
                },
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 1840,
                "output_tokens": 412,
                "total_tokens": 2252,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "tool_usage": {
                "image_gen": {
                    "input_tokens": 0,
                    "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                    "total_tokens": 0,
                },
                "web_search": {"num_requests": 1},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.021488,
        input_cost=0.00322,
        output_cost=0.005768,
        prompt_tokens=1840,
        completion_tokens=412,
        tool_usage_cost=0.0125,
    ),
)

GPT_5_6_RESPONSES_WEB_SEARCH_REPORTED_COUNT_OPEN_PAGE_ONLY: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_web_search_reported_count_open_page_only",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={
        "model": "$MODEL",
        "input": "search this text",
        "tools": [{"type": "web_search_preview", "search_context_size": "medium"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_0_$REQUEST_ID",
                    "status": "completed",
                    "action": {"type": "open_page", "url": "https://scripted.example/a"},
                },
                {
                    "type": "web_search_call",
                    "id": "ws_1_$REQUEST_ID",
                    "status": "completed",
                    "action": {"type": "find_in_page", "pattern": "release", "url": "https://scripted.example/a"},
                },
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 1840,
                "output_tokens": 412,
                "total_tokens": 2252,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "tool_usage": {
                "image_gen": {
                    "input_tokens": 0,
                    "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                    "total_tokens": 0,
                },
                "web_search": {"num_requests": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.008988,
        input_cost=0.00322,
        output_cost=0.005768,
        prompt_tokens=1840,
        completion_tokens=412,
        tool_usage_cost=0.0,
    ),
)

GPT_5_6_RESPONSES_WEB_SEARCH_REPORTED_COUNT_MIXED_ACTIONS_STREAM: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_web_search_reported_count_mixed_actions_stream",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={
        "model": "$MODEL",
        "input": "search this text",
        "tools": [{"type": "web_search_preview", "search_context_size": "medium"}],
        "stream": True,
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_$REQUEST_ID","object":"response","created_at":1700000000,"status":"in_progress","model":"gpt-5.6","output":[{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}]}],"usage":null}}',
            'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_$REQUEST_ID","status":"in_progress","role":"assistant","content":[]}}',
            'event: response.content_part.added\ndata: {"type":"response.content_part.added","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"part":{"type":"content_part","text":""}}',
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"delta":"scripted "}',
            'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"delta":"response"}',
            'event: response.output_text.done\ndata: {"type":"response.output_text.done","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"text":"scripted response"}',
            'event: response.content_part.done\ndata: {"type":"response.content_part.done","item_id":"msg_$REQUEST_ID","output_index":0,"content_index":0,"part":{"type":"output_text","text":"scripted response","annotations":[]}}',
            'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}]}}',
            'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_$REQUEST_ID","object":"response","created_at":1700000000,"status":"completed","model":"gpt-5.6","output":[{"type":"web_search_call","id":"ws_0_$REQUEST_ID","status":"completed","action":{"type":"search","query":"scripted query"}},{"type":"web_search_call","id":"ws_1_$REQUEST_ID","status":"completed","action":{"type":"open_page","url":"https://scripted.example/a"}},{"type":"web_search_call","id":"ws_2_$REQUEST_ID","status":"completed","action":{"type":"open_page","url":"https://scripted.example/b"}},{"type":"web_search_call","id":"ws_3_$REQUEST_ID","status":"completed","action":{"type":"find_in_page","pattern":"release","url":"https://scripted.example/a"}},{"type":"message","id":"msg_$REQUEST_ID","status":"completed","role":"assistant","content":[{"type":"output_text","text":"scripted response","annotations":[]}] }],"usage":{"input_tokens":1840,"output_tokens":412,"total_tokens":2252,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}},"tool_usage":{"image_gen":{"input_tokens":0,"input_tokens_details":{"image_tokens":0,"text_tokens":0},"output_tokens":0,"output_tokens_details":{"image_tokens":0,"text_tokens":0},"total_tokens":0},"web_search":{"num_requests":1}}}}',
        ),
    ),
    expected=ExactExpected(
        spend=0.021488,
        input_cost=0.00322,
        output_cost=0.005768,
        prompt_tokens=1840,
        completion_tokens=412,
        tool_usage_cost=0.0125,
    ),
)

GPT_5_6_RESPONSES_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "flex text", "service_tier": "flex"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1840,
                "output_tokens": 412,
                "total_tokens": 2252,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
                "service_tier": "flex",
            },
            "service_tier": "flex",
        },
    ),
    expected=ExactExpected(
        spend=0.004494, input_cost=0.00161, output_cost=0.002884, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_RESPONSES_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "priority text", "service_tier": "priority"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 1840,
                "output_tokens": 412,
                "total_tokens": 2252,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
                "service_tier": "priority",
            },
            "service_tier": "priority",
        },
    ),
    expected=ExactExpected(
        spend=0.017976, input_cost=0.00644, output_cost=0.011536, prompt_tokens=1840, completion_tokens=412
    ),
)

OPENAI_DEPLOYMENT_PRICING_OVERRIDE: Final = CostTrackingTestCase(
    name="openai-deployment-pricing-override",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    deployment=Deployment(model="openai/cc-custom-model", input_cost_per_token=7e-06, output_cost_per_token=2.1e-05),
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "openai-deployment-pricing-override"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "$MODEL",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted response"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "default",
        },
    ),
    expected=ExactExpected(
        spend=0.021532, input_cost=0.01288, output_cost=0.008652, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_UPSTREAM_400_ZERO_SPEND: Final = CostTrackingTestCase(
    name="gpt-5.6-upstream_400_zero_spend",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={"error": {"message": "scripted upstream failure 400", "type": "server_error", "code": "400"}},
        status=400,
    ),
    expected=FailureExpected(failure=FailureDetails(status=400)),
)

GPT_5_6_UPSTREAM_401_ZERO_SPEND: Final = CostTrackingTestCase(
    name="gpt-5.6-upstream_401_zero_spend",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={"error": {"message": "scripted upstream failure 401", "type": "server_error", "code": "401"}},
        status=401,
    ),
    expected=FailureExpected(failure=FailureDetails(status=401)),
)

GPT_5_6_UPSTREAM_500_STREAM_REQUEST_ZERO_SPEND: Final = CostTrackingTestCase(
    name="gpt-5.6-upstream_500_stream_request_zero_spend",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": True},
    response=JsonResponse(
        content_type="application/json",
        body={"error": {"message": "scripted upstream failure 500", "type": "server_error", "code": "500"}},
        status=500,
    ),
    expected=FailureExpected(failure=FailureDetails(status=500)),
)

GPT_5_6_RESPONSES_UPSTREAM_500_ZERO_SPEND: Final = CostTrackingTestCase(
    name="gpt-5.6-responses_upstream_500_zero_spend",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "proxy behaviour probe", "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={"error": {"message": "scripted upstream failure 500", "type": "server_error", "code": "500"}},
        status=500,
    ),
    expected=FailureExpected(failure=FailureDetails(status=500)),
)

CLAUDE_SONNET_5_MESSAGES_UPSTREAM_500_ZERO_SPEND: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_upstream_500_zero_spend",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    endpoint="/v1/messages",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "max_tokens": 412},
    response=JsonResponse(
        content_type="application/json",
        body={"error": {"message": "scripted upstream failure 500", "type": "server_error", "code": "500"}},
        status=500,
    ),
    expected=FailureExpected(failure=FailureDetails(status=500)),
)

GPT_5_6_FALLBACK_BILLED_TO_ANSWERING_DEPLOYMENT: Final = CostTrackingTestCase(
    name="gpt-5.6-fallback_billed_to_answering_deployment",
    covers="quota_management.spend_tracking.routing.fallback_billing",
    model="gpt-5.6",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted answer"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
    fallback_from=JsonResponse(
        content_type="application/json",
        body={"error": {"message": "scripted upstream failure 500", "type": "server_error", "code": "500"}},
        status=500,
    ),
)

GPT_5_6_N_2_CHOICES: Final = CostTrackingTestCase(
    name="gpt-5.6-n_2_choices",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted answer"}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "second choice"}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_FINISH_REASON_LENGTH: Final = CostTrackingTestCase(
    name="gpt-5.6-finish_reason_length",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "truncated"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_STREAM_USAGE_IN_EMPTY_CHOICES_CHUNK: Final = CostTrackingTestCase(
    name="gpt-5.6-stream_usage_in_empty_choices_chunk",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "proxy behaviour probe"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"scripted answer"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[],"usage":{"prompt_tokens":1840,"completion_tokens":412,"total_tokens":2252}}',
            "data: [DONE]",
        ),
    ),
    expected=ExactExpected(
        spend=0.008988,
        input_cost=0.00322,
        output_cost=0.005768,
        prompt_tokens=1840,
        completion_tokens=412,
        cost_header=False,
    ),
)

GPT_5_6_STREAM_USAGE_IN_LAST_DELTA_CHUNK: Final = CostTrackingTestCase(
    name="gpt-5.6-stream_usage_in_last_delta_chunk",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gpt-5.6",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "proxy behaviour probe"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"scripted answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1840,"completion_tokens":412,"total_tokens":2252}}',
            "data: [DONE]",
        ),
    ),
    expected=ExactExpected(
        spend=0.008988,
        input_cost=0.00322,
        output_cost=0.005768,
        prompt_tokens=1840,
        completion_tokens=412,
        cost_header=False,
    ),
)

GPT_5_6_UNKNOWN_MODEL_RESPONSE_MODEL_UNKNOWN: Final = CostTrackingTestCase(
    name="gpt-5.6-unknown_model_response_model_unknown",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    deployment=Deployment(model="openai/not-in-any-map-xyz"),
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "not-in-any-map-xyz",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted answer"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0, input_cost=0.0, output_cost=0.0, prompt_tokens=1840, completion_tokens=412, cost_header=False
    ),
)

GPT_5_6_UNKNOWN_MODEL_RESPONSE_MODEL_KNOWN: Final = CostTrackingTestCase(
    name="gpt-5.6-unknown_model_response_model_known",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-5.6",
    deployment=Deployment(model="openai/not-in-any-map-xyz"),
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "gpt-5.6",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted answer"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.008988, input_cost=0.00322, output_cost=0.005768, prompt_tokens=1840, completion_tokens=412
    ),
)

GPT_5_6_CLIENT_DISCONNECT_MID_STREAM: Final = CostTrackingTestCase(
    name="gpt-5.6-client_disconnect_mid_stream",
    covers="quota_management.spend_tracking.scripted_wire.client_disconnect",
    model="gpt-5.6",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "proxy behaviour probe"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-0"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-1"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-2"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-3"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-4"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-5"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-6"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-7"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-8"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-9"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-10"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-11"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-12"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-13"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-14"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-15"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-16"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-17"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-18"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1789788263,"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"frame-19"},"finish_reason":null}],"usage":null}',
            "data: [DONE]",
        ),
        frame_delay_ms=200,
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=1.75e-06, output_cost_per_token=1.4e-05),
        prompt_tokens=10,
        min_completion_tokens=9,
        max_completion_tokens=30,
    ),
    disconnect_after_frames=3,
)


# whisper-next
WHISPER_NEXT_TRANSCRIPTIONS_PER_SECOND: Final = CostTrackingTestCase(
    name="whisper-next-transcriptions-per-second",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="whisper-next",
    endpoint="/v1/audio/transcriptions",
    upload=WavUpload(kind="wav", seconds=3.5),
    request={"language": "en", "response_format": "json"},
    response=JsonResponse(content_type="application/json", body={"text": "hello"}),
    expected=ExactExpected(spend=0.00035, input_cost=0.00035, output_cost=0, prompt_tokens=0, completion_tokens=0),
)


# whisper-verbose-next
WHISPER_VERBOSE_NEXT_TRANSCRIPTIONS_DURATION: Final = CostTrackingTestCase(
    name="whisper-verbose-next-transcriptions-duration",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="whisper-verbose-next",
    endpoint="/v1/audio/transcriptions",
    upload=WavUpload(kind="wav", seconds=3.5),
    request={"response_format": "verbose_json"},
    response=JsonResponse(content_type="application/json", body={"text": "hello", "duration": 12.25}),
    expected=ExactExpected(spend=0.00245, input_cost=0.00245, output_cost=0, prompt_tokens=0, completion_tokens=0),
)


# gpt-4o-transcribe-next
GPT_4O_TRANSCRIBE_NEXT_TRANSCRIPTIONS_TOKENS: Final = CostTrackingTestCase(
    name="gpt-4o-transcribe-next-transcriptions-tokens",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-4o-transcribe-next",
    endpoint="/v1/audio/transcriptions",
    upload=WavUpload(kind="wav", seconds=1.0),
    request={"response_format": "json"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "text": "hello",
            "usage": {
                "type": "tokens",
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "input_token_details": {"text_tokens": 2, "audio_tokens": 8},
            },
        },
    ),
    expected=ExactExpected(
        spend=9.044e-05, input_cost=8.422e-05, output_cost=6.22e-06, prompt_tokens=10, completion_tokens=2
    ),
)


# tts-next
TTS_NEXT_SPEECH_PER_CHARACTER: Final = CostTrackingTestCase(
    name="tts-next-speech-per-character",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="tts-next",
    endpoint="/v1/audio/speech",
    request={"input": "hello world", "voice": "alloy", "response_format": "mp3"},
    response=BinaryResponse(content_type="audio/mpeg", length=2048),
    expected=ExactExpected(spend=0.0001, input_cost=0.0001, output_cost=0, prompt_tokens=0, completion_tokens=0),
)


# tts-next-hd
TTS_NEXT_HD_SPEECH_PER_CHARACTER: Final = CostTrackingTestCase(
    name="tts-next-hd-speech-per-character",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="tts-next-hd",
    endpoint="/v1/audio/speech",
    request={"input": "hello world", "voice": "alloy", "response_format": "mp3"},
    response=BinaryResponse(content_type="audio/mpeg", length=2048),
    expected=ExactExpected(spend=0.0002, input_cost=0.0002, output_cost=0, prompt_tokens=0, completion_tokens=0),
)


# 1024-x-1024/dall-e-3-next
DALL_E_3_NEXT_IMAGES_STANDARD: Final = CostTrackingTestCase(
    name="dall-e-3-next-images-standard",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="1024-x-1024/dall-e-3-next",
    endpoint="/v1/images/generations",
    deployment=Deployment(model="openai/dall-e-3-next"),
    request={"prompt": "a deterministic square", "size": "1024x1024", "quality": "standard", "n": 1},
    response=JsonResponse(
        content_type="application/json", body={"created": 1700000000, "data": [{"url": "https://x/1.png"}]}
    ),
    expected=ExactExpected(
        spend=0.04, input_cost=0.04, output_cost=0, prompt_tokens=0, completion_tokens=0, breakdown_persisted=False
    ),
)

DALL_E_3_NEXT_IMAGES_TWO: Final = CostTrackingTestCase(
    name="dall-e-3-next-images-two",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="1024-x-1024/dall-e-3-next",
    endpoint="/v1/images/generations",
    deployment=Deployment(model="openai/dall-e-3-next"),
    request={"prompt": "two deterministic squares", "size": "1024x1024", "quality": "standard", "n": 2},
    response=JsonResponse(
        content_type="application/json",
        body={"created": 1700000003, "data": [{"url": "https://x/1.png"}, {"url": "https://x/2.png"}]},
    ),
    expected=ExactExpected(
        spend=0.08, input_cost=0.08, output_cost=0, prompt_tokens=0, completion_tokens=0, breakdown_persisted=False
    ),
)


# hd/1024-x-1024/dall-e-3-next
DALL_E_3_NEXT_IMAGES_HD: Final = CostTrackingTestCase(
    name="dall-e-3-next-images-hd",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="hd/1024-x-1024/dall-e-3-next",
    endpoint="/v1/images/generations",
    deployment=Deployment(model="openai/dall-e-3-next"),
    request={"prompt": "a deterministic square", "size": "1024x1024", "quality": "hd", "n": 1},
    response=JsonResponse(
        content_type="application/json", body={"created": 1700000001, "data": [{"url": "https://x/1.png"}]}
    ),
    expected=ExactExpected(
        spend=0.08, input_cost=0.08, output_cost=0, prompt_tokens=0, completion_tokens=0, breakdown_persisted=False
    ),
)


# 1792-x-1024/dall-e-3-next
DALL_E_3_NEXT_IMAGES_WIDE: Final = CostTrackingTestCase(
    name="dall-e-3-next-images-wide",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="1792-x-1024/dall-e-3-next",
    endpoint="/v1/images/generations",
    deployment=Deployment(model="openai/dall-e-3-next"),
    request={"prompt": "a deterministic wide image", "size": "1792x1024", "quality": "standard", "n": 1},
    response=JsonResponse(
        content_type="application/json", body={"created": 1700000002, "data": [{"url": "https://x/1.png"}]}
    ),
    expected=ExactExpected(
        spend=0.06, input_cost=0.06, output_cost=0, prompt_tokens=0, completion_tokens=0, breakdown_persisted=False
    ),
)


# gpt-image-next
GPT_IMAGE_NEXT_IMAGES_LOW: Final = CostTrackingTestCase(
    name="gpt-image-next-images-low",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-image-next",
    endpoint="/v1/images/generations",
    deployment=Deployment(model="openai/gpt-image-next"),
    request={"prompt": "a deterministic generated image", "size": "1024x1024", "quality": "low", "n": 1},
    response=JsonResponse(
        content_type="application/json",
        body={
            "created": 1700000004,
            "data": [{"b64_json": "AA=="}],
            "usage": {
                "total_tokens": 30,
                "input_tokens": 10,
                "output_tokens": 20,
                "input_tokens_details": {"text_tokens": 10, "image_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.0001191,
        input_cost=1.71e-05,
        output_cost=0.000102,
        prompt_tokens=10,
        completion_tokens=20,
        breakdown_persisted=False,
    ),
)


# low/1024-x-1024/gpt-image-next
GPT_IMAGE_NEXT_IMAGES_EDIT: Final = CostTrackingTestCase(
    name="gpt-image-next-images-edit",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="low/1024-x-1024/gpt-image-next",
    endpoint="/v1/images/edits",
    deployment=Deployment(model="openai/gpt-image-next"),
    upload=PngUpload(kind="png"),
    request={"prompt": "edit this deterministic image", "size": "1024x1024", "quality": "low", "n": 1},
    response=JsonResponse(
        content_type="application/json",
        body={
            "created": 1700000005,
            "data": [{"b64_json": "AA=="}],
            "usage": {
                "total_tokens": 30,
                "input_tokens": 10,
                "output_tokens": 20,
                "input_tokens_details": {"text_tokens": 10, "image_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.000119,
        input_cost=1.7e-05,
        output_cost=0.000102,
        prompt_tokens=10,
        completion_tokens=20,
        breakdown_persisted=False,
    ),
)


# text-embedding-4-small
TEXT_EMBEDDINGS_4_SMALL_SINGLE: Final = CostTrackingTestCase(
    name="text-embeddings-4-small-single",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="text-embedding-4-small",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": "one embedding"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-4-small",
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        },
    ),
    expected=ExactExpected(spend=7.07e-06, input_cost=7.07e-06, output_cost=0.0, prompt_tokens=7, completion_tokens=0),
)

TEXT_EMBEDDINGS_4_SMALL_BATCH: Final = CostTrackingTestCase(
    name="text-embeddings-4-small-batch",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="text-embedding-4-small",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": ["one", "two", "three"]},
    response=JsonResponse(
        content_type="application/json",
        body={
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 1},
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 2},
            ],
            "model": "text-embedding-4-small",
            "usage": {"prompt_tokens": 21, "total_tokens": 21},
        },
    ),
    expected=ExactExpected(
        spend=2.1210000000000002e-05,
        input_cost=2.1210000000000002e-05,
        output_cost=0.0,
        prompt_tokens=21,
        completion_tokens=0,
    ),
)

TEXT_EMBEDDINGS_4_SMALL_TOKEN_ARRAY: Final = CostTrackingTestCase(
    name="text-embeddings-4-small-token-array",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="text-embedding-4-small",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": [1, 2, 3, 4]},
    response=JsonResponse(
        content_type="application/json",
        body={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-4-small",
            "usage": {"prompt_tokens": 9, "total_tokens": 9},
        },
    ),
    expected=ExactExpected(
        spend=9.090000000000001e-06,
        input_cost=9.090000000000001e-06,
        output_cost=0.0,
        prompt_tokens=9,
        completion_tokens=0,
    ),
)


# text-embedding-3-large-next
TEXT_EMBEDDINGS_3_LARGE_DIMENSIONS: Final = CostTrackingTestCase(
    name="text-embeddings-3-large-dimensions",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="text-embedding-3-large-next",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": "large embedding", "dimensions": 3},
    response=JsonResponse(
        content_type="application/json",
        body={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-3-large-next",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        },
    ),
    expected=ExactExpected(spend=8.16e-06, input_cost=8.16e-06, output_cost=0.0, prompt_tokens=8, completion_tokens=0),
)


# omni-moderation-next
OMNI_MODERATIONS_NEXT_SINGLE: Final = CostTrackingTestCase(
    name="omni-moderations-next-single",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="omni-moderation-next",
    endpoint="/v1/moderations",
    request={"model": "$MODEL", "input": "safe text"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "modr-single-$REQUEST_ID",
            "model": "omni-moderation-next",
            "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
        },
    ),
    expected=ExactExpected(spend=0.0, input_cost=0.0, output_cost=0.0, prompt_tokens=0, completion_tokens=0),
)

OMNI_MODERATIONS_NEXT_LIST: Final = CostTrackingTestCase(
    name="omni-moderations-next-list",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="omni-moderation-next",
    endpoint="/v1/moderations",
    request={"model": "$MODEL", "input": ["safe text", "more safe text"]},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "modr-list-$REQUEST_ID",
            "model": "omni-moderation-next",
            "results": [
                {"flagged": False, "categories": {}, "category_scores": {}},
                {"flagged": False, "categories": {}, "category_scores": {}},
            ],
        },
    ),
    expected=ExactExpected(spend=0.0, input_cost=0.0, output_cost=0.0, prompt_tokens=0, completion_tokens=0),
)


# text-embedding-3-large
GPT_5_6_CHAT_REQUEST_TO_EMBEDDING_ENTRY: Final = CostTrackingTestCase(
    name="gpt-5.6-chat_request_to_embedding_entry",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="text-embedding-3-large",
    deployment=Deployment(model="openai/text-embedding-3-large"),
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "proxy behaviour probe"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1789788262,
            "model": "text-embedding-3-large",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted answer"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0002392, input_cost=0.0002392, output_cost=0.0, prompt_tokens=1840, completion_tokens=412
    ),
)


# gpt-6.1-sol
GPT_6_1_SOL_INPUT_TEXT: Final = CostTrackingTestCase(
    name="gpt-6.1-sol-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-6.1-sol",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-6.1-sol",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0078, input_cost=0.00368, output_cost=0.00412, prompt_tokens=1840, completion_tokens=412
    ),
)


CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    GPT_5_3_CODEX_INPUT_TEXT,
    GPT_5_3_CODEX_CACHE_READ,
    GPT_5_3_CODEX_REASONING,
    GPT_5_3_CODEX_SERVICE_TIER_FLEX,
    GPT_5_3_CODEX_SERVICE_TIER_PRIORITY,
    GPT_5_3_CODEX_WEB_SEARCH_MEDIUM,
    GPT_5_3_CODEX_WEB_SEARCH_LOW,
    GPT_5_3_CODEX_WEB_SEARCH_HIGH,
    GPT_5_3_CODEX_FILE_SEARCH,
    GPT_5_3_CODEX_STREAM,
    GPT_5_3_CODEX_STREAM_NO_USAGE,
    GPT_5_3_CODEX_STREAM_NO_USAGE_TOOL_CALL,
    GPT_5_3_CODEX_STREAM_NO_USAGE_IMAGE_INPUT,
    GPT_5_3_CODEX_STREAM_INCOMPLETE,
    GPT_5_3_CODEX_STREAM_NO_USAGE_INCOMPLETE,
    GPT_5_3_CODEX_STREAM_UNVALIDATED,
    GPT_5_3_CODEX_STREAM_NO_USAGE_UNVALIDATED,
    GPT_5_3_CODEX_RESPONSE_MODEL_OVERRIDE,
    GPT_5_3_CODEX_STREAM_RESPONSE_MODEL_OVERRIDE,
    GPT_5_3_CODEX_TOOL_CALL,
    GPT_5_3_CODEX_STREAM_TOOL_CALL,
    GPT_5_3_CODEX_STREAM_FULL_USAGE,
    GPT_5_4_MINI_INPUT_TEXT,
    GPT_5_4_MINI_CACHE_READ,
    GPT_5_4_MINI_AUDIO_INPUT,
    GPT_5_4_MINI_AUDIO_OUTPUT,
    GPT_5_4_MINI_REASONING,
    GPT_5_4_MINI_SERVICE_TIER_FLEX,
    GPT_5_4_MINI_SERVICE_TIER_PRIORITY,
    GPT_5_4_MINI_WEB_SEARCH_MEDIUM,
    GPT_5_4_MINI_WEB_SEARCH_LOW,
    GPT_5_4_MINI_WEB_SEARCH_HIGH,
    GPT_5_4_MINI_STREAM,
    GPT_5_4_MINI_STREAM_NO_USAGE,
    GPT_5_4_MINI_STREAM_NO_USAGE_TOOL_CALL,
    GPT_5_4_MINI_STREAM_NO_USAGE_IMAGE_INPUT,
    GPT_5_4_MINI_RESPONSE_MODEL_OVERRIDE,
    GPT_5_4_MINI_STREAM_RESPONSE_MODEL_OVERRIDE,
    GPT_5_4_MINI_TOOL_CALL,
    GPT_5_4_MINI_STREAM_TOOL_CALL,
    GPT_5_4_MINI_STREAM_FULL_USAGE,
    GPT_5_5_PRO_INPUT_TEXT,
    GPT_5_5_PRO_CACHE_READ,
    GPT_5_5_PRO_REASONING,
    GPT_5_5_PRO_SERVICE_TIER_FLEX,
    GPT_5_5_PRO_SERVICE_TIER_PRIORITY,
    GPT_5_5_PRO_WEB_SEARCH_MEDIUM,
    GPT_5_5_PRO_WEB_SEARCH_LOW,
    GPT_5_5_PRO_WEB_SEARCH_HIGH,
    GPT_5_5_PRO_FILE_SEARCH,
    GPT_5_5_PRO_STREAM,
    GPT_5_5_PRO_STREAM_NO_USAGE,
    GPT_5_5_PRO_STREAM_NO_USAGE_TOOL_CALL,
    GPT_5_5_PRO_STREAM_NO_USAGE_IMAGE_INPUT,
    GPT_5_5_PRO_STREAM_INCOMPLETE,
    GPT_5_5_PRO_STREAM_NO_USAGE_INCOMPLETE,
    GPT_5_5_PRO_STREAM_UNVALIDATED,
    GPT_5_5_PRO_STREAM_NO_USAGE_UNVALIDATED,
    GPT_5_5_PRO_RESPONSE_MODEL_OVERRIDE,
    GPT_5_5_PRO_STREAM_RESPONSE_MODEL_OVERRIDE,
    GPT_5_5_PRO_TOOL_CALL,
    GPT_5_5_PRO_STREAM_TOOL_CALL,
    GPT_5_5_PRO_STREAM_FULL_USAGE,
    GPT_5_6_INPUT_TEXT,
    GPT_5_6_CACHE_READ,
    GPT_5_6_AUDIO_INPUT,
    GPT_5_6_AUDIO_OUTPUT,
    GPT_5_6_REASONING,
    GPT_5_6_SERVICE_TIER_FLEX,
    GPT_5_6_SERVICE_TIER_PRIORITY,
    GPT_5_6_WEB_SEARCH_MEDIUM,
    GPT_5_6_WEB_SEARCH_LOW,
    GPT_5_6_WEB_SEARCH_HIGH,
    GPT_5_6_STREAM,
    GPT_5_6_STREAM_NO_USAGE,
    GPT_5_6_STREAM_NO_USAGE_TOOL_CALL,
    GPT_5_6_STREAM_NO_USAGE_IMAGE_INPUT,
    GPT_5_6_RESPONSE_MODEL_OVERRIDE,
    GPT_5_6_STREAM_RESPONSE_MODEL_OVERRIDE,
    GPT_5_6_TOOL_CALL,
    GPT_5_6_STREAM_TOOL_CALL,
    GPT_5_6_STREAM_FULL_USAGE,
    GPT_5_6_RESPONSES_NATIVE_JSON,
    GPT_5_6_UPSTREAM_500_ZERO_SPEND,
    GPT_5_6_UPSTREAM_429_ZERO_SPEND,
    WHISPER_NEXT_TRANSCRIPTIONS_PER_SECOND,
    WHISPER_VERBOSE_NEXT_TRANSCRIPTIONS_DURATION,
    GPT_4O_TRANSCRIBE_NEXT_TRANSCRIPTIONS_TOKENS,
    TTS_NEXT_SPEECH_PER_CHARACTER,
    TTS_NEXT_HD_SPEECH_PER_CHARACTER,
    DALL_E_3_NEXT_IMAGES_STANDARD,
    DALL_E_3_NEXT_IMAGES_HD,
    DALL_E_3_NEXT_IMAGES_WIDE,
    DALL_E_3_NEXT_IMAGES_TWO,
    GPT_IMAGE_NEXT_IMAGES_LOW,
    GPT_IMAGE_NEXT_IMAGES_EDIT,
    TEXT_EMBEDDINGS_4_SMALL_SINGLE,
    TEXT_EMBEDDINGS_4_SMALL_BATCH,
    TEXT_EMBEDDINGS_4_SMALL_TOKEN_ARRAY,
    TEXT_EMBEDDINGS_3_LARGE_DIMENSIONS,
    OMNI_MODERATIONS_NEXT_SINGLE,
    OMNI_MODERATIONS_NEXT_LIST,
    GPT_5_6_RESPONSES_CACHE_READ,
    GPT_5_6_RESPONSES_REASONING,
    GPT_5_6_RESPONSES_STREAM,
    GPT_5_6_RESPONSES_STREAM_CACHE_READ,
    GPT_5_6_RESPONSES_INCOMPLETE,
    GPT_5_6_RESPONSES_PREVIOUS_RESPONSE_ID,
    GPT_5_6_RESPONSES_WEB_SEARCH_MEDIUM,
    GPT_5_6_RESPONSES_WEB_SEARCH_REPORTED_COUNT_MIXED_ACTIONS,
    GPT_5_6_RESPONSES_WEB_SEARCH_REPORTED_COUNT_OPEN_PAGE_ONLY,
    GPT_5_6_RESPONSES_WEB_SEARCH_REPORTED_COUNT_MIXED_ACTIONS_STREAM,
    GPT_5_3_CODEX_RESPONSES_FILE_SEARCH,
    GPT_5_6_RESPONSES_SERVICE_TIER_FLEX,
    GPT_5_6_RESPONSES_SERVICE_TIER_PRIORITY,
    OPENAI_DEPLOYMENT_PRICING_OVERRIDE,
    GPT_5_6_UPSTREAM_400_ZERO_SPEND,
    GPT_5_6_UPSTREAM_401_ZERO_SPEND,
    GPT_5_6_UPSTREAM_500_STREAM_REQUEST_ZERO_SPEND,
    GPT_5_6_RESPONSES_UPSTREAM_500_ZERO_SPEND,
    CLAUDE_SONNET_5_MESSAGES_UPSTREAM_500_ZERO_SPEND,
    GPT_5_6_FALLBACK_BILLED_TO_ANSWERING_DEPLOYMENT,
    GPT_5_6_N_2_CHOICES,
    GPT_5_6_FINISH_REASON_LENGTH,
    GPT_5_6_STREAM_USAGE_IN_EMPTY_CHOICES_CHUNK,
    GPT_5_6_STREAM_USAGE_IN_LAST_DELTA_CHUNK,
    GPT_5_6_UNKNOWN_MODEL_RESPONSE_MODEL_UNKNOWN,
    GPT_5_6_UNKNOWN_MODEL_RESPONSE_MODEL_KNOWN,
    GPT_5_6_CHAT_REQUEST_TO_EMBEDDING_ENTRY,
    GPT_5_6_CLIENT_DISCONNECT_MID_STREAM,
    GPT_6_1_SOL_INPUT_TEXT,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(plain=GPT_5_3_CODEX_INPUT_TEXT, streamed=GPT_5_3_CODEX_STREAM),
    StreamParityTestCase(plain=GPT_5_3_CODEX_INPUT_TEXT, streamed=GPT_5_3_CODEX_STREAM_INCOMPLETE),
    StreamParityTestCase(plain=GPT_5_3_CODEX_INPUT_TEXT, streamed=GPT_5_3_CODEX_STREAM_UNVALIDATED),
    StreamParityTestCase(
        plain=GPT_5_3_CODEX_RESPONSE_MODEL_OVERRIDE, streamed=GPT_5_3_CODEX_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=GPT_5_3_CODEX_TOOL_CALL, streamed=GPT_5_3_CODEX_STREAM_TOOL_CALL),
    StreamParityTestCase(plain=GPT_5_4_MINI_INPUT_TEXT, streamed=GPT_5_4_MINI_STREAM),
    StreamParityTestCase(
        plain=GPT_5_4_MINI_RESPONSE_MODEL_OVERRIDE, streamed=GPT_5_4_MINI_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=GPT_5_4_MINI_TOOL_CALL, streamed=GPT_5_4_MINI_STREAM_TOOL_CALL),
    StreamParityTestCase(plain=GPT_5_5_PRO_INPUT_TEXT, streamed=GPT_5_5_PRO_STREAM),
    StreamParityTestCase(plain=GPT_5_5_PRO_INPUT_TEXT, streamed=GPT_5_5_PRO_STREAM_INCOMPLETE),
    StreamParityTestCase(plain=GPT_5_5_PRO_INPUT_TEXT, streamed=GPT_5_5_PRO_STREAM_UNVALIDATED),
    StreamParityTestCase(
        plain=GPT_5_5_PRO_RESPONSE_MODEL_OVERRIDE, streamed=GPT_5_5_PRO_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=GPT_5_5_PRO_TOOL_CALL, streamed=GPT_5_5_PRO_STREAM_TOOL_CALL),
    StreamParityTestCase(plain=GPT_5_6_RESPONSE_MODEL_OVERRIDE, streamed=GPT_5_6_STREAM_RESPONSE_MODEL_OVERRIDE),
    StreamParityTestCase(plain=GPT_5_6_TOOL_CALL, streamed=GPT_5_6_STREAM_TOOL_CALL),
)
