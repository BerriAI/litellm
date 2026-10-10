"""azure cost tracking cases, one CostTrackingTestCase literal per request shape (moved from cost_tracking_cases.json).

PARITY pairs the streamed case with its plain twin where both bill the same row."""

from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    BinaryResponse,
    CostTrackingTestCase,
    Deployment,
    ExactExpected,
    JsonResponse,
    RecountExpected,
    RecountRates,
    SseResponse,
    WavUpload,
)
from integration.cost_calculation.stream_parity.case import StreamParityTestCase, sse_frames


# azure/gpt-5.4-mini
AZURE_GPT_5_4_MINI_INPUT_TEXT: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "28b0c4ce80d6 summarize the attached material in one line and name the city weather",
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
            "created": 1789788217,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 28b0c4ce80d6"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_CACHE_READ: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "33fdcb306184 summarize the attached material in one line and name the city weather",
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
            "created": 1789788218,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 33fdcb306184"},
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
        spend=0.001767168, input_cost=0.000672768, output_cost=0.0010944, prompt_tokens=12928, completion_tokens=380
    ),
)

AZURE_GPT_5_4_MINI_AUDIO_INPUT: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-audio_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "28a136ca9579 summarize the attached material in one line and name the city weather",
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
            "created": 1789788220,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 28a136ca9579"},
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
        spend=0.01586436, input_cost=0.01525956, output_cost=0.0006048, prompt_tokens=1546, completion_tokens=210
    ),
)

AZURE_GPT_5_4_MINI_AUDIO_OUTPUT: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-audio_output",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "2bebbaa4e254 summarize the attached material in one line and name the city weather",
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
            "created": 1789788222,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 2bebbaa4e254"},
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
        spend=0.0241176, input_cost=7.92e-05, output_cost=0.0240384, prompt_tokens=220, completion_tokens=1300
    ),
)

AZURE_GPT_5_4_MINI_REASONING: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "6af41b14ef04 summarize the attached material in one line and name the city weather",
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
            "created": 1789788222,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 6af41b14ef04"},
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
        spend=0.0135432, input_cost=0.0004464, output_cost=0.0130968, prompt_tokens=1240, completion_tokens=4040
    ),
)

AZURE_GPT_5_4_MINI_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "60a03b8b6237 summarize the attached material in one line and name the city weather",
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
            "created": 1789788225,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 60a03b8b6237"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "flex",
        },
    ),
    expected=ExactExpected(
        spend=0.00092448, input_cost=0.0003312, output_cost=0.00059328, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "3922bd062f4a summarize the attached material in one line and name the city weather",
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
            "created": 1789788226,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 3922bd062f4a"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "priority",
        },
    ),
    expected=ExactExpected(
        spend=0.00369792, input_cost=0.0013248, output_cost=0.00237312, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "26430574f63b summarize the attached material in one line and name the city weather",
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
            "created": 1789788211,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 26430574f63b",
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
        spend=0.01434896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_WEB_SEARCH_LOW: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-web_search_low",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "5792eab53e4c summarize the attached material in one line and name the city weather",
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
            "created": 1789788213,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 5792eab53e4c",
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
        spend=0.01184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_WEB_SEARCH_HIGH: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-web_search_high",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "7a7fc7488611 summarize the attached material in one line and name the city weather",
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
            "created": 1789788215,
            "model": "cc-pinned-deployment",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 7a7fc7488611",
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
        spend=0.01684896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_STREAM: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "35a730eefc00 summarize the attached material in one line and name the city weather",
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
                "created": 1789788215,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788215,
                "model": "cc-pinned-deployment",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 35a730eefc00"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788215,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788215,
                "model": "cc-pinned-deployment",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "969d5ff8918e summarize the attached material in one line and name the city weather",
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
                "created": 1789788216,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788216,
                "model": "cc-pinned-deployment",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 969d5ff8918e"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788216,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.6e-07, output_cost_per_token=2.88e-06)),
)

AZURE_GPT_5_4_MINI_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "db6294a8264b summarize the attached material in one line and name the city weather",
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
                "created": 1789788218,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788218,
                "model": "cc-pinned-deployment",
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
                "created": 1789788218,
                "model": "cc-pinned-deployment",
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
                "created": 1789788218,
                "model": "cc-pinned-deployment",
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
                "created": 1789788218,
                "model": "cc-pinned-deployment",
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
                "created": 1789788218,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.6e-07, output_cost_per_token=2.88e-06)),
)

AZURE_GPT_5_4_MINI_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "524f8c567f64 summarize the attached material in one line and name the city weather",
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
                "created": 1789788220,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788220,
                "model": "cc-pinned-deployment",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 524f8c567f64"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788220,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.6e-07, output_cost_per_token=2.88e-06)),
)

AZURE_GPT_5_4_MINI_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "109128998398 summarize the attached material in one line and name the city weather",
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
            "created": 1789788221,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 109128998398"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "e5bffdbd65e2 summarize the attached material in one line and name the city weather",
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
                "created": 1789788221,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788221,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer e5bffdbd65e2"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788221,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788221,
                "model": "gpt-5.6",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_TOOL_CALL: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "75003151c7de summarize the attached material in one line and name the city weather",
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
            "created": 1789788223,
            "model": "cc-pinned-deployment",
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
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "e3e7c45697d3 summarize the attached material in one line and name the city weather",
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
                "created": 1789788225,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788225,
                "model": "cc-pinned-deployment",
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
                "created": 1789788225,
                "model": "cc-pinned-deployment",
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
                "created": 1789788225,
                "model": "cc-pinned-deployment",
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
                "created": 1789788225,
                "model": "cc-pinned-deployment",
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
                "created": 1789788225,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788225,
                "model": "cc-pinned-deployment",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_4_MINI_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="azure-gpt-5.4-mini-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
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
                        "text": "906fb0b08ba9 summarize the attached material in one line and name the city weather",
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
                "created": 1789788211,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788211,
                "model": "cc-pinned-deployment",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 906fb0b08ba9"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788211,
                "model": "cc-pinned-deployment",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788211,
                "model": "cc-pinned-deployment",
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
        spend=0.014385144, input_cost=0.004348584, output_cost=0.01003656, prompt_tokens=8314, completion_tokens=1592
    ),
)

AZURE_PINNED_GPT_5_4_MINI_STREAM: Final = CostTrackingTestCase(
    name="azure-pinned-gpt-5.4-mini-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.4-mini",
    deployment=Deployment(model="azure/cc-pinned-deployment", base_model="azure/gpt-5.4-mini"),
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "azure-pinned-gpt-5.4-mini-stream"}],
        "stream": True,
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1,"model":"$MODEL","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1,"model":"$MODEL","choices":[],"usage":{"prompt_tokens":1840,"completion_tokens":412,"total_tokens":2252}}\n\n',
            "data: [DONE]\n\n",
        ),
    ),
    expected=ExactExpected(
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)


# azure/gpt-5.6
AZURE_GPT_5_6_INPUT_TEXT: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "7dc90bf2ab07 summarize the attached material in one line and name the city weather",
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
            "created": 1789788212,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 7dc90bf2ab07"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0092448, input_cost=0.003312, output_cost=0.0059328, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_CACHE_READ: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "c84b90d4fd99 summarize the attached material in one line and name the city weather",
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
            "created": 1789788214,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer c84b90d4fd99"},
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
        spend=0.00883584, input_cost=0.00336384, output_cost=0.005472, prompt_tokens=12928, completion_tokens=380
    ),
)

AZURE_GPT_5_6_AUDIO_INPUT: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-audio_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "1744b6a5bab3 summarize the attached material in one line and name the city weather",
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
            "created": 1789788215,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 1744b6a5bab3"},
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
        spend=0.0626468, input_cost=0.0596228, output_cost=0.003024, prompt_tokens=1546, completion_tokens=210
    ),
)

AZURE_GPT_5_6_AUDIO_OUTPUT: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-audio_output",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "dfb52830c629 summarize the attached material in one line and name the city weather",
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
            "created": 1789788217,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer dfb52830c629"},
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
        spend=0.094828, input_cost=0.000396, output_cost=0.094432, prompt_tokens=220, completion_tokens=1300
    ),
)

AZURE_GPT_5_6_REASONING: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "5c6865969ae9 summarize the attached material in one line and name the city weather",
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
            "created": 1789788218,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 5c6865969ae9"},
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
        spend=0.067716,
        input_cost=0.002232,
        output_cost=0.065484,
        prompt_tokens=1240,
        completion_tokens=4040,
        reasoning_cost=0.05742,
    ),
)

AZURE_GPT_5_6_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "b67dcd189cdd summarize the attached material in one line and name the city weather",
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
            "created": 1789788221,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer b67dcd189cdd"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "flex",
        },
    ),
    expected=ExactExpected(
        spend=0.0046224, input_cost=0.001656, output_cost=0.0029664, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "01c77d1ef23d summarize the attached material in one line and name the city weather",
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
            "created": 1789788222,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 01c77d1ef23d"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "priority",
        },
    ),
    expected=ExactExpected(
        spend=0.0184896, input_cost=0.006624, output_cost=0.0118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "d4efeea706ac summarize the attached material in one line and name the city weather",
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
            "created": 1789788222,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer d4efeea706ac",
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
        spend=0.0217448,
        input_cost=0.003312,
        output_cost=0.0059328,
        prompt_tokens=1840,
        completion_tokens=412,
        tool_usage_cost=0.0125,
    ),
)

AZURE_GPT_5_6_WEB_SEARCH_LOW: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-web_search_low",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "73458bfd2358 summarize the attached material in one line and name the city weather",
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
            "created": 1789788225,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 73458bfd2358",
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
        spend=0.0192448, input_cost=0.003312, output_cost=0.0059328, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_WEB_SEARCH_HIGH: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-web_search_high",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "0aebd59315f2 summarize the attached material in one line and name the city weather",
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
            "created": 1789788226,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "scripted answer 0aebd59315f2",
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
        spend=0.0242448, input_cost=0.003312, output_cost=0.0059328, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_STREAM: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "9df3f46fd138 summarize the attached material in one line and name the city weather",
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
                "created": 1789788211,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788211,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer 9df3f46fd138"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788211,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788211,
                "model": "gpt-5.6",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.0092448, input_cost=0.003312, output_cost=0.0059328, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "b54d5959e61f summarize the attached material in one line and name the city weather",
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
                "created": 1789788213,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788213,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer b54d5959e61f"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788213,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.8e-06, output_cost_per_token=1.44e-05)),
)

AZURE_GPT_5_6_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "23eb3226fc23 summarize the attached material in one line and name the city weather",
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
                    "created_at": 1789788215,
                    "model": "gpt-5.6",
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
                    "created_at": 1789788215,
                    "status": "completed",
                    "model": "gpt-5.6",
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
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.8e-06, output_cost_per_token=1.44e-05)),
)

AZURE_GPT_5_6_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "fb83549ab4c5 summarize the attached material in one line and name the city weather",
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
                "created": 1789788215,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788215,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer fb83549ab4c5"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788215,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            done=True,
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1.8e-06, output_cost_per_token=1.44e-05)),
)

AZURE_GPT_5_6_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "74e949a94e0f summarize the attached material in one line and name the city weather",
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
            "created": 1789788216,
            "model": "gpt-5.4-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "scripted answer 74e949a94e0f"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "af89dddadd12 summarize the attached material in one line and name the city weather",
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
                "created": 1789788218,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788218,
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer af89dddadd12"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788218,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788218,
                "model": "gpt-5.4-mini",
                "choices": [],
                "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=0.00184896, input_cost=0.0006624, output_cost=0.00118656, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_TOOL_CALL: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "30d4deb9b74f summarize the attached material in one line and name the city weather",
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
            "created_at": 1789788220,
            "status": "completed",
            "model": "gpt-5.6",
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
        spend=0.0092448, input_cost=0.003312, output_cost=0.0059328, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "ef44a3525238 summarize the attached material in one line and name the city weather",
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
                    "created_at": 1789788221,
                    "model": "gpt-5.6",
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
                    "created_at": 1789788221,
                    "status": "completed",
                    "model": "gpt-5.6",
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
        spend=0.0092448, input_cost=0.003312, output_cost=0.0059328, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_GPT_5_6_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="azure-gpt-5.6-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure/gpt-5.6",
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
                        "text": "e440709770ad summarize the attached material in one line and name the city weather",
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
                "created": 1789788221,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788221,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "scripted answer e440709770ad"},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788221,
                "model": "gpt-5.6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": "chatcmpl-$REQUEST_ID",
                "object": "chat.completion.chunk",
                "created": 1789788221,
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
        spend=0.06169072, input_cost=0.01794792, output_cost=0.0437428, prompt_tokens=8314, completion_tokens=1592
    ),
)


# azure/whisper-next
AZURE_WHISPER_NEXT_TRANSCRIPTIONS_DEPLOYMENT: Final = CostTrackingTestCase(
    name="azure-whisper-next-transcriptions-deployment",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/whisper-next",
    endpoint="/v1/audio/transcriptions",
    deployment=Deployment(model="azure/cc-whisper-deployment", base_model="azure/whisper-next"),
    upload=WavUpload(kind="wav", seconds=3.5),
    request={"response_format": "json"},
    response=JsonResponse(content_type="application/json", body={"text": "hello"}),
    expected=ExactExpected(spend=0.000385, input_cost=0.000385, output_cost=0, prompt_tokens=0, completion_tokens=0),
)


# azure/tts-next
AZURE_TTS_NEXT_SPEECH_DEPLOYMENT: Final = CostTrackingTestCase(
    name="azure-tts-next-speech-deployment",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/tts-next",
    endpoint="/v1/audio/speech",
    deployment=Deployment(model="azure/cc-tts-deployment", base_model="azure/tts-next"),
    request={"input": "hello world", "voice": "alloy", "response_format": "mp3"},
    response=BinaryResponse(content_type="audio/mpeg", length=2048),
    expected=ExactExpected(spend=0.00011, input_cost=0.00011, output_cost=0, prompt_tokens=0, completion_tokens=0),
)


# azure/text-embedding-4-large
AZURE_TEXT_EMBEDDINGS_4_LARGE_DEPLOYMENT: Final = CostTrackingTestCase(
    name="azure-text-embeddings-4-large-deployment",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure/text-embedding-4-large",
    endpoint="/v1/embeddings",
    deployment=Deployment(model="azure/cc-pinned-embedding-deployment", base_model="azure/text-embedding-4-large"),
    request={"model": "$MODEL", "input": "azure embedding"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "azure/text-embedding-4-large",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        },
    ),
    expected=ExactExpected(spend=8.24e-06, input_cost=8.24e-06, output_cost=0.0, prompt_tokens=8, completion_tokens=0),
)


CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    AZURE_GPT_5_4_MINI_INPUT_TEXT,
    AZURE_GPT_5_4_MINI_CACHE_READ,
    AZURE_GPT_5_4_MINI_AUDIO_INPUT,
    AZURE_GPT_5_4_MINI_AUDIO_OUTPUT,
    AZURE_GPT_5_4_MINI_REASONING,
    AZURE_GPT_5_4_MINI_SERVICE_TIER_FLEX,
    AZURE_GPT_5_4_MINI_SERVICE_TIER_PRIORITY,
    AZURE_GPT_5_4_MINI_WEB_SEARCH_MEDIUM,
    AZURE_GPT_5_4_MINI_WEB_SEARCH_LOW,
    AZURE_GPT_5_4_MINI_WEB_SEARCH_HIGH,
    AZURE_GPT_5_4_MINI_STREAM,
    AZURE_GPT_5_4_MINI_STREAM_NO_USAGE,
    AZURE_GPT_5_4_MINI_STREAM_NO_USAGE_TOOL_CALL,
    AZURE_GPT_5_4_MINI_STREAM_NO_USAGE_IMAGE_INPUT,
    AZURE_GPT_5_4_MINI_RESPONSE_MODEL_OVERRIDE,
    AZURE_GPT_5_4_MINI_STREAM_RESPONSE_MODEL_OVERRIDE,
    AZURE_GPT_5_4_MINI_TOOL_CALL,
    AZURE_GPT_5_4_MINI_STREAM_TOOL_CALL,
    AZURE_GPT_5_4_MINI_STREAM_FULL_USAGE,
    AZURE_GPT_5_6_INPUT_TEXT,
    AZURE_GPT_5_6_CACHE_READ,
    AZURE_GPT_5_6_AUDIO_INPUT,
    AZURE_GPT_5_6_AUDIO_OUTPUT,
    AZURE_GPT_5_6_REASONING,
    AZURE_GPT_5_6_SERVICE_TIER_FLEX,
    AZURE_GPT_5_6_SERVICE_TIER_PRIORITY,
    AZURE_GPT_5_6_WEB_SEARCH_MEDIUM,
    AZURE_GPT_5_6_WEB_SEARCH_LOW,
    AZURE_GPT_5_6_WEB_SEARCH_HIGH,
    AZURE_GPT_5_6_STREAM,
    AZURE_GPT_5_6_STREAM_NO_USAGE,
    AZURE_GPT_5_6_STREAM_NO_USAGE_TOOL_CALL,
    AZURE_GPT_5_6_STREAM_NO_USAGE_IMAGE_INPUT,
    AZURE_GPT_5_6_RESPONSE_MODEL_OVERRIDE,
    AZURE_GPT_5_6_STREAM_RESPONSE_MODEL_OVERRIDE,
    AZURE_GPT_5_6_TOOL_CALL,
    AZURE_GPT_5_6_STREAM_TOOL_CALL,
    AZURE_GPT_5_6_STREAM_FULL_USAGE,
    AZURE_WHISPER_NEXT_TRANSCRIPTIONS_DEPLOYMENT,
    AZURE_TTS_NEXT_SPEECH_DEPLOYMENT,
    AZURE_TEXT_EMBEDDINGS_4_LARGE_DEPLOYMENT,
    AZURE_PINNED_GPT_5_4_MINI_STREAM,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(plain=AZURE_GPT_5_6_INPUT_TEXT, streamed=AZURE_GPT_5_6_STREAM),
    StreamParityTestCase(
        plain=AZURE_GPT_5_6_RESPONSE_MODEL_OVERRIDE, streamed=AZURE_GPT_5_6_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=AZURE_GPT_5_6_TOOL_CALL, streamed=AZURE_GPT_5_6_STREAM_TOOL_CALL),
)
