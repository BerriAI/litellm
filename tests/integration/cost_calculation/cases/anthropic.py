"""anthropic cost tracking cases, one CostTrackingTestCase literal per request shape (moved from cost_tracking_cases.json).

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


# claude-haiku-4-5
CLAUDE_HAIKU_4_5_INPUT_TEXT: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "83d8e1f3f711 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted answer 83d8e1f3f711"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0039, input_cost=0.00184, output_cost=0.00206, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_CACHE_READ: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-haiku-4-5",
    request={
        "model": "$MODEL",
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
                        "text": "e56cd6ddbc3b summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted answer e56cd6ddbc3b"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 640, "output_tokens": 380, "cache_read_input_tokens": 12288},
        },
    ),
    expected=ExactExpected(
        spend=0.0037688,
        input_cost=0.0018688,
        output_cost=0.0019,
        prompt_tokens=12928,
        completion_tokens=380,
        cache_read_cost=0.0012288,
    ),
)

CLAUDE_HAIKU_4_5_CACHE_WRITE_5M: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-cache_write_5m",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-haiku-4-5",
    request={
        "model": "$MODEL",
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
                        "text": "aead4d429a63 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted answer aead4d429a63"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 512,
                "output_tokens": 350,
                "cache_creation_input_tokens": 9216,
                "cache_creation": {"ephemeral_5m_input_tokens": 9216, "ephemeral_1h_input_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.013782, input_cost=0.012032, output_cost=0.00175, prompt_tokens=9728, completion_tokens=350
    ),
)

CLAUDE_HAIKU_4_5_CACHE_WRITE_1H: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-cache_write_1h",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-haiku-4-5",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "8defd838f26f summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted answer 8defd838f26f"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 512,
                "output_tokens": 350,
                "cache_creation_input_tokens": 9216,
                "cache_creation": {"ephemeral_5m_input_tokens": 2048, "ephemeral_1h_input_tokens": 7168},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.019158, input_cost=0.017408, output_cost=0.00175, prompt_tokens=9728, completion_tokens=350
    ),
)

CLAUDE_HAIKU_4_5_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "f7dcd0281161 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted answer f7dcd0281161"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "service_tier": "priority"},
        },
    ),
    expected=ExactExpected(
        spend=0.004875, input_cost=0.0023, output_cost=0.002575, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_ANTHROPIC_US_INFERENCE: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-anthropic_us_inference",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "1c0a1a2e155f summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted answer 1c0a1a2e155f"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "inference_geo": "us"},
        },
    ),
    expected=ExactExpected(
        spend=0.00429, input_cost=0.002024, output_cost=0.002266, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "540998778abd summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted answer 540998778abd"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "server_tool_use": {"web_search_requests": 3}},
        },
    ),
    expected=ExactExpected(
        spend=0.0339, input_cost=0.00184, output_cost=0.00206, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_STREAM: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "8feb52d222c0 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer 8feb52d222c0"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0039, input_cost=0.00184, output_cost=0.00206, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "ac21e9843010 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer ac21e9843010"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1e-06, output_cost_per_token=5e-06)),
)

CLAUDE_HAIKU_4_5_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "596ca026b176 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_$REQUEST_ID", "name": "get_weather", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1e-06, output_cost_per_token=5e-06)),
)

CLAUDE_HAIKU_4_5_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "89c83ea0f121 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer 89c83ea0f121"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=1e-06, output_cost_per_token=5e-06)),
)

CLAUDE_HAIKU_4_5_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "d5df85778fb1 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer d5df85778fb1"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0195, input_cost=0.0092, output_cost=0.0103, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "2f3ca8c25a81 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer 2f3ca8c25a81"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0195, input_cost=0.0092, output_cost=0.0103, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "74403961022c summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_$REQUEST_ID",
                    "name": "get_weather",
                    "input": {
                        "city": "Berlin",
                        "days": 7,
                        "units": "metric",
                        "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                    },
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0039, input_cost=0.00184, output_cost=0.00206, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
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
                        "text": "5a30a53bb4d6 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_$REQUEST_ID", "name": "get_weather", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0039, input_cost=0.00184, output_cost=0.00206, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-haiku-4-5",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "f89827fda6c5 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": 1840,
                        "cache_read_input_tokens": 6144,
                        "cache_creation_input_tokens": 3072,
                        "cache_creation": {"ephemeral_5m_input_tokens": 2048, "ephemeral_1h_input_tokens": 1024},
                    },
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer f89827fda6c5"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0091224, input_cost=0.0070624, output_cost=0.00206, prompt_tokens=11056, completion_tokens=412
    ),
)

CLAUDE_HAIKU_4_5_MESSAGES_INPUT_TEXT: Final = CostTrackingTestCase(
    name="claude-haiku-4-5-messages_input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-haiku-4-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0039, input_cost=0.00184, output_cost=0.00206, prompt_tokens=1840, completion_tokens=412
    ),
)


# claude-opus-5
CLAUDE_OPUS_5_INPUT_TEXT: Final = CostTrackingTestCase(
    name="claude-opus-5-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
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
                        "text": "d4916e93889c summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer d4916e93889c"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0195, input_cost=0.0092, output_cost=0.0103, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_CACHE_READ: Final = CostTrackingTestCase(
    name="claude-opus-5-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
    request={
        "model": "$MODEL",
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
                        "text": "b83799f51ed7 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer b83799f51ed7"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 640, "output_tokens": 380, "cache_read_input_tokens": 12288},
        },
    ),
    expected=ExactExpected(
        spend=0.018844, input_cost=0.009344, output_cost=0.0095, prompt_tokens=12928, completion_tokens=380
    ),
)

CLAUDE_OPUS_5_CACHE_WRITE_5M: Final = CostTrackingTestCase(
    name="claude-opus-5-cache_write_5m",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
    request={
        "model": "$MODEL",
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
                        "text": "5cfdc176130a summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer 5cfdc176130a"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 512,
                "output_tokens": 350,
                "cache_creation_input_tokens": 9216,
                "cache_creation": {"ephemeral_5m_input_tokens": 9216, "ephemeral_1h_input_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.06891, input_cost=0.06016, output_cost=0.00875, prompt_tokens=9728, completion_tokens=350
    ),
)

CLAUDE_OPUS_5_CACHE_WRITE_1H: Final = CostTrackingTestCase(
    name="claude-opus-5-cache_write_1h",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "6f9107c3b3ff summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer 6f9107c3b3ff"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 512,
                "output_tokens": 350,
                "cache_creation_input_tokens": 9216,
                "cache_creation": {"ephemeral_5m_input_tokens": 2048, "ephemeral_1h_input_tokens": 7168},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.09579, input_cost=0.08704, output_cost=0.00875, prompt_tokens=9728, completion_tokens=350
    ),
)

CLAUDE_OPUS_5_TIERED_INPUT_ABOVE_200K: Final = CostTrackingTestCase(
    name="claude-opus-5-tiered_input_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
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
                        "text": "f7bec63ac6ff summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer f7bec63ac6ff"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 204800, "output_tokens": 620},
        },
    ),
    expected=ExactExpected(
        spend=2.07125, input_cost=2.048, output_cost=0.02325, prompt_tokens=204800, completion_tokens=620
    ),
)

CLAUDE_OPUS_5_TIERED_CACHE_READ_ABOVE_200K: Final = CostTrackingTestCase(
    name="claude-opus-5-tiered_cache_read_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
    request={
        "model": "$MODEL",
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
                        "text": "58ab3f8e01f6 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer 58ab3f8e01f6"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4096, "output_tokens": 480, "cache_read_input_tokens": 201728},
        },
    ),
    expected=ExactExpected(
        spend=0.260688, input_cost=0.242688, output_cost=0.018, prompt_tokens=205824, completion_tokens=480
    ),
)

CLAUDE_OPUS_5_TIERED_CACHE_WRITE_ABOVE_200K: Final = CostTrackingTestCase(
    name="claude-opus-5-tiered_cache_write_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
    request={
        "model": "$MODEL",
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
                        "text": "1a923968b132 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer 1a923968b132"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 4096,
                "output_tokens": 480,
                "cache_creation_input_tokens": 200704,
                "cache_creation": {"ephemeral_5m_input_tokens": 200704, "ephemeral_1h_input_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=2.56776, input_cost=2.54976, output_cost=0.018, prompt_tokens=204800, completion_tokens=480
    ),
)

CLAUDE_OPUS_5_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="claude-opus-5-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
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
                        "text": "928b583c6a13 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer 928b583c6a13"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "service_tier": "priority"},
        },
    ),
    expected=ExactExpected(
        spend=0.024375, input_cost=0.0115, output_cost=0.012875, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_ANTHROPIC_FAST_MODE: Final = CostTrackingTestCase(
    name="claude-opus-5-anthropic_fast_mode",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
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
                        "text": "dd7504ab4a95 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer dd7504ab4a95"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "speed": "fast"},
        },
    ),
    expected=ExactExpected(
        spend=0.117, input_cost=0.0552, output_cost=0.0618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_ANTHROPIC_US_INFERENCE: Final = CostTrackingTestCase(
    name="claude-opus-5-anthropic_us_inference",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
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
                        "text": "7dcf31884733 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer 7dcf31884733"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "inference_geo": "us"},
        },
    ),
    expected=ExactExpected(
        spend=0.02145, input_cost=0.01012, output_cost=0.01133, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="claude-opus-5-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5",
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
                        "text": "e63cb0e28801 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "scripted answer e63cb0e28801"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "server_tool_use": {"web_search_requests": 3}},
        },
    ),
    expected=ExactExpected(
        spend=0.0495, input_cost=0.0092, output_cost=0.0103, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_STREAM: Final = CostTrackingTestCase(
    name="claude-opus-5-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
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
                        "text": "1fcb7b21debc summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer 1fcb7b21debc"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0195, input_cost=0.0092, output_cost=0.0103, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="claude-opus-5-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
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
                        "text": "e5bff69088af summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer e5bff69088af"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5e-06, output_cost_per_token=2.5e-05)),
)

CLAUDE_OPUS_5_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-opus-5-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
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
                        "text": "b6ef7189d74f summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_$REQUEST_ID", "name": "get_weather", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5e-06, output_cost_per_token=2.5e-05)),
)

CLAUDE_OPUS_5_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="claude-opus-5-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
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
                        "text": "9901e704cc69 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer 9901e704cc69"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5e-06, output_cost_per_token=2.5e-05)),
)

CLAUDE_OPUS_5_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="claude-opus-5-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
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
                        "text": "14075d9902ec summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer 14075d9902ec"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0117, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="claude-opus-5-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
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
                        "text": "aa3357727723 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer aa3357727723"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0117, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-opus-5-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
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
                        "text": "8e5f37db0dfc summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_$REQUEST_ID",
                    "name": "get_weather",
                    "input": {
                        "city": "Berlin",
                        "days": 7,
                        "units": "metric",
                        "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                    },
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0195, input_cost=0.0092, output_cost=0.0103, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-opus-5-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
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
                        "text": "7cfe98295218 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_$REQUEST_ID", "name": "get_weather", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0195, input_cost=0.0092, output_cost=0.0103, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_OPUS_5_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="claude-opus-5-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-opus-5",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "0bacb827a61a summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": 1840,
                        "cache_read_input_tokens": 6144,
                        "cache_creation_input_tokens": 3072,
                        "cache_creation": {"ephemeral_5m_input_tokens": 2048, "ephemeral_1h_input_tokens": 1024},
                    },
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer 0bacb827a61a"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.045612, input_cost=0.035312, output_cost=0.0103, prompt_tokens=11056, completion_tokens=412
    ),
)


# claude-sonnet-5
CLAUDE_SONNET_5_INPUT_TEXT: Final = CostTrackingTestCase(
    name="claude-sonnet-5-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "e672859760ae summarize the attached material in one line and name the city weather",
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
            "id": "msg_$UNIQUE_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer e672859760ae"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0117, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412, rollups=True
    ),
)

CLAUDE_SONNET_5_CACHE_READ: Final = CostTrackingTestCase(
    name="claude-sonnet-5-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    request={
        "model": "$MODEL",
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
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
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
    expected=ExactExpected(
        spend=0.0113064, input_cost=0.0056064, output_cost=0.0057, prompt_tokens=12928, completion_tokens=380
    ),
)

CLAUDE_SONNET_5_CACHE_WRITE_5M: Final = CostTrackingTestCase(
    name="claude-sonnet-5-cache_write_5m",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    request={
        "model": "$MODEL",
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
                        "text": "212f38c1ea0d summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer 212f38c1ea0d"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 512,
                "output_tokens": 350,
                "cache_creation_input_tokens": 9216,
                "cache_creation": {"ephemeral_5m_input_tokens": 9216, "ephemeral_1h_input_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.041346, input_cost=0.036096, output_cost=0.00525, prompt_tokens=9728, completion_tokens=350
    ),
)

CLAUDE_SONNET_5_CACHE_WRITE_1H: Final = CostTrackingTestCase(
    name="claude-sonnet-5-cache_write_1h",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "638e0a865af7 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer 638e0a865af7"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 512,
                "output_tokens": 350,
                "cache_creation_input_tokens": 9216,
                "cache_creation": {"ephemeral_5m_input_tokens": 2048, "ephemeral_1h_input_tokens": 7168},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.057474, input_cost=0.052224, output_cost=0.00525, prompt_tokens=9728, completion_tokens=350
    ),
)

CLAUDE_SONNET_5_TIERED_INPUT_ABOVE_200K: Final = CostTrackingTestCase(
    name="claude-sonnet-5-tiered_input_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "5ccef99d1220 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer 5ccef99d1220"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 204800, "output_tokens": 620},
        },
    ),
    expected=ExactExpected(
        spend=1.24275, input_cost=1.2288, output_cost=0.01395, prompt_tokens=204800, completion_tokens=620
    ),
)

CLAUDE_SONNET_5_TIERED_CACHE_READ_ABOVE_200K: Final = CostTrackingTestCase(
    name="claude-sonnet-5-tiered_cache_read_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    request={
        "model": "$MODEL",
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
                        "text": "aaa479b1e950 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer aaa479b1e950"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4096, "output_tokens": 480, "cache_read_input_tokens": 201728},
        },
    ),
    expected=ExactExpected(
        spend=0.1564128, input_cost=0.1456128, output_cost=0.0108, prompt_tokens=205824, completion_tokens=480
    ),
)

CLAUDE_SONNET_5_TIERED_CACHE_WRITE_ABOVE_200K: Final = CostTrackingTestCase(
    name="claude-sonnet-5-tiered_cache_write_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    request={
        "model": "$MODEL",
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
                        "text": "f3c0e1d4dedd summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer f3c0e1d4dedd"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 4096,
                "output_tokens": 480,
                "cache_creation_input_tokens": 200704,
                "cache_creation": {"ephemeral_5m_input_tokens": 200704, "ephemeral_1h_input_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=1.540656, input_cost=1.529856, output_cost=0.0108, prompt_tokens=204800, completion_tokens=480
    ),
)

CLAUDE_SONNET_5_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="claude-sonnet-5-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "9863908ec91f summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer 9863908ec91f"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "service_tier": "priority"},
        },
    ),
    expected=ExactExpected(
        spend=0.014625, input_cost=0.0069, output_cost=0.007725, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_ANTHROPIC_US_INFERENCE: Final = CostTrackingTestCase(
    name="claude-sonnet-5-anthropic_us_inference",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "bb37086ce8e6 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer bb37086ce8e6"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "inference_geo": "us"},
        },
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="claude-sonnet-5-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "ddcbec1b7eb2 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted answer ddcbec1b7eb2"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412, "server_tool_use": {"web_search_requests": 3}},
        },
    ),
    expected=ExactExpected(
        spend=0.0417, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_STREAM: Final = CostTrackingTestCase(
    name="claude-sonnet-5-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "ca259a6916f2 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer ca259a6916f2"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0117, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="claude-sonnet-5-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "b14b060d38cc summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer b14b060d38cc"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=3e-06, output_cost_per_token=1.5e-05),
        prompt_tokens=48,
        completion_tokens=10,
    ),
)

CLAUDE_SONNET_5_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-sonnet-5-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "23d6e2f6eb94 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_fixture_0001", "name": "get_weather", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=3e-06, output_cost_per_token=1.5e-05), min_completion_tokens=60
    ),
)

CLAUDE_SONNET_5_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="claude-sonnet-5-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "f803710311e5 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer f803710311e5"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=3e-06, output_cost_per_token=1.5e-05),
        prompt_tokens=303,
        completion_tokens=10,
    ),
)

CLAUDE_SONNET_5_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="claude-sonnet-5-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "04c8cd550f99 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "scripted answer 04c8cd550f99"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0039, input_cost=0.00184, output_cost=0.00206, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="claude-sonnet-5-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "3ca6439c3e0d summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer 3ca6439c3e0d"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0039, input_cost=0.00184, output_cost=0.00206, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-sonnet-5-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "e32fe8463152 summarize the attached material in one line and name the city weather",
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
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_$REQUEST_ID",
                    "name": "get_weather",
                    "input": {
                        "city": "Berlin",
                        "days": 7,
                        "units": "metric",
                        "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                    },
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0117, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="claude-sonnet-5-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
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
                        "text": "11512728994f summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1840},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_$REQUEST_ID", "name": "get_weather", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil',
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0117, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="claude-sonnet-5-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "543a97cebc29 summarize the attached material in one line and name the city weather",
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
                "type": "message_start",
                "message": {
                    "id": "msg_$REQUEST_ID",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": 1840,
                        "cache_read_input_tokens": 6144,
                        "cache_creation_input_tokens": 3072,
                        "cache_creation": {"ephemeral_5m_input_tokens": 2048, "ephemeral_1h_input_tokens": 1024},
                    },
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "scripted answer 543a97cebc29"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 412}},
            {"type": "message_stop"},
        ),
    ),
    expected=ExactExpected(
        spend=0.0273672, input_cost=0.0211872, output_cost=0.00618, prompt_tokens=11056, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_MESSAGES_INPUT_TEXT: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0117, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_MESSAGES_CACHE_READ: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "cached text", "cache_control": {"type": "ephemeral"}}],
            }
        ],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 640, "output_tokens": 380, "cache_read_input_tokens": 12288},
        },
    ),
    expected=ExactExpected(
        spend=0.0113064,
        input_cost=0.0056064,
        output_cost=0.0057,
        prompt_tokens=12928,
        completion_tokens=380,
        cache_read_cost=0.0036864,
    ),
)

CLAUDE_SONNET_5_MESSAGES_CACHE_WRITE_5M: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_cache_write_5m",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 350,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "cache this text", "cache_control": {"type": "ephemeral"}}],
            }
        ],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 512,
                "output_tokens": 350,
                "cache_creation_input_tokens": 9216,
                "cache_creation": {"ephemeral_5m_input_tokens": 9216, "ephemeral_1h_input_tokens": 0},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.041346,
        input_cost=0.036096,
        output_cost=0.00525,
        prompt_tokens=9728,
        completion_tokens=350,
        cache_creation_cost=0.03456,
    ),
)

CLAUDE_SONNET_5_MESSAGES_CACHE_WRITE_1H: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_cache_write_1h",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 350,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "cache this text for an hour",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        ],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 512,
                "output_tokens": 350,
                "cache_creation_input_tokens": 9216,
                "cache_creation": {"ephemeral_5m_input_tokens": 2048, "ephemeral_1h_input_tokens": 7168},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.057474,
        input_cost=0.052224,
        output_cost=0.00525,
        prompt_tokens=9728,
        completion_tokens=350,
        cache_creation_cost=0.050688,
    ),
)

CLAUDE_SONNET_5_MESSAGES_WEB_SEARCH: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_web_search",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srv_$REQUEST_ID",
                    "name": "web_search",
                    "input": {"query": "scripted query"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv_$REQUEST_ID",
                    "content": [
                        {"type": "web_search_result", "title": "scripted result", "url": "https://scripted.example"}
                    ],
                },
                {"type": "text", "text": "scripted response"},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1840, "output_tokens": 412, "server_tool_use": {"web_search_requests": 2}},
        },
    ),
    expected=ExactExpected(
        spend=0.0317,
        input_cost=0.00552,
        output_cost=0.00618,
        prompt_tokens=1840,
        completion_tokens=412,
        tool_usage_cost=0.02,
    ),
)

CLAUDE_SONNET_5_MESSAGES_STREAM: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
        "stream": True,
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_$REQUEST_ID","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1840}}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"scripted "}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"response"}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":412}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ),
    ),
    expected=ExactExpected(
        spend=0.0117, input_cost=0.00552, output_cost=0.00618, prompt_tokens=1840, completion_tokens=412
    ),
)

CLAUDE_SONNET_5_MESSAGES_STREAM_CACHE_READ: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_stream_cache_read",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="claude-sonnet-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 380,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
        "stream": True,
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_$REQUEST_ID","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":640,"cache_read_input_tokens":12288}}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"scripted "}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"response"}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":380}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ),
    ),
    expected=ExactExpected(
        spend=0.0113064,
        input_cost=0.0056064,
        output_cost=0.0057,
        prompt_tokens=12928,
        completion_tokens=380,
        cache_read_cost=0.0036864,
    ),
)

CLAUDE_SONNET_5_MESSAGES_TIERED_INPUT_ABOVE_200K: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages_tiered_input_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 620,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 210000, "output_tokens": 620},
        },
    ),
    expected=ExactExpected(
        spend=1.27395, input_cost=1.26, output_cost=0.01395, prompt_tokens=210000, completion_tokens=620
    ),
)

CLAUDE_SONNET_5_PASSTHROUGH_MESSAGES: Final = CostTrackingTestCase(
    name="claude-sonnet-5-passthrough-messages",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    endpoint="/anthropic/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0117,
        input_cost=0.00552,
        output_cost=0.00618,
        prompt_tokens=1840,
        completion_tokens=412,
        breakdown_persisted=False,
        cost_header=False,
    ),
)

CLAUDE_SONNET_5_PASSTHROUGH_MESSAGES_CACHE_READ: Final = CostTrackingTestCase(
    name="claude-sonnet-5-passthrough-messages_cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    endpoint="/anthropic/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "cached text", "cache_control": {"type": "ephemeral"}}],
            }
        ],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 640, "output_tokens": 380, "cache_read_input_tokens": 12288},
        },
    ),
    expected=ExactExpected(
        spend=0.0113064,
        input_cost=0.0056064,
        output_cost=0.0057,
        prompt_tokens=12928,
        completion_tokens=380,
        cache_read_cost=0.0036864,
        breakdown_persisted=False,
        cost_header=False,
    ),
)

CLAUDE_SONNET_5_CACHE_CREATION_1H_ABOVE_200K: Final = CostTrackingTestCase(
    name="claude-sonnet-5-cache_creation_1h_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-sonnet-5",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "one hour cache"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 150000,
                "cache_creation_input_tokens": 60000,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 60000},
                "cache_read_input_tokens": 0,
                "output_tokens": 412,
            },
        },
    ),
    expected=ExactExpected(
        spend=1.62927,
        input_cost=1.62,
        output_cost=0.00927,
        cache_creation_cost=0.72,
        prompt_tokens=210000,
        completion_tokens=412,
    ),
)


# claude-opus-5-5
CLAUDE_OPUS_5_5_INPUT_TEXT: Final = CostTrackingTestCase(
    name="claude-opus-5-5-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="claude-opus-5-5",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "Hello."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.0156, input_cost=0.00736, output_cost=0.00824, prompt_tokens=1840, completion_tokens=412
    ),
)


CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    CLAUDE_HAIKU_4_5_INPUT_TEXT,
    CLAUDE_HAIKU_4_5_CACHE_READ,
    CLAUDE_HAIKU_4_5_CACHE_WRITE_5M,
    CLAUDE_HAIKU_4_5_CACHE_WRITE_1H,
    CLAUDE_HAIKU_4_5_SERVICE_TIER_PRIORITY,
    CLAUDE_HAIKU_4_5_ANTHROPIC_US_INFERENCE,
    CLAUDE_HAIKU_4_5_WEB_SEARCH_MEDIUM,
    CLAUDE_HAIKU_4_5_STREAM,
    CLAUDE_HAIKU_4_5_STREAM_NO_USAGE,
    CLAUDE_HAIKU_4_5_STREAM_NO_USAGE_TOOL_CALL,
    CLAUDE_HAIKU_4_5_STREAM_NO_USAGE_IMAGE_INPUT,
    CLAUDE_HAIKU_4_5_RESPONSE_MODEL_OVERRIDE,
    CLAUDE_HAIKU_4_5_STREAM_RESPONSE_MODEL_OVERRIDE,
    CLAUDE_HAIKU_4_5_TOOL_CALL,
    CLAUDE_HAIKU_4_5_STREAM_TOOL_CALL,
    CLAUDE_HAIKU_4_5_STREAM_FULL_USAGE,
    CLAUDE_OPUS_5_INPUT_TEXT,
    CLAUDE_OPUS_5_CACHE_READ,
    CLAUDE_OPUS_5_CACHE_WRITE_5M,
    CLAUDE_OPUS_5_CACHE_WRITE_1H,
    CLAUDE_OPUS_5_TIERED_INPUT_ABOVE_200K,
    CLAUDE_OPUS_5_TIERED_CACHE_READ_ABOVE_200K,
    CLAUDE_OPUS_5_TIERED_CACHE_WRITE_ABOVE_200K,
    CLAUDE_OPUS_5_SERVICE_TIER_PRIORITY,
    CLAUDE_OPUS_5_ANTHROPIC_FAST_MODE,
    CLAUDE_OPUS_5_ANTHROPIC_US_INFERENCE,
    CLAUDE_OPUS_5_WEB_SEARCH_MEDIUM,
    CLAUDE_OPUS_5_STREAM,
    CLAUDE_OPUS_5_STREAM_NO_USAGE,
    CLAUDE_OPUS_5_STREAM_NO_USAGE_TOOL_CALL,
    CLAUDE_OPUS_5_STREAM_NO_USAGE_IMAGE_INPUT,
    CLAUDE_OPUS_5_RESPONSE_MODEL_OVERRIDE,
    CLAUDE_OPUS_5_STREAM_RESPONSE_MODEL_OVERRIDE,
    CLAUDE_OPUS_5_TOOL_CALL,
    CLAUDE_OPUS_5_STREAM_TOOL_CALL,
    CLAUDE_OPUS_5_STREAM_FULL_USAGE,
    CLAUDE_SONNET_5_INPUT_TEXT,
    CLAUDE_SONNET_5_CACHE_READ,
    CLAUDE_SONNET_5_CACHE_WRITE_5M,
    CLAUDE_SONNET_5_CACHE_WRITE_1H,
    CLAUDE_SONNET_5_TIERED_INPUT_ABOVE_200K,
    CLAUDE_SONNET_5_TIERED_CACHE_READ_ABOVE_200K,
    CLAUDE_SONNET_5_TIERED_CACHE_WRITE_ABOVE_200K,
    CLAUDE_SONNET_5_SERVICE_TIER_PRIORITY,
    CLAUDE_SONNET_5_ANTHROPIC_US_INFERENCE,
    CLAUDE_SONNET_5_WEB_SEARCH_MEDIUM,
    CLAUDE_SONNET_5_STREAM,
    CLAUDE_SONNET_5_STREAM_NO_USAGE,
    CLAUDE_SONNET_5_STREAM_NO_USAGE_TOOL_CALL,
    CLAUDE_SONNET_5_STREAM_NO_USAGE_IMAGE_INPUT,
    CLAUDE_SONNET_5_RESPONSE_MODEL_OVERRIDE,
    CLAUDE_SONNET_5_STREAM_RESPONSE_MODEL_OVERRIDE,
    CLAUDE_SONNET_5_TOOL_CALL,
    CLAUDE_SONNET_5_STREAM_TOOL_CALL,
    CLAUDE_SONNET_5_STREAM_FULL_USAGE,
    CLAUDE_SONNET_5_MESSAGES_INPUT_TEXT,
    CLAUDE_SONNET_5_MESSAGES_CACHE_READ,
    CLAUDE_SONNET_5_MESSAGES_CACHE_WRITE_5M,
    CLAUDE_SONNET_5_MESSAGES_CACHE_WRITE_1H,
    CLAUDE_SONNET_5_MESSAGES_WEB_SEARCH,
    CLAUDE_SONNET_5_MESSAGES_STREAM,
    CLAUDE_SONNET_5_MESSAGES_STREAM_CACHE_READ,
    CLAUDE_SONNET_5_MESSAGES_TIERED_INPUT_ABOVE_200K,
    CLAUDE_HAIKU_4_5_MESSAGES_INPUT_TEXT,
    CLAUDE_SONNET_5_PASSTHROUGH_MESSAGES,
    CLAUDE_SONNET_5_PASSTHROUGH_MESSAGES_CACHE_READ,
    CLAUDE_SONNET_5_CACHE_CREATION_1H_ABOVE_200K,
    CLAUDE_OPUS_5_5_INPUT_TEXT,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(plain=CLAUDE_HAIKU_4_5_INPUT_TEXT, streamed=CLAUDE_HAIKU_4_5_STREAM),
    StreamParityTestCase(
        plain=CLAUDE_HAIKU_4_5_RESPONSE_MODEL_OVERRIDE, streamed=CLAUDE_HAIKU_4_5_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=CLAUDE_HAIKU_4_5_TOOL_CALL, streamed=CLAUDE_HAIKU_4_5_STREAM_TOOL_CALL),
    StreamParityTestCase(plain=CLAUDE_OPUS_5_INPUT_TEXT, streamed=CLAUDE_OPUS_5_STREAM),
    StreamParityTestCase(
        plain=CLAUDE_OPUS_5_RESPONSE_MODEL_OVERRIDE, streamed=CLAUDE_OPUS_5_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=CLAUDE_OPUS_5_TOOL_CALL, streamed=CLAUDE_OPUS_5_STREAM_TOOL_CALL),
    StreamParityTestCase(
        plain=CLAUDE_SONNET_5_RESPONSE_MODEL_OVERRIDE, streamed=CLAUDE_SONNET_5_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=CLAUDE_SONNET_5_TOOL_CALL, streamed=CLAUDE_SONNET_5_STREAM_TOOL_CALL),
    StreamParityTestCase(plain=CLAUDE_SONNET_5_MESSAGES_INPUT_TEXT, streamed=CLAUDE_SONNET_5_MESSAGES_STREAM),
)
