"""vertex_ai cost tracking cases, one CostTrackingTestCase literal per request shape (moved from cost_tracking_cases.json).

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


# gemini-3.1-pro
GEMINI_3_1_PRO_INPUT_TEXT: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "5fdf6b7dd9b9 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 5fdf6b7dd9b9"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.0090552, input_cost=0.003864, output_cost=0.0051912, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_CACHE_READ: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "bb9b95a5e878 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer bb9b95a5e878"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12928,
                "candidatesTokenCount": 380,
                "totalTokenCount": 13308,
                "cachedContentTokenCount": 12288,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 12928}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.00871248, input_cost=0.00392448, output_cost=0.004788, prompt_tokens=12928, completion_tokens=380
    ),
)

GEMINI_3_1_PRO_AUDIO_INPUT: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-audio_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "eccb8318be2d summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer eccb8318be2d"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1546,
                "candidatesTokenCount": 210,
                "totalTokenCount": 1756,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 96},
                    {"modality": "AUDIO", "tokenCount": 1450},
                ],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.0067626, input_cost=0.0041166, output_cost=0.002646, prompt_tokens=1546, completion_tokens=210
    ),
)

GEMINI_3_1_PRO_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-image_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "f50f723a74f1 summarize the attached material in one line and name the city weather",
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
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer f50f723a74f1"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 2116,
                "candidatesTokenCount": 240,
                "totalTokenCount": 2356,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 310},
                    {"modality": "IMAGE", "tokenCount": 1806},
                ],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.0078288, input_cost=0.0048048, output_cost=0.003024, prompt_tokens=2116, completion_tokens=240
    ),
)

GEMINI_3_1_PRO_REASONING: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "a09586282605 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer a09586282605"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1240,
                "candidatesTokenCount": 560,
                "thoughtsTokenCount": 3480,
                "totalTokenCount": 5280,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1240}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.05664, input_cost=0.002604, output_cost=0.054036, prompt_tokens=1240, completion_tokens=4040
    ),
)

GEMINI_3_1_PRO_TIERED_INPUT_ABOVE_200K: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-tiered_input_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "7972fad2f18c summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 7972fad2f18c"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 204800,
                "candidatesTokenCount": 620,
                "totalTokenCount": 205420,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 204800}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.871878, input_cost=0.86016, output_cost=0.011718, prompt_tokens=204800, completion_tokens=620
    ),
)

GEMINI_3_1_PRO_TIERED_CACHE_READ_ABOVE_200K: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-tiered_cache_read_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "c6ae4eac7c46 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer c6ae4eac7c46"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 205824,
                "candidatesTokenCount": 480,
                "totalTokenCount": 206304,
                "cachedContentTokenCount": 201728,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 205824}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.11100096, input_cost=0.10192896, output_cost=0.009072, prompt_tokens=205824, completion_tokens=480
    ),
)

GEMINI_3_1_PRO_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "afc20048852d summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer afc20048852d"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                "trafficType": "ON_DEMAND_FLEX",
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.0045276, input_cost=0.001932, output_cost=0.0025956, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "c45311f260f5 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer c45311f260f5"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                "trafficType": "ON_DEMAND_PRIORITY",
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.011319, input_cost=0.00483, output_cost=0.006489, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_WEB_SEARCH_MEDIUM: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-web_search_medium",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "73631ea17d2b summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"googleSearch": {}}],
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 73631ea17d2b"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "groundingMetadata": {"webSearchQueries": ["query 0", "query 1", "query 2"]},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.1140552, input_cost=0.003864, output_cost=0.0051912, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_GOOGLE_MAPS_GROUNDING: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-google_maps_grounding",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "5c6ab7918d5a summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"googleMaps": {}}],
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 5c6ab7918d5a"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "groundingMetadata": {
                        "webSearchQueries": ["maps query 0"],
                        "groundingChunks": [{"maps": {"uri": "https://maps.google.com/?cid=0"}}],
                        "googleMapsWidgetContextToken": "token_$REQUEST_ID",
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.0340552, input_cost=0.003864, output_cost=0.0051912, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_FALLBACK_VIDEO_TOKENS_AT_INPUT_RATE: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-fallback_video_tokens_at_input_rate",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "5fc254e9189f summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:video/mp4;base64,AAAAGGZ0eXBpc29tAAACAGlzb21pc282AAACCG1kYXQNFBsiKTA3PkVMU1phaG92fYSLkpmgp661vMPK0djf5u30+wIJEBceJSwzOkFIT1ZdZGtyeYCHjpWco6qxuL/GzdTb4unw9/4FDBMaISgvNj1ES1JZYGdudXyDipGYn6attLvCydDX3uXs8/oBCA8WHSQrMjlAR05VXGNqcXh/ho2Um6KpsLe+xczT2uHo7/b9BAsSGSAnLjU8Q0pRWF9mbXR7gomQl56lrLO6wcjP1t3k6/L5AAcOFRwjKjE4P0ZNVFtiaXB3foWMk5qhqK+2vcTL0tng5+71/AMKERgfJi00O0JJUFdeZWxzeoGIj5adpKuyucDHztXc4+rx+P8GDRQbIikwNz5FTFNaYWhvdn2Ei5KZoKeutbzDytHY3+bt9PsCCRAXHiUsMzpBSE9WXWRrcnmAh46VnKOqsbi/xs3U2+Lp8Pf+BQwTGiEoLzY9REtSWWBnbnV8g4qRmJ+mrbS7wsnQ197l7PP6AQgPFh0kKzI5QEdOVVxjanF4f4aNlJuiqbC3vsXM09rh6O/2/QQLEhkgJy41PENKUVhfZm10e4KJkJeepayzusHIz9bd5Ovy+QAHDhUcIyoxOD9GTVRbYmlwd36FjJOaoaivtr3Ey9LZ4Ofu9fwDChEYHyYtNDtCSVBXXmVsc3qBiI+WnaSrsrnAx87V3OPq8fj/Bg==",
                            "format": "mp4",
                        },
                    },
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 5fc254e9189f"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8060,
                "candidatesTokenCount": 300,
                "totalTokenCount": 8360,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 140},
                    {"modality": "VIDEO", "tokenCount": 7920},
                ],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.020706, input_cost=0.016926, output_cost=0.00378, prompt_tokens=8060, completion_tokens=300
    ),
)

GEMINI_3_1_PRO_STREAM: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "52b6a80ff038 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 52b6a80ff038"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.1-pro",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 412,
                    "totalTokenCount": 2252,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.1-pro",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.0090552, input_cost=0.003864, output_cost=0.0051912, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "4985d6423ec4 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 4985d6423ec4"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.1-pro",
            }
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=2.1e-06, output_cost_per_token=1.26e-05)),
)

GEMINI_3_1_PRO_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "11bca0892f81 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {
                                            "city": "Berlin",
                                            "days": 7,
                                            "units": "metric",
                                            "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                        },
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.1-pro",
            }
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=2.1e-06, output_cost_per_token=1.26e-05)),
)

GEMINI_3_1_PRO_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "4c92224d4b84 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 4c92224d4b84"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.1-pro",
            }
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=2.1e-06, output_cost_per_token=1.26e-05)),
)

GEMINI_3_1_PRO_PROMPT_BLOCKED: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-prompt_blocked",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "6c159519a099 summarize the attached material in one line and name the city weather",
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
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "HIGH", "blocked": True}],
            },
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 0,
                "totalTokenCount": 1840,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.003864, input_cost=0.003864, output_cost=0.0, prompt_tokens=1840, completion_tokens=0
    ),
)

GEMINI_3_1_PRO_STREAM_PROMPT_BLOCKED: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-stream_prompt_blocked",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "7df769816861 summarize the attached material in one line and name the city weather",
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
                "promptFeedback": {
                    "blockReason": "SAFETY",
                    "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "HIGH", "blocked": True}],
                },
                "modelVersion": "gemini-3.1-pro",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 1840,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.1-pro",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.003864, input_cost=0.003864, output_cost=0.0, prompt_tokens=1840, completion_tokens=0
    ),
)

GEMINI_3_1_PRO_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "956e05125691 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 956e05125691"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.00224224, input_cost=0.0009568, output_cost=0.00128544, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "2048ef936293 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 2048ef936293"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.8-flash",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 412,
                    "totalTokenCount": 2252,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.8-flash",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.00224224, input_cost=0.0009568, output_cost=0.00128544, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_TOOL_CALL: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "2a68816dc8ea summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {
                                        "city": "Berlin",
                                        "days": 7,
                                        "units": "metric",
                                        "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                    },
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.0090552, input_cost=0.003864, output_cost=0.0051912, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "7ba6668f10df summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {
                                            "city": "Berlin",
                                            "days": 7,
                                            "units": "metric",
                                            "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                        },
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.1-pro",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 412,
                    "totalTokenCount": 2252,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.1-pro",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.0090552, input_cost=0.003864, output_cost=0.0051912, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_1_PRO_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
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
                        "text": "54e6d8c321ef summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 54e6d8c321ef"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.1-pro",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 8314,
                    "candidatesTokenCount": 412,
                    "thoughtsTokenCount": 900,
                    "totalTokenCount": 9626,
                    "cachedContentTokenCount": 6144,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 7984},
                        {"modality": "AUDIO", "tokenCount": 330},
                    ],
                },
                "modelVersion": "gemini-3.1-pro",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.02338644, input_cost=0.00604524, output_cost=0.0173412, prompt_tokens=8314, completion_tokens=1312
    ),
)

GEMINI_3_1_PRO_PASSTHROUGH_STREAM_GENERATE_CONTENT_PRICED_VIA_VERTEX_KEY: Final = CostTrackingTestCase(
    name="gemini-3.1-pro-passthrough-stream_generate_content_priced_via_vertex_key",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.1-pro",
    endpoint="/gemini/v1beta/models/$MODEL:streamGenerateContent?alt=sse",
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
                        "text": "52b6a80ff038 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 52b6a80ff038"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.1-pro",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 412,
                    "totalTokenCount": 2252,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.1-pro",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.0090552,
        input_cost=0.003864,
        output_cost=0.0051912,
        prompt_tokens=1840,
        completion_tokens=412,
        breakdown_persisted=False,
        cost_header=False,
    ),
)


# gemini-3.8-flash
GEMINI_3_8_FLASH_INPUT_TEXT: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "e7c21c357fb0 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer e7c21c357fb0"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.00224224, input_cost=0.0009568, output_cost=0.00128544, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_CACHE_READ: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "c58ea8fe6a99 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer c58ea8fe6a99"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12928,
                "candidatesTokenCount": 380,
                "totalTokenCount": 13308,
                "cachedContentTokenCount": 12288,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 12928}],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.002157376, input_cost=0.000971776, output_cost=0.0011856, prompt_tokens=12928, completion_tokens=380
    ),
)

GEMINI_3_8_FLASH_AUDIO_INPUT: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-audio_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "3e33892c4f9f summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 3e33892c4f9f"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1546,
                "candidatesTokenCount": 210,
                "totalTokenCount": 1756,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 96},
                    {"modality": "AUDIO", "tokenCount": 1450},
                ],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.00221312, input_cost=0.00155792, output_cost=0.0006552, prompt_tokens=1546, completion_tokens=210
    ),
)

GEMINI_3_8_FLASH_AUDIO_OUTPUT: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-audio_output",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "30bf1c0de6fe summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 30bf1c0de6fe"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 220,
                "candidatesTokenCount": 1300,
                "totalTokenCount": 1520,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 220}],
                "candidatesTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 180},
                    {"modality": "AUDIO", "tokenCount": 1120},
                ],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.0076648, input_cost=0.0001144, output_cost=0.0075504, prompt_tokens=220, completion_tokens=1300
    ),
)

GEMINI_3_8_FLASH_VIDEO_INPUT: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-video_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "5f5da5957185 summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:video/mp4;base64,AAAAGGZ0eXBpc29tAAACAGlzb21pc282AAACCG1kYXQNFBsiKTA3PkVMU1phaG92fYSLkpmgp661vMPK0djf5u30+wIJEBceJSwzOkFIT1ZdZGtyeYCHjpWco6qxuL/GzdTb4unw9/4FDBMaISgvNj1ES1JZYGdudXyDipGYn6attLvCydDX3uXs8/oBCA8WHSQrMjlAR05VXGNqcXh/ho2Um6KpsLe+xczT2uHo7/b9BAsSGSAnLjU8Q0pRWF9mbXR7gomQl56lrLO6wcjP1t3k6/L5AAcOFRwjKjE4P0ZNVFtiaXB3foWMk5qhqK+2vcTL0tng5+71/AMKERgfJi00O0JJUFdeZWxzeoGIj5adpKuyucDHztXc4+rx+P8GDRQbIikwNz5FTFNaYWhvdn2Ei5KZoKeutbzDytHY3+bt9PsCCRAXHiUsMzpBSE9WXWRrcnmAh46VnKOqsbi/xs3U2+Lp8Pf+BQwTGiEoLzY9REtSWWBnbnV8g4qRmJ+mrbS7wsnQ197l7PP6AQgPFh0kKzI5QEdOVVxjanF4f4aNlJuiqbC3vsXM09rh6O/2/QQLEhkgJy41PENKUVhfZm10e4KJkJeepayzusHIz9bd5Ovy+QAHDhUcIyoxOD9GTVRbYmlwd36FjJOaoaivtr3Ey9LZ4Ofu9fwDChEYHyYtNDtCSVBXXmVsc3qBiI+WnaSrsrnAx87V3OPq8fj/Bg==",
                            "format": "mp4",
                        },
                    },
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 5f5da5957185"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8060,
                "candidatesTokenCount": 300,
                "totalTokenCount": 8360,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 140},
                    {"modality": "VIDEO", "tokenCount": 7920},
                ],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.0059192, input_cost=0.0049832, output_cost=0.000936, prompt_tokens=8060, completion_tokens=300
    ),
)

GEMINI_3_8_FLASH_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "0009f5ac891e summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 0009f5ac891e"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                "trafficType": "ON_DEMAND_FLEX",
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.00112112, input_cost=0.0004784, output_cost=0.00064272, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "057d8b15b597 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 057d8b15b597"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                "trafficType": "ON_DEMAND_PRIORITY",
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.0028028, input_cost=0.001196, output_cost=0.0016068, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_WEB_SEARCH_PER_PROMPT: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-web_search_per_prompt",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "c409248006ff summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"googleSearch": {}}],
        "allowed_openai_params": ["web_search_options"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer c409248006ff"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "groundingMetadata": {"webSearchQueries": ["query 0", "query 1", "query 2"]},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.03724224, input_cost=0.0009568, output_cost=0.00128544, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_GOOGLE_MAPS_GROUNDING: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-google_maps_grounding",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "cdcfe11184ca summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [{"googleMaps": {}}],
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer cdcfe11184ca"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "groundingMetadata": {
                        "webSearchQueries": ["maps query 0"],
                        "groundingChunks": [{"maps": {"uri": "https://maps.google.com/?cid=0"}}],
                        "googleMapsWidgetContextToken": "token_$REQUEST_ID",
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.02724224, input_cost=0.0009568, output_cost=0.00128544, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_FALLBACK_REASONING_AT_OUTPUT_RATE: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-fallback_reasoning_at_output_rate",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "f92946792f44 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer f92946792f44"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1240,
                "candidatesTokenCount": 560,
                "thoughtsTokenCount": 3480,
                "totalTokenCount": 5280,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1240}],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.0132496, input_cost=0.0006448, output_cost=0.0126048, prompt_tokens=1240, completion_tokens=4040
    ),
)

GEMINI_3_8_FLASH_FALLBACK_IMAGE_TOKENS_AT_INPUT_RATE: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-fallback_image_tokens_at_input_rate",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "59106006ecc4 summarize the attached material in one line and name the city weather",
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
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 59106006ecc4"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 2116,
                "candidatesTokenCount": 240,
                "totalTokenCount": 2356,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 310},
                    {"modality": "IMAGE", "tokenCount": 1806},
                ],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.00184912, input_cost=0.00110032, output_cost=0.0007488, prompt_tokens=2116, completion_tokens=240
    ),
)

GEMINI_3_8_FLASH_STREAM: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "02cc764f4300 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 02cc764f4300"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.8-flash",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 412,
                    "totalTokenCount": 2252,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.8-flash",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.00224224, input_cost=0.0009568, output_cost=0.00128544, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "60f7b65abfa3 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 60f7b65abfa3"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.8-flash",
            }
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=5.2e-07, output_cost_per_token=3.12e-06),
        prompt_tokens=48,
        completion_tokens=12,
    ),
)

GEMINI_3_8_FLASH_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "18632b64dd03 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {
                                            "city": "Berlin",
                                            "days": 7,
                                            "units": "metric",
                                            "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                        },
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.8-flash",
            }
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=5.2e-07, output_cost_per_token=3.12e-06), min_completion_tokens=60
    ),
)

GEMINI_3_8_FLASH_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "a3126f19100d summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer a3126f19100d"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.8-flash",
            }
        ),
    ),
    expected=RecountExpected(
        recount=RecountRates(input_cost_per_token=5.2e-07, output_cost_per_token=3.12e-06),
        prompt_tokens=302,
        completion_tokens=10,
    ),
)

GEMINI_3_8_FLASH_PROMPT_BLOCKED: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-prompt_blocked",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "899380691bc7 summarize the attached material in one line and name the city weather",
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
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "HIGH", "blocked": True}],
            },
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 0,
                "totalTokenCount": 1840,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.0009568, input_cost=0.0009568, output_cost=0.0, prompt_tokens=1840, completion_tokens=0
    ),
)

GEMINI_3_8_FLASH_STREAM_PROMPT_BLOCKED: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-stream_prompt_blocked",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "4c9331e5da39 summarize the attached material in one line and name the city weather",
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
                "promptFeedback": {
                    "blockReason": "SAFETY",
                    "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "HIGH", "blocked": True}],
                },
                "modelVersion": "gemini-3.8-flash",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 1840,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.8-flash",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.0009568, input_cost=0.0009568, output_cost=0.0, prompt_tokens=1840, completion_tokens=0
    ),
)

GEMINI_3_8_FLASH_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "4972a11cd52d summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {"parts": [{"text": "scripted answer 4972a11cd52d"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.1-pro",
        },
    ),
    expected=ExactExpected(
        spend=0.0090552, input_cost=0.003864, output_cost=0.0051912, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "0908445fc9e7 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer 0908445fc9e7"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.1-pro",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 412,
                    "totalTokenCount": 2252,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.1-pro",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.0090552, input_cost=0.003864, output_cost=0.0051912, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_TOOL_CALL: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "151d9709f7f7 summarize the attached material in one line and name the city weather",
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
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {
                                        "city": "Berlin",
                                        "days": 7,
                                        "units": "metric",
                                        "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                    },
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1840,
                "candidatesTokenCount": 412,
                "totalTokenCount": 2252,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
            },
            "modelVersion": "gemini-3.8-flash",
        },
    ),
    expected=ExactExpected(
        spend=0.00224224, input_cost=0.0009568, output_cost=0.00128544, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "9253170bf979 summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {
                                            "city": "Berlin",
                                            "days": 7,
                                            "units": "metric",
                                            "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                        },
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.8-flash",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 1840,
                    "candidatesTokenCount": 412,
                    "totalTokenCount": 2252,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1840}],
                },
                "modelVersion": "gemini-3.8-flash",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.00224224, input_cost=0.0009568, output_cost=0.00128544, prompt_tokens=1840, completion_tokens=412
    ),
)

GEMINI_3_8_FLASH_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="gemini-3.8-flash-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="gemini-3.8-flash",
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
                        "text": "e31c97cab9cc summarize the attached material in one line and name the city weather",
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
                "candidates": [
                    {
                        "content": {"parts": [{"text": "scripted answer e31c97cab9cc"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-3.8-flash",
            },
            {
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 8314,
                    "candidatesTokenCount": 692,
                    "thoughtsTokenCount": 900,
                    "totalTokenCount": 9906,
                    "cachedContentTokenCount": 6144,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 7984},
                        {"modality": "AUDIO", "tokenCount": 330},
                    ],
                    "candidatesTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 412},
                        {"modality": "AUDIO", "tokenCount": 280},
                    ],
                },
                "modelVersion": "gemini-3.8-flash",
            },
        ),
    ),
    expected=ExactExpected(
        spend=0.007460128, input_cost=0.001619488, output_cost=0.00584064, prompt_tokens=8314, completion_tokens=1592
    ),
)


# 1024-x-1024/imagen-next
IMAGEN_NEXT_IMAGES_ONE: Final = CostTrackingTestCase(
    name="imagen-next-images-one",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="1024-x-1024/imagen-next",
    endpoint="/v1/images/generations",
    request={"prompt": "a deterministic vertex image", "sampleCount": 1},
    response=JsonResponse(
        content_type="application/json", body={"predictions": [{"bytesBase64Encoded": "AA==", "mimeType": "image/png"}]}
    ),
    expected=ExactExpected(
        spend=0.05, input_cost=0.05, output_cost=0, prompt_tokens=0, completion_tokens=0, breakdown_persisted=False
    ),
)


# text-embedding-006
VERTEX_EMBEDDINGS_TEXT_006: Final = CostTrackingTestCase(
    name="vertex-embeddings-text-006",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="text-embedding-006",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": "vertex embedding"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2, 0.3], "statistics": {"token_count": 7, "truncated": False}}}
            ]
        },
    ),
    expected=ExactExpected(
        spend=7.4899999999999994e-06,
        input_cost=7.4899999999999994e-06,
        output_cost=0.0,
        prompt_tokens=7,
        completion_tokens=0,
    ),
)


CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    GEMINI_3_1_PRO_INPUT_TEXT,
    GEMINI_3_1_PRO_CACHE_READ,
    GEMINI_3_1_PRO_AUDIO_INPUT,
    GEMINI_3_1_PRO_IMAGE_INPUT,
    GEMINI_3_1_PRO_REASONING,
    GEMINI_3_1_PRO_TIERED_INPUT_ABOVE_200K,
    GEMINI_3_1_PRO_TIERED_CACHE_READ_ABOVE_200K,
    GEMINI_3_1_PRO_SERVICE_TIER_FLEX,
    GEMINI_3_1_PRO_SERVICE_TIER_PRIORITY,
    GEMINI_3_1_PRO_WEB_SEARCH_MEDIUM,
    GEMINI_3_1_PRO_GOOGLE_MAPS_GROUNDING,
    GEMINI_3_1_PRO_FALLBACK_VIDEO_TOKENS_AT_INPUT_RATE,
    GEMINI_3_1_PRO_STREAM,
    GEMINI_3_1_PRO_STREAM_NO_USAGE,
    GEMINI_3_1_PRO_STREAM_NO_USAGE_TOOL_CALL,
    GEMINI_3_1_PRO_STREAM_NO_USAGE_IMAGE_INPUT,
    GEMINI_3_1_PRO_PROMPT_BLOCKED,
    GEMINI_3_1_PRO_STREAM_PROMPT_BLOCKED,
    GEMINI_3_1_PRO_RESPONSE_MODEL_OVERRIDE,
    GEMINI_3_1_PRO_STREAM_RESPONSE_MODEL_OVERRIDE,
    GEMINI_3_1_PRO_TOOL_CALL,
    GEMINI_3_1_PRO_STREAM_TOOL_CALL,
    GEMINI_3_1_PRO_STREAM_FULL_USAGE,
    GEMINI_3_8_FLASH_INPUT_TEXT,
    GEMINI_3_8_FLASH_CACHE_READ,
    GEMINI_3_8_FLASH_AUDIO_INPUT,
    GEMINI_3_8_FLASH_AUDIO_OUTPUT,
    GEMINI_3_8_FLASH_VIDEO_INPUT,
    GEMINI_3_8_FLASH_SERVICE_TIER_FLEX,
    GEMINI_3_8_FLASH_SERVICE_TIER_PRIORITY,
    GEMINI_3_8_FLASH_WEB_SEARCH_PER_PROMPT,
    GEMINI_3_8_FLASH_GOOGLE_MAPS_GROUNDING,
    GEMINI_3_8_FLASH_FALLBACK_REASONING_AT_OUTPUT_RATE,
    GEMINI_3_8_FLASH_FALLBACK_IMAGE_TOKENS_AT_INPUT_RATE,
    GEMINI_3_8_FLASH_STREAM,
    GEMINI_3_8_FLASH_STREAM_NO_USAGE,
    GEMINI_3_8_FLASH_STREAM_NO_USAGE_TOOL_CALL,
    GEMINI_3_8_FLASH_STREAM_NO_USAGE_IMAGE_INPUT,
    GEMINI_3_8_FLASH_PROMPT_BLOCKED,
    GEMINI_3_8_FLASH_STREAM_PROMPT_BLOCKED,
    GEMINI_3_8_FLASH_RESPONSE_MODEL_OVERRIDE,
    GEMINI_3_8_FLASH_STREAM_RESPONSE_MODEL_OVERRIDE,
    GEMINI_3_8_FLASH_TOOL_CALL,
    GEMINI_3_8_FLASH_STREAM_TOOL_CALL,
    GEMINI_3_8_FLASH_STREAM_FULL_USAGE,
    IMAGEN_NEXT_IMAGES_ONE,
    VERTEX_EMBEDDINGS_TEXT_006,
    GEMINI_3_1_PRO_PASSTHROUGH_STREAM_GENERATE_CONTENT_PRICED_VIA_VERTEX_KEY,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(plain=GEMINI_3_1_PRO_INPUT_TEXT, streamed=GEMINI_3_1_PRO_STREAM),
    StreamParityTestCase(plain=GEMINI_3_1_PRO_PROMPT_BLOCKED, streamed=GEMINI_3_1_PRO_STREAM_PROMPT_BLOCKED),
    StreamParityTestCase(
        plain=GEMINI_3_1_PRO_RESPONSE_MODEL_OVERRIDE, streamed=GEMINI_3_1_PRO_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=GEMINI_3_1_PRO_TOOL_CALL, streamed=GEMINI_3_1_PRO_STREAM_TOOL_CALL),
    StreamParityTestCase(plain=GEMINI_3_8_FLASH_INPUT_TEXT, streamed=GEMINI_3_8_FLASH_STREAM),
    StreamParityTestCase(plain=GEMINI_3_8_FLASH_PROMPT_BLOCKED, streamed=GEMINI_3_8_FLASH_STREAM_PROMPT_BLOCKED),
    StreamParityTestCase(
        plain=GEMINI_3_8_FLASH_RESPONSE_MODEL_OVERRIDE, streamed=GEMINI_3_8_FLASH_STREAM_RESPONSE_MODEL_OVERRIDE
    ),
    StreamParityTestCase(plain=GEMINI_3_8_FLASH_TOOL_CALL, streamed=GEMINI_3_8_FLASH_STREAM_TOOL_CALL),
)
