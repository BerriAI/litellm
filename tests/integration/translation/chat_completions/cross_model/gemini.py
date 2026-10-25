from dataclasses import replace
from typing import Final

from integration._support.client import list_value
from integration.translation.case import SSEFrame, StreamingTranslationTestCase
from integration.translation.chat_completions.bases.gemini import (
    GEMINI_3_1_PRO_PREVIEW_TEST_CASE,
    GEMINI_3_8_FLASH_TEST_CASE,
)
from integration.translation.chat_completions.cross_model.contracts import followup_request
from integration.translation.chat_completions.templates import (
    CODING_AGENT_REQUEST,
    CODING_AGENT_RESPONSE,
    coding_agent_stream_message,
)
from integration.translation.templates.vertex_ai import VERTEX_AI_AGENT_REQUEST

GEMINI_SKIP_VALIDATION_SIGNATURE: Final = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="

GEMINI_3_1_PRO_PREVIEW_THOUGHT: Final = (
    "Okay, here's my summary, framed as the internal thought process:\n\n"
    "**Concurrent Debugging of `retries_remaining` Usage**\n\n"
    "Alright, the objective is clear: I need to track down the root cause. My gut tells me this "
    "`retries_remaining` issue isn't isolated. My first move will be to perform a comprehensive search "
    "for all instances where `retries_remaining` is referenced within the codebase. We're talking "
    "about a full, concurrent search. I don't want to waste time waiting.\n\n"
    "Once I have a list, the plan is to launch the relevant test suite for each of those callers, "
    "*concurrently*. That's the only way to reproduce and identify the specific failure scenario. "
    "Speed and thoroughness are key here. I'll need to instrument each test run to capture as much "
    "diagnostic information as possible - logging, stack traces, maybe even memory dumps if the issue "
    "seems complex. I need a clear picture of what's happening within those retry loops to figure out "
    "what's causing the problem.\n\n\n"
)

GEMINI_3_8_FLASH_FOLLOWUP_TEXT: Final = (
    "The function `retries_remaining` erroneously adds `limit` and `attempts` instead of subtracting "
    "them with a minimum of zero, causing `test_retries_remaining` to fail when "
    "`retries_remaining(3, 1)` returns `4` instead of `2`. In addition to the tests, `src/worker.py` "
    "was found as a caller invoking `retries_remaining(3, attempts)`."
)

GEMINI_3_1_PRO_PREVIEW_FOLLOWUP_THOUGHT: Final = (
    "**My Analysis of the Bug Report**\n\n"
    "Okay, let's break this down efficiently. The key is to avoid unnecessary verbosity and, "
    "importantly, not invoke any tooling or editing operations. This is about summarizing the defect "
    "itself. I see the caller is in `src/worker.py` and the failing test is "
    "`tests/test_retry.py::test_retries_remaining`.\n\n"
    "The core issue appears to be an incorrect calculation within the `retries_remaining` function. "
    "It looks like the implementation adds attempts to the limit instead of subtracting them, leading "
    "to a logical error. This directly impacts the `test_retries_remaining` test and, I presume, causes "
    "issues in the `src/worker.py` module. To fix this, the return value needs to be changed to "
    "`max(0, limit - attempts)` to properly calculate the remaining retries. This correction should "
    "resolve the test failure and ensure the correct behavior of the retry logic called from the worker.\n\n\n"
)

GEMINI_3_1_PRO_PREVIEW_FOLLOWUP_TEXT: Final = (
    "The bug is that `retries_remaining` incorrectly adds `attempts` to `limit` instead of subtracting "
    "it, causing `test_retries_remaining` to fail by returning 4 instead of 2. Fixing the implementation "
    "to return `max(0, limit - attempts)` will resolve the test failure and correct the behavior for its "
    "caller in `src/worker.py`."
)

GEMINI_3_8_FLASH_AGENT_PROVIDER_RESPONSE: Final = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "name": "search_code",
                            "args": {"query": "retries_remaining"},
                            "id": "call_4323393",
                        },
                        "thoughtSignature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                    },
                    {
                        "functionCall": {
                            "name": "run_tests",
                            "args": {"path": "tests/test_retry.py"},
                            "id": "call_4323394",
                        }
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 516,
        "candidatesTokenCount": 41,
        "totalTokenCount": 557,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 516}],
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "yyHWapj7Br7sz7IPte2L2QU",
}

GEMINI_3_1_PRO_PREVIEW_AGENT_PROVIDER_RESPONSE: Final = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": GEMINI_3_1_PRO_PREVIEW_THOUGHT, "thought": True},
                    {
                        "functionCall": {
                            "name": "search_code",
                            "args": {"query": "retries_remaining"},
                            "id": "call_4206648",
                        },
                        "thoughtSignature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                    },
                    {
                        "functionCall": {
                            "name": "run_tests",
                            "args": {"path": "tests/test_retry.py"},
                            "id": "call_4206649",
                        }
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 516,
        "candidatesTokenCount": 41,
        "totalTokenCount": 602,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 516}],
        "thoughtsTokenCount": 45,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.1-pro-preview",
    "responseId": "0CHWas7iDZLqz7IPsLnekAs",
}

GEMINI_3_8_FLASH_AGENT_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "gemini/gemini-3.8-flash"},
    expected_provider_endpoint="/models/gemini-3.8-flash:generateContent",
    expected_provider_request=VERTEX_AI_AGENT_REQUEST,
    mock_provider_response=GEMINI_3_8_FLASH_AGENT_PROVIDER_RESPONSE,
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "gemini/gemini-3.8-flash",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                            },
                            "function": {
                                "arguments": '{"query": "retries_remaining"}',
                                "name": "search_code",
                            },
                            "id": "call_4323393__thought__" + GEMINI_SKIP_VALIDATION_SIGNATURE,
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {
                                "arguments": '{"path": "tests/test_retry.py"}',
                                "name": "run_tests",
                            },
                            "id": "call_4323394",
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "thinking_blocks": [],
                    "provider_specific_fields": {
                        "thought_signatures": [GEMINI_SKIP_VALIDATION_SIGNATURE],
                    },
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 41,
            "prompt_tokens": 516,
            "total_tokens": 557,
            "completion_tokens_details": {"text_tokens": 41},
            "prompt_tokens_details": {"text_tokens": 516},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)

GEMINI_3_1_PRO_PREVIEW_AGENT_TEST_CASE: Final = replace(
    GEMINI_3_1_PRO_PREVIEW_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "gemini/gemini-3.1-pro-preview"},
    expected_provider_endpoint="/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_request=VERTEX_AI_AGENT_REQUEST,
    mock_provider_response=GEMINI_3_1_PRO_PREVIEW_AGENT_PROVIDER_RESPONSE,
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "gemini/gemini-3.1-pro-preview",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                            },
                            "function": {
                                "arguments": '{"query": "retries_remaining"}',
                                "name": "search_code",
                            },
                            "id": "call_4206648__thought__" + GEMINI_SKIP_VALIDATION_SIGNATURE,
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {
                                "arguments": '{"path": "tests/test_retry.py"}',
                                "name": "run_tests",
                            },
                            "id": "call_4206649",
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "reasoning_content": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                        }
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [GEMINI_SKIP_VALIDATION_SIGNATURE],
                    },
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 86,
            "prompt_tokens": 516,
            "total_tokens": 602,
            "completion_tokens_details": {"reasoning_tokens": 45, "text_tokens": 41},
            "prompt_tokens_details": {"text_tokens": 516},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)

GEMINI_3_8_FLASH_AGENT_FOLLOWUP_PROVIDER_REQUEST: Final = {
    **VERTEX_AI_AGENT_REQUEST,
    "contents": [
        *list_value(VERTEX_AI_AGENT_REQUEST["contents"]),
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "name": "search_code",
                        "args": {"query": "retries_remaining"},
                        "id": "call_4323393",
                    },
                    "thoughtSignature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                },
                {
                    "function_call": {
                        "name": "run_tests",
                        "args": {"path": "tests/test_retry.py"},
                        "id": "call_4323394",
                    }
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "search_code",
                        "response": {
                            "matches": [
                                {
                                    "path": "src/worker.py",
                                    "line": 12,
                                    "text": "remaining = retries_remaining(3, attempts)",
                                }
                            ]
                        },
                        "id": "call_4323393",
                    }
                },
                {
                    "function_response": {
                        "name": "run_tests",
                        "response": {
                            "exit_code": 1,
                            "output": "FAILED tests/test_retry.py::test_retries_remaining - assert 4 == 2",
                        },
                        "id": "call_4323394",
                    }
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Stop after these diagnostics. Do not call more tools or change files. "
                        "Summarize the bug, the caller found and the failed test in two short sentences."
                    )
                }
            ],
        },
    ],
}

GEMINI_3_1_PRO_PREVIEW_AGENT_FOLLOWUP_PROVIDER_REQUEST: Final = {
    **GEMINI_3_8_FLASH_AGENT_FOLLOWUP_PROVIDER_REQUEST,
    "contents": [
        *list_value(VERTEX_AI_AGENT_REQUEST["contents"]),
        {
            "role": "model",
            "parts": [
                {"text": GEMINI_3_1_PRO_PREVIEW_THOUGHT, "thought": True},
                {
                    "function_call": {
                        "name": "search_code",
                        "args": {"query": "retries_remaining"},
                        "id": "call_4206648",
                    },
                    "thoughtSignature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                },
                {
                    "function_call": {
                        "name": "run_tests",
                        "args": {"path": "tests/test_retry.py"},
                        "id": "call_4206649",
                    }
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "search_code",
                        "response": {
                            "matches": [
                                {
                                    "path": "src/worker.py",
                                    "line": 12,
                                    "text": "remaining = retries_remaining(3, attempts)",
                                }
                            ]
                        },
                        "id": "call_4206648",
                    }
                },
                {
                    "function_response": {
                        "name": "run_tests",
                        "response": {
                            "exit_code": 1,
                            "output": "FAILED tests/test_retry.py::test_retries_remaining - assert 4 == 2",
                        },
                        "id": "call_4206649",
                    }
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Stop after these diagnostics. Do not call more tools or change files. "
                        "Summarize the bug, the caller found and the failed test in two short sentences."
                    )
                }
            ],
        },
    ],
}

GEMINI_3_8_FLASH_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GEMINI_3_8_FLASH_AGENT_TEST_CASE.litellm_request,
        GEMINI_3_8_FLASH_AGENT_TEST_CASE.expected_litellm_response,
    ),
    expected_provider_endpoint="/models/gemini-3.8-flash:generateContent",
    expected_provider_request=GEMINI_3_8_FLASH_AGENT_FOLLOWUP_PROVIDER_REQUEST,
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": GEMINI_3_8_FLASH_FOLLOWUP_TEXT,
                            "thoughtSignature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 670,
            "candidatesTokenCount": 89,
            "totalTokenCount": 759,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 670}],
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.8-flash",
        "responseId": "nyLWatOHEKHoz7IPnICMuAs",
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "gemini/gemini-3.8-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": GEMINI_3_8_FLASH_FOLLOWUP_TEXT,
                    "role": "assistant",
                    "images": [],
                    "thinking_blocks": [],
                    "provider_specific_fields": {
                        "thought_signatures": [GEMINI_SKIP_VALIDATION_SIGNATURE],
                    },
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 89,
            "prompt_tokens": 670,
            "total_tokens": 759,
            "completion_tokens_details": {"text_tokens": 89},
            "prompt_tokens_details": {"text_tokens": 670},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)

GEMINI_3_1_PRO_PREVIEW_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GEMINI_3_1_PRO_PREVIEW_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GEMINI_3_1_PRO_PREVIEW_AGENT_TEST_CASE.litellm_request,
        GEMINI_3_1_PRO_PREVIEW_AGENT_TEST_CASE.expected_litellm_response,
    ),
    expected_provider_endpoint="/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_request=GEMINI_3_1_PRO_PREVIEW_AGENT_FOLLOWUP_PROVIDER_REQUEST,
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": GEMINI_3_1_PRO_PREVIEW_FOLLOWUP_THOUGHT, "thought": True},
                        {
                            "text": GEMINI_3_1_PRO_PREVIEW_FOLLOWUP_TEXT,
                            "thoughtSignature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                        },
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 887,
            "candidatesTokenCount": 78,
            "totalTokenCount": 1329,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 887}],
            "thoughtsTokenCount": 364,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.1-pro-preview",
        "responseId": "oiLWauy3Kr2Ez7IPse_RwAs",
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "gemini/gemini-3.1-pro-preview",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": GEMINI_3_1_PRO_PREVIEW_FOLLOWUP_TEXT,
                    "role": "assistant",
                    "images": [],
                    "reasoning_content": GEMINI_3_1_PRO_PREVIEW_FOLLOWUP_THOUGHT,
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": GEMINI_3_1_PRO_PREVIEW_FOLLOWUP_THOUGHT,
                        }
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [GEMINI_SKIP_VALIDATION_SIGNATURE],
                    },
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 442,
            "prompt_tokens": 887,
            "total_tokens": 1329,
            "completion_tokens_details": {"reasoning_tokens": 364, "text_tokens": 78},
            "prompt_tokens_details": {"text_tokens": 887},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)

GEMINI_3_8_FLASH_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GEMINI_3_8_FLASH_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GEMINI_3_8_FLASH_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GEMINI_3_8_FLASH_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/models/gemini-3.8-flash:streamGenerateContent?alt=sse",
    expected_provider_request=VERTEX_AI_AGENT_REQUEST,
    mock_provider_frames=(
        SSEFrame(
            data={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "search_code",
                                        "args": {"query": "retries_remaining"},
                                        "id": "call_5122749",
                                    },
                                    "thoughtSignature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                                }
                            ],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 516,
                    "candidatesTokenCount": 19,
                    "totalTokenCount": 535,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 516}],
                    "serviceTier": "standard",
                },
                "modelVersion": "gemini-3.8-flash",
                "responseId": "zSHWaoPJKvPrz7IP7dqb-AM",
            }
        ),
        SSEFrame(
            data={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "run_tests",
                                        "args": {"path": "tests/test_retry.py"},
                                        "id": "call_5122752",
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 516,
                    "candidatesTokenCount": 41,
                    "totalTokenCount": 557,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 516}],
                    "serviceTier": "standard",
                },
                "modelVersion": "gemini-3.8-flash",
                "responseId": "zSHWaoPJKvPrz7IP7dqb-AM",
            }
        ),
        SSEFrame(
            data={
                "candidates": [
                    {
                        "content": {"parts": [{"text": ""}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 516,
                    "candidatesTokenCount": 41,
                    "totalTokenCount": 557,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 516}],
                    "serviceTier": "standard",
                },
                "modelVersion": "gemini-3.8-flash",
                "responseId": "zSHWaoPJKvPrz7IP7dqb-AM",
            }
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_5122749__thought__" + GEMINI_SKIP_VALIDATION_SIGNATURE, "call_5122752"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 516,
        "completion_tokens": 41,
        "total_tokens": 557,
        "prompt_tokens_details": {"text_tokens": 516},
        "completion_tokens_details": {"text_tokens": 41},
    },
)

GEMINI_3_1_PRO_PREVIEW_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GEMINI_3_1_PRO_PREVIEW_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GEMINI_3_1_PRO_PREVIEW_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GEMINI_3_1_PRO_PREVIEW_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse",
    expected_provider_request=VERTEX_AI_AGENT_REQUEST,
    mock_provider_frames=(
        SSEFrame(
            data={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "search_code",
                                        "args": {"query": "retries_remaining"},
                                        "id": "call_6112941",
                                    },
                                    "thoughtSignature": GEMINI_SKIP_VALIDATION_SIGNATURE,
                                }
                            ],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 516,
                    "candidatesTokenCount": 19,
                    "totalTokenCount": 599,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 516}],
                    "thoughtsTokenCount": 64,
                    "serviceTier": "standard",
                },
                "modelVersion": "gemini-3.1-pro-preview",
                "responseId": "1SHWaoD1Ode1qtsPo6rR4AI",
            }
        ),
        SSEFrame(
            data={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "run_tests",
                                        "args": {"path": "tests/test_retry.py"},
                                        "id": "call_6112947",
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 516,
                    "candidatesTokenCount": 41,
                    "totalTokenCount": 621,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 516}],
                    "thoughtsTokenCount": 64,
                    "serviceTier": "standard",
                },
                "modelVersion": "gemini-3.1-pro-preview",
                "responseId": "1SHWaoD1Ode1qtsPo6rR4AI",
            }
        ),
        SSEFrame(
            data={
                "candidates": [
                    {
                        "content": {"parts": [{"text": ""}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 516,
                    "candidatesTokenCount": 41,
                    "totalTokenCount": 621,
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 516}],
                    "thoughtsTokenCount": 64,
                    "serviceTier": "standard",
                },
                "modelVersion": "gemini-3.1-pro-preview",
                "responseId": "1SHWaoD1Ode1qtsPo6rR4AI",
            }
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_6112941__thought__" + GEMINI_SKIP_VALIDATION_SIGNATURE, "call_6112947"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 516,
        "completion_tokens": 105,
        "total_tokens": 621,
        "prompt_tokens_details": {"text_tokens": 516},
        "completion_tokens_details": {"reasoning_tokens": 64, "text_tokens": 41},
    },
)
