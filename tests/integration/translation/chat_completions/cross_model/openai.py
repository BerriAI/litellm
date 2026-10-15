from dataclasses import replace
from typing import Final

from integration._support.client import list_value, object_value
from integration.translation.case import (
    SSEFrame,
    StreamingTranslationTestCase,
)
from integration.translation.chat_completions.cross_model.contracts import followup_request
from integration.translation.chat_completions.bases.openai import (
    GPT_5_4_MINI_TEST_CASE,
    GPT_5_6_LUNA_TEST_CASE,
    GPT_5_6_SOL_TEST_CASE,
    GPT_5_6_TERRA_TEST_CASE,
    GPT_5_MINI_TEST_CASE,
    GPT_6_1_SOL_TEST_CASE,
    GPT_6_ASTRA_TEST_CASE,
    GPT_6_LUNA_TEST_CASE,
    GPT_6_SOL_TEST_CASE,
)
from integration.translation.chat_completions.templates import (
    CODING_AGENT_REQUEST,
    coding_agent_stream_message,
    CODING_AGENT_RESPONSE,
)
from integration.translation.templates.openai import (
    OPENAI_CHAT_AGENT_REQUEST,
    OPENAI_CHAT_AGENT_RESPONSE,
    OPENAI_RESPONSES_AGENT_REQUEST,
    OPENAI_RESPONSES_AGENT_RESPONSE,
)

GPT_5_MINI_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_MINI_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_MINI_TEST_CASE.litellm_request, GPT_5_MINI_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/chat/completions",
    expected_provider_request={
        **OPENAI_CHAT_AGENT_REQUEST,
        "messages": [
            *list_value(OPENAI_CHAT_AGENT_REQUEST["messages"]),
            {
                "content": "I'll search for callers and run the specific test.",
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {"arguments": '{"query": "retries_remaining("}', "name": "search_code"},
                        "id": "call_mFie59uQ12eusnoPzdw7qlho",
                        "type": "function",
                    },
                    {
                        "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                        "id": "call_hEFM0TcGu1J2VeKD4SjqoBg2",
                        "type": "function",
                    },
                ],
                "annotations": [],
            },
            {
                "role": "tool",
                "tool_call_id": "call_mFie59uQ12eusnoPzdw7qlho",
                "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = retries_remaining(3, '
                'attempts)"}]}',
            },
            {
                "role": "tool",
                "tool_call_id": "call_hEFM0TcGu1J2VeKD4SjqoBg2",
                "content": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - assert 4 == '
                '2"}',
            },
            {
                "role": "user",
                "content": "Stop after these diagnostics. Do not call more tools or change files. Summarize the bug, the "
                "caller found and the failed test in two short sentences.",
            },
        ],
        "model": "gpt-5-mini",
    },
    mock_provider_response={
        **OPENAI_CHAT_AGENT_RESPONSE,
        "id": "chatcmpl-EZ6ZnzJxYtvI0dT0Ek5WxIoYCSBuF",
        "created": 1792035467,
        "model": "gpt-5-mini-2025-08-07",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The function incorrectly returns limit + attempts (giving 4 for "
                    "retries_remaining(3, 1)) instead of the number of retries left; it should return "
                    "max(limit - attempts, 0) so it never goes negative. The caller is src/worker.py "
                    "(line using retries_remaining(3, attempts)), and the test tests/test_retry.py "
                    "failed asserting retries_remaining(3, 1) == 2 (actual 4).",
                    "refusal": None,
                    "annotations": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            **object_value(OPENAI_CHAT_AGENT_RESPONSE["usage"]),
            "prompt_tokens": 712,
            "completion_tokens": 160,
            "total_tokens": 872,
            "completion_tokens_details": {
                **object_value(object_value(OPENAI_CHAT_AGENT_RESPONSE["usage"])["completion_tokens_details"]),
                "reasoning_tokens": 64,
            },
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "chatcmpl-EZ6ZnzJxYtvI0dT0Ek5WxIoYCSBuF",
        "model": "openai/gpt-5-mini",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The function incorrectly returns limit + attempts (giving 4 for "
                    "retries_remaining(3, 1)) instead of the number of retries left; it should return "
                    "max(limit - attempts, 0) so it never goes negative. The caller is src/worker.py "
                    "(line using retries_remaining(3, attempts)), and the test tests/test_retry.py "
                    "failed asserting retries_remaining(3, 1) == 2 (actual 4).",
                    "role": "assistant",
                    "annotations": [],
                    "refusal": None,
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 160,
            "prompt_tokens": 712,
            "total_tokens": 872,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 64,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)


GPT_5_MINI_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_5_MINI_TEST_CASE.litellm_endpoint,
    litellm_request={**GPT_5_MINI_TEST_CASE.litellm_request, "stream": True, "stream_options": {"include_usage": True}},
    expected_provider_headers=GPT_5_MINI_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/chat/completions",
    expected_provider_request={
        **OPENAI_CHAT_AGENT_REQUEST,
        "model": "gpt-5-mini",
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}],
                "obfuscation": "66",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_cfsTFwIpo5Pw1Ycjne6cu6rd",
                                    "type": "function",
                                    "function": {"name": "search_code", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "5",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"qu'}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "r2XI9kO1",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ery":'}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "Flu2rEC",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "retr'}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "xa2B96",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ies_"}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "CcMgkpZ0u",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "remai"}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "pWPji9Pa",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ning("'}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "o0ffHn",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "ZibrP37NuHYx",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_w0dR7zl3FEWVVOUvxe2qRhw3",
                                    "type": "function",
                                    "function": {"name": "run_tests", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "S6L",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": '{"pa'}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "MOX33Kx9",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": 'th": '}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "wllLLks",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": '"tests'}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "aZmbqx",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "/tes"}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "eM2MAKKyu",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "t_ret"}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "Jnv7RFXG",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": 'ry.py"'}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "JKkZAU",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "}"}}]},
                        "finish_reason": None,
                    }
                ],
                "obfuscation": "6iPuHABVnS8x",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": None,
                "obfuscation": "XBrG5eHvNp1",
            }
        ),
        SSEFrame(
            data={
                "id": "chatcmpl-EZ6ZmdVdvBZtqGJXMTyRP3skJs724",
                "object": "chat.completion.chunk",
                "created": 1792035466,
                "model": "gpt-5-mini-2025-08-07",
                "service_tier": "default",
                "system_fingerprint": None,
                "choices": [],
                "usage": {
                    "prompt_tokens": 529,
                    "completion_tokens": 53,
                    "total_tokens": 582,
                    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                        "accepted_prediction_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                },
                "obfuscation": "FKzs",
            }
        ),
        SSEFrame(data="[DONE]"),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_cfsTFwIpo5Pw1Ycjne6cu6rd", "call_w0dR7zl3FEWVVOUvxe2qRhw3"),
        arguments=({"query": "retries_remaining("}, {"path": "tests/test_retry.py"}),
    ),
    expected_litellm_usage={
        "prompt_tokens": 529,
        "completion_tokens": 53,
        "total_tokens": 582,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
)


GPT_5_4_MINI_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_4_MINI_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_4_MINI_TEST_CASE.litellm_request, GPT_5_4_MINI_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/responses",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-5.4-mini",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I’ll search for other callers and run the retry tests now, then inspect the results "
                        "before patching.",
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_KG0GOlKRaaLKw5nHIZoptT4K",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining("}',
            },
            {
                "type": "function_call",
                "call_id": "call_pMG9sdoFy24PVF4ZP95Mo6ay",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_KG0GOlKRaaLKw5nHIZoptT4K",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_pMG9sdoFy24PVF4ZP95Mo6ay",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_09416b73f1fe04c2006ad04a8d112c87d0867e37246b001702",
        "created_at": 1792035469,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035470,
        "model": "gpt-5.4-mini-2026-03-17",
        "output": [
            {
                "id": "msg_09416b73f1fe04c2006ad04a8d743487d09dc5ba9d22daca8f",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` in `src/retry.py` is incorrect: it adds `limit + attempts`, so "
                        "it can grow instead of counting down and can return the wrong value. I found one "
                        "caller in `src/worker.py` (`remaining = retries_remaining(3, attempts)`), and "
                        "`tests/test_retry.py` failed with `assert 4 == 2` for `retries_remaining(3, 1)`.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "current_turn"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 648,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 92,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 740,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_09416b73f1fe04c2006ad04a8d112c87d0867e37246b001702",
        "model": "openai/gpt-5.4-mini",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` in `src/retry.py` is incorrect: it adds `limit + attempts`, so "
                    "it can grow instead of counting down and can return the wrong value. I found one "
                    "caller in `src/worker.py` (`remaining = retries_remaining(3, attempts)`), and "
                    "`tests/test_retry.py` failed with `assert 4 == 2` for `retries_remaining(3, 1)`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 92,
            "prompt_tokens": 648,
            "total_tokens": 740,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_5_4_MINI_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_5_4_MINI_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_5_4_MINI_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_5_4_MINI_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/responses",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.4-mini", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_00c4cf409ce163c2006ad04a8c09f487d0b3e9a66eda650928",
                    "created_at": 1792035468,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-5.4-mini-2026-03-17",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {
                        **object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]),
                        "context": "current_turn",
                    },
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 0,
            },
            event="response.created",
        ),
        SSEFrame(
            data={
                "type": "response.in_progress",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_00c4cf409ce163c2006ad04a8c09f487d0b3e9a66eda650928",
                    "created_at": 1792035468,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-5.4-mini-2026-03-17",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {
                        **object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]),
                        "context": "current_turn",
                    },
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 1,
            },
            event="response.in_progress",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                    "type": "message",
                    "status": "in_progress",
                    "content": [],
                    "phase": "commentary",
                    "role": "assistant",
                },
                "output_index": 0,
                "sequence_number": 2,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.content_part.added",
                "content_index": 0,
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "output_index": 0,
                "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": ""},
                "sequence_number": 3,
            },
            event="response.content_part.added",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "I",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "eHDRHRFaSgOxgLt",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": "’ll",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "QBDpQ96r3XDxM",
                "output_index": 0,
                "sequence_number": 5,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " search",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "g6hcfgivG",
                "output_index": 0,
                "sequence_number": 6,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " for",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "bArHAfwcNjdS",
                "output_index": 0,
                "sequence_number": 7,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " callers",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "HZOXGK6D",
                "output_index": 0,
                "sequence_number": 8,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " and",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "Y3y5ohUx2iuA",
                "output_index": 0,
                "sequence_number": 9,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " run",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "gXLZAWReGI4d",
                "output_index": 0,
                "sequence_number": 10,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " the",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "rRRzDh3BIfpu",
                "output_index": 0,
                "sequence_number": 11,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " targeted",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "w28urHB",
                "output_index": 0,
                "sequence_number": 12,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " test",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "89E2Aos73NW",
                "output_index": 0,
                "sequence_number": 13,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " to",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "WTjRqaN4MSCk4",
                "output_index": 0,
                "sequence_number": 14,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " reproduce",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "0OyNC2",
                "output_index": 0,
                "sequence_number": 15,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " the",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "euS5RzGukPf6",
                "output_index": 0,
                "sequence_number": 16,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " failure",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "RxidlllQ",
                "output_index": 0,
                "sequence_number": 17,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": ".",
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "obfuscation": "hYdq83ygJvfR23D",
                "output_index": 0,
                "sequence_number": 18,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.done",
                "content_index": 0,
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "logprobs": [],
                "output_index": 0,
                "sequence_number": 19,
                "text": "I’ll search for callers and run the targeted test to reproduce the failure.",
            },
            event="response.output_text.done",
        ),
        SSEFrame(
            data={
                "type": "response.content_part.done",
                "content_index": 0,
                "item_id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                "output_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": "I’ll search for callers and run the targeted test to reproduce the failure.",
                },
                "sequence_number": 20,
            },
            event="response.content_part.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [],
                            "logprobs": [],
                            "text": "I’ll search for callers and run the targeted test to reproduce the failure.",
                        }
                    ],
                    "phase": "commentary",
                    "role": "assistant",
                },
                "output_index": 0,
                "sequence_number": 21,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_00c4cf409ce163c2006ad04a8cbe2087d094443cec70ed7857",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_8ucGVxGVa1SVMbD1mwPwwE3c",
                    "name": "search_code",
                },
                "output_index": 1,
                "sequence_number": 22,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining("}',
                "item_id": "fc_00c4cf409ce163c2006ad04a8cbe2087d094443cec70ed7857",
                "obfuscation": "Tc",
                "output_index": 1,
                "sequence_number": 23,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining("}',
                "item_id": "fc_00c4cf409ce163c2006ad04a8cbe2087d094443cec70ed7857",
                "output_index": 1,
                "sequence_number": 24,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_00c4cf409ce163c2006ad04a8cbe2087d094443cec70ed7857",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining("}',
                    "call_id": "call_8ucGVxGVa1SVMbD1mwPwwE3c",
                    "name": "search_code",
                },
                "output_index": 1,
                "sequence_number": 25,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_00c4cf409ce163c2006ad04a8cbe3487d098ddc0743d835252",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_NyYyHGphM3cB0Z3KkJRhhIfB",
                    "name": "run_tests",
                },
                "output_index": 2,
                "sequence_number": 26,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_00c4cf409ce163c2006ad04a8cbe3487d098ddc0743d835252",
                "obfuscation": "35",
                "output_index": 2,
                "sequence_number": 27,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_00c4cf409ce163c2006ad04a8cbe3487d098ddc0743d835252",
                "output_index": 2,
                "sequence_number": 28,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_00c4cf409ce163c2006ad04a8cbe3487d098ddc0743d835252",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_NyYyHGphM3cB0Z3KkJRhhIfB",
                    "name": "run_tests",
                },
                "output_index": 2,
                "sequence_number": 29,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_00c4cf409ce163c2006ad04a8c09f487d0b3e9a66eda650928",
                    "created_at": 1792035468,
                    "status": "completed",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": 1792035468,
                    "model": "gpt-5.4-mini-2026-03-17",
                    "output": [
                        {
                            "id": "msg_00c4cf409ce163c2006ad04a8c5b3887d0ade3c9adfd1c5550",
                            "type": "message",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "annotations": [],
                                    "logprobs": [],
                                    "text": "I’ll search for callers and run the targeted test to reproduce the failure.",
                                }
                            ],
                            "phase": "commentary",
                            "role": "assistant",
                        },
                        {
                            "id": "fc_00c4cf409ce163c2006ad04a8cbe2087d094443cec70ed7857",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining("}',
                            "call_id": "call_8ucGVxGVa1SVMbD1mwPwwE3c",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_00c4cf409ce163c2006ad04a8cbe3487d098ddc0743d835252",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_NyYyHGphM3cB0Z3KkJRhhIfB",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {
                        **object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]),
                        "context": "current_turn",
                    },
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 76,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 524,
                    },
                },
                "sequence_number": 30,
            },
            event="response.completed",
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_8ucGVxGVa1SVMbD1mwPwwE3c", "call_NyYyHGphM3cB0Z3KkJRhhIfB"),
        arguments=({"query": "retries_remaining("}, {"path": "tests/test_retry.py"}),
        content="I’ll search for callers and run the targeted test to reproduce the failure.",
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 76,
        "total_tokens": 524,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


GPT_5_6_LUNA_AGENT_TEST_CASE: Final = replace(
    GPT_5_6_LUNA_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "openai/gpt-5.6-luna"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-luna"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0f14003909426a48006ad04a8d23a887d081fe3e6b3590c981",
        "created_at": 1792035469,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035470,
        "model": "gpt-5.6-luna",
        "output": [
            {
                "id": "rs_0f14003909426a48006ad04a8de18887d08c1b94c6c89b1874",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "synthetic-signature",
                "summary": [],
            },
            {
                "id": "fc_0f14003909426a48006ad04a8e515c87d0a5775da42fe4bd46",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_3GVFEuVtRLOckfUM9aGO3BWw",
                "name": "search_code",
            },
            {
                "id": "fc_0f14003909426a48006ad04a8e517087d0861ab0d982d00589",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_ZGwThtiIkm6uppdIfybZZyes",
                "name": "run_tests",
            },
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 448,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 65,
            "output_tokens_details": {"reasoning_tokens": 10},
            "total_tokens": 513,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0f14003909426a48006ad04a8d23a887d081fe3e6b3590c981",
        "model": "openai/gpt-5.6-luna",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query":"retries_remaining"}', "name": "search_code"},
                            "id": "call_3GVFEuVtRLOckfUM9aGO3BWw",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_ZGwThtiIkm6uppdIfybZZyes",
                            "type": "function",
                        },
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_0f14003909426a48006ad04a8de18887d08c1b94c6c89b1874",
                            "encrypted_content": "synthetic-signature",
                            "summary": [],
                        }
                    ],
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 65,
            "prompt_tokens": 448,
            "total_tokens": 513,
            "completion_tokens_details": {"reasoning_tokens": 10},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_5_6_LUNA_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_6_LUNA_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_6_LUNA_AGENT_TEST_CASE.litellm_request, GPT_5_6_LUNA_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/responses",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-5.6-luna",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "id": "rs_0f14003909426a48006ad04a8de18887d08c1b94c6c89b1874",
                "summary": [],
                "type": "reasoning",
                "encrypted_content": "synthetic-signature",
            },
            {
                "type": "function_call",
                "call_id": "call_3GVFEuVtRLOckfUM9aGO3BWw",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_ZGwThtiIkm6uppdIfybZZyes",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_3GVFEuVtRLOckfUM9aGO3BWw",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_ZGwThtiIkm6uppdIfybZZyes",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0f14003909426a48006ad04a9045a087d0964813f7654b35a1",
        "created_at": 1792035472,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035473,
        "model": "gpt-5.6-luna",
        "output": [
            {
                "id": "msg_0f14003909426a48006ad04a90ca6c87d0be2c6013a3971814",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly adds `limit` and `attempts`, returning 4 instead of "
                        "2 and potentially negative values are not clamped. It is called by "
                        "`src/worker.py:12`, and `tests/test_retry.py::test_retries_remaining` failed.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 631,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 61,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 692,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0f14003909426a48006ad04a9045a087d0964813f7654b35a1",
        "model": "openai/gpt-5.6-luna",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds `limit` and `attempts`, returning 4 instead "
                    "of 2 and potentially negative values are not clamped. It is called by "
                    "`src/worker.py:12`, and `tests/test_retry.py::test_retries_remaining` failed.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 61,
            "prompt_tokens": 631,
            "total_tokens": 692,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_5_6_LUNA_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_5_6_LUNA_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_5_6_LUNA_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_5_6_LUNA_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/responses",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-luna", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_02bfb16fa00695cf006ad04a8eedb487d096750f7c7c45a477",
                    "created_at": 1792035470,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-5.6-luna",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 0,
            },
            event="response.created",
        ),
        SSEFrame(
            data={
                "type": "response.in_progress",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_02bfb16fa00695cf006ad04a8eedb487d096750f7c7c45a477",
                    "created_at": 1792035470,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-5.6-luna",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 1,
            },
            event="response.in_progress",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "rs_02bfb16fa00695cf006ad04a8f621887d0bc262e1d904dd464",
                    "type": "reasoning",
                    "content": [],
                    "encrypted_content": "synthetic-signature",
                    "summary": [],
                },
                "output_index": 0,
                "sequence_number": 2,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "rs_02bfb16fa00695cf006ad04a8f621887d0bc262e1d904dd464",
                    "type": "reasoning",
                    "content": [],
                    "encrypted_content": "synthetic-signature",
                    "summary": [],
                },
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_02bfb16fa00695cf006ad04a8fddc087d0a24f77988613c342",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_caBz8iaQgGxQPczEdr5gE18A",
                    "name": "search_code",
                },
                "output_index": 1,
                "sequence_number": 4,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining"}',
                "item_id": "fc_02bfb16fa00695cf006ad04a8fddc087d0a24f77988613c342",
                "obfuscation": "QP1",
                "output_index": 1,
                "sequence_number": 5,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_02bfb16fa00695cf006ad04a8fddc087d0a24f77988613c342",
                "output_index": 1,
                "sequence_number": 6,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_02bfb16fa00695cf006ad04a8fddc087d0a24f77988613c342",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_caBz8iaQgGxQPczEdr5gE18A",
                    "name": "search_code",
                },
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_02bfb16fa00695cf006ad04a8fddd487d08191a0749adf899d",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_Bg8gxxs2CV57YzN0vxup4DvY",
                    "name": "run_tests",
                },
                "output_index": 2,
                "sequence_number": 8,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_02bfb16fa00695cf006ad04a8fddd487d08191a0749adf899d",
                "obfuscation": "Y0",
                "output_index": 2,
                "sequence_number": 9,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_02bfb16fa00695cf006ad04a8fddd487d08191a0749adf899d",
                "output_index": 2,
                "sequence_number": 10,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_02bfb16fa00695cf006ad04a8fddd487d08191a0749adf899d",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_Bg8gxxs2CV57YzN0vxup4DvY",
                    "name": "run_tests",
                },
                "output_index": 2,
                "sequence_number": 11,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_02bfb16fa00695cf006ad04a8eedb487d096750f7c7c45a477",
                    "created_at": 1792035470,
                    "status": "completed",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": 1792035471,
                    "model": "gpt-5.6-luna",
                    "output": [
                        {
                            "id": "rs_02bfb16fa00695cf006ad04a8f621887d0bc262e1d904dd464",
                            "type": "reasoning",
                            "content": [],
                            "encrypted_content": "synthetic-signature",
                            "summary": [],
                        },
                        {
                            "id": "fc_02bfb16fa00695cf006ad04a8fddc087d0a24f77988613c342",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_caBz8iaQgGxQPczEdr5gE18A",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_02bfb16fa00695cf006ad04a8fddd487d08191a0749adf899d",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_Bg8gxxs2CV57YzN0vxup4DvY",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 67,
                        "output_tokens_details": {"reasoning_tokens": 12},
                        "total_tokens": 515,
                    },
                },
                "sequence_number": 12,
            },
            event="response.completed",
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_caBz8iaQgGxQPczEdr5gE18A", "call_Bg8gxxs2CV57YzN0vxup4DvY"),
        reasoning_items=(
            {
                "id": "rs_02bfb16fa00695cf006ad04a8f621887d0bc262e1d904dd464",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "synthetic-signature",
                "summary": [],
            },
        ),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 67,
        "total_tokens": 515,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 12},
    },
)


GPT_5_6_SOL_AGENT_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "openai/gpt-5.6-sol"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-sol"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0d5f7a80fc3395a0006ad04a9e6d0887d0ae4fb1f1c2c82d05",
        "created_at": 1792035486,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035487,
        "model": "gpt-5.6-sol",
        "output": [
            {
                "id": "fc_0d5f7a80fc3395a0006ad04a9f5fc087d088ad5fac4a144541",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_zImu1qJwA3vjFH9oAm11Hmu9",
                "name": "search_code",
            },
            {
                "id": "fc_0d5f7a80fc3395a0006ad04a9f5fd487d08863403c471f66a2",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_D6WIuRv4nU3S9ROhWYxh3e7U",
                "name": "run_tests",
            },
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 448,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 53,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0d5f7a80fc3395a0006ad04a9e6d0887d0ae4fb1f1c2c82d05",
        "model": "openai/gpt-5.6-sol",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query":"retries_remaining"}', "name": "search_code"},
                            "id": "call_zImu1qJwA3vjFH9oAm11Hmu9",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_D6WIuRv4nU3S9ROhWYxh3e7U",
                            "type": "function",
                        },
                    ],
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_5_6_SOL_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_6_SOL_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_6_SOL_AGENT_TEST_CASE.litellm_request, GPT_5_6_SOL_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/responses",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-5.6-sol",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_zImu1qJwA3vjFH9oAm11Hmu9",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_D6WIuRv4nU3S9ROhWYxh3e7U",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_zImu1qJwA3vjFH9oAm11Hmu9",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_D6WIuRv4nU3S9ROhWYxh3e7U",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_00b169780b921f8d006ad04aa121b087d0b9ba7a302cc95525",
        "created_at": 1792035489,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035491,
        "model": "gpt-5.6-sol",
        "output": [
            {
                "id": "msg_00b169780b921f8d006ad04aa2b2e887d0a17f0ac4a6c9e0ea",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly adds `limit + attempts`; it should compute "
                        "`max(limit - attempts, 0)`. The caller is `src/worker.py:12`, and "
                        "`tests/test_retry.py` failed because `retries_remaining(3, 1)` returned `4` instead "
                        "of `2`.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 619,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 68,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 687,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_00b169780b921f8d006ad04aa121b087d0b9ba7a302cc95525",
        "model": "openai/gpt-5.6-sol",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds `limit + attempts`; it should compute "
                    "`max(limit - attempts, 0)`. The caller is `src/worker.py:12`, and "
                    "`tests/test_retry.py` failed because `retries_remaining(3, 1)` returned `4` "
                    "instead of `2`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 68,
            "prompt_tokens": 619,
            "total_tokens": 687,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_5_6_SOL_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_5_6_SOL_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_5_6_SOL_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_5_6_SOL_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/responses",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-sol", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0ebd0518833b954d006ad04a9fb3b887d099f87343fa94eb4c",
                    "created_at": 1792035487,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-5.6-sol",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 0,
            },
            event="response.created",
        ),
        SSEFrame(
            data={
                "type": "response.in_progress",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0ebd0518833b954d006ad04a9fb3b887d099f87343fa94eb4c",
                    "created_at": 1792035487,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-5.6-sol",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 1,
            },
            event="response.in_progress",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_0ebd0518833b954d006ad04aa0b1bc87d088f28baa27dae3db",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_RHePNBeoQ4fn5WnJUzavuVVF",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 2,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining"}',
                "item_id": "fc_0ebd0518833b954d006ad04aa0b1bc87d088f28baa27dae3db",
                "obfuscation": "lsF",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0ebd0518833b954d006ad04aa0b1bc87d088f28baa27dae3db",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0ebd0518833b954d006ad04aa0b1bc87d088f28baa27dae3db",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_RHePNBeoQ4fn5WnJUzavuVVF",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 5,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_0ebd0518833b954d006ad04aa0b1cc87d0abe535b36be72b7d",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_1h6M2yMno8h8DYlCJsImg3N4",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 6,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0ebd0518833b954d006ad04aa0b1cc87d0abe535b36be72b7d",
                "obfuscation": "5m",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0ebd0518833b954d006ad04aa0b1cc87d0abe535b36be72b7d",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0ebd0518833b954d006ad04aa0b1cc87d0abe535b36be72b7d",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_1h6M2yMno8h8DYlCJsImg3N4",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 9,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0ebd0518833b954d006ad04a9fb3b887d099f87343fa94eb4c",
                    "created_at": 1792035487,
                    "status": "completed",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": 1792035488,
                    "model": "gpt-5.6-sol",
                    "output": [
                        {
                            "id": "fc_0ebd0518833b954d006ad04aa0b1bc87d088f28baa27dae3db",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_RHePNBeoQ4fn5WnJUzavuVVF",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0ebd0518833b954d006ad04aa0b1cc87d0abe535b36be72b7d",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_1h6M2yMno8h8DYlCJsImg3N4",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 53,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 501,
                    },
                },
                "sequence_number": 10,
            },
            event="response.completed",
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_RHePNBeoQ4fn5WnJUzavuVVF", "call_1h6M2yMno8h8DYlCJsImg3N4"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


GPT_5_6_TERRA_AGENT_TEST_CASE: Final = replace(
    GPT_5_6_TERRA_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "openai/gpt-5.6-terra"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-terra"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0be594400e6793a9006ad04aa8479487d0a036daf89a003d98",
        "created_at": 1792035496,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035497,
        "model": "gpt-5.6-terra",
        "output": [
            {
                "id": "fc_0be594400e6793a9006ad04aa91fb487d0a879ab47aea4a82f",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_ouugWUOqgTyW3gYHpwTMK8d9",
                "name": "search_code",
            },
            {
                "id": "fc_0be594400e6793a9006ad04aa91fc887d0b9f5335792f10a68",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_8fATNhdv39Db1hljGzU7PHcW",
                "name": "run_tests",
            },
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 448,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 53,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0be594400e6793a9006ad04aa8479487d0a036daf89a003d98",
        "model": "openai/gpt-5.6-terra",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query":"retries_remaining"}', "name": "search_code"},
                            "id": "call_ouugWUOqgTyW3gYHpwTMK8d9",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_8fATNhdv39Db1hljGzU7PHcW",
                            "type": "function",
                        },
                    ],
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_5_6_TERRA_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_6_TERRA_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_6_TERRA_AGENT_TEST_CASE.litellm_request, GPT_5_6_TERRA_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/responses",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-5.6-terra",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_ouugWUOqgTyW3gYHpwTMK8d9",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_8fATNhdv39Db1hljGzU7PHcW",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_ouugWUOqgTyW3gYHpwTMK8d9",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_8fATNhdv39Db1hljGzU7PHcW",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0fe2d1d5d2bfdea9006ad04aaaeaf887d0b804e06e9f2a88bb",
        "created_at": 1792035498,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035500,
        "model": "gpt-5.6-terra",
        "output": [
            {
                "id": "msg_0fe2d1d5d2bfdea9006ad04aab8a0087d0889580abf3a2b2c0",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly returns `limit + attempts`; it should subtract "
                        "attempts and clamp the result at zero. The only caller is `src/worker.py`, and "
                        "`tests/test_retry.py` fails because `retries_remaining(3, 1)` returns `4` instead of "
                        "`2`.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 619,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 65,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 684,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0fe2d1d5d2bfdea9006ad04aaaeaf887d0b804e06e9f2a88bb",
        "model": "openai/gpt-5.6-terra",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly returns `limit + attempts`; it should subtract "
                    "attempts and clamp the result at zero. The only caller is `src/worker.py`, and "
                    "`tests/test_retry.py` fails because `retries_remaining(3, 1)` returns `4` instead "
                    "of `2`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 65,
            "prompt_tokens": 619,
            "total_tokens": 684,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_5_6_TERRA_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_5_6_TERRA_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_5_6_TERRA_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_5_6_TERRA_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/responses",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-terra", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_07964387f422fad6006ad04aa9820c87d0b00c0f7e8179e669",
                    "created_at": 1792035497,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-5.6-terra",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 0,
            },
            event="response.created",
        ),
        SSEFrame(
            data={
                "type": "response.in_progress",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_07964387f422fad6006ad04aa9820c87d0b00c0f7e8179e669",
                    "created_at": 1792035497,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-5.6-terra",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 1,
            },
            event="response.in_progress",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_07964387f422fad6006ad04aaa99d087d09bf5301d4def0fb9",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_xDdtGVzn464uEE7X8AuVRiHJ",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 2,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining"}',
                "item_id": "fc_07964387f422fad6006ad04aaa99d087d09bf5301d4def0fb9",
                "obfuscation": "IFA",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_07964387f422fad6006ad04aaa99d087d09bf5301d4def0fb9",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_07964387f422fad6006ad04aaa99d087d09bf5301d4def0fb9",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_xDdtGVzn464uEE7X8AuVRiHJ",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 5,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_07964387f422fad6006ad04aaa99e487d0adda87c8afeac85a",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_yPhpkZWP5B2zJo4SPhCEAxW4",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 6,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_07964387f422fad6006ad04aaa99e487d0adda87c8afeac85a",
                "obfuscation": "RH",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_07964387f422fad6006ad04aaa99e487d0adda87c8afeac85a",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_07964387f422fad6006ad04aaa99e487d0adda87c8afeac85a",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_yPhpkZWP5B2zJo4SPhCEAxW4",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 9,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_07964387f422fad6006ad04aa9820c87d0b00c0f7e8179e669",
                    "created_at": 1792035497,
                    "status": "completed",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": 1792035498,
                    "model": "gpt-5.6-terra",
                    "output": [
                        {
                            "id": "fc_07964387f422fad6006ad04aaa99d087d09bf5301d4def0fb9",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_xDdtGVzn464uEE7X8AuVRiHJ",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_07964387f422fad6006ad04aaa99e487d0adda87c8afeac85a",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_yPhpkZWP5B2zJo4SPhCEAxW4",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 53,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 501,
                    },
                },
                "sequence_number": 10,
            },
            event="response.completed",
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_xDdtGVzn464uEE7X8AuVRiHJ", "call_yPhpkZWP5B2zJo4SPhCEAxW4"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


GPT_6_ASTRA_AGENT_TEST_CASE: Final = replace(
    GPT_6_ASTRA_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "openai/gpt-6-astra"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-astra"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0c460ec382f21326006ad04aba1fc887d095a25a4a60b52123",
        "created_at": 1792035514,
        "status": "completed",
        "access_programs": None,
        "billing": {"payer": "developer"},
        "completed_at": 1792035515,
        "model": "gpt-6-astra",
        "output": [
            {
                "id": "fc_0c460ec382f21326006ad04abbb08c87d0afe6415b0f3dc234",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_zlQhijQRYcsyaid7PMS9OLr8",
                "name": "search_code",
            },
            {
                "id": "fc_0c460ec382f21326006ad04abbb0a487d09860b37cc0b5a25e",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_HhTD5sPm6PxiPB1xX8sGLZid",
                "name": "run_tests",
            },
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 448,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 53,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0c460ec382f21326006ad04aba1fc887d095a25a4a60b52123",
        "model": "openai/gpt-6-astra",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query":"retries_remaining"}', "name": "search_code"},
                            "id": "call_zlQhijQRYcsyaid7PMS9OLr8",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_HhTD5sPm6PxiPB1xX8sGLZid",
                            "type": "function",
                        },
                    ],
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_6_ASTRA_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_6_ASTRA_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_6_ASTRA_AGENT_TEST_CASE.litellm_request, GPT_6_ASTRA_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/responses",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-6-astra",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_zlQhijQRYcsyaid7PMS9OLr8",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_HhTD5sPm6PxiPB1xX8sGLZid",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_zlQhijQRYcsyaid7PMS9OLr8",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_HhTD5sPm6PxiPB1xX8sGLZid",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0eb40bf438f81237006ad04abe048087d097e1ca444166aa82",
        "created_at": 1792035518,
        "status": "completed",
        "access_programs": None,
        "billing": {"payer": "developer"},
        "completed_at": 1792035520,
        "model": "gpt-6-astra",
        "output": [
            {
                "id": "msg_0eb40bf438f81237006ad04abf240087d0b599ea1d1e38c6ce",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly adds attempts to the limit; it should subtract "
                        "attempts and clamp the result to zero. A caller was found at `src/worker.py:12`, and "
                        "`tests/test_retry.py::test_retries_remaining` failed because it returned 4 instead "
                        "of 2.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 619,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 63,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 682,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0eb40bf438f81237006ad04abe048087d097e1ca444166aa82",
        "model": "openai/gpt-6-astra",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds attempts to the limit; it should subtract "
                    "attempts and clamp the result to zero. A caller was found at `src/worker.py:12`, "
                    "and `tests/test_retry.py::test_retries_remaining` failed because it returned 4 "
                    "instead of 2.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 63,
            "prompt_tokens": 619,
            "total_tokens": 682,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_6_ASTRA_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_6_ASTRA_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_6_ASTRA_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_6_ASTRA_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/responses",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-astra", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0e6cd25a5edf0b7f006ad04abc0b8087d08587ae2062f213ba",
                    "created_at": 1792035516,
                    "status": "in_progress",
                    "access_programs": None,
                    "completed_at": None,
                    "model": "gpt-6-astra",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 0,
            },
            event="response.created",
        ),
        SSEFrame(
            data={
                "type": "response.in_progress",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0e6cd25a5edf0b7f006ad04abc0b8087d08587ae2062f213ba",
                    "created_at": 1792035516,
                    "status": "in_progress",
                    "access_programs": None,
                    "completed_at": None,
                    "model": "gpt-6-astra",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 1,
            },
            event="response.in_progress",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_0e6cd25a5edf0b7f006ad04abdb3cc87d087c65ee2c89879b1",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_rdLoAf1TMckmLuY7a9q3qGV2",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 2,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining"}',
                "item_id": "fc_0e6cd25a5edf0b7f006ad04abdb3cc87d087c65ee2c89879b1",
                "obfuscation": "szK",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0e6cd25a5edf0b7f006ad04abdb3cc87d087c65ee2c89879b1",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0e6cd25a5edf0b7f006ad04abdb3cc87d087c65ee2c89879b1",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_rdLoAf1TMckmLuY7a9q3qGV2",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 5,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_0e6cd25a5edf0b7f006ad04abdb3e087d0a1023113b3068644",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_BCEXFCGlvI1hhltFXxKMhOQj",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 6,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0e6cd25a5edf0b7f006ad04abdb3e087d0a1023113b3068644",
                "obfuscation": "T3",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0e6cd25a5edf0b7f006ad04abdb3e087d0a1023113b3068644",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0e6cd25a5edf0b7f006ad04abdb3e087d0a1023113b3068644",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_BCEXFCGlvI1hhltFXxKMhOQj",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 9,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0e6cd25a5edf0b7f006ad04abc0b8087d08587ae2062f213ba",
                    "created_at": 1792035516,
                    "status": "completed",
                    "access_programs": None,
                    "completed_at": 1792035517,
                    "model": "gpt-6-astra",
                    "output": [
                        {
                            "id": "fc_0e6cd25a5edf0b7f006ad04abdb3cc87d087c65ee2c89879b1",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_rdLoAf1TMckmLuY7a9q3qGV2",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0e6cd25a5edf0b7f006ad04abdb3e087d0a1023113b3068644",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_BCEXFCGlvI1hhltFXxKMhOQj",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 53,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 501,
                    },
                },
                "sequence_number": 10,
            },
            event="response.completed",
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_rdLoAf1TMckmLuY7a9q3qGV2", "call_BCEXFCGlvI1hhltFXxKMhOQj"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


GPT_6_LUNA_AGENT_TEST_CASE: Final = replace(
    GPT_6_LUNA_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "openai/gpt-6-luna"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-luna"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_02084d7da75674a2006ad04a8e5ba087d09bcd2d078980b34f",
        "created_at": 1792035470,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035473,
        "model": "gpt-6-luna",
        "output": [
            {
                "id": "fc_02084d7da75674a2006ad04a91d01087d092d71e6d7c9d6cc5",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_UlhZOD1wA0vr4DfyZTAHc8xz",
                "name": "search_code",
            },
            {
                "id": "fc_02084d7da75674a2006ad04a91d04087d084b80063891ba7cf",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_iVnLyAFnnvoEcpuai2jPibro",
                "name": "run_tests",
            },
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 448,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 53,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_02084d7da75674a2006ad04a8e5ba087d09bcd2d078980b34f",
        "model": "openai/gpt-6-luna",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query":"retries_remaining"}', "name": "search_code"},
                            "id": "call_UlhZOD1wA0vr4DfyZTAHc8xz",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_iVnLyAFnnvoEcpuai2jPibro",
                            "type": "function",
                        },
                    ],
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_6_LUNA_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_6_LUNA_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_6_LUNA_AGENT_TEST_CASE.litellm_request, GPT_6_LUNA_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/responses",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-6-luna",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_UlhZOD1wA0vr4DfyZTAHc8xz",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_iVnLyAFnnvoEcpuai2jPibro",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_UlhZOD1wA0vr4DfyZTAHc8xz",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_iVnLyAFnnvoEcpuai2jPibro",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_09aa77587dce4423006ad04a966dd087d0887015ed690aa27c",
        "created_at": 1792035478,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035481,
        "model": "gpt-6-luna",
        "output": [
            {
                "id": "msg_09aa77587dce4423006ad04a98665c87d0b324abdcf3b3c685",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` adds `limit` and `attempts` instead of subtracting attempts from "
                        "the limit, so it can return a negative value. The caller is `src/worker.py`; "
                        "`tests/test_retry.py` failed because `retries_remaining(3, 1)` returned 4 instead of "
                        "2.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 619,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 70,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 689,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_09aa77587dce4423006ad04a966dd087d0887015ed690aa27c",
        "model": "openai/gpt-6-luna",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds `limit` and `attempts` instead of subtracting attempts "
                    "from the limit, so it can return a negative value. The caller is `src/worker.py`; "
                    "`tests/test_retry.py` failed because `retries_remaining(3, 1)` returned 4 instead "
                    "of 2.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 70,
            "prompt_tokens": 619,
            "total_tokens": 689,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_6_LUNA_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_6_LUNA_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_6_LUNA_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_6_LUNA_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/responses",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-luna", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0dd28c98133095a3006ad04a92a73887d0958056861d8bfcff",
                    "created_at": 1792035474,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-6-luna",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 0,
            },
            event="response.created",
        ),
        SSEFrame(
            data={
                "type": "response.in_progress",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0dd28c98133095a3006ad04a92a73887d0958056861d8bfcff",
                    "created_at": 1792035474,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-6-luna",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 1,
            },
            event="response.in_progress",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_0dd28c98133095a3006ad04a96253487d090fa557a2e0cf905",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_BXnB0UfiU2LW63KiDbncliyC",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 2,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining"}',
                "item_id": "fc_0dd28c98133095a3006ad04a96253487d090fa557a2e0cf905",
                "obfuscation": "Ios",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0dd28c98133095a3006ad04a96253487d090fa557a2e0cf905",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0dd28c98133095a3006ad04a96253487d090fa557a2e0cf905",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_BXnB0UfiU2LW63KiDbncliyC",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 5,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_0dd28c98133095a3006ad04a96254887d0b68dd02014ba2ca8",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_rN9W4duhkWVD0ycWn3xo03Dl",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 6,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0dd28c98133095a3006ad04a96254887d0b68dd02014ba2ca8",
                "obfuscation": "m1",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0dd28c98133095a3006ad04a96254887d0b68dd02014ba2ca8",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0dd28c98133095a3006ad04a96254887d0b68dd02014ba2ca8",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_rN9W4duhkWVD0ycWn3xo03Dl",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 9,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0dd28c98133095a3006ad04a92a73887d0958056861d8bfcff",
                    "created_at": 1792035474,
                    "status": "completed",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": 1792035478,
                    "model": "gpt-6-luna",
                    "output": [
                        {
                            "id": "fc_0dd28c98133095a3006ad04a96253487d090fa557a2e0cf905",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_BXnB0UfiU2LW63KiDbncliyC",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0dd28c98133095a3006ad04a96254887d0b68dd02014ba2ca8",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_rN9W4duhkWVD0ycWn3xo03Dl",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 53,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 501,
                    },
                },
                "sequence_number": 10,
            },
            event="response.completed",
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_BXnB0UfiU2LW63KiDbncliyC", "call_rN9W4duhkWVD0ycWn3xo03Dl"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


GPT_6_SOL_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_6_SOL_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_6_SOL_TEST_CASE.litellm_request, GPT_6_SOL_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/responses",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-6-sol",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_R9OwdkIw3MYvLUzFLkNUgcof",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_9jA0N8trabVxGFS04x6FbMQ4",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_R9OwdkIw3MYvLUzFLkNUgcof",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_9jA0N8trabVxGFS04x6FbMQ4",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_002863be50c3c198006ad04aa69ba487d0a65f207d45bc6d04",
        "created_at": 1792035494,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "completed_at": 1792035495,
        "model": "gpt-6-sol",
        "output": [
            {
                "id": "msg_002863be50c3c198006ad04aa739c887d0a2e4214f4a5a76ac",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` adds `attempts` to `limit` instead of subtracting it and "
                        "clamping the result at zero; `src/worker.py` calls it. `tests/test_retry.py` failed "
                        "because `retries_remaining(3, 1)` returned 4 instead of 2.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 619,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 66,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 685,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_002863be50c3c198006ad04aa69ba487d0a65f207d45bc6d04",
        "model": "openai/gpt-6-sol",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds `attempts` to `limit` instead of subtracting it and "
                    "clamping the result at zero; `src/worker.py` calls it. `tests/test_retry.py` "
                    "failed because `retries_remaining(3, 1)` returned 4 instead of 2.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 66,
            "prompt_tokens": 619,
            "total_tokens": 685,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_6_SOL_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_6_SOL_TEST_CASE.litellm_endpoint,
    litellm_request={**GPT_6_SOL_TEST_CASE.litellm_request, "stream": True, "stream_options": {"include_usage": True}},
    expected_provider_headers=GPT_6_SOL_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/responses",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-sol", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_007fb930e79ce091006ad04aa52aac87d08c4ac9fdaeedc479",
                    "created_at": 1792035493,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-6-sol",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 0,
            },
            event="response.created",
        ),
        SSEFrame(
            data={
                "type": "response.in_progress",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_007fb930e79ce091006ad04aa52aac87d08c4ac9fdaeedc479",
                    "created_at": 1792035493,
                    "status": "in_progress",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": None,
                    "model": "gpt-6-sol",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 1,
            },
            event="response.in_progress",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_007fb930e79ce091006ad04aa636fc87d0ad9672a977254ff7",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_ISAnqtLEGvg6OfXg9D16OBHr",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 2,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining"}',
                "item_id": "fc_007fb930e79ce091006ad04aa636fc87d0ad9672a977254ff7",
                "obfuscation": "qIu",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_007fb930e79ce091006ad04aa636fc87d0ad9672a977254ff7",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_007fb930e79ce091006ad04aa636fc87d0ad9672a977254ff7",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_ISAnqtLEGvg6OfXg9D16OBHr",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 5,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_007fb930e79ce091006ad04aa6370c87d088a9a43b41d1d60f",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_h6Z6YQpd9m7NsG5iiF05Wue9",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 6,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_007fb930e79ce091006ad04aa6370c87d088a9a43b41d1d60f",
                "obfuscation": "ip",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_007fb930e79ce091006ad04aa6370c87d088a9a43b41d1d60f",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_007fb930e79ce091006ad04aa6370c87d088a9a43b41d1d60f",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_h6Z6YQpd9m7NsG5iiF05Wue9",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 9,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_007fb930e79ce091006ad04aa52aac87d08c4ac9fdaeedc479",
                    "created_at": 1792035493,
                    "status": "completed",
                    "access_programs": {"cyber": "daybreak_blue"},
                    "completed_at": 1792035494,
                    "model": "gpt-6-sol",
                    "output": [
                        {
                            "id": "fc_007fb930e79ce091006ad04aa636fc87d0ad9672a977254ff7",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_ISAnqtLEGvg6OfXg9D16OBHr",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_007fb930e79ce091006ad04aa6370c87d088a9a43b41d1d60f",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_h6Z6YQpd9m7NsG5iiF05Wue9",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 53,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 501,
                    },
                },
                "sequence_number": 10,
            },
            event="response.completed",
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_ISAnqtLEGvg6OfXg9D16OBHr", "call_h6Z6YQpd9m7NsG5iiF05Wue9"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


GPT_6_1_SOL_AGENT_TEST_CASE: Final = replace(
    GPT_6_1_SOL_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "openai/gpt-6.1-sol"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6.1-sol"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0bf6d56fb68ed25e006ad04aa4d03c87d0bd8428d6dc163509",
        "created_at": 1792035492,
        "status": "completed",
        "access_programs": None,
        "billing": {"payer": "developer"},
        "completed_at": 1792035494,
        "model": "gpt-6.1-sol",
        "output": [
            {
                "id": "fc_0bf6d56fb68ed25e006ad04aa6488087d0a469e018c1b152d6",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_oG4NqCvca99WyRvP8UbjOvgO",
                "name": "search_code",
            },
            {
                "id": "fc_0bf6d56fb68ed25e006ad04aa6489087d08af67d6330cba787",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_TCvHHb4Vi9q7QfL6YkWGcKoT",
                "name": "run_tests",
            },
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 448,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 53,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0bf6d56fb68ed25e006ad04aa4d03c87d0bd8428d6dc163509",
        "model": "openai/gpt-6.1-sol",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query":"retries_remaining"}', "name": "search_code"},
                            "id": "call_oG4NqCvca99WyRvP8UbjOvgO",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_TCvHHb4Vi9q7QfL6YkWGcKoT",
                            "type": "function",
                        },
                    ],
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_6_1_SOL_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_6_1_SOL_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_6_1_SOL_AGENT_TEST_CASE.litellm_request, GPT_6_1_SOL_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/responses",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-6.1-sol",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_oG4NqCvca99WyRvP8UbjOvgO",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_TCvHHb4Vi9q7QfL6YkWGcKoT",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_oG4NqCvca99WyRvP8UbjOvgO",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_TCvHHb4Vi9q7QfL6YkWGcKoT",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_09b6501350a8bfd9006ad04aa944dc87d089844d553ed99ad5",
        "created_at": 1792035497,
        "status": "completed",
        "access_programs": None,
        "billing": {"payer": "developer"},
        "completed_at": 1792035499,
        "model": "gpt-6.1-sol",
        "output": [
            {
                "id": "msg_09b6501350a8bfd9006ad04aaa2bf087d08631dc13aa2d46dc",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly adds attempts to the limit; it should subtract "
                        "attempts and clamp the result to zero, and a caller was found at `src/worker.py:12`. "
                        "`tests/test_retry.py::test_retries_remaining` failed because it returned 4 instead "
                        "of 2; no files were changed.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "24h",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 619,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 68,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 687,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_09b6501350a8bfd9006ad04aa944dc87d089844d553ed99ad5",
        "model": "openai/gpt-6.1-sol",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds attempts to the limit; it should subtract "
                    "attempts and clamp the result to zero, and a caller was found at "
                    "`src/worker.py:12`. `tests/test_retry.py::test_retries_remaining` failed because "
                    "it returned 4 instead of 2; no files were changed.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 68,
            "prompt_tokens": 619,
            "total_tokens": 687,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
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
)


GPT_6_1_SOL_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_6_1_SOL_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_6_1_SOL_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_6_1_SOL_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/responses",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6.1-sol", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_034cb270fc22c3d0006ad04aa69ab887d0a5d949bf2af3dd07",
                    "created_at": 1792035494,
                    "status": "in_progress",
                    "access_programs": None,
                    "completed_at": None,
                    "model": "gpt-6.1-sol",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 0,
            },
            event="response.created",
        ),
        SSEFrame(
            data={
                "type": "response.in_progress",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_034cb270fc22c3d0006ad04aa69ab887d0a5d949bf2af3dd07",
                    "created_at": 1792035494,
                    "status": "in_progress",
                    "access_programs": None,
                    "completed_at": None,
                    "model": "gpt-6.1-sol",
                    "output": [],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "auto",
                    "usage": None,
                },
                "sequence_number": 1,
            },
            event="response.in_progress",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_034cb270fc22c3d0006ad04aa8c81487d093c417f3350d851e",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_dRjirW1hJv1g0OIFLZi21Ssq",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 2,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining"}',
                "item_id": "fc_034cb270fc22c3d0006ad04aa8c81487d093c417f3350d851e",
                "obfuscation": "2nb",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_034cb270fc22c3d0006ad04aa8c81487d093c417f3350d851e",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_034cb270fc22c3d0006ad04aa8c81487d093c417f3350d851e",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_dRjirW1hJv1g0OIFLZi21Ssq",
                    "name": "search_code",
                },
                "output_index": 0,
                "sequence_number": 5,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_034cb270fc22c3d0006ad04aa8c82487d0a353e420b4fe4315",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_4Rb17RKwt3mIAswOKn3gi8Im",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 6,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_034cb270fc22c3d0006ad04aa8c82487d0a353e420b4fe4315",
                "obfuscation": "Cc",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_034cb270fc22c3d0006ad04aa8c82487d0a353e420b4fe4315",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_034cb270fc22c3d0006ad04aa8c82487d0a353e420b4fe4315",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_4Rb17RKwt3mIAswOKn3gi8Im",
                    "name": "run_tests",
                },
                "output_index": 1,
                "sequence_number": 9,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_034cb270fc22c3d0006ad04aa69ab887d0a5d949bf2af3dd07",
                    "created_at": 1792035494,
                    "status": "completed",
                    "access_programs": None,
                    "completed_at": 1792035496,
                    "model": "gpt-6.1-sol",
                    "output": [
                        {
                            "id": "fc_034cb270fc22c3d0006ad04aa8c81487d093c417f3350d851e",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_dRjirW1hJv1g0OIFLZi21Ssq",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_034cb270fc22c3d0006ad04aa8c82487d0a353e420b4fe4315",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_4Rb17RKwt3mIAswOKn3gi8Im",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 53,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 501,
                    },
                },
                "sequence_number": 10,
            },
            event="response.completed",
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_dRjirW1hJv1g0OIFLZi21Ssq", "call_4Rb17RKwt3mIAswOKn3gi8Im"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)
