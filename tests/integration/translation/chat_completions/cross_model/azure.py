from dataclasses import replace
from typing import Final

from integration._support.client import list_value, object_value
from integration.translation.case import (
    SSEFrame,
    StreamingTranslationTestCase,
)
from integration.translation.chat_completions.cross_model.contracts import followup_request
from integration.translation.chat_completions.bases.azure import (
    GPT_5_4_MINI_TEST_CASE,
    GPT_5_6_LUNA_TEST_CASE,
    GPT_5_6_SOL_TEST_CASE,
    GPT_5_6_TERRA_TEST_CASE,
    GPT_6_1_SOL_TEST_CASE,
    GPT_6_ASTRA_TEST_CASE,
    GPT_6_LUNA_TEST_CASE,
)
from integration.translation.chat_completions.templates import (
    CODING_AGENT_REQUEST,
    CODING_AGENT_RESPONSE,
)
from integration.translation.templates.openai import (
    OPENAI_RESPONSES_AGENT_REQUEST,
    OPENAI_RESPONSES_AGENT_RESPONSE,
)

GPT_5_4_MINI_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_4_MINI_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_4_MINI_TEST_CASE.litellm_request, GPT_5_4_MINI_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
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
                        "text": "I’ll search for callers and run the targeted test before patching.",
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_d8z67veoQLDjeKe3peLFNYvC",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining("}',
            },
            {
                "type": "function_call",
                "call_id": "call_yL8qxv5wZi6Eoesr69xs2UBd",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_d8z67veoQLDjeKe3peLFNYvC",
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
                "call_id": "call_yL8qxv5wZi6Eoesr69xs2UBd",
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
        "id": "resp_06ec32a505685f87006ad04ad0730881908ab0c1abc319010a",
        "created_at": 1792035536,
        "status": "completed",
        "completed_at": 1792035537,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1544, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 327, "check_offset": 0},
            },
        ],
        "model": "gpt-5.4-mini",
        "output": [
            {
                "id": "msg_06ec32a505685f87006ad04ad1429481909d8c3580ca80b194",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` is implemented as `limit + attempts`, so it returns the wrong "
                        "value and can exceed the retry limit instead of counting retries left, which should "
                        "never go below zero. I found one caller in `src/worker.py` (`remaining = "
                        "retries_remaining(3, attempts)`), and `tests/test_retry.py` failed with `assert 4 == "
                        "2`.",
                    }
                ],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "prompt_cache_retention": "in_memory",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "current_turn"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 641,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 81,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 722,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_06ec32a505685f87006ad04ad0730881908ab0c1abc319010a",
        "model": "azure/gpt-5.4-mini",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` is implemented as `limit + attempts`, so it returns the wrong "
                    "value and can exceed the retry limit instead of counting retries left, which "
                    "should never go below zero. I found one caller in `src/worker.py` (`remaining = "
                    "retries_remaining(3, attempts)`), and `tests/test_retry.py` failed with `assert 4 "
                    "== 2`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 81,
            "prompt_tokens": 641,
            "total_tokens": 722,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1544, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 327, "check_offset": 0},
            },
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.4-mini", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0ea6c1cbfcc73010006ad04acd01d481939812185df9b09745",
                    "created_at": 1792035533,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
                    "model": "gpt-5.4-mini",
                    "output": [],
                    "prompt_cache_retention": "in_memory",
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
                    "id": "resp_0ea6c1cbfcc73010006ad04acd01d481939812185df9b09745",
                    "created_at": 1792035533,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
                    "model": "gpt-5.4-mini",
                    "output": [],
                    "prompt_cache_retention": "in_memory",
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
                    "id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
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
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
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
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "pPj5QsU4fqIYwwd",
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
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "h6F0Twp72vyNf",
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
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "dsVhK4t0D",
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
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "eSos9BMVKRKg",
                "output_index": 0,
                "sequence_number": 7,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " other",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "iMPuYxpimb",
                "output_index": 0,
                "sequence_number": 8,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " callers",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "sR6ai0lK",
                "output_index": 0,
                "sequence_number": 9,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " and",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "wC38sb0I6W5P",
                "output_index": 0,
                "sequence_number": 10,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " run",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "j8HyEZhFj59X",
                "output_index": 0,
                "sequence_number": 11,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " the",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "Bg48dNsGNXu9",
                "output_index": 0,
                "sequence_number": 12,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " targeted",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "Xdciymf",
                "output_index": 0,
                "sequence_number": 13,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " test",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "k9v1ZTQZZC0",
                "output_index": 0,
                "sequence_number": 14,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " before",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "vEB8FyB7z",
                "output_index": 0,
                "sequence_number": 15,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": " editing",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "rJnUpEPt",
                "output_index": 0,
                "sequence_number": 16,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": ".",
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "obfuscation": "7BEy69GpMxtXCGN",
                "output_index": 0,
                "sequence_number": 17,
            },
            event="response.output_text.delta",
        ),
        SSEFrame(
            data={
                "type": "response.output_text.done",
                "content_index": 0,
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "logprobs": [],
                "output_index": 0,
                "sequence_number": 18,
                "text": "I’ll search for other callers and run the targeted test before editing.",
            },
            event="response.output_text.done",
        ),
        SSEFrame(
            data={
                "type": "response.content_part.done",
                "content_index": 0,
                "item_id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                "output_index": 0,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": "I’ll search for other callers and run the targeted test before editing.",
                },
                "sequence_number": 19,
            },
            event="response.content_part.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [],
                            "logprobs": [],
                            "text": "I’ll search for other callers and run the targeted test before editing.",
                        }
                    ],
                    "phase": "commentary",
                    "role": "assistant",
                },
                "output_index": 0,
                "sequence_number": 20,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_0ea6c1cbfcc73010006ad04ad01dd881939b6cd34e31efbadd",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_fIwA6Y6RbSeLJr9xUjjM4dZG",
                    "name": "search_code",
                },
                "output_index": 1,
                "sequence_number": 21,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"query":"retries_remaining("}',
                "item_id": "fc_0ea6c1cbfcc73010006ad04ad01dd881939b6cd34e31efbadd",
                "obfuscation": "0m",
                "output_index": 1,
                "sequence_number": 22,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining("}',
                "item_id": "fc_0ea6c1cbfcc73010006ad04ad01dd881939b6cd34e31efbadd",
                "output_index": 1,
                "sequence_number": 23,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0ea6c1cbfcc73010006ad04ad01dd881939b6cd34e31efbadd",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining("}',
                    "call_id": "call_fIwA6Y6RbSeLJr9xUjjM4dZG",
                    "name": "search_code",
                },
                "output_index": 1,
                "sequence_number": 24,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.added",
                "item": {
                    "id": "fc_0ea6c1cbfcc73010006ad04ad01df881939bd566e3a41578a8",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_7E8zG0YBqOXl7kVzO8QtrLMA",
                    "name": "run_tests",
                },
                "output_index": 2,
                "sequence_number": 25,
            },
            event="response.output_item.added",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0ea6c1cbfcc73010006ad04ad01df881939bd566e3a41578a8",
                "obfuscation": "wa",
                "output_index": 2,
                "sequence_number": 26,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0ea6c1cbfcc73010006ad04ad01df881939bd566e3a41578a8",
                "output_index": 2,
                "sequence_number": 27,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0ea6c1cbfcc73010006ad04ad01df881939bd566e3a41578a8",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_7E8zG0YBqOXl7kVzO8QtrLMA",
                    "name": "run_tests",
                },
                "output_index": 2,
                "sequence_number": 28,
            },
            event="response.output_item.done",
        ),
        SSEFrame(
            data={
                "type": "response.completed",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0ea6c1cbfcc73010006ad04acd01d481939812185df9b09745",
                    "created_at": 1792035533,
                    "status": "completed",
                    "completed_at": 1792035536,
                    "content_filters": [
                        {
                            "blocked": False,
                            "source_type": "prompt",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                                "jailbreak": {"detected": False, "filtered": False},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
                        },
                        {
                            "blocked": False,
                            "source_type": "completion",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "protected_material_code": {"detected": False, "filtered": False},
                                "protected_material_text": {"detected": False, "filtered": False},
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 258, "check_offset": 0},
                        },
                    ],
                    "model": "gpt-5.4-mini",
                    "output": [
                        {
                            "id": "msg_0ea6c1cbfcc73010006ad04acfeac081939b6cd3f18339bc69",
                            "type": "message",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "annotations": [],
                                    "logprobs": [],
                                    "text": "I’ll search for other callers and run the targeted test before editing.",
                                }
                            ],
                            "phase": "commentary",
                            "role": "assistant",
                        },
                        {
                            "id": "fc_0ea6c1cbfcc73010006ad04ad01dd881939b6cd34e31efbadd",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining("}',
                            "call_id": "call_fIwA6Y6RbSeLJr9xUjjM4dZG",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0ea6c1cbfcc73010006ad04ad01df881939bd566e3a41578a8",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_7E8zG0YBqOXl7kVzO8QtrLMA",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "in_memory",
                    "reasoning": {
                        **object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]),
                        "context": "current_turn",
                    },
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 75,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 523,
                    },
                },
                "sequence_number": 29,
            },
            event="response.completed",
        ),
    ),
)


GPT_5_6_LUNA_AGENT_TEST_CASE: Final = replace(
    GPT_5_6_LUNA_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure/gpt-5.6-luna"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-luna"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0d8b01a729695f11006ad04aca09c88195846f1bb1afe76a46",
        "created_at": 1792035530,
        "status": "completed",
        "completed_at": 1792035531,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_text": {"detected": False, "filtered": False},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            },
        ],
        "model": "gpt-5.6-luna",
        "output": [
            {
                "id": "fc_0d8b01a729695f11006ad04acae5a88195a985d9e92905e5c2",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_4MLZtLJ85gCWH7mi9P1twZB4",
                "name": "search_code",
            },
            {
                "id": "fc_0d8b01a729695f11006ad04acae5d08195a3985d7c3c690bea",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_RZ9vuE6JysYX2WnCJdW5uDET",
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
        "id": "resp_0d8b01a729695f11006ad04aca09c88195846f1bb1afe76a46",
        "model": "azure/gpt-5.6-luna",
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
                            "id": "call_4MLZtLJ85gCWH7mi9P1twZB4",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_RZ9vuE6JysYX2WnCJdW5uDET",
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
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_text": {"detected": False, "filtered": False},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            },
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-5.6-luna",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_4MLZtLJ85gCWH7mi9P1twZB4",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_RZ9vuE6JysYX2WnCJdW5uDET",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_4MLZtLJ85gCWH7mi9P1twZB4",
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
                "call_id": "call_RZ9vuE6JysYX2WnCJdW5uDET",
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
        "id": "resp_01e24fa82f710b30006ad04acd0e0c8197a4c00208165b3317",
        "created_at": 1792035533,
        "status": "completed",
        "completed_at": 1792035535,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 266, "check_offset": 0},
            }
        ],
        "model": "gpt-5.6-luna",
        "output": [
            {
                "id": "msg_01e24fa82f710b30006ad04acd93248197a159fe6aaff1057f",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly returns `limit + attempts` instead of the "
                        "nonnegative number of retries left, causing `retries_remaining(3, 1)` to return 4 "
                        "rather than 2. It is called by `src/worker.py` line 12, and `tests/test_retry.py` "
                        "failed with `assert 4 == 2`.",
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
            "output_tokens": 77,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 696,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_01e24fa82f710b30006ad04acd0e0c8197a4c00208165b3317",
        "model": "azure/gpt-5.6-luna",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly returns `limit + attempts` instead of the "
                    "nonnegative number of retries left, causing `retries_remaining(3, 1)` to return 4 "
                    "rather than 2. It is called by `src/worker.py` line 12, and `tests/test_retry.py` "
                    "failed with `assert 4 == 2`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 77,
            "prompt_tokens": 619,
            "total_tokens": 696,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 266, "check_offset": 0},
            }
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-luna", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0313e1970d397ba0006ad04acb68788196846bbf2ed8dbf632",
                    "created_at": 1792035531,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "resp_0313e1970d397ba0006ad04acb68788196846bbf2ed8dbf632",
                    "created_at": 1792035531,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "fc_0313e1970d397ba0006ad04acc90308196a9d2e07c220113e9",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_6iIYBzCfGUHOV79jPQDBiCUh",
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
                "item_id": "fc_0313e1970d397ba0006ad04acc90308196a9d2e07c220113e9",
                "obfuscation": "tDl",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0313e1970d397ba0006ad04acc90308196a9d2e07c220113e9",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0313e1970d397ba0006ad04acc90308196a9d2e07c220113e9",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_6iIYBzCfGUHOV79jPQDBiCUh",
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
                    "id": "fc_0313e1970d397ba0006ad04acc91148196b0db0164f3187270",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_lYnjsUXj9gGS81amv8I8TyFx",
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
                "item_id": "fc_0313e1970d397ba0006ad04acc91148196b0db0164f3187270",
                "obfuscation": "VD",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0313e1970d397ba0006ad04acc91148196b0db0164f3187270",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0313e1970d397ba0006ad04acc91148196b0db0164f3187270",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_lYnjsUXj9gGS81amv8I8TyFx",
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
                    "id": "resp_0313e1970d397ba0006ad04acb68788196846bbf2ed8dbf632",
                    "created_at": 1792035531,
                    "status": "completed",
                    "completed_at": 1792035532,
                    "content_filters": [
                        {
                            "blocked": False,
                            "source_type": "prompt",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                                "jailbreak": {"detected": False, "filtered": False},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
                        },
                        {
                            "blocked": False,
                            "source_type": "completion",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "protected_material_code": {"detected": False, "filtered": False},
                                "protected_material_text": {"detected": False, "filtered": False},
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
                        },
                    ],
                    "model": "gpt-5.6-luna",
                    "output": [
                        {
                            "id": "fc_0313e1970d397ba0006ad04acc90308196a9d2e07c220113e9",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_6iIYBzCfGUHOV79jPQDBiCUh",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0313e1970d397ba0006ad04acc91148196b0db0164f3187270",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_lYnjsUXj9gGS81amv8I8TyFx",
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
)


GPT_5_6_SOL_AGENT_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure/gpt-5.6-sol"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-sol"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_056246822b3d6bbe006ad04acf85708195985564a55f34ca39",
        "created_at": 1792035535,
        "status": "completed",
        "completed_at": 1792035537,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            }
        ],
        "model": "gpt-5.6-sol",
        "output": [
            {
                "id": "fc_056246822b3d6bbe006ad04ad0a7d88195b00b9c397503645d",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_Axfr9xZFZpF7vnPZpcKc0h7R",
                "name": "search_code",
            },
            {
                "id": "fc_056246822b3d6bbe006ad04ad0a7f08195825e685f14f00061",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_ETq10m3CDCDbOZnHquM329UY",
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
        "id": "resp_056246822b3d6bbe006ad04acf85708195985564a55f34ca39",
        "model": "azure/gpt-5.6-sol",
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
                            "id": "call_Axfr9xZFZpF7vnPZpcKc0h7R",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_ETq10m3CDCDbOZnHquM329UY",
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
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            }
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-5.6-sol",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_Axfr9xZFZpF7vnPZpcKc0h7R",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_ETq10m3CDCDbOZnHquM329UY",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_Axfr9xZFZpF7vnPZpcKc0h7R",
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
                "call_id": "call_ETq10m3CDCDbOZnHquM329UY",
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
        "id": "resp_0df3b5195ad659b1006ad04ad41fb88194a4a6dab5489198c8",
        "created_at": 1792035540,
        "status": "completed",
        "completed_at": 1792035542,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 244, "check_offset": 0},
            }
        ],
        "model": "gpt-5.6-sol",
        "output": [
            {
                "id": "msg_0df3b5195ad659b1006ad04ad50e90819485ac4767aeb3bb7a",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly adds `limit + attempts`; it should subtract attempts "
                        "and clamp the result at zero. The caller is `src/worker.py:12`, and "
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
            "output_tokens": 66,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 685,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0df3b5195ad659b1006ad04ad41fb88194a4a6dab5489198c8",
        "model": "azure/gpt-5.6-sol",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds `limit + attempts`; it should subtract "
                    "attempts and clamp the result at zero. The caller is `src/worker.py:12`, and "
                    "`tests/test_retry.py` failed because `retries_remaining(3, 1)` returned `4` "
                    "instead of `2`.",
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
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 244, "check_offset": 0},
            }
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-sol", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0cad649bd12ea37c006ad04ad2064c8190acc63dd66e6e2c6e",
                    "created_at": 1792035538,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "resp_0cad649bd12ea37c006ad04ad2064c8190acc63dd66e6e2c6e",
                    "created_at": 1792035538,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "fc_0cad649bd12ea37c006ad04ad34be481909667a3e3c797a56f",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_IAvLgDLvbWIqJpT6QDPScqZ5",
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
                "item_id": "fc_0cad649bd12ea37c006ad04ad34be481909667a3e3c797a56f",
                "obfuscation": "nBn",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0cad649bd12ea37c006ad04ad34be481909667a3e3c797a56f",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0cad649bd12ea37c006ad04ad34be481909667a3e3c797a56f",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_IAvLgDLvbWIqJpT6QDPScqZ5",
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
                    "id": "fc_0cad649bd12ea37c006ad04ad34c08819099f00ec1103f279f",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_LZ5lIeqQwip4VqN3Nb456AvB",
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
                "item_id": "fc_0cad649bd12ea37c006ad04ad34c08819099f00ec1103f279f",
                "obfuscation": "h7",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0cad649bd12ea37c006ad04ad34c08819099f00ec1103f279f",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0cad649bd12ea37c006ad04ad34c08819099f00ec1103f279f",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_LZ5lIeqQwip4VqN3Nb456AvB",
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
                    "id": "resp_0cad649bd12ea37c006ad04ad2064c8190acc63dd66e6e2c6e",
                    "created_at": 1792035538,
                    "status": "completed",
                    "completed_at": 1792035539,
                    "content_filters": [
                        {
                            "blocked": False,
                            "source_type": "prompt",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                                "jailbreak": {"detected": False, "filtered": False},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
                        }
                    ],
                    "model": "gpt-5.6-sol",
                    "output": [
                        {
                            "id": "fc_0cad649bd12ea37c006ad04ad34be481909667a3e3c797a56f",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_IAvLgDLvbWIqJpT6QDPScqZ5",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0cad649bd12ea37c006ad04ad34c08819099f00ec1103f279f",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_LZ5lIeqQwip4VqN3Nb456AvB",
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
)


GPT_5_6_TERRA_AGENT_TEST_CASE: Final = replace(
    GPT_5_6_TERRA_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure/gpt-5.6-terra"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-terra"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_021cbaff805a5105006ad04ad314608196900f4e35d47b7c80",
        "created_at": 1792035539,
        "status": "completed",
        "completed_at": 1792035540,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_text": {"detected": False, "filtered": False},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            },
        ],
        "model": "gpt-5.6-terra",
        "output": [
            {
                "id": "fc_021cbaff805a5105006ad04ad4692081968b2da0357cef3c87",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_AQnAECWqPckZwylX3qZBzlY5",
                "name": "search_code",
            },
            {
                "id": "fc_021cbaff805a5105006ad04ad469508196b6eaec87d50240ea",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_msUupLfFYClJSMSOaK3Oq57h",
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
        "id": "resp_021cbaff805a5105006ad04ad314608196900f4e35d47b7c80",
        "model": "azure/gpt-5.6-terra",
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
                            "id": "call_AQnAECWqPckZwylX3qZBzlY5",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_msUupLfFYClJSMSOaK3Oq57h",
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
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_text": {"detected": False, "filtered": False},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            },
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-5.6-terra",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_AQnAECWqPckZwylX3qZBzlY5",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_msUupLfFYClJSMSOaK3Oq57h",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_AQnAECWqPckZwylX3qZBzlY5",
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
                "call_id": "call_msUupLfFYClJSMSOaK3Oq57h",
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
        "id": "resp_0642d30b87023209006ad04ad7a8648195bc09816464eaf68e",
        "created_at": 1792035544,
        "status": "completed",
        "completed_at": 1792035546,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1440, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 265, "check_offset": 0},
            },
        ],
        "model": "gpt-5.6-terra",
        "output": [
            {
                "id": "msg_0642d30b87023209006ad04ad9e2c881958fcd9b5846e7e116",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly adds `limit` and `attempts`; it should subtract "
                        "attempts from the limit and clamp the result at zero. The only caller found is "
                        "`src/worker.py`, and `tests/test_retry.py` fails because `retries_remaining(3, 1)` "
                        "returns 4 instead of 2.",
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
            "output_tokens": 71,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 690,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0642d30b87023209006ad04ad7a8648195bc09816464eaf68e",
        "model": "azure/gpt-5.6-terra",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds `limit` and `attempts`; it should subtract "
                    "attempts from the limit and clamp the result at zero. The only caller found is "
                    "`src/worker.py`, and `tests/test_retry.py` fails because `retries_remaining(3, 1)` "
                    "returns 4 instead of 2.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 71,
            "prompt_tokens": 619,
            "total_tokens": 690,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1440, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 265, "check_offset": 0},
            },
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.6-terra", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0514fff817a3c850006ad04ad4e2608193b487419ab254ce3b",
                    "created_at": 1792035540,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "resp_0514fff817a3c850006ad04ad4e2608193b487419ab254ce3b",
                    "created_at": 1792035540,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "rs_0514fff817a3c850006ad04ad600108193a7f0973d7f0edb02",
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
                    "id": "rs_0514fff817a3c850006ad04ad600108193a7f0973d7f0edb02",
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
                    "id": "fc_0514fff817a3c850006ad04ad657a08193a133625fd099c618",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_yLjqnSPdFaGv8U8E75rQWQLC",
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
                "item_id": "fc_0514fff817a3c850006ad04ad657a08193a133625fd099c618",
                "obfuscation": "2jp",
                "output_index": 1,
                "sequence_number": 5,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0514fff817a3c850006ad04ad657a08193a133625fd099c618",
                "output_index": 1,
                "sequence_number": 6,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0514fff817a3c850006ad04ad657a08193a133625fd099c618",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_yLjqnSPdFaGv8U8E75rQWQLC",
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
                    "id": "fc_0514fff817a3c850006ad04ad657fc8193899c6082c0c899fc",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_eSkcn73McIf21NZhszW14NXh",
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
                "item_id": "fc_0514fff817a3c850006ad04ad657fc8193899c6082c0c899fc",
                "obfuscation": "9V",
                "output_index": 2,
                "sequence_number": 9,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0514fff817a3c850006ad04ad657fc8193899c6082c0c899fc",
                "output_index": 2,
                "sequence_number": 10,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0514fff817a3c850006ad04ad657fc8193899c6082c0c899fc",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_eSkcn73McIf21NZhszW14NXh",
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
                    "id": "resp_0514fff817a3c850006ad04ad4e2608193b487419ab254ce3b",
                    "created_at": 1792035540,
                    "status": "completed",
                    "completed_at": 1792035543,
                    "content_filters": [
                        {
                            "blocked": False,
                            "source_type": "completion",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "protected_material_text": {"detected": False, "filtered": False},
                                "protected_material_code": {"detected": False, "filtered": False},
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 201, "check_offset": 0},
                        }
                    ],
                    "model": "gpt-5.6-terra",
                    "output": [
                        {
                            "id": "rs_0514fff817a3c850006ad04ad600108193a7f0973d7f0edb02",
                            "type": "reasoning",
                            "content": [],
                            "encrypted_content": "synthetic-signature",
                            "summary": [],
                        },
                        {
                            "id": "fc_0514fff817a3c850006ad04ad657a08193a133625fd099c618",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_yLjqnSPdFaGv8U8E75rQWQLC",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0514fff817a3c850006ad04ad657fc8193899c6082c0c899fc",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_eSkcn73McIf21NZhszW14NXh",
                            "name": "run_tests",
                        },
                    ],
                    "prompt_cache_retention": "24h",
                    "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "all_turns"},
                    "service_tier": "default",
                    "usage": {
                        "input_tokens": 448,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 63,
                        "output_tokens_details": {"reasoning_tokens": 8},
                        "total_tokens": 511,
                    },
                },
                "sequence_number": 12,
            },
            event="response.completed",
        ),
    ),
)


GPT_6_ASTRA_AGENT_TEST_CASE: Final = replace(
    GPT_6_ASTRA_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure/gpt-6-astra"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-astra"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0e1aa1a45b752917006ad04adab8a08193bf9a431cf996fc8c",
        "created_at": 1792035546,
        "status": "completed",
        "completed_at": 1792035548,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            },
        ],
        "model": "gpt-6-astra",
        "output": [
            {
                "id": "fc_0e1aa1a45b752917006ad04adc75588193998f33b19d72ab67",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_gqwDRArJmmOo4biwCSHfRKC3",
                "name": "search_code",
            },
            {
                "id": "fc_0e1aa1a45b752917006ad04adc75748193ac66d67791c9f4c3",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_ATuKGX5x8rBXnJcB1sWNJGGW",
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
        "id": "resp_0e1aa1a45b752917006ad04adab8a08193bf9a431cf996fc8c",
        "model": "azure/gpt-6-astra",
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
                            "id": "call_gqwDRArJmmOo4biwCSHfRKC3",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_ATuKGX5x8rBXnJcB1sWNJGGW",
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
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            },
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-6-astra",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_gqwDRArJmmOo4biwCSHfRKC3",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_ATuKGX5x8rBXnJcB1sWNJGGW",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_gqwDRArJmmOo4biwCSHfRKC3",
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
                "call_id": "call_ATuKGX5x8rBXnJcB1sWNJGGW",
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
        "id": "resp_06e17202c665e7e6006ad04adf41d48196bae319c820b8d7c9",
        "created_at": 1792035551,
        "status": "completed",
        "completed_at": 1792035556,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1440, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 207, "check_offset": 0},
            },
        ],
        "model": "gpt-6-astra",
        "output": [
            {
                "id": "msg_06e17202c665e7e6006ad04ae231188196b719641af24c136d",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` adds attempts to the limit instead of subtracting and clamping "
                        "at zero; a caller was found at `src/worker.py:12`. `tests/test_retry.py` failed with "
                        "`assert 4 == 2`; no files were changed.",
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
            "output_tokens": 58,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 677,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_06e17202c665e7e6006ad04adf41d48196bae319c820b8d7c9",
        "model": "azure/gpt-6-astra",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds attempts to the limit instead of subtracting and clamping "
                    "at zero; a caller was found at `src/worker.py:12`. `tests/test_retry.py` failed "
                    "with `assert 4 == 2`; no files were changed.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 58,
            "prompt_tokens": 619,
            "total_tokens": 677,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1440, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 207, "check_offset": 0},
            },
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-astra", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0a0b09b257351892006ad04adcf4d08190b69ea01d0aaddae7",
                    "created_at": 1792035549,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "resp_0a0b09b257351892006ad04adcf4d08190b69ea01d0aaddae7",
                    "created_at": 1792035549,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "fc_0a0b09b257351892006ad04ade8e5c8190800f23a7f173f2aa",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_KOcCFTjmrhmIPy65glI8Xu9L",
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
                "item_id": "fc_0a0b09b257351892006ad04ade8e5c8190800f23a7f173f2aa",
                "obfuscation": "HA4",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0a0b09b257351892006ad04ade8e5c8190800f23a7f173f2aa",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0a0b09b257351892006ad04ade8e5c8190800f23a7f173f2aa",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_KOcCFTjmrhmIPy65glI8Xu9L",
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
                    "id": "fc_0a0b09b257351892006ad04ade8e7881908e1b26fcb5c22b40",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_CykEVxHeCcGE0RKZ1ZMjUUrL",
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
                "item_id": "fc_0a0b09b257351892006ad04ade8e7881908e1b26fcb5c22b40",
                "obfuscation": "Z9",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0a0b09b257351892006ad04ade8e7881908e1b26fcb5c22b40",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0a0b09b257351892006ad04ade8e7881908e1b26fcb5c22b40",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_CykEVxHeCcGE0RKZ1ZMjUUrL",
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
                    "id": "resp_0a0b09b257351892006ad04adcf4d08190b69ea01d0aaddae7",
                    "created_at": 1792035549,
                    "status": "completed",
                    "completed_at": 1792035550,
                    "content_filters": [
                        {
                            "blocked": False,
                            "source_type": "prompt",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                                "jailbreak": {"detected": False, "filtered": False},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
                        },
                        {
                            "blocked": False,
                            "source_type": "completion",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "protected_material_code": {"detected": False, "filtered": False},
                                "protected_material_text": {"detected": False, "filtered": False},
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
                        },
                    ],
                    "model": "gpt-6-astra",
                    "output": [
                        {
                            "id": "fc_0a0b09b257351892006ad04ade8e5c8190800f23a7f173f2aa",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_KOcCFTjmrhmIPy65glI8Xu9L",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0a0b09b257351892006ad04ade8e7881908e1b26fcb5c22b40",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_CykEVxHeCcGE0RKZ1ZMjUUrL",
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
)


GPT_6_LUNA_AGENT_TEST_CASE: Final = replace(
    GPT_6_LUNA_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure/gpt-6-luna"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-luna"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0c24cb3c910732b2006ad04aca9fbc8194a44eee0786d73cf2",
        "created_at": 1792035530,
        "status": "completed",
        "completed_at": 1792035532,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            }
        ],
        "model": "gpt-6-luna",
        "output": [
            {
                "id": "fc_0c24cb3c910732b2006ad04acbbe448194a853526f61cffa91",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_eXvJbbJr4vYTWZVzcZEWWfes",
                "name": "search_code",
            },
            {
                "id": "fc_0c24cb3c910732b2006ad04acbbe5c8194bd29577a3a6e5a19",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_JVDRMpVXHX7CaSEnKsvodfeO",
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
        "id": "resp_0c24cb3c910732b2006ad04aca9fbc8194a44eee0786d73cf2",
        "model": "azure/gpt-6-luna",
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
                            "id": "call_eXvJbbJr4vYTWZVzcZEWWfes",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_JVDRMpVXHX7CaSEnKsvodfeO",
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
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            }
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-6-luna",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_eXvJbbJr4vYTWZVzcZEWWfes",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_JVDRMpVXHX7CaSEnKsvodfeO",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_eXvJbbJr4vYTWZVzcZEWWfes",
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
                "call_id": "call_JVDRMpVXHX7CaSEnKsvodfeO",
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
        "id": "resp_0f3e2451550b29f5006ad04acf531c8196a57251326e10b7f2",
        "created_at": 1792035535,
        "status": "completed",
        "completed_at": 1792035538,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1440, "check_offset": 0},
            }
        ],
        "model": "gpt-6-luna",
        "output": [
            {
                "id": "msg_0f3e2451550b29f5006ad04ad1519881969543782dd26fc553",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` adds `limit` and `attempts` instead of returning retries left, "
                        "so it can return negative-relevant incorrect values. The caller is `src/worker.py`; "
                        "`tests/test_retry.py` failed because it got `4` instead of `2`.",
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
            "output_tokens": 61,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 680,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0f3e2451550b29f5006ad04acf531c8196a57251326e10b7f2",
        "model": "azure/gpt-6-luna",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds `limit` and `attempts` instead of returning retries left, "
                    "so it can return negative-relevant incorrect values. The caller is "
                    "`src/worker.py`; `tests/test_retry.py` failed because it got `4` instead of `2`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 61,
            "prompt_tokens": 619,
            "total_tokens": 680,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1440, "check_offset": 0},
            }
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6-luna", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0c2198d02c07d92c006ad04acd13948193b18a9566a5c2da02",
                    "created_at": 1792035533,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "resp_0c2198d02c07d92c006ad04acd13948193b18a9566a5c2da02",
                    "created_at": 1792035533,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "fc_0c2198d02c07d92c006ad04aceeb54819397ae8fdbd9fca4e6",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_eMkw1KRyoMPyzl7zjNvCLaIg",
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
                "item_id": "fc_0c2198d02c07d92c006ad04aceeb54819397ae8fdbd9fca4e6",
                "obfuscation": "faN",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0c2198d02c07d92c006ad04aceeb54819397ae8fdbd9fca4e6",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0c2198d02c07d92c006ad04aceeb54819397ae8fdbd9fca4e6",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_eMkw1KRyoMPyzl7zjNvCLaIg",
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
                    "id": "fc_0c2198d02c07d92c006ad04aceeb7081938a08631af84d9fef",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_JL9islb4uAm6WuVzOzHtyx6D",
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
                "item_id": "fc_0c2198d02c07d92c006ad04aceeb7081938a08631af84d9fef",
                "obfuscation": "Kb",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0c2198d02c07d92c006ad04aceeb7081938a08631af84d9fef",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0c2198d02c07d92c006ad04aceeb7081938a08631af84d9fef",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_JL9islb4uAm6WuVzOzHtyx6D",
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
                    "id": "resp_0c2198d02c07d92c006ad04acd13948193b18a9566a5c2da02",
                    "created_at": 1792035533,
                    "status": "completed",
                    "completed_at": 1792035535,
                    "content_filters": [
                        {
                            "blocked": False,
                            "source_type": "prompt",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                                "jailbreak": {"detected": False, "filtered": False},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
                        },
                        {
                            "blocked": False,
                            "source_type": "completion",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "protected_material_code": {"detected": False, "filtered": False},
                                "protected_material_text": {"detected": False, "filtered": False},
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
                        },
                    ],
                    "model": "gpt-6-luna",
                    "output": [
                        {
                            "id": "fc_0c2198d02c07d92c006ad04aceeb54819397ae8fdbd9fca4e6",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_eMkw1KRyoMPyzl7zjNvCLaIg",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0c2198d02c07d92c006ad04aceeb7081938a08631af84d9fef",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_JL9islb4uAm6WuVzOzHtyx6D",
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
)


GPT_6_1_SOL_AGENT_TEST_CASE: Final = replace(
    GPT_6_1_SOL_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure/gpt-6.1-sol"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6.1-sol"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_01a631a5f160dbc9006ad04ad218dc8194974d60c91352ff3d",
        "created_at": 1792035538,
        "status": "completed",
        "completed_at": 1792035540,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            },
        ],
        "model": "gpt-6.1-sol",
        "output": [
            {
                "id": "fc_01a631a5f160dbc9006ad04ad423688194a5ac6f3f3b1d2cda",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining"}',
                "call_id": "call_dqmp0EOXveDO6b806CZWmBpr",
                "name": "search_code",
            },
            {
                "id": "fc_01a631a5f160dbc9006ad04ad4238081948de286ed3888178c",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_FFRmRg2YAqlhfkzgSDVzQq6u",
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
        "id": "resp_01a631a5f160dbc9006ad04ad218dc8194974d60c91352ff3d",
        "model": "azure/gpt-6.1-sol",
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
                            "id": "call_dqmp0EOXveDO6b806CZWmBpr",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_FFRmRg2YAqlhfkzgSDVzQq6u",
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
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
            },
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={
        **OPENAI_RESPONSES_AGENT_REQUEST,
        "model": "gpt-6.1-sol",
        "input": [
            *list_value(OPENAI_RESPONSES_AGENT_REQUEST["input"]),
            {
                "type": "function_call",
                "call_id": "call_dqmp0EOXveDO6b806CZWmBpr",
                "name": "search_code",
                "arguments": '{"query":"retries_remaining"}',
            },
            {
                "type": "function_call",
                "call_id": "call_FFRmRg2YAqlhfkzgSDVzQq6u",
                "name": "run_tests",
                "arguments": '{"path":"tests/test_retry.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_dqmp0EOXveDO6b806CZWmBpr",
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
                "call_id": "call_FFRmRg2YAqlhfkzgSDVzQq6u",
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
        "id": "resp_04571a38ec90e20b006ad04ad7d5808195955c3e687ba442b1",
        "created_at": 1792035543,
        "status": "completed",
        "completed_at": 1792035546,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1440, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_text": {"detected": False, "filtered": False},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 256, "check_offset": 0},
            },
        ],
        "model": "gpt-6.1-sol",
        "output": [
            {
                "id": "msg_04571a38ec90e20b006ad04ad90c7c81959445ad517dcae3d0",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "`retries_remaining` incorrectly adds attempts to the limit instead of subtracting "
                        "and clamping at zero; a caller was found in `src/worker.py:12`. "
                        "`tests/test_retry.py` failed because `retries_remaining(3, 1)` returned 4 instead of "
                        "2; no files were changed.",
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
            "output_tokens": 69,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 688,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_04571a38ec90e20b006ad04ad7d5808195955c3e687ba442b1",
        "model": "azure/gpt-6.1-sol",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds attempts to the limit instead of subtracting "
                    "and clamping at zero; a caller was found in `src/worker.py:12`. "
                    "`tests/test_retry.py` failed because `retries_remaining(3, 1)` returned 4 instead "
                    "of 2; no files were changed.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 69,
            "prompt_tokens": 619,
            "total_tokens": 688,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 1440, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_text": {"detected": False, "filtered": False},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 256, "check_offset": 0},
            },
        ],
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
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-6.1-sol", "stream": True},
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "response.created",
                "response": {
                    **OPENAI_RESPONSES_AGENT_RESPONSE,
                    "id": "resp_0d2627310f3a7aa6006ad04ad554d48190be01b0c171757b4f",
                    "created_at": 1792035541,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "resp_0d2627310f3a7aa6006ad04ad554d48190be01b0c171757b4f",
                    "created_at": 1792035541,
                    "status": "in_progress",
                    "completed_at": None,
                    "content_filters": None,
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
                    "id": "fc_0d2627310f3a7aa6006ad04ad7806c8190bb0bcc797f55cb25",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_5EftqTN9zDBp87YG60xu9vKB",
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
                "item_id": "fc_0d2627310f3a7aa6006ad04ad7806c8190bb0bcc797f55cb25",
                "obfuscation": "JvY",
                "output_index": 0,
                "sequence_number": 3,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"query":"retries_remaining"}',
                "item_id": "fc_0d2627310f3a7aa6006ad04ad7806c8190bb0bcc797f55cb25",
                "output_index": 0,
                "sequence_number": 4,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0d2627310f3a7aa6006ad04ad7806c8190bb0bcc797f55cb25",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"query":"retries_remaining"}',
                    "call_id": "call_5EftqTN9zDBp87YG60xu9vKB",
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
                    "id": "fc_0d2627310f3a7aa6006ad04ad780888190ae854fb4c17b7d5c",
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": "call_cmFZIGDGEFRBReyEVlw7hifi",
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
                "item_id": "fc_0d2627310f3a7aa6006ad04ad780888190ae854fb4c17b7d5c",
                "obfuscation": "lL",
                "output_index": 1,
                "sequence_number": 7,
            },
            event="response.function_call_arguments.delta",
        ),
        SSEFrame(
            data={
                "type": "response.function_call_arguments.done",
                "arguments": '{"path":"tests/test_retry.py"}',
                "item_id": "fc_0d2627310f3a7aa6006ad04ad780888190ae854fb4c17b7d5c",
                "output_index": 1,
                "sequence_number": 8,
            },
            event="response.function_call_arguments.done",
        ),
        SSEFrame(
            data={
                "type": "response.output_item.done",
                "item": {
                    "id": "fc_0d2627310f3a7aa6006ad04ad780888190ae854fb4c17b7d5c",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"path":"tests/test_retry.py"}',
                    "call_id": "call_cmFZIGDGEFRBReyEVlw7hifi",
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
                    "id": "resp_0d2627310f3a7aa6006ad04ad554d48190be01b0c171757b4f",
                    "created_at": 1792035541,
                    "status": "completed",
                    "completed_at": 1792035543,
                    "content_filters": [
                        {
                            "blocked": False,
                            "source_type": "prompt",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                                "jailbreak": {"detected": False, "filtered": False},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
                        },
                        {
                            "blocked": False,
                            "source_type": "completion",
                            "content_filter_raw": [],
                            "content_filter_results": {
                                "protected_material_text": {"detected": False, "filtered": False},
                                "protected_material_code": {"detected": False, "filtered": False},
                                "hate": {"filtered": False, "severity": "safe"},
                                "sexual": {"filtered": False, "severity": "safe"},
                                "violence": {"filtered": False, "severity": "safe"},
                                "self_harm": {"filtered": False, "severity": "safe"},
                            },
                            "content_filter_offsets": {"start_offset": 0, "end_offset": 186, "check_offset": 0},
                        },
                    ],
                    "model": "gpt-6.1-sol",
                    "output": [
                        {
                            "id": "fc_0d2627310f3a7aa6006ad04ad7806c8190bb0bcc797f55cb25",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"query":"retries_remaining"}',
                            "call_id": "call_5EftqTN9zDBp87YG60xu9vKB",
                            "name": "search_code",
                        },
                        {
                            "id": "fc_0d2627310f3a7aa6006ad04ad780888190ae854fb4c17b7d5c",
                            "type": "function_call",
                            "status": "completed",
                            "arguments": '{"path":"tests/test_retry.py"}',
                            "call_id": "call_cmFZIGDGEFRBReyEVlw7hifi",
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
)
