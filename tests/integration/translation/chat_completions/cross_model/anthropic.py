from dataclasses import replace
from typing import Final

from integration._support.client import list_value, object_value
from integration.translation.case import (
    SSEFrame,
    StreamingTranslationTestCase,
)
from integration.translation.chat_completions.cross_model.contracts import followup_request
from integration.translation.chat_completions.bases.anthropic import (
    CLAUDE_FABLE_5_1_TEST_CASE,
    CLAUDE_FABLE_5_TEST_CASE,
    CLAUDE_HAIKU_4_5_TEST_CASE,
    CLAUDE_HAIKU_5_5_TEST_CASE,
    CLAUDE_OPUS_5_5_TEST_CASE,
    CLAUDE_OPUS_5_TEST_CASE,
)
from integration.translation.chat_completions.templates import (
    CODING_AGENT_REQUEST,
    coding_agent_stream_message,
    CODING_AGENT_RESPONSE,
)
from integration.translation.templates.anthropic import (
    ANTHROPIC_AGENT_REQUEST,
    ANTHROPIC_AGENT_RESPONSE,
)

CLAUDE_FABLE_5_1_AGENT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "anthropic/claude-fable-5-1"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-fable-5-1",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_011Cg3KsJgDqYavGG6sHCiWE",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01Ceyd8UuHDATjxTVpqhDNXG",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01HpLFUAkerRSH1ENkpiqHfz",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
                "caller": {"type": "direct"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1184,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-fable-5-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_01Ceyd8UuHDATjxTVpqhDNXG",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_01HpLFUAkerRSH1ENkpiqHfz",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1184,
            "total_tokens": 1289,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1184,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


CLAUDE_FABLE_5_1_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_FABLE_5_1_AGENT_TEST_CASE.litellm_request, CLAUDE_FABLE_5_1_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-fable-5-1",
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01Ceyd8UuHDATjxTVpqhDNXG",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01HpLFUAkerRSH1ENkpiqHfz",
                        "name": "run_tests",
                        "input": {"path": "tests/test_retry.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01Ceyd8UuHDATjxTVpqhDNXG",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01HpLFUAkerRSH1ENkpiqHfz",
                        "content": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    },
                    {
                        "type": "text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    },
                ],
            },
        ],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_011Cg3KsWsCSjykB1RWPyPir",
        "content": [
            {
                "type": "text",
                "text": "`retries_remaining` adds `limit + attempts` instead of subtracting and clamping to zero, so "
                "`retries_remaining(3, 1)` returns 4 and the test fails with `assert 4 == 2`. The only caller is "
                "`src/worker.py` line 12 (`retries_remaining(3, attempts)`), which would also receive wrong "
                "values; the fix is `max(limit - attempts, 0)`, but no changes have been made.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1492,
            "output_tokens": 143,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-fable-5-1",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds `limit + attempts` instead of subtracting and clamping to "
                    "zero, so `retries_remaining(3, 1)` returns 4 and the test fails with `assert 4 == "
                    "2`. The only caller is `src/worker.py` line 12 (`retries_remaining(3, attempts)`), "
                    "which would also receive wrong values; the fix is `max(limit - attempts, 0)`, but "
                    "no changes have been made.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 143,
            "prompt_tokens": 1492,
            "total_tokens": 1635,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 143},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1492,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


CLAUDE_FABLE_5_1_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=CLAUDE_FABLE_5_1_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **CLAUDE_FABLE_5_1_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=CLAUDE_FABLE_5_1_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-fable-5-1",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-fable-5-1",
                    "id": "msg_011Cg3KsQdeGAsFKPGVfT7PQ",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 16,
                        "service_tier": "standard",
                        "inference_geo": "global",
                    },
                    "diagnostics": None,
                },
            },
            event="message_start",
        ),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01324ezmdSmGSpmxGX55GvYF",
                    "name": "search_code",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"query', '": "ret', "ries_re", "maining", '"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 0}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01MxHYehH7UYVY2A8WkRJZPu",
                    "name": "run_tests",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"path"', ': "test', "s/test_", "retry.p", 'y"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 1}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None, "stop_details": None, "container": None},
                "usage": {
                    "input_tokens": 1184,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 105,
                    "output_tokens_details": {"thinking_tokens": 0},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("toolu_01324ezmdSmGSpmxGX55GvYF", "toolu_01MxHYehH7UYVY2A8WkRJZPu"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1184,
        "completion_tokens": 105,
        "total_tokens": 1289,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


CLAUDE_FABLE_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_FABLE_5_TEST_CASE.litellm_request, CLAUDE_FABLE_5_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-fable-5",
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_017kyzUjeWYmUvPe1JuA3sP9",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01LJUyCenUD37DbG5waJEG3r",
                        "name": "run_tests",
                        "input": {"path": "tests/test_retry.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_017kyzUjeWYmUvPe1JuA3sP9",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01LJUyCenUD37DbG5waJEG3r",
                        "content": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    },
                    {
                        "type": "text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    },
                ],
            },
        ],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-fable-5",
        "id": "msg_011Cg3KrtY9hrfGJYFycP3xp",
        "content": [
            {
                "type": "text",
                "text": "The bug: `retries_remaining` adds `limit + attempts` instead of subtracting and clamping, so "
                "`retries_remaining(3, 1)` returns 4 instead of 2, failing "
                "`tests/test_retry.py::test_retries_remaining` (exit code 1). The only caller is `src/worker.py` "
                "line 12, which uses `retries_remaining(3, attempts)`; the fix should be `max(limit - attempts, "
                "0)`.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1490,
            "output_tokens": 147,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-fable-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug: `retries_remaining` adds `limit + attempts` instead of subtracting and "
                    "clamping, so `retries_remaining(3, 1)` returns 4 instead of 2, failing "
                    "`tests/test_retry.py::test_retries_remaining` (exit code 1). The only caller is "
                    "`src/worker.py` line 12, which uses `retries_remaining(3, attempts)`; the fix "
                    "should be `max(limit - attempts, 0)`.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 147,
            "prompt_tokens": 1490,
            "total_tokens": 1637,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 147},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1490,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


CLAUDE_FABLE_5_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=CLAUDE_FABLE_5_TEST_CASE.litellm_endpoint,
    litellm_request={
        **CLAUDE_FABLE_5_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=CLAUDE_FABLE_5_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-fable-5",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-fable-5",
                    "id": "msg_011Cg3Krm6BajhLzpyzF5p7E",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 17,
                        "service_tier": "standard",
                        "inference_geo": "global",
                    },
                    "diagnostics": None,
                },
            },
            event="message_start",
        ),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01UXBJZGdCaY4vxon1j4uapy",
                    "name": "search_code",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"query', '": "ret', "ries_re", "maining", '"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 0}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_0124RYbAPnwVoGtdzfPCNPc6",
                    "name": "run_tests",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"path"', ': "test', "s/test_", "retry.p", 'y"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 1}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None, "stop_details": None, "container": None},
                "usage": {
                    "input_tokens": 1184,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 105,
                    "output_tokens_details": {"thinking_tokens": 0},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("toolu_01UXBJZGdCaY4vxon1j4uapy", "toolu_0124RYbAPnwVoGtdzfPCNPc6"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1184,
        "completion_tokens": 105,
        "total_tokens": 1289,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


CLAUDE_HAIKU_4_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_4_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "anthropic/claude-haiku-4-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-haiku-4-5",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_011Cg3KoVnbE5kuSs7bjQdRq",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants me to:\n"
                "1. Search for other callers of retries_remaining\n"
                "2. Run the tests in tests/test_retry.py to see the current failures\n"
                "\n"
                "These are independent operations, so I can make both calls together.",
                "signature": "synthetic-signature",
            },
            {
                "type": "tool_use",
                "id": "toolu_01Af9eZ9UqMBcsWTtdnCdbTY",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01NFkCH8SPp5bfkDQ5eXNgeY",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
                "caller": {"type": "direct"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1237,
            "output_tokens": 158,
            "output_tokens_details": {"thinking_tokens": 57},
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-haiku-4-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_01Af9eZ9UqMBcsWTtdnCdbTY",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_01NFkCH8SPp5bfkDQ5eXNgeY",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "The user wants me to:\n"
                    "1. Search for other callers of retries_remaining\n"
                    "2. Run the tests in tests/test_retry.py to see the current failures\n"
                    "\n"
                    "These are independent operations, so I can make both calls together.",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user wants me to:\n"
                            "1. Search for other callers of retries_remaining\n"
                            "2. Run the tests in tests/test_retry.py to see the current "
                            "failures\n"
                            "\n"
                            "These are independent operations, so I can make both calls "
                            "together.",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "The user wants me to:\n"
                                "1. Search for other callers of "
                                "retries_remaining\n"
                                "2. Run the tests in "
                                "tests/test_retry.py to see the "
                                "current failures\n"
                                "\n"
                                "These are independent "
                                "operations, so I can make both "
                                "calls together.",
                                "signature": "synthetic-signature",
                            }
                        ],
                    },
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 158,
            "prompt_tokens": 1237,
            "total_tokens": 1395,
            "completion_tokens_details": {"reasoning_tokens": 57, "text_tokens": 101},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1237,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "not_available",
            "service_tier": "standard",
        },
    },
)


CLAUDE_HAIKU_4_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_4_5_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_HAIKU_4_5_AGENT_TEST_CASE.litellm_request, CLAUDE_HAIKU_4_5_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-haiku-4-5",
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The user wants me to:\n"
                        "1. Search for other callers of retries_remaining\n"
                        "2. Run the tests in tests/test_retry.py to see the current failures\n"
                        "\n"
                        "These are independent operations, so I can make both calls together.",
                        "signature": "synthetic-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01Af9eZ9UqMBcsWTtdnCdbTY",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01NFkCH8SPp5bfkDQ5eXNgeY",
                        "name": "run_tests",
                        "input": {"path": "tests/test_retry.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01Af9eZ9UqMBcsWTtdnCdbTY",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01NFkCH8SPp5bfkDQ5eXNgeY",
                        "content": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    },
                    {
                        "type": "text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    },
                ],
            },
        ],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_011Cg3Kojd3LkheWFkkvejbS",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants me to summarize the findings without making any changes or running more "
                "tools.\n"
                "\n"
                "Let me analyze what I found:\n"
                "\n"
                "1. Bug: The current implementation `return limit + attempts` is wrong. It should calculate "
                "retries remaining as `limit - attempts`, and never return a negative number.\n"
                "\n"
                "2. Test failure: The test expects `retries_remaining(3, 1) == 2` (3 retries limit, 1 attempt "
                "made = 2 remaining), but the current implementation returns `3 + 1 = 4`.\n"
                "\n"
                "3. Caller: Found in `src/worker.py` at line 12: `remaining = retries_remaining(3, attempts)` "
                "- this is used to track remaining retries.\n"
                "\n"
                "Let me summarize in two short sentences.",
                "signature": "synthetic-signature",
            },
            {
                "type": "text",
                "text": "The bug is that `retries_remaining` uses `limit + attempts` instead of `limit - attempts`, so it "
                "returns an incorrect value (4 instead of 2 for the test case). The function is called in "
                "`src/worker.py:12` and the test fails with `assert 4 == 2`, confirming the logic needs to "
                "subtract attempts from the limit and clamp to zero.",
            },
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1511,
            "output_tokens": 271,
            "output_tokens_details": {"thinking_tokens": 178},
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-haiku-4-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` uses `limit + attempts` instead of `limit - "
                    "attempts`, so it returns an incorrect value (4 instead of 2 for the test case). "
                    "The function is called in `src/worker.py:12` and the test fails with `assert 4 == "
                    "2`, confirming the logic needs to subtract attempts from the limit and clamp to "
                    "zero.",
                    "role": "assistant",
                    "reasoning_content": "The user wants me to summarize the findings without making any changes "
                    "or running more tools.\n"
                    "\n"
                    "Let me analyze what I found:\n"
                    "\n"
                    "1. Bug: The current implementation `return limit + attempts` is wrong. "
                    "It should calculate retries remaining as `limit - attempts`, and never "
                    "return a negative number.\n"
                    "\n"
                    "2. Test failure: The test expects `retries_remaining(3, 1) == 2` (3 "
                    "retries limit, 1 attempt made = 2 remaining), but the current "
                    "implementation returns `3 + 1 = 4`.\n"
                    "\n"
                    "3. Caller: Found in `src/worker.py` at line 12: `remaining = "
                    "retries_remaining(3, attempts)` - this is used to track remaining "
                    "retries.\n"
                    "\n"
                    "Let me summarize in two short sentences.",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user wants me to summarize the findings without making "
                            "any changes or running more tools.\n"
                            "\n"
                            "Let me analyze what I found:\n"
                            "\n"
                            "1. Bug: The current implementation `return limit + attempts` "
                            "is wrong. It should calculate retries remaining as `limit - "
                            "attempts`, and never return a negative number.\n"
                            "\n"
                            "2. Test failure: The test expects `retries_remaining(3, 1) "
                            "== 2` (3 retries limit, 1 attempt made = 2 remaining), but "
                            "the current implementation returns `3 + 1 = 4`.\n"
                            "\n"
                            "3. Caller: Found in `src/worker.py` at line 12: `remaining = "
                            "retries_remaining(3, attempts)` - this is used to track "
                            "remaining retries.\n"
                            "\n"
                            "Let me summarize in two short sentences.",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "The user wants me to summarize "
                                "the findings without making any "
                                "changes or running more tools.\n"
                                "\n"
                                "Let me analyze what I found:\n"
                                "\n"
                                "1. Bug: The current "
                                "implementation `return limit + "
                                "attempts` is wrong. It should "
                                "calculate retries remaining as "
                                "`limit - attempts`, and never "
                                "return a negative number.\n"
                                "\n"
                                "2. Test failure: The test "
                                "expects `retries_remaining(3, "
                                "1) == 2` (3 retries limit, 1 "
                                "attempt made = 2 remaining), "
                                "but the current implementation "
                                "returns `3 + 1 = 4`.\n"
                                "\n"
                                "3. Caller: Found in "
                                "`src/worker.py` at line 12: "
                                "`remaining = "
                                "retries_remaining(3, attempts)` "
                                "- this is used to track "
                                "remaining retries.\n"
                                "\n"
                                "Let me summarize in two short "
                                "sentences.",
                                "signature": "synthetic-signature",
                            }
                        ],
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 271,
            "prompt_tokens": 1511,
            "total_tokens": 1782,
            "completion_tokens_details": {"reasoning_tokens": 178, "text_tokens": 93},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1511,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "not_available",
            "service_tier": "standard",
        },
    },
)


CLAUDE_HAIKU_4_5_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=CLAUDE_HAIKU_4_5_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **CLAUDE_HAIKU_4_5_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=CLAUDE_HAIKU_4_5_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-haiku-4-5",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-haiku-4-5-20251001",
                    "id": "msg_011Cg3Kocj4JYpPR3uTXuiXw",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1237,
                        "output_tokens": 1,
                        "service_tier": "standard",
                        "inference_geo": "not_available",
                    },
                    "diagnostics": None,
                },
            },
            event="message_start",
        ),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            event="content_block_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": thinking},
                },
                event="content_block_delta",
            )
            for thinking in (
                "The",
                " user wants me to",
                ":\n1.",
                " Search",
                " for other callers",
                " of retries_",
                "remaining",
                "\n2. Run the",
                " tests to see",
                " the",
                " failure",
                "\n\nThese",
                " are independent,",
                " so I can do",
                " both together",
                ".",
            )
        ),
        SSEFrame(
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "synthetic-signature"},
            },
            event="content_block_delta",
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 0}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01LQnJgiochDrKWke4nfjbS7",
                    "name": "search_code",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"query', '": "ret', "ries_re", "maining", '"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 1}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01SVGxc454FPrL4yRwc3QMV7",
                    "name": "run_tests",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"path"', ': "test', "s/test_", "retry.p", 'y"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 2}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None, "stop_details": None, "container": None},
                "usage": {
                    "input_tokens": 1237,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 147,
                    "output_tokens_details": {"thinking_tokens": 46},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("toolu_01LQnJgiochDrKWke4nfjbS7", "toolu_01SVGxc454FPrL4yRwc3QMV7"),
        reasoning_content=(
            "The user wants me to:\n"
            "1. Search for other callers of retries_remaining\n"
            "2. Run the tests to see the failure\n"
            "\n"
            "These are independent, so I can do both together."
        ),
        thinking_blocks=(
            {
                "type": "thinking",
                "thinking": "The user wants me to:\n"
                "1. Search for other callers of retries_remaining\n"
                "2. Run the tests to see the failure\n"
                "\n"
                "These are independent, so I can do both together.",
                "signature": "synthetic-signature",
            },
        ),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1237,
        "completion_tokens": 147,
        "total_tokens": 1384,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 46},
    },
)


CLAUDE_HAIKU_5_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "anthropic/claude-haiku-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-haiku-5-5",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-haiku-5-5",
        "id": "msg_011Cg3KowbsLXtHBaLnCbhQH",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01Gpi96LYCH1cZxDsQHHShB3",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01T9K6cra9y93JkMxr1HjLWY",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
                "caller": {"type": "direct"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1184,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_01Gpi96LYCH1cZxDsQHHShB3",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_01T9K6cra9y93JkMxr1HjLWY",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1184,
            "total_tokens": 1289,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1184,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


CLAUDE_HAIKU_5_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_HAIKU_5_5_AGENT_TEST_CASE.litellm_request, CLAUDE_HAIKU_5_5_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-haiku-5-5",
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01Gpi96LYCH1cZxDsQHHShB3",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01T9K6cra9y93JkMxr1HjLWY",
                        "name": "run_tests",
                        "input": {"path": "tests/test_retry.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01Gpi96LYCH1cZxDsQHHShB3",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01T9K6cra9y93JkMxr1HjLWY",
                        "content": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    },
                    {
                        "type": "text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    },
                ],
            },
        ],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-haiku-5-5",
        "id": "msg_011Cg3Kp4sRi7kxBgQHRJNpc",
        "content": [
            {
                "type": "text",
                "text": "The bug is that `retries_remaining` returns `limit + attempts` instead of subtracting attempts "
                "from limit, and it has no floor at zero, so `retries_remaining(3, 1)` yields 4 where the test "
                "expects 2. The only other caller is `src/worker.py` line 12 (`remaining = retries_remaining(3, "
                "attempts)`), and `tests/test_retry.py` fails with `assert 4 == 2`.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1492,
            "output_tokens": 142,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` returns `limit + attempts` instead of "
                    "subtracting attempts from limit, and it has no floor at zero, so "
                    "`retries_remaining(3, 1)` yields 4 where the test expects 2. The only other caller "
                    "is `src/worker.py` line 12 (`remaining = retries_remaining(3, attempts)`), and "
                    "`tests/test_retry.py` fails with `assert 4 == 2`.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 142,
            "prompt_tokens": 1492,
            "total_tokens": 1634,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 142},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1492,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


CLAUDE_HAIKU_5_5_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=CLAUDE_HAIKU_5_5_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **CLAUDE_HAIKU_5_5_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=CLAUDE_HAIKU_5_5_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-haiku-5-5",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-haiku-5-5",
                    "id": "msg_011Cg3Kp1JcDNc38fnmQ7xF4",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 16,
                        "service_tier": "standard",
                        "inference_geo": "global",
                    },
                    "diagnostics": None,
                },
            },
            event="message_start",
        ),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_0177HUb25ArysRAu8eF3u4DJ",
                    "name": "search_code",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"query', '": "ret', "ries_re", "maining", '"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 0}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_011gegGWvYoxPwxWY7LSK4Uk",
                    "name": "run_tests",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"path"', ': "test', "s/test_", "retry.p", 'y"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 1}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None, "stop_details": None, "container": None},
                "usage": {
                    "input_tokens": 1184,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 105,
                    "output_tokens_details": {"thinking_tokens": 0},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("toolu_0177HUb25ArysRAu8eF3u4DJ", "toolu_011gegGWvYoxPwxWY7LSK4Uk"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1184,
        "completion_tokens": 105,
        "total_tokens": 1289,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


CLAUDE_OPUS_5_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_OPUS_5_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "anthropic/claude-opus-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-opus-5-5",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_011Cg3KqG1vFQyp6DD67uewg",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01Dyzoq9Gpzjwsm1CFPLYsas",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_011gpP8bxJ5JfFzmtNY6QrnX",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
                "caller": {"type": "direct"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1184,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-opus-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_01Dyzoq9Gpzjwsm1CFPLYsas",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_011gpP8bxJ5JfFzmtNY6QrnX",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1184,
            "total_tokens": 1289,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1184,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


CLAUDE_OPUS_5_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_OPUS_5_5_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_OPUS_5_5_AGENT_TEST_CASE.litellm_request, CLAUDE_OPUS_5_5_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-opus-5-5",
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01Dyzoq9Gpzjwsm1CFPLYsas",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_011gpP8bxJ5JfFzmtNY6QrnX",
                        "name": "run_tests",
                        "input": {"path": "tests/test_retry.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01Dyzoq9Gpzjwsm1CFPLYsas",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_011gpP8bxJ5JfFzmtNY6QrnX",
                        "content": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    },
                    {
                        "type": "text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    },
                ],
            },
        ],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_011Cg3KqTXyJVhzWyFQZsXhd",
        "content": [
            {
                "type": "text",
                "text": "The bug is that `retries_remaining` in `src/retry.py` returns `limit + attempts` when it should "
                "return `max(0, limit - attempts)`, and the only other caller is `src/worker.py` line 12 "
                "(`retries_remaining(3, attempts)`). Running `tests/test_retry.py` failed with exit code 1 at "
                "`assert 4 == 2`, and I haven't changed any files.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1492,
            "output_tokens": 132,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-opus-5-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` in `src/retry.py` returns `limit + attempts` "
                    "when it should return `max(0, limit - attempts)`, and the only other caller is "
                    "`src/worker.py` line 12 (`retries_remaining(3, attempts)`). Running "
                    "`tests/test_retry.py` failed with exit code 1 at `assert 4 == 2`, and I haven't "
                    "changed any files.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 132,
            "prompt_tokens": 1492,
            "total_tokens": 1624,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 132},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1492,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


CLAUDE_OPUS_5_5_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=CLAUDE_OPUS_5_5_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **CLAUDE_OPUS_5_5_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=CLAUDE_OPUS_5_5_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-opus-5-5",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-opus-5-5",
                    "id": "msg_011Cg3KqMnwKVvkz1hizi61h",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 16,
                        "service_tier": "standard",
                        "inference_geo": "global",
                    },
                    "diagnostics": None,
                },
            },
            event="message_start",
        ),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_013mPRCxrKKH6KzYGHKPukwv",
                    "name": "search_code",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"query', '": "ret', "ries_re", "maining", '"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 0}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01FkXiaw4cTiXua3PoMd4wCu",
                    "name": "run_tests",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"path"', ': "test', "s/test_", "retry.p", 'y"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 1}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None, "stop_details": None, "container": None},
                "usage": {
                    "input_tokens": 1184,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 105,
                    "output_tokens_details": {"thinking_tokens": 0},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("toolu_013mPRCxrKKH6KzYGHKPukwv", "toolu_01FkXiaw4cTiXua3PoMd4wCu"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1184,
        "completion_tokens": 105,
        "total_tokens": 1289,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)


CLAUDE_OPUS_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_OPUS_5_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_OPUS_5_TEST_CASE.litellm_request, CLAUDE_OPUS_5_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-opus-5",
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01NFinjr4AyFZJJ1g44QYJJJ",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_018ZYT8DgJn6QPmdWqASvZnS",
                        "name": "run_tests",
                        "input": {"path": "tests/test_retry.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01NFinjr4AyFZJJ1g44QYJJJ",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_018ZYT8DgJn6QPmdWqASvZnS",
                        "content": '{"exit_code": 1, "output": "FAILED tests/test_retry.py::test_retries_remaining - '
                        'assert 4 == 2"}',
                    },
                    {
                        "type": "text",
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences.",
                    },
                ],
            },
        ],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-opus-5",
        "id": "msg_011Cg3KpvQD7r8esiTQdqnJr",
        "content": [
            {
                "type": "text",
                "text": "The bug: `retries_remaining` adds `limit + attempts` instead of subtracting, so "
                "`retries_remaining(3, 1)` returns 4 instead of 2, and it has no floor at zero; the fix would be "
                "`max(0, limit - attempts)`.\n"
                "\n"
                "Diagnostics: the only caller is `src/worker.py` line 12 (`remaining = retries_remaining(3, "
                "attempts)`), and `tests/test_retry.py::test_retries_remaining` fails with `assert 4 == 2`. No "
                "files were changed.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1486,
            "output_tokens": 174,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-opus-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug: `retries_remaining` adds `limit + attempts` instead of subtracting, so "
                    "`retries_remaining(3, 1)` returns 4 instead of 2, and it has no floor at zero; the "
                    "fix would be `max(0, limit - attempts)`.\n"
                    "\n"
                    "Diagnostics: the only caller is `src/worker.py` line 12 (`remaining = "
                    "retries_remaining(3, attempts)`), and "
                    "`tests/test_retry.py::test_retries_remaining` fails with `assert 4 == 2`. No files "
                    "were changed.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 174,
            "prompt_tokens": 1486,
            "total_tokens": 1660,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 174},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1486,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


CLAUDE_OPUS_5_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=CLAUDE_OPUS_5_TEST_CASE.litellm_endpoint,
    litellm_request={
        **CLAUDE_OPUS_5_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=CLAUDE_OPUS_5_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-opus-5",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-opus-5",
                    "id": "msg_011Cg3KpoR2dKR4xpZrcN55A",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1180,
                        "output_tokens": 17,
                        "service_tier": "standard",
                        "inference_geo": "global",
                    },
                    "diagnostics": None,
                },
            },
            event="message_start",
        ),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_017r3PoQsgodP2mgcgDnDLxU",
                    "name": "search_code",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"query', '": "ret', "ries_re", "maining", '"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 0}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01T8ECEb4XwFyozSnCKUiuB2",
                    "name": "run_tests",
                    "input": {},
                    "caller": {"type": "direct"},
                },
            },
            event="content_block_start",
        ),
        *(
            SSEFrame(
                data={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
                event="content_block_delta",
            )
            for partial_json in ("", '{"path"', ': "test', "s/test_", "retry.p", 'y"}')
        ),
        SSEFrame(data={"type": "content_block_stop", "index": 1}, event="content_block_stop"),
        SSEFrame(
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None, "stop_details": None, "container": None},
                "usage": {
                    "input_tokens": 1180,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 105,
                    "output_tokens_details": {"thinking_tokens": 0},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("toolu_017r3PoQsgodP2mgcgDnDLxU", "toolu_01T8ECEb4XwFyozSnCKUiuB2"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1180,
        "completion_tokens": 105,
        "total_tokens": 1285,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    },
)
