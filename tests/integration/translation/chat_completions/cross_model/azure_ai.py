from dataclasses import replace
from typing import Final

from integration._support.client import list_value, object_value
from integration.translation.case import (
    SSEFrame,
    StreamingTranslationTestCase,
)
from integration.translation.chat_completions.cross_model.contracts import followup_request
from integration.translation.chat_completions.bases.azure_ai import (
    CLAUDE_FABLE_5_1_TEST_CASE,
    CLAUDE_FABLE_5_TEST_CASE,
    CLAUDE_HAIKU_4_5_TEST_CASE,
    CLAUDE_HAIKU_5_5_TEST_CASE,
)
from integration.translation.chat_completions.templates import (
    CODING_AGENT_REQUEST,
    CODING_AGENT_RESPONSE,
)
from integration.translation.templates.anthropic import (
    ANTHROPIC_AGENT_REQUEST,
    ANTHROPIC_AGENT_RESPONSE,
)

CLAUDE_FABLE_5_1_AGENT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure_ai/claude-fable-5-1"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "api-key": "synthetic-azure-ai-key",
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
        "id": "msg_011Cg3Kv32UNwGKGLFiQp2ZG",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01FC3ve1J7EqbgpW7pDmfNKV",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01EmRmNW1Y26WDxEp8DaBbFW",
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
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "azure_ai/claude-fable-5-1",
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
                            "id": "toolu_01FC3ve1J7EqbgpW7pDmfNKV",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_01EmRmNW1Y26WDxEp8DaBbFW",
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
            "inference_geo": "not_available",
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
    expected_provider_endpoint="/anthropic/v1/messages",
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
                        "id": "toolu_01FC3ve1J7EqbgpW7pDmfNKV",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01EmRmNW1Y26WDxEp8DaBbFW",
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
                        "tool_use_id": "toolu_01FC3ve1J7EqbgpW7pDmfNKV",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01EmRmNW1Y26WDxEp8DaBbFW",
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
        "id": "msg_011Cg3KvJUtp38k6exsQR5Do",
        "content": [
            {
                "type": "text",
                "text": "`retries_remaining` adds `limit + attempts` instead of subtracting and clamping at zero, so "
                "`retries_remaining(3, 1)` returns 4 and `test_retries_remaining` fails with `assert 4 == 2`. The "
                "only other caller is `src/worker.py` line 12 (`retries_remaining(3, attempts)`), which would "
                "receive inflated, never-zero counts; the fix would be `return max(0, limit - attempts)`.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1492,
            "output_tokens": 150,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "azure_ai/claude-fable-5-1",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds `limit + attempts` instead of subtracting and clamping at "
                    "zero, so `retries_remaining(3, 1)` returns 4 and `test_retries_remaining` fails "
                    "with `assert 4 == 2`. The only other caller is `src/worker.py` line 12 "
                    "(`retries_remaining(3, attempts)`), which would receive inflated, never-zero "
                    "counts; the fix would be `return max(0, limit - attempts)`.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 150,
            "prompt_tokens": 1492,
            "total_tokens": 1642,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 150},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1492,
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


CLAUDE_FABLE_5_1_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=CLAUDE_FABLE_5_1_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **CLAUDE_FABLE_5_1_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=CLAUDE_FABLE_5_1_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/anthropic/v1/messages",
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
                    "id": "msg_011Cg3KvAerJShbdfNKh1sQE",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 16,
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
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01PfMSQLj6cQzebc2vGbdsV8",
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
                    "id": "toolu_01SVc4RwmEaMiDK33hUEbTZW",
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
)


CLAUDE_FABLE_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_FABLE_5_TEST_CASE.litellm_request, CLAUDE_FABLE_5_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/anthropic/v1/messages",
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
                        "id": "toolu_013SS65unrdz97WLKGBouZc1",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01TVCw5m95kexg5C2n728prK",
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
                        "tool_use_id": "toolu_013SS65unrdz97WLKGBouZc1",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01TVCw5m95kexg5C2n728prK",
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
        "id": "msg_011Cg3KvBvWS6jcrzMQB9W5K",
        "content": [
            {
                "type": "text",
                "text": "The bug: `retries_remaining` adds `limit + attempts` instead of returning `max(limit - attempts, "
                "0)`, causing `retries_remaining(3, 1)` to return 4 instead of 2 and failing "
                "`tests/test_retry.py::test_retries_remaining`. The only caller is `src/worker.py` line 12, which "
                "uses `retries_remaining(3, attempts)`.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1490,
            "output_tokens": 131,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "azure_ai/claude-fable-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug: `retries_remaining` adds `limit + attempts` instead of returning "
                    "`max(limit - attempts, 0)`, causing `retries_remaining(3, 1)` to return 4 instead "
                    "of 2 and failing `tests/test_retry.py::test_retries_remaining`. The only caller is "
                    "`src/worker.py` line 12, which uses `retries_remaining(3, attempts)`.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 131,
            "prompt_tokens": 1490,
            "total_tokens": 1621,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 131},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1490,
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


CLAUDE_FABLE_5_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=CLAUDE_FABLE_5_TEST_CASE.litellm_endpoint,
    litellm_request={
        **CLAUDE_FABLE_5_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=CLAUDE_FABLE_5_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/anthropic/v1/messages",
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
                    "id": "msg_011Cg3KuzPxKvpGokQyXz3pp",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 22,
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
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01DprHn2GeJDAWFEyTqBzy2Z",
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
                    "id": "toolu_016kava9hqtKw71W4vBEedfW",
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
)


CLAUDE_HAIKU_4_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_4_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure_ai/claude-haiku-4-5"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "api-key": "synthetic-azure-ai-key",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-haiku-4-5",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_011Cg3KtJMuQx5gfeoXFGZPS",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants me to:\n"
                "1. Search for other callers of retries_remaining\n"
                "2. Run the tests in tests/test_retry.py\n"
                "\n"
                "These are independent operations, so I can call them together. Let me do that now.",
                "signature": "synthetic-signature",
            },
            {
                "type": "tool_use",
                "id": "toolu_015u5cmERPFinecDhXTzL9ZK",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01XT5pVcHehbc1e7GAkRPtj9",
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
        "model": "azure_ai/claude-haiku-4-5",
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
                            "id": "toolu_015u5cmERPFinecDhXTzL9ZK",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_01XT5pVcHehbc1e7GAkRPtj9",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "The user wants me to:\n"
                    "1. Search for other callers of retries_remaining\n"
                    "2. Run the tests in tests/test_retry.py\n"
                    "\n"
                    "These are independent operations, so I can call them together. Let me do "
                    "that now.",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user wants me to:\n"
                            "1. Search for other callers of retries_remaining\n"
                            "2. Run the tests in tests/test_retry.py\n"
                            "\n"
                            "These are independent operations, so I can call them "
                            "together. Let me do that now.",
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
                                "tests/test_retry.py\n"
                                "\n"
                                "These are independent "
                                "operations, so I can call them "
                                "together. Let me do that now.",
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
    expected_provider_endpoint="/anthropic/v1/messages",
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
                        "2. Run the tests in tests/test_retry.py\n"
                        "\n"
                        "These are independent operations, so I can call them together. Let me do that "
                        "now.",
                        "signature": "synthetic-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_015u5cmERPFinecDhXTzL9ZK",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01XT5pVcHehbc1e7GAkRPtj9",
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
                        "tool_use_id": "toolu_015u5cmERPFinecDhXTzL9ZK",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01XT5pVcHehbc1e7GAkRPtj9",
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
        "id": "msg_011Cg3KtaHMLgMNzMjsRsrTC",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants me to summarize the bug, the caller, and the failed test in two short "
                "sentences without making any changes.\n"
                "\n"
                "Let me analyze:\n"
                "\n"
                "1. **The bug**: The function `retries_remaining(limit: int, attempts: int)` currently returns "
                "`limit + attempts`, but it should return `limit - attempts` (the number of retries left). The "
                "test shows that with a limit of 3 and 1 attempt, it should return 2 (3 - 1 = 2), but "
                "currently returns 4 (3 + 1 = 4).\n"
                "\n"
                "2. **The caller**: `src/worker.py` at line 12 calls `retries_remaining(3, attempts)`.\n"
                "\n"
                "3. **The failed test**: `test_retries_remaining` failed because `retries_remaining(3, 1)` "
                "returned 4 instead of 2. Additionally, the function should never return negative numbers, but "
                "with the current implementation it could if attempts exceeds limit.\n"
                "\n"
                "Let me write this concisely in two short sentences.",
                "signature": "synthetic-signature",
            },
            {
                "type": "text",
                "text": "**Bug Summary:**\n"
                "\n"
                "The function `retries_remaining()` incorrectly adds `limit + attempts` instead of subtracting "
                "`limit - attempts`, causing the test to fail: it returned 4 instead of 2 for "
                "`retries_remaining(3, 1)`. The caller in `src/worker.py` line 12 will receive wrong values, and "
                "the function needs to ensure it never returns negative numbers (using `max(0, limit - "
                "attempts)`).",
            },
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1511,
            "output_tokens": 351,
            "output_tokens_details": {"thinking_tokens": 241},
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "azure_ai/claude-haiku-4-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "**Bug Summary:**\n"
                    "\n"
                    "The function `retries_remaining()` incorrectly adds `limit + attempts` instead of "
                    "subtracting `limit - attempts`, causing the test to fail: it returned 4 instead of "
                    "2 for `retries_remaining(3, 1)`. The caller in `src/worker.py` line 12 will "
                    "receive wrong values, and the function needs to ensure it never returns negative "
                    "numbers (using `max(0, limit - attempts)`).",
                    "role": "assistant",
                    "reasoning_content": "The user wants me to summarize the bug, the caller, and the failed test "
                    "in two short sentences without making any changes.\n"
                    "\n"
                    "Let me analyze:\n"
                    "\n"
                    "1. **The bug**: The function `retries_remaining(limit: int, attempts: "
                    "int)` currently returns `limit + attempts`, but it should return `limit "
                    "- attempts` (the number of retries left). The test shows that with a "
                    "limit of 3 and 1 attempt, it should return 2 (3 - 1 = 2), but currently "
                    "returns 4 (3 + 1 = 4).\n"
                    "\n"
                    "2. **The caller**: `src/worker.py` at line 12 calls "
                    "`retries_remaining(3, attempts)`.\n"
                    "\n"
                    "3. **The failed test**: `test_retries_remaining` failed because "
                    "`retries_remaining(3, 1)` returned 4 instead of 2. Additionally, the "
                    "function should never return negative numbers, but with the current "
                    "implementation it could if attempts exceeds limit.\n"
                    "\n"
                    "Let me write this concisely in two short sentences.",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user wants me to summarize the bug, the caller, and the "
                            "failed test in two short sentences without making any "
                            "changes.\n"
                            "\n"
                            "Let me analyze:\n"
                            "\n"
                            "1. **The bug**: The function `retries_remaining(limit: int, "
                            "attempts: int)` currently returns `limit + attempts`, but it "
                            "should return `limit - attempts` (the number of retries "
                            "left). The test shows that with a limit of 3 and 1 attempt, "
                            "it should return 2 (3 - 1 = 2), but currently returns 4 (3 + "
                            "1 = 4).\n"
                            "\n"
                            "2. **The caller**: `src/worker.py` at line 12 calls "
                            "`retries_remaining(3, attempts)`.\n"
                            "\n"
                            "3. **The failed test**: `test_retries_remaining` failed "
                            "because `retries_remaining(3, 1)` returned 4 instead of 2. "
                            "Additionally, the function should never return negative "
                            "numbers, but with the current implementation it could if "
                            "attempts exceeds limit.\n"
                            "\n"
                            "Let me write this concisely in two short sentences.",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "The user wants me to summarize "
                                "the bug, the caller, and the "
                                "failed test in two short "
                                "sentences without making any "
                                "changes.\n"
                                "\n"
                                "Let me analyze:\n"
                                "\n"
                                "1. **The bug**: The function "
                                "`retries_remaining(limit: int, "
                                "attempts: int)` currently "
                                "returns `limit + attempts`, but "
                                "it should return `limit - "
                                "attempts` (the number of "
                                "retries left). The test shows "
                                "that with a limit of 3 and 1 "
                                "attempt, it should return 2 (3 "
                                "- 1 = 2), but currently returns "
                                "4 (3 + 1 = 4).\n"
                                "\n"
                                "2. **The caller**: "
                                "`src/worker.py` at line 12 "
                                "calls `retries_remaining(3, "
                                "attempts)`.\n"
                                "\n"
                                "3. **The failed test**: "
                                "`test_retries_remaining` failed "
                                "because `retries_remaining(3, "
                                "1)` returned 4 instead of 2. "
                                "Additionally, the function "
                                "should never return negative "
                                "numbers, but with the current "
                                "implementation it could if "
                                "attempts exceeds limit.\n"
                                "\n"
                                "Let me write this concisely in "
                                "two short sentences.",
                                "signature": "synthetic-signature",
                            }
                        ],
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 351,
            "prompt_tokens": 1511,
            "total_tokens": 1862,
            "completion_tokens_details": {"reasoning_tokens": 241, "text_tokens": 110},
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
    expected_provider_endpoint="/anthropic/v1/messages",
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
                    "id": "msg_011Cg3KtSW3j5SS95KZvtDgz",
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
                " Search for other cal",
                "lers of ret",
                "ries_remaining",
                "\n2. Run the",
                " tests in",
                " tests",
                "/test_retry",
                ".py to see",
                " the",
                " failure",
                "\n\nThese",
                " are independent,",
                " so I can call",
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
                    "id": "toolu_01WZqSaeccXLYKfSvcihGBVb",
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
                    "id": "toolu_019gRbhpTDQDLs6Dvp7kmkYi",
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
                    "output_tokens": 155,
                    "output_tokens_details": {"thinking_tokens": 54},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
)


CLAUDE_HAIKU_5_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure_ai/claude-haiku-5-5"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "api-key": "synthetic-azure-ai-key",
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
        "id": "msg_011Cg3KtRR3vNYhPw7iG4Wer",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll kick off the search and test run in parallel, then wait for both to finish.\n\n",
                "signature": "synthetic-signature",
            },
            {
                "type": "tool_use",
                "id": "toolu_01QtUaPjQ2fuKF1k3CVuQsfk",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01LXC3mf15XifCD4Pr4tuia7",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
                "caller": {"type": "direct"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1184,
            "output_tokens": 131,
            "output_tokens_details": {"thinking_tokens": 25},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "azure_ai/claude-haiku-5-5",
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
                            "id": "toolu_01QtUaPjQ2fuKF1k3CVuQsfk",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_01LXC3mf15XifCD4Pr4tuia7",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "I'll kick off the search and test run in parallel, then wait for both to "
                    "finish.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I'll kick off the search and test run in parallel, then wait "
                            "for both to finish.\n"
                            "\n",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I'll kick off the search and "
                                "test run in parallel, then wait "
                                "for both to finish.\n"
                                "\n",
                                "signature": "synthetic-signature",
                            }
                        ],
                    },
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 131,
            "prompt_tokens": 1184,
            "total_tokens": 1315,
            "completion_tokens_details": {"reasoning_tokens": 25, "text_tokens": 106},
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
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-haiku-5-5",
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I'll kick off the search and test run in parallel, then wait for both to finish.\n"
                        "\n",
                        "signature": "synthetic-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01QtUaPjQ2fuKF1k3CVuQsfk",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01LXC3mf15XifCD4Pr4tuia7",
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
                        "tool_use_id": "toolu_01QtUaPjQ2fuKF1k3CVuQsfk",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01LXC3mf15XifCD4Pr4tuia7",
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
        "id": "msg_011Cg3KtjRHKEu1uskhTfSJP",
        "content": [
            {
                "type": "text",
                "text": "The bug is that `retries_remaining` returns `limit + attempts` instead of subtracting the "
                "attempts used, and it has no floor at zero. The one other caller is `src/worker.py` line 12 "
                "(`remaining = retries_remaining(3, attempts)`), and `tests/test_retry.py` fails with `assert 4 == "
                "2`, confirming the bug. No files were changed.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1518,
            "output_tokens": 126,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "azure_ai/claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` returns `limit + attempts` instead of "
                    "subtracting the attempts used, and it has no floor at zero. The one other caller "
                    "is `src/worker.py` line 12 (`remaining = retries_remaining(3, attempts)`), and "
                    "`tests/test_retry.py` fails with `assert 4 == 2`, confirming the bug. No files "
                    "were changed.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 126,
            "prompt_tokens": 1518,
            "total_tokens": 1644,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 126},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1518,
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
    expected_provider_endpoint="/anthropic/v1/messages",
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
                    "id": "msg_011Cg3KtZGbKrbhkS1hx1ZvP",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 3,
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
                "I",
                "'ll run the search and test in",
                " parallel.",
                " I'",
                "m spotting the bug now",
                ": the function returns limit +",
                " attempts instead of max(",
                "0, limit - attempts",
                "), so for (",
                "3,1)",
                " it should give",
                " 2 but gives 4",
                ", and for",
                " (3,5) it should give",
                " 0 but gives 8",
                " ",
                "— the test will fail as written",
                ". Let me make those calls",
                " now.",
                "\n\n",
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
                    "id": "toolu_01B7tauZtVgnVUmrvUE2Wykb",
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
                    "id": "toolu_01MFTSxJgqqX3MimPYpa5pHs",
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
                    "input_tokens": 1184,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 196,
                    "output_tokens_details": {"thinking_tokens": 90},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
)
