from dataclasses import replace
from typing import Final

from integration._support.client import list_value, object_value
from integration.translation.case import (
    SSEFrame,
    StreamingTranslationTestCase,
)
from integration.translation.chat_completions.cross_model.contracts import followup_request
from integration.translation.chat_completions.bases.vertex_ai import (
    CLAUDE_FABLE_5_1_TEST_CASE,
    CLAUDE_FABLE_5_TEST_CASE,
    CLAUDE_HAIKU_4_5_TEST_CASE,
    CLAUDE_OPUS_5_5_TEST_CASE,
    CLAUDE_OPUS_5_TEST_CASE,
    GEMINI_4_ARGON_TEST_CASE,
)
from integration.translation.chat_completions.templates import (
    CODING_AGENT_REQUEST,
    CODING_AGENT_RESPONSE,
)
from integration.translation.templates.anthropic import (
    ANTHROPIC_AGENT_REQUEST,
    ANTHROPIC_AGENT_RESPONSE,
)
from integration.translation.templates.vertex_ai import (
    VERTEX_AI_AGENT_REQUEST,
    VERTEX_AI_AGENT_RESPONSE,
)

CLAUDE_FABLE_5_1_AGENT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "vertex_ai/claude-fable-5-1@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5-1@default:rawPredict",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "scripted-token",
        "anthropic-version": "2023-06-01",
        "authorization": "Bearer scripted-token",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_vrtx_011Cg3KsVe2McbhunNuC9818",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01Ecfd3DKgAtvQGQ523HWVAk",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_016Zr3Ac4cejCSamgQfWEnUs",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1184,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-fable-5-1@default",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_vrtx_01Ecfd3DKgAtvQGQ523HWVAk",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_vrtx_016Zr3Ac4cejCSamgQfWEnUs",
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
        },
    },
)


CLAUDE_FABLE_5_1_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_FABLE_5_1_AGENT_TEST_CASE.litellm_request, CLAUDE_FABLE_5_1_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5-1@default:rawPredict",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_01Ecfd3DKgAtvQGQ523HWVAk",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_016Zr3Ac4cejCSamgQfWEnUs",
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
                        "tool_use_id": "toolu_vrtx_01Ecfd3DKgAtvQGQ523HWVAk",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_vrtx_016Zr3Ac4cejCSamgQfWEnUs",
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
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_vrtx_011Cg3KsjzEeVbGa7nQM5WSM",
        "content": [
            {
                "type": "text",
                "text": "`retries_remaining` in src/retry.py adds `limit + attempts` instead of subtracting and clamping, "
                "so `retries_remaining(3, 1)` returns 4 and the test fails with `assert 4 == 2`. The only other "
                "caller is src/worker.py line 12 (`retries_remaining(3, attempts)`), which is therefore also "
                "receiving inflated values; the fix should be `max(limit - attempts, 0)`.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1492,
            "output_tokens": 143,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-fable-5-1@default",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` in src/retry.py adds `limit + attempts` instead of subtracting "
                    "and clamping, so `retries_remaining(3, 1)` returns 4 and the test fails with "
                    "`assert 4 == 2`. The only other caller is src/worker.py line 12 "
                    "(`retries_remaining(3, attempts)`), which is therefore also receiving inflated "
                    "values; the fix should be `max(limit - attempts, 0)`.",
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
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5-1@default:streamRawPredict?alt=sse",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "anthropic_version": "vertex-2023-10-16",
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-fable-5-1",
                    "id": "msg_vrtx_011Cg3KscHPJ1FG8f2MpvVM2",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 16,
                    },
                },
            },
            event="message_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_vrtx_01Fb8AC6UB4uVzYa1b3zemi6",
                    "name": "search_code",
                    "input": {},
                },
            },
            event="content_block_start",
        ),
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
                    "id": "toolu_vrtx_01JYVwxmecP9JbZ1kUktCC85",
                    "name": "run_tests",
                    "input": {},
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
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5@default:rawPredict",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_01HkGRMBmWWCqTp5Ck2nZnJp",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_01ASaiWVEwF3LUJU8KDzBdrr",
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
                        "tool_use_id": "toolu_vrtx_01HkGRMBmWWCqTp5Ck2nZnJp",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_vrtx_01ASaiWVEwF3LUJU8KDzBdrr",
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
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-fable-5",
        "id": "msg_vrtx_011Cg3KsKgkMrw3BurZAmkNj",
        "content": [
            {
                "type": "text",
                "text": "The bug: `retries_remaining` adds `limit + attempts` instead of subtracting and clamping, so "
                "`retries_remaining(3, 1)` returns 4 instead of 2, failing "
                "`tests/test_retry.py::test_retries_remaining` (exit code 1). The only caller is `src/worker.py` "
                "line 12 (`retries_remaining(3, attempts)`); the fix would be `max(limit - attempts, 0)`.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1490,
            "output_tokens": 145,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-fable-5@default",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug: `retries_remaining` adds `limit + attempts` instead of subtracting and "
                    "clamping, so `retries_remaining(3, 1)` returns 4 instead of 2, failing "
                    "`tests/test_retry.py::test_retries_remaining` (exit code 1). The only caller is "
                    "`src/worker.py` line 12 (`retries_remaining(3, attempts)`); the fix would be "
                    "`max(limit - attempts, 0)`.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 145,
            "prompt_tokens": 1490,
            "total_tokens": 1635,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 145},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1490,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
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
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5@default:streamRawPredict?alt=sse",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "anthropic_version": "vertex-2023-10-16",
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-fable-5",
                    "id": "msg_vrtx_011Cg3KsDk4JLF2SnBKwPmy8",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 17,
                    },
                },
            },
            event="message_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_vrtx_01V4czREisiGafV3679XxFXp",
                    "name": "search_code",
                    "input": {},
                },
            },
            event="content_block_start",
        ),
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
                    "id": "toolu_vrtx_01453AfT3eHq6RMWZyQrb4KX",
                    "name": "run_tests",
                    "input": {},
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
    litellm_request={**CODING_AGENT_REQUEST, "model": "vertex_ai/claude-haiku-4-5@20251001"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "scripted-token",
        "anthropic-version": "2023-06-01",
        "authorization": "Bearer scripted-token",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_vrtx_011Cg3KoZJgBQ9mSu1GnHuso",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants me to:\n"
                "1. Search for other callers of retries_remaining\n"
                "2. Run tests/test_retry.py to see the failures\n"
                "\n"
                "These are independent, so I can call them together.",
                "signature": "synthetic-signature",
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_0136Y5muJivfopS9jATU7WKm",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01H4LqQBBsENeC8fC37BhX2J",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1237,
            "output_tokens": 152,
            "output_tokens_details": {"thinking_tokens": 51},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_vrtx_0136Y5muJivfopS9jATU7WKm",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_vrtx_01H4LqQBBsENeC8fC37BhX2J",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "The user wants me to:\n"
                    "1. Search for other callers of retries_remaining\n"
                    "2. Run tests/test_retry.py to see the failures\n"
                    "\n"
                    "These are independent, so I can call them together.",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user wants me to:\n"
                            "1. Search for other callers of retries_remaining\n"
                            "2. Run tests/test_retry.py to see the failures\n"
                            "\n"
                            "These are independent, so I can call them together.",
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
                                "2. Run tests/test_retry.py to "
                                "see the failures\n"
                                "\n"
                                "These are independent, so I can "
                                "call them together.",
                                "signature": "synthetic-signature",
                            }
                        ],
                    },
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 152,
            "prompt_tokens": 1237,
            "total_tokens": 1389,
            "completion_tokens_details": {"reasoning_tokens": 51, "text_tokens": 101},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1237,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


CLAUDE_HAIKU_4_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_4_5_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_HAIKU_4_5_AGENT_TEST_CASE.litellm_request, CLAUDE_HAIKU_4_5_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The user wants me to:\n"
                        "1. Search for other callers of retries_remaining\n"
                        "2. Run tests/test_retry.py to see the failures\n"
                        "\n"
                        "These are independent, so I can call them together.",
                        "signature": "synthetic-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_0136Y5muJivfopS9jATU7WKm",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_01H4LqQBBsENeC8fC37BhX2J",
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
                        "tool_use_id": "toolu_vrtx_0136Y5muJivfopS9jATU7WKm",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_vrtx_01H4LqQBBsENeC8fC37BhX2J",
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
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_vrtx_011Cg3KokugR8m6v5WoXYy5q",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants me to summarize findings without making changes or calling more tools.\n"
                "\n"
                "Let me analyze:\n"
                "1. The bug: `retries_remaining(limit: int, attempts: int)` currently returns `limit + "
                "attempts`, but it should return the number of retries left. Based on the test, "
                "`retries_remaining(3, 1)` should return 2, and `retries_remaining(3, 5)` should return 0. So "
                "it should return `max(0, limit - attempts)`.\n"
                "\n"
                "2. The caller: There's a caller in `src/worker.py` at line 12 that calls "
                "`retries_remaining(3, attempts)`.\n"
                "\n"
                "3. The failed test: `test_retries_remaining` fails because `retries_remaining(3, 1)` returns "
                "4 instead of 2.\n"
                "\n"
                "Let me write this in two short sentences.",
                "signature": "synthetic-signature",
            },
            {
                "type": "text",
                "text": "The bug: `retries_remaining()` currently adds `limit + attempts` instead of subtracting attempts "
                "from the limit and clamping to zero (should be `max(0, limit - attempts)`). The caller in "
                "`src/worker.py` line 12 uses it to track remaining retries, and the test fails because "
                "`retries_remaining(3, 1)` returns 4 instead of the expected 2.",
            },
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1511,
            "output_tokens": 304,
            "output_tokens_details": {"thinking_tokens": 203},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug: `retries_remaining()` currently adds `limit + attempts` instead of "
                    "subtracting attempts from the limit and clamping to zero (should be `max(0, limit "
                    "- attempts)`). The caller in `src/worker.py` line 12 uses it to track remaining "
                    "retries, and the test fails because `retries_remaining(3, 1)` returns 4 instead of "
                    "the expected 2.",
                    "role": "assistant",
                    "reasoning_content": "The user wants me to summarize findings without making changes or "
                    "calling more tools.\n"
                    "\n"
                    "Let me analyze:\n"
                    "1. The bug: `retries_remaining(limit: int, attempts: int)` currently "
                    "returns `limit + attempts`, but it should return the number of retries "
                    "left. Based on the test, `retries_remaining(3, 1)` should return 2, and "
                    "`retries_remaining(3, 5)` should return 0. So it should return `max(0, "
                    "limit - attempts)`.\n"
                    "\n"
                    "2. The caller: There's a caller in `src/worker.py` at line 12 that calls "
                    "`retries_remaining(3, attempts)`.\n"
                    "\n"
                    "3. The failed test: `test_retries_remaining` fails because "
                    "`retries_remaining(3, 1)` returns 4 instead of 2.\n"
                    "\n"
                    "Let me write this in two short sentences.",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user wants me to summarize findings without making "
                            "changes or calling more tools.\n"
                            "\n"
                            "Let me analyze:\n"
                            "1. The bug: `retries_remaining(limit: int, attempts: int)` "
                            "currently returns `limit + attempts`, but it should return "
                            "the number of retries left. Based on the test, "
                            "`retries_remaining(3, 1)` should return 2, and "
                            "`retries_remaining(3, 5)` should return 0. So it should "
                            "return `max(0, limit - attempts)`.\n"
                            "\n"
                            "2. The caller: There's a caller in `src/worker.py` at line "
                            "12 that calls `retries_remaining(3, attempts)`.\n"
                            "\n"
                            "3. The failed test: `test_retries_remaining` fails because "
                            "`retries_remaining(3, 1)` returns 4 instead of 2.\n"
                            "\n"
                            "Let me write this in two short sentences.",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "The user wants me to summarize "
                                "findings without making changes "
                                "or calling more tools.\n"
                                "\n"
                                "Let me analyze:\n"
                                "1. The bug: "
                                "`retries_remaining(limit: int, "
                                "attempts: int)` currently "
                                "returns `limit + attempts`, but "
                                "it should return the number of "
                                "retries left. Based on the "
                                "test, `retries_remaining(3, 1)` "
                                "should return 2, and "
                                "`retries_remaining(3, 5)` "
                                "should return 0. So it should "
                                "return `max(0, limit - "
                                "attempts)`.\n"
                                "\n"
                                "2. The caller: There's a caller "
                                "in `src/worker.py` at line 12 "
                                "that calls "
                                "`retries_remaining(3, "
                                "attempts)`.\n"
                                "\n"
                                "3. The failed test: "
                                "`test_retries_remaining` fails "
                                "because `retries_remaining(3, "
                                "1)` returns 4 instead of 2.\n"
                                "\n"
                                "Let me write this in two short "
                                "sentences.",
                                "signature": "synthetic-signature",
                            }
                        ],
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 304,
            "prompt_tokens": 1511,
            "total_tokens": 1815,
            "completion_tokens_details": {"reasoning_tokens": 203, "text_tokens": 101},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1511,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
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
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-haiku-4-5@20251001:streamRawPredict?alt=sse",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "anthropic_version": "vertex-2023-10-16",
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-haiku-4-5-20251001",
                    "id": "msg_vrtx_011Cg3KofLaGqudJVwCcL4Q1",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1237,
                        "output_tokens": 6,
                    },
                },
            },
            event="message_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            event="content_block_start",
        ),
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
                "The user wants me to:",
                "\n1. Search for other",
                " callers of ret",
                "ries_remaining",
                "\n2. Run the",
                " tests in",
                " tests",
                "/test_retry.py",
                "\n\nThese",
                " are independent so",
                " I can run",
                " both together.",
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
                    "id": "toolu_vrtx_015N8ohQNJaWXj7Uifk3jR3V",
                    "name": "search_code",
                    "input": {},
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
                    "id": "toolu_vrtx_01XkT66qy9agHfZJKbt7ACxA",
                    "name": "run_tests",
                    "input": {},
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
                    "output_tokens": 150,
                    "output_tokens_details": {"thinking_tokens": 49},
                },
            },
            event="message_delta",
        ),
        SSEFrame(data={"type": "message_stop"}, event="message_stop"),
    ),
)


CLAUDE_OPUS_5_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_OPUS_5_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "vertex_ai/claude-opus-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5-5@default:rawPredict",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "scripted-token",
        "anthropic-version": "2023-06-01",
        "authorization": "Bearer scripted-token",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_vrtx_011Cg3KqcHKNHvvYAwm8yZ7r",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01WdFyxtHEbXZNdX9XCEzyVv",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01HKZB5dfbH7JEjk9Cq58jT2",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1184,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-opus-5-5@default",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_vrtx_01WdFyxtHEbXZNdX9XCEzyVv",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_vrtx_01HKZB5dfbH7JEjk9Cq58jT2",
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
        },
    },
)


CLAUDE_OPUS_5_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_OPUS_5_5_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_OPUS_5_5_AGENT_TEST_CASE.litellm_request, CLAUDE_OPUS_5_5_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5-5@default:rawPredict",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_01WdFyxtHEbXZNdX9XCEzyVv",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_01HKZB5dfbH7JEjk9Cq58jT2",
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
                        "tool_use_id": "toolu_vrtx_01WdFyxtHEbXZNdX9XCEzyVv",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_vrtx_01HKZB5dfbH7JEjk9Cq58jT2",
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
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_vrtx_011Cg3KqrsQxizktHqKwWv5m",
        "content": [
            {
                "type": "thinking",
                "thinking": "I found the bug: it's returning limit + attempts instead of max(0, limit - attempts), which "
                "is why the caller in worker.py line 12 fails the test expecting 2 but getting 4.\n"
                "\n",
                "signature": "synthetic-signature",
            },
            {
                "type": "text",
                "text": "The bug is that `retries_remaining` returns `limit + attempts` when it should return `max(0, "
                "limit - attempts)`, so `tests/test_retry.py::test_retries_remaining` fails with `assert 4 == 2` "
                "(exit code 1). The only other caller is `src/worker.py` line 12 (`remaining = "
                "retries_remaining(3, attempts)`), and I haven't changed any files.",
            },
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1492,
            "output_tokens": 195,
            "output_tokens_details": {"thinking_tokens": 62},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-opus-5-5@default",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` returns `limit + attempts` when it should "
                    "return `max(0, limit - attempts)`, so "
                    "`tests/test_retry.py::test_retries_remaining` fails with `assert 4 == 2` (exit "
                    "code 1). The only other caller is `src/worker.py` line 12 (`remaining = "
                    "retries_remaining(3, attempts)`), and I haven't changed any files.",
                    "role": "assistant",
                    "reasoning_content": "I found the bug: it's returning limit + attempts instead of max(0, limit "
                    "- attempts), which is why the caller in worker.py line 12 fails the test "
                    "expecting 2 but getting 4.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I found the bug: it's returning limit + attempts instead of "
                            "max(0, limit - attempts), which is why the caller in "
                            "worker.py line 12 fails the test expecting 2 but getting 4.\n"
                            "\n",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I found the bug: it's returning "
                                "limit + attempts instead of "
                                "max(0, limit - attempts), which "
                                "is why the caller in worker.py "
                                "line 12 fails the test "
                                "expecting 2 but getting 4.\n"
                                "\n",
                                "signature": "synthetic-signature",
                            }
                        ],
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 195,
            "prompt_tokens": 1492,
            "total_tokens": 1687,
            "completion_tokens_details": {"reasoning_tokens": 62, "text_tokens": 133},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1492,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
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
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5-5@default:streamRawPredict?alt=sse",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "anthropic_version": "vertex-2023-10-16",
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-opus-5-5",
                    "id": "msg_vrtx_011Cg3KqiHUaCbe6RcbVi5G6",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1184,
                        "output_tokens": 16,
                    },
                },
            },
            event="message_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_vrtx_01LefrYS5RoUrFshmxExy7Hj",
                    "name": "search_code",
                    "input": {},
                },
            },
            event="content_block_start",
        ),
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
                    "id": "toolu_vrtx_01KRry1KaahxK2NvecpzvsdN",
                    "name": "run_tests",
                    "input": {},
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


CLAUDE_OPUS_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_OPUS_5_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_OPUS_5_TEST_CASE.litellm_request, CLAUDE_OPUS_5_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5@default:rawPredict",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "messages": [
            *list_value(ANTHROPIC_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_014y2somv8RVnNoX3s7i7RwH",
                        "name": "search_code",
                        "input": {"query": "retries_remaining"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_vrtx_01B8D9djnyoPnNQSccNW99Hi",
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
                        "tool_use_id": "toolu_vrtx_014y2somv8RVnNoX3s7i7RwH",
                        "content": '{"matches": [{"path": "src/worker.py", "line": 12, "text": "remaining = '
                        'retries_remaining(3, attempts)"}]}',
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_vrtx_01B8D9djnyoPnNQSccNW99Hi",
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
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-opus-5",
        "id": "msg_vrtx_011Cg3KqRVw65mPo6JXq7wkN",
        "content": [
            {
                "type": "text",
                "text": "Bug: `retries_remaining` adds `limit + attempts` instead of subtracting, so `retries_remaining(3, "
                "1)` returned 4 instead of 2, and it has no floor to prevent negative results when `attempts > "
                "limit`.\n"
                "\n"
                "Caller/test: the only caller is `src/worker.py:12` (`remaining = retries_remaining(3, "
                "attempts)`), and `tests/test_retry.py::test_retries_remaining` fails with `assert 4 == 2`. The "
                "fix would be `max(0, limit - attempts)`, but no files were changed.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1486,
            "output_tokens": 185,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-opus-5@default",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Bug: `retries_remaining` adds `limit + attempts` instead of subtracting, so "
                    "`retries_remaining(3, 1)` returned 4 instead of 2, and it has no floor to prevent "
                    "negative results when `attempts > limit`.\n"
                    "\n"
                    "Caller/test: the only caller is `src/worker.py:12` (`remaining = "
                    "retries_remaining(3, attempts)`), and "
                    "`tests/test_retry.py::test_retries_remaining` fails with `assert 4 == 2`. The fix "
                    "would be `max(0, limit - attempts)`, but no files were changed.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 185,
            "prompt_tokens": 1486,
            "total_tokens": 1671,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 185},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1486,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
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
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5@default:streamRawPredict?alt=sse",
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "anthropic_version": "vertex-2023-10-16",
        "stream": True,
    },
    mock_provider_frames=(
        SSEFrame(
            data={
                "type": "message_start",
                "message": {
                    **ANTHROPIC_AGENT_RESPONSE,
                    "model": "claude-opus-5",
                    "id": "msg_vrtx_011Cg3KqJD8Qj5A1iw6AEhMR",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
                        "input_tokens": 1180,
                        "output_tokens": 16,
                    },
                },
            },
            event="message_start",
        ),
        SSEFrame(data={"type": "ping"}, event="ping"),
        SSEFrame(
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_vrtx_01U7KysFJdXDySG6SViytraf",
                    "name": "search_code",
                    "input": {},
                },
            },
            event="content_block_start",
        ),
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
                    "id": "toolu_vrtx_01Tph9GZmNEQ49UUuE6tQKLt",
                    "name": "run_tests",
                    "input": {},
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
)


GEMINI_4_ARGON_AGENT_TEST_CASE: Final = replace(
    GEMINI_4_ARGON_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "vertex_ai/gemini-4-argon"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-4-argon:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={**VERTEX_AI_AGENT_REQUEST},
    mock_provider_response={
        **VERTEX_AI_AGENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "I will search for callers of `retries_remaining` and run the test suite to "
                            "reproduce the failure."
                        },
                        {
                            "functionCall": {
                                "name": "search_code",
                                "args": {"query": "retries_remaining"},
                                "id": "call_318300",
                            },
                            "thoughtSignature": "synthetic-signature",
                        },
                        {
                            "functionCall": {
                                "name": "run_tests",
                                "args": {"path": "tests/test_retry.py"},
                                "id": "call_318301",
                            }
                        },
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            **object_value(VERTEX_AI_AGENT_RESPONSE["usageMetadata"]),
            "promptTokenCount": 582,
            "candidatesTokenCount": 69,
            "totalTokenCount": 651,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 582}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 69}],
        },
        "createTime": "2026-10-15T03:38:40.834141Z",
        "responseId": "wErQat30Mp-hpucPhIzf2AM",
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "wErQat30Mp-hpucPhIzf2AM",
        "model": "vertex_ai/gemini-4-argon",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I will search for callers of `retries_remaining` and run the test suite to "
                    "reproduce the failure.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {"thought_signature": "synthetic-signature"},
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "call_318300__thought__synthetic-signature",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_318301",
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "thinking_blocks": [],
                    "provider_specific_fields": {"thought_signatures": ["synthetic-signature"]},
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 69,
            "prompt_tokens": 582,
            "total_tokens": 651,
            "completion_tokens_details": {"text_tokens": 69},
            "prompt_tokens_details": {"text_tokens": 582},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)


GEMINI_4_ARGON_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GEMINI_4_ARGON_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GEMINI_4_ARGON_AGENT_TEST_CASE.litellm_request, GEMINI_4_ARGON_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-4-argon:generateContent",
    expected_provider_request={
        **VERTEX_AI_AGENT_REQUEST,
        "contents": [
            *list_value(VERTEX_AI_AGENT_REQUEST["contents"]),
            {
                "role": "model",
                "parts": [
                    {
                        "text": "I will search for callers of `retries_remaining` and run the test suite to reproduce "
                        "the failure."
                    },
                    {
                        "function_call": {
                            "name": "search_code",
                            "args": {"query": "retries_remaining"},
                            "id": "call_318300",
                        },
                        "thoughtSignature": "synthetic-signature",
                    },
                    {
                        "function_call": {
                            "name": "run_tests",
                            "args": {"path": "tests/test_retry.py"},
                            "id": "call_318301",
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
                            "id": "call_318300",
                        }
                    },
                    {
                        "function_response": {
                            "name": "run_tests",
                            "response": {
                                "exit_code": 1,
                                "output": "FAILED tests/test_retry.py::test_retries_remaining - assert 4 == 2",
                            },
                            "id": "call_318301",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    }
                ],
            },
        ],
    },
    mock_provider_response={
        **VERTEX_AI_AGENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "`retries_remaining` in `src/retry.py` erroneously adds `limit + attempts` "
                            "instead of clamping `limit - attempts` to a minimum of 0, causing "
                            "`tests/test_retry.py::test_retries_remaining` to fail with `assert 4 == "
                            "2`. A search found one caller in `src/worker.py` at line 12 (`remaining = "
                            "retries_remaining(3, attempts)`).",
                            "thoughtSignature": "synthetic-signature",
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            **object_value(VERTEX_AI_AGENT_RESPONSE["usageMetadata"]),
            "promptTokenCount": 780,
            "candidatesTokenCount": 96,
            "totalTokenCount": 876,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 780}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 96}],
        },
        "createTime": "2026-10-15T03:38:45.481873Z",
        "responseId": "xUrQatG0HcubpucPg4md8Ag",
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "xUrQatG0HcubpucPg4md8Ag",
        "model": "vertex_ai/gemini-4-argon",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` in `src/retry.py` erroneously adds `limit + attempts` instead "
                    "of clamping `limit - attempts` to a minimum of 0, causing "
                    "`tests/test_retry.py::test_retries_remaining` to fail with `assert 4 == 2`. A "
                    "search found one caller in `src/worker.py` at line 12 (`remaining = "
                    "retries_remaining(3, attempts)`).",
                    "role": "assistant",
                    "images": [],
                    "thinking_blocks": [],
                    "provider_specific_fields": {"thought_signatures": ["synthetic-signature"]},
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 96,
            "prompt_tokens": 780,
            "total_tokens": 876,
            "completion_tokens_details": {"text_tokens": 96},
            "prompt_tokens_details": {"text_tokens": 780},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)


GEMINI_4_ARGON_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GEMINI_4_ARGON_AGENT_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GEMINI_4_ARGON_AGENT_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GEMINI_4_ARGON_AGENT_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-4-argon:streamGenerateContent?alt=sse",
    expected_provider_request={**VERTEX_AI_AGENT_REQUEST},
    mock_provider_frames=(
        SSEFrame(
            data={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "search_code",
                                        "args": {"query": "retries_remaining"},
                                        "id": "call_337037",
                                    },
                                    "thoughtSignature": "synthetic-signature",
                                }
                            ],
                        }
                    }
                ],
                "usageMetadata": {"trafficType": "ON_DEMAND"},
                "modelVersion": "gemini-4-argon",
                "createTime": "2026-10-15T03:38:43.235481Z",
                "responseId": "w0rQatmvDsCQ998PtIbRqA0",
            }
        ),
        SSEFrame(
            data={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "run_tests",
                                        "args": {"path": "tests/test_retry.py"},
                                        "id": "call_337046",
                                    }
                                }
                            ],
                        }
                    }
                ],
                "usageMetadata": {"trafficType": "ON_DEMAND"},
                "modelVersion": "gemini-4-argon",
                "createTime": "2026-10-15T03:38:43.235481Z",
                "responseId": "w0rQatmvDsCQ998PtIbRqA0",
            }
        ),
        SSEFrame(
            data={
                "candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 582,
                    "candidatesTokenCount": 48,
                    "totalTokenCount": 630,
                    "trafficType": "ON_DEMAND",
                    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 582}],
                    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 48}],
                },
                "modelVersion": "gemini-4-argon",
                "createTime": "2026-10-15T03:38:43.235481Z",
                "responseId": "w0rQatmvDsCQ998PtIbRqA0",
            }
        ),
    ),
)
