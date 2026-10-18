from dataclasses import replace
from typing import Final

from integration._support.client import list_value, object_value
from integration.translation.case import (
    EventStreamFrame,
    StreamingTranslationTestCase,
)
from integration.translation.chat_completions.cross_model.contracts import followup_request
from integration.translation.chat_completions.bases.bedrock_converse import (
    CLAUDE_FABLE_5_1_TEST_CASE,
    CLAUDE_FABLE_5_TEST_CASE,
    CLAUDE_HAIKU_4_5_TEST_CASE,
    CLAUDE_HAIKU_5_5_TEST_CASE,
    CLAUDE_OPUS_5_5_TEST_CASE,
    CLAUDE_OPUS_5_TEST_CASE,
    GPT_5_6_LUNA_TEST_CASE,
    GPT_5_6_SOL_TEST_CASE,
    GPT_5_6_TERRA_TEST_CASE,
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
from integration.translation.templates.bedrock_converse import (
    BEDROCK_CONVERSE_AGENT_REQUEST,
    BEDROCK_CONVERSE_AGENT_RESPONSE,
)

CLAUDE_FABLE_5_1_AGENT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.anthropic.claude-fable-5-1"},
    expected_provider_endpoint="/model/global.anthropic.claude-fable-5-1/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
        "metrics": {"latencyMs": 3779},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_pfGpYRvkYgW7G6gt3h6QUl",
                            "type": "tool_use",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_LfHkrgNn9Gf2LjuKrQ0b5m",
                            "type": "tool_use",
                        }
                    },
                ],
            },
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1184,
            "outputTokens": 105,
            "totalTokens": 1289,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-fable-5-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "tooluse_pfGpYRvkYgW7G6gt3h6QUl",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "tooluse_LfHkrgNn9Gf2LjuKrQ0b5m",
                            "type": "function",
                        },
                    ],
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
    expected_provider_endpoint="/model/global.anthropic.claude-fable-5-1/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_pfGpYRvkYgW7G6gt3h6QUl",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_LfHkrgNn9Gf2LjuKrQ0b5m",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "tooluse_pfGpYRvkYgW7G6gt3h6QUl",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "tooluse_LfHkrgNn9Gf2LjuKrQ0b5m",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
        "metrics": {"latencyMs": 4671},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "`retries_remaining` adds `limit + attempts` instead of subtracting and clamping, so "
                        "`retries_remaining(3, 1)` returns 4 and the test fails with `assert 4 == 2` (the second case, "
                        "`(3, 5)`, would also go negative instead of returning 0). The only caller is `src/worker.py` line "
                        "12 (`retries_remaining(3, attempts)`), which would receive inflated counts; the fix is `max(0, "
                        "limit - attempts)`, but I have not applied it per your instruction."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1492,
            "outputTokens": 170,
            "totalTokens": 1662,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-fable-5-1",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds `limit + attempts` instead of subtracting and clamping, "
                    "so `retries_remaining(3, 1)` returns 4 and the test fails with `assert 4 == 2` "
                    "(the second case, `(3, 5)`, would also go negative instead of returning 0). The "
                    "only caller is `src/worker.py` line 12 (`retries_remaining(3, attempts)`), which "
                    "would receive inflated counts; the fix is `max(0, limit - attempts)`, but I have "
                    "not applied it per your instruction.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 170,
            "prompt_tokens": 1492,
            "total_tokens": 1662,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 170},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1492,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
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
    expected_provider_endpoint="/model/global.anthropic.claude-fable-5-1/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqr", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvwxyzABC",
                "start": {
                    "toolUse": {
                        "name": "search_code",
                        "toolUseId": "tooluse_1scOGH6yqKg2qgAQuDrITO",
                        "type": "tool_use",
                    }
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": ""}}, "p": "abcdefghijklm"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"qu'}}, "p": "abc"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": 'ery":'}}, "p": "abcdefghi"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": ' "retries_'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "rema"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": 'ining"}'}}, "p": "abcdefghijklmnopqrst"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefg"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0",
                "start": {
                    "toolUse": {"name": "run_tests", "toolUseId": "tooluse_e14xz2GhG71KhrznKTu7ZR", "type": "tool_use"}
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": ""}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"path":'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFG",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ' "tests/test'}}, "p": "abcdefghijkl"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "_retr"}},
                "p": "abcdefghijklmnopqrstuvwxyzABC",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": 'y.py"}'}}, "p": "abcd"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "ab"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO",
                "stopReason": "tool_use",
            },
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 5201},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL",
                "usage": {"inputTokens": 1184, "outputTokens": 105, "serverToolUsage": {}, "totalTokens": 1289},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("tooluse_1scOGH6yqKg2qgAQuDrITO", "tooluse_e14xz2GhG71KhrznKTu7ZR"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1184,
        "completion_tokens": 105,
        "total_tokens": 1289,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 1184,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
    },
)


CLAUDE_FABLE_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_FABLE_5_TEST_CASE.litellm_request, CLAUDE_FABLE_5_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.anthropic.claude-fable-5/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_8F5NPCLa3Auzb12CgAn28Q",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_0GdPjDJbx5w7EzbCQi49uO",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "tooluse_8F5NPCLa3Auzb12CgAn28Q",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "tooluse_0GdPjDJbx5w7EzbCQi49uO",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
        "metrics": {"latencyMs": 2973},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "The bug is that `retries_remaining` adds `limit + attempts` instead of subtracting and clamping "
                        "at zero, so `retries_remaining(3, 1)` returns 4 instead of 2, failing "
                        "`tests/test_retry.py::test_retries_remaining`. The only caller is `src/worker.py` line 12 "
                        "(`retries_remaining(3, attempts)`); the fix would be `max(limit - attempts, 0)`."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1490,
            "outputTokens": 143,
            "totalTokens": 1633,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-fable-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` adds `limit + attempts` instead of subtracting "
                    "and clamping at zero, so `retries_remaining(3, 1)` returns 4 instead of 2, failing "
                    "`tests/test_retry.py::test_retries_remaining`. The only caller is `src/worker.py` "
                    "line 12 (`retries_remaining(3, attempts)`); the fix would be `max(limit - "
                    "attempts, 0)`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 143,
            "prompt_tokens": 1490,
            "total_tokens": 1633,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 143},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1490,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
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
    expected_provider_endpoint="/model/global.anthropic.claude-fable-5/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM",
                "start": {
                    "toolUse": {
                        "name": "search_code",
                        "toolUseId": "tooluse_7je3KtzalxPnlZ5FGwfYSx",
                        "type": "tool_use",
                    }
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": ""}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"quer'}}, "p": "abcdefghijklm"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": 'y"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": ': "r'}}, "p": "abcde"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "etries_rema"}}, "p": "abcdefghijklmnopq"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": 'ining"}'}}, "p": "abcdefghij"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrstuvwxyzABCDEF"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuvwxyzABC",
                "start": {
                    "toolUse": {"name": "run_tests", "toolUseId": "tooluse_VfjLWMwJQpLJ9b1u33bw0F", "type": "tool_use"}
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ""}}, "p": "abcdefghijklmnop"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"pat'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": 'h": "te'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "sts/tes"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "t_retry.p"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": 'y"}'}}, "p": "abc"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abc"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012",
                "stopReason": "tool_use",
            },
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 2522},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO",
                "usage": {"inputTokens": 1184, "outputTokens": 105, "serverToolUsage": {}, "totalTokens": 1289},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("tooluse_7je3KtzalxPnlZ5FGwfYSx", "tooluse_VfjLWMwJQpLJ9b1u33bw0F"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1184,
        "completion_tokens": 105,
        "total_tokens": 1289,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 1184,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
    },
)


CLAUDE_HAIKU_4_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_4_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={
        **CODING_AGENT_REQUEST,
        "model": "bedrock/converse/global.anthropic.claude-haiku-4-5-20251001-v1:0",
    },
    expected_provider_endpoint="/model/global.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tool_choice": {"disable_parallel_tool_use": False},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 55}}},
        "metrics": {"latencyMs": 1318},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": "synthetic-signature",
                                "text": "The user wants me to:\n"
                                "1. Search for other callers of retries_remaining\n"
                                "2. Run the tests in tests/test_retry.py to reproduce the "
                                "failure\n"
                                "\n"
                                "These are independent, so I can make both calls "
                                "together.",
                            }
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_pKAmZuowSdrVLHAcP64czO",
                            "type": "tool_use",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_2ZZoepPMFZrCmojWjnuxMr",
                            "type": "tool_use",
                        }
                    },
                ],
            },
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1237,
            "outputTokens": 156,
            "totalTokens": 1393,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "tooluse_pKAmZuowSdrVLHAcP64czO",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "tooluse_2ZZoepPMFZrCmojWjnuxMr",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "The user wants me to:\n"
                    "1. Search for other callers of retries_remaining\n"
                    "2. Run the tests in tests/test_retry.py to reproduce the failure\n"
                    "\n"
                    "These are independent, so I can make both calls together.",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user wants me to:\n"
                            "1. Search for other callers of retries_remaining\n"
                            "2. Run the tests in tests/test_retry.py to reproduce the "
                            "failure\n"
                            "\n"
                            "These are independent, so I can make both calls together.",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "reasoningContentBlocks": [
                            {
                                "reasoningText": {
                                    "signature": "synthetic-signature",
                                    "text": "The user "
                                    "wants me "
                                    "to:\n"
                                    "1. Search "
                                    "for other "
                                    "callers of "
                                    "retries_remaining\n"
                                    "2. Run the "
                                    "tests in "
                                    "tests/test_retry.py "
                                    "to "
                                    "reproduce "
                                    "the "
                                    "failure\n"
                                    "\n"
                                    "These are "
                                    "independent, "
                                    "so I can "
                                    "make both "
                                    "calls "
                                    "together.",
                                }
                            }
                        ]
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 156,
            "prompt_tokens": 1237,
            "total_tokens": 1393,
            "completion_tokens_details": {"reasoning_tokens": 55, "text_tokens": 101},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1237,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
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
    expected_provider_endpoint="/model/global.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "The user wants me to:\n"
                                "1. Search for other callers of "
                                "retries_remaining\n"
                                "2. Run the tests in tests/test_retry.py to "
                                "reproduce the failure\n"
                                "\n"
                                "These are independent, so I can make both "
                                "calls together.",
                                "signature": "synthetic-signature",
                            }
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_pKAmZuowSdrVLHAcP64czO",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_2ZZoepPMFZrCmojWjnuxMr",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "tooluse_pKAmZuowSdrVLHAcP64czO",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "tooluse_2ZZoepPMFZrCmojWjnuxMr",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tool_choice": {"disable_parallel_tool_use": False},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 236}}},
        "metrics": {"latencyMs": 3009},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": "synthetic-signature",
                                "text": "Let me analyze what I found:\n"
                                "\n"
                                "1. **The bug**: The current implementation is `return "
                                "limit + attempts`, but it should return `limit - "
                                "attempts`. The test expects that with limit=3 and "
                                "attempts=1, there should be 2 retries remaining (3-1=2). "
                                "Currently it returns 4 (3+1=4).\n"
                                "\n"
                                "2. **The caller**: `src/worker.py` line 12 calls "
                                "`retries_remaining(3, attempts)` and stores the result in "
                                "`remaining`.\n"
                                "\n"
                                "3. **The failed test**: The test "
                                "`test_retries_remaining()` expects `retries_remaining(3, "
                                "1) == 2` but the current implementation returns 4. "
                                "Additionally, the second assertion expects "
                                "`retries_remaining(3, 5) == 0` (meaning when you've "
                                "attempted more than the limit, it should return 0, not "
                                "negative).\n"
                                "\n"
                                "The user wants a two-sentence summary of:\n"
                                "- The bug\n"
                                "- The caller found\n"
                                "- The failed test\n"
                                "\n"
                                "Let me write concisely:",
                            }
                        }
                    },
                    {
                        "text": "The bug: `retries_remaining()` currently adds `limit + attempts` instead of subtracting `limit - "
                        "attempts`, and doesn't prevent negative results. The caller in `src/worker.py` line 12 uses "
                        "`retries_remaining(3, attempts)` to track remaining retries, and the test fails because "
                        "`retries_remaining(3, 1)` returns 4 instead of the expected 2."
                    },
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1511,
            "outputTokens": 336,
            "totalTokens": 1847,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug: `retries_remaining()` currently adds `limit + attempts` instead of "
                    "subtracting `limit - attempts`, and doesn't prevent negative results. The caller "
                    "in `src/worker.py` line 12 uses `retries_remaining(3, attempts)` to track "
                    "remaining retries, and the test fails because `retries_remaining(3, 1)` returns 4 "
                    "instead of the expected 2.",
                    "role": "assistant",
                    "reasoning_content": "Let me analyze what I found:\n"
                    "\n"
                    "1. **The bug**: The current implementation is `return limit + attempts`, "
                    "but it should return `limit - attempts`. The test expects that with "
                    "limit=3 and attempts=1, there should be 2 retries remaining (3-1=2). "
                    "Currently it returns 4 (3+1=4).\n"
                    "\n"
                    "2. **The caller**: `src/worker.py` line 12 calls `retries_remaining(3, "
                    "attempts)` and stores the result in `remaining`.\n"
                    "\n"
                    "3. **The failed test**: The test `test_retries_remaining()` expects "
                    "`retries_remaining(3, 1) == 2` but the current implementation returns 4. "
                    "Additionally, the second assertion expects `retries_remaining(3, 5) == "
                    "0` (meaning when you've attempted more than the limit, it should return "
                    "0, not negative).\n"
                    "\n"
                    "The user wants a two-sentence summary of:\n"
                    "- The bug\n"
                    "- The caller found\n"
                    "- The failed test\n"
                    "\n"
                    "Let me write concisely:",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "Let me analyze what I found:\n"
                            "\n"
                            "1. **The bug**: The current implementation is `return limit "
                            "+ attempts`, but it should return `limit - attempts`. The "
                            "test expects that with limit=3 and attempts=1, there should "
                            "be 2 retries remaining (3-1=2). Currently it returns 4 "
                            "(3+1=4).\n"
                            "\n"
                            "2. **The caller**: `src/worker.py` line 12 calls "
                            "`retries_remaining(3, attempts)` and stores the result in "
                            "`remaining`.\n"
                            "\n"
                            "3. **The failed test**: The test `test_retries_remaining()` "
                            "expects `retries_remaining(3, 1) == 2` but the current "
                            "implementation returns 4. Additionally, the second assertion "
                            "expects `retries_remaining(3, 5) == 0` (meaning when you've "
                            "attempted more than the limit, it should return 0, not "
                            "negative).\n"
                            "\n"
                            "The user wants a two-sentence summary of:\n"
                            "- The bug\n"
                            "- The caller found\n"
                            "- The failed test\n"
                            "\n"
                            "Let me write concisely:",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "reasoningContentBlocks": [
                            {
                                "reasoningText": {
                                    "signature": "synthetic-signature",
                                    "text": "Let me "
                                    "analyze "
                                    "what I "
                                    "found:\n"
                                    "\n"
                                    "1. **The "
                                    "bug**: The "
                                    "current "
                                    "implementation "
                                    "is `return "
                                    "limit + "
                                    "attempts`, "
                                    "but it "
                                    "should "
                                    "return "
                                    "`limit - "
                                    "attempts`. "
                                    "The test "
                                    "expects "
                                    "that with "
                                    "limit=3 "
                                    "and "
                                    "attempts=1, "
                                    "there "
                                    "should be "
                                    "2 retries "
                                    "remaining "
                                    "(3-1=2). "
                                    "Currently "
                                    "it returns "
                                    "4 "
                                    "(3+1=4).\n"
                                    "\n"
                                    "2. **The "
                                    "caller**: "
                                    "`src/worker.py` "
                                    "line 12 "
                                    "calls "
                                    "`retries_remaining(3, "
                                    "attempts)` "
                                    "and stores "
                                    "the result "
                                    "in "
                                    "`remaining`.\n"
                                    "\n"
                                    "3. **The "
                                    "failed "
                                    "test**: "
                                    "The test "
                                    "`test_retries_remaining()` "
                                    "expects "
                                    "`retries_remaining(3, "
                                    "1) == 2` "
                                    "but the "
                                    "current "
                                    "implementation "
                                    "returns 4. "
                                    "Additionally, "
                                    "the second "
                                    "assertion "
                                    "expects "
                                    "`retries_remaining(3, "
                                    "5) == 0` "
                                    "(meaning "
                                    "when "
                                    "you've "
                                    "attempted "
                                    "more than "
                                    "the limit, "
                                    "it should "
                                    "return 0, "
                                    "not "
                                    "negative).\n"
                                    "\n"
                                    "The user "
                                    "wants a "
                                    "two-sentence "
                                    "summary "
                                    "of:\n"
                                    "- The bug\n"
                                    "- The "
                                    "caller "
                                    "found\n"
                                    "- The "
                                    "failed "
                                    "test\n"
                                    "\n"
                                    "Let me "
                                    "write "
                                    "concisely:",
                                }
                            }
                        ]
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 336,
            "prompt_tokens": 1511,
            "total_tokens": 1847,
            "completion_tokens_details": {"reasoning_tokens": 236, "text_tokens": 100},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1511,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
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
    expected_provider_endpoint="/model/global.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tool_choice": {"disable_parallel_tool_use": False},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                                "additionalProperties": False,
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxy", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "The user wants me to:\n1"}},
                "p": "ab",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": ". Search"}}, "p": "abcdefg"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " for other callers of retries_"}},
                "p": "abcdefghijklmnopqrstuv",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "remaining\n2. Run the"}}, "p": "ab"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " tests in"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " tests"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "/test_retry.py to see"}},
                "p": "abcdefghijklmnop",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " the"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " failure"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "\n\nLet me make"}}, "p": "abcdefg"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": " both"}}, "p": "abcdef"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " calls together since they're independent."}},
                "p": "abcdefghijklmnopqrstuvwxyzABCD",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": ""}}, "p": "abcde"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"signature": "synthetic-signature"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuv",
                "start": {
                    "toolUse": {
                        "name": "search_code",
                        "toolUseId": "tooluse_gjENiEmnd8v2Cwbb0CS4iu",
                        "type": "tool_use",
                    }
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ""}}, "p": "abcdefghij"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"query": '}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"retries_re'}}, "p": "ab"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "mainin"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": 'g"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456",
                "start": {
                    "toolUse": {"name": "run_tests", "toolUseId": "tooluse_i14f63N6Z3OKUAGDUZyNvd", "type": "tool_use"}
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": ""}}, "p": "abcdefghijkl"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "delta": {"toolUse": {"input": '{"path":'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": ' "tests/te'}}, "p": "abcdef"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": "st_retry.py"}}, "p": "abcdefghijklmnopq"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "delta": {"toolUse": {"input": '"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 54}}},
                "p": "abcdefghijklmnopqr",
                "stopReason": "tool_use",
            },
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 1288},
                "p": "abcdefghijklmnopqrstuvwxy",
                "usage": {"inputTokens": 1237, "outputTokens": 155, "serverToolUsage": {}, "totalTokens": 1392},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("tooluse_gjENiEmnd8v2Cwbb0CS4iu", "tooluse_i14f63N6Z3OKUAGDUZyNvd"),
        reasoning_content=(
            "The user wants me to:\n"
            "1. Search for other callers of retries_remaining\n"
            "2. Run the tests in tests/test_retry.py to see the failure\n"
            "\n"
            "Let me make both calls together since they're independent."
        ),
        thinking_blocks=(
            {
                "type": "thinking",
                "thinking": "The user wants me to:\n"
                "1. Search for other callers of retries_remaining\n"
                "2. Run the tests in tests/test_retry.py to see the failure\n"
                "\n"
                "Let me make both calls together since they're independent.",
                "signature": "synthetic-signature",
            },
        ),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1237,
        "completion_tokens": 155,
        "total_tokens": 1392,
        "completion_tokens_details": {"reasoning_tokens": 54, "text_tokens": 101},
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 1237,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
    },
)


CLAUDE_HAIKU_5_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.anthropic.claude-haiku-5-5"},
    expected_provider_endpoint="/model/global.anthropic.claude-haiku-5-5/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 24}}},
        "metrics": {"latencyMs": 2486},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": "synthetic-signature",
                                "text": "I'll search for the callers and run the tests together "
                                "now, then wait for results.\n"
                                "\n",
                            }
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_SdXtOgdBswzwk48AWnrJey",
                            "type": "tool_use",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_jYXgvkM3sX3RQTILr0zOtF",
                            "type": "tool_use",
                        }
                    },
                ],
            },
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1144,
            "outputTokens": 130,
            "totalTokens": 1274,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "tooluse_SdXtOgdBswzwk48AWnrJey",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "tooluse_jYXgvkM3sX3RQTILr0zOtF",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "I'll search for the callers and run the tests together now, then wait "
                    "for results.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I'll search for the callers and run the tests together now, "
                            "then wait for results.\n"
                            "\n",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "reasoningContentBlocks": [
                            {
                                "reasoningText": {
                                    "signature": "synthetic-signature",
                                    "text": "I'll "
                                    "search for "
                                    "the "
                                    "callers "
                                    "and run "
                                    "the tests "
                                    "together "
                                    "now, then "
                                    "wait for "
                                    "results.\n"
                                    "\n",
                                }
                            }
                        ]
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 130,
            "prompt_tokens": 1144,
            "total_tokens": 1274,
            "completion_tokens_details": {"reasoning_tokens": 24, "text_tokens": 106},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1144,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


CLAUDE_HAIKU_5_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_AGENT_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_HAIKU_5_5_AGENT_TEST_CASE.litellm_request, CLAUDE_HAIKU_5_5_AGENT_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.anthropic.claude-haiku-5-5/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "I'll search for the callers and run the tests "
                                "together now, then wait for results.\n"
                                "\n",
                                "signature": "synthetic-signature",
                            }
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_SdXtOgdBswzwk48AWnrJey",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_jYXgvkM3sX3RQTILr0zOtF",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "tooluse_SdXtOgdBswzwk48AWnrJey",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "tooluse_jYXgvkM3sX3RQTILr0zOtF",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
        "metrics": {"latencyMs": 1331},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "The bug is that `retries_remaining` returns `limit + attempts` instead of the retries left, so it "
                        "can go negative or grow when attempts exceed the limit. The only caller I found is "
                        "`src/worker.py` line 12 (`remaining = retries_remaining(3, attempts)`), and `tests/test_retry.py` "
                        "fails with `assert 4 == 2`, confirming the bug."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1477,
            "outputTokens": 126,
            "totalTokens": 1603,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` returns `limit + attempts` instead of the "
                    "retries left, so it can go negative or grow when attempts exceed the limit. The "
                    "only caller I found is `src/worker.py` line 12 (`remaining = retries_remaining(3, "
                    "attempts)`), and `tests/test_retry.py` fails with `assert 4 == 2`, confirming the "
                    "bug.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 126,
            "prompt_tokens": 1477,
            "total_tokens": 1603,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 126},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1477,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
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
    expected_provider_endpoint="/model/global.anthropic.claude-haiku-5-5/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghi", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "I"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "'ll run"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " the"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " search and the t"}},
                "p": "abcdefghijklmnopqrst",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "ests concurrently,"}},
                "p": "abcdefg",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " then wait for both"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " to finish before contin"}},
                "p": "abcdefghijklmnopqrstu",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "uing.\n\n"}}, "p": "abcdefghij"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"signature": "synthetic-signature"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrst"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrs",
                "start": {
                    "toolUse": {
                        "name": "search_code",
                        "toolUseId": "tooluse_vcdBZHtRE7Rcccknxsqz8v",
                        "type": "tool_use",
                    }
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ""}}, "p": "abcdefgh"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"query": "'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "retr"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "ies_"}}, "p": "abcdefghijklmnopqrstuvw"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": 'remaining"}'}}, "p": "ab"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "p": "abcdefghijklmnopqrstuvwxyzABCDE",
                "start": {
                    "toolUse": {"name": "run_tests", "toolUseId": "tooluse_RQ4zQ01K7BjSGttKrLDP3p", "type": "tool_use"}
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": ""}}, "p": "abc"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "delta": {"toolUse": {"input": '{"path": "t'}},
                "p": "abcdefghijklmnopqrstuvwxyzABC",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": "ests/test_re"}}, "p": "abcdefghijkl"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "delta": {"toolUse": {"input": 'try.py"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 23}}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY",
                "stopReason": "tool_use",
            },
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 2150},
                "p": "abcdefghijklmnop",
                "usage": {"inputTokens": 1144, "outputTokens": 129, "serverToolUsage": {}, "totalTokens": 1273},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("tooluse_vcdBZHtRE7Rcccknxsqz8v", "tooluse_RQ4zQ01K7BjSGttKrLDP3p"),
        reasoning_content=(
            "I'll run the search and the tests concurrently, then wait for both to finish before continuing.\n\n"
        ),
        thinking_blocks=(
            {
                "type": "thinking",
                "thinking": "I'll run the search and the tests concurrently, then wait for both to finish before "
                "continuing.\n"
                "\n",
                "signature": "synthetic-signature",
            },
        ),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1144,
        "completion_tokens": 129,
        "total_tokens": 1273,
        "completion_tokens_details": {"reasoning_tokens": 23, "text_tokens": 106},
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 1144,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
    },
)


CLAUDE_OPUS_5_5_AGENT_TEST_CASE: Final = replace(
    CLAUDE_OPUS_5_5_TEST_CASE,
    scenario="agent_tools",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.anthropic.claude-opus-5-5"},
    expected_provider_endpoint="/model/global.anthropic.claude-opus-5-5/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
        "metrics": {"latencyMs": 1770},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_nPKoHBRfRQFaSg6GFObaN1",
                            "type": "tool_use",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_wZSOklAhzSmLcXXCHzgA6Q",
                            "type": "tool_use",
                        }
                    },
                ],
            },
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1144,
            "outputTokens": 105,
            "totalTokens": 1249,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-opus-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "tooluse_nPKoHBRfRQFaSg6GFObaN1",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "tooluse_wZSOklAhzSmLcXXCHzgA6Q",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1144,
            "total_tokens": 1249,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1144,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
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
    expected_provider_endpoint="/model/global.anthropic.claude-opus-5-5/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_nPKoHBRfRQFaSg6GFObaN1",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_wZSOklAhzSmLcXXCHzgA6Q",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "tooluse_nPKoHBRfRQFaSg6GFObaN1",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "tooluse_wZSOklAhzSmLcXXCHzgA6Q",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 19}}},
        "metrics": {"latencyMs": 2612},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": "synthetic-signature",
                                "text": "The user is asking me to stop and not use any more tools, "
                                "so I should respect that and wrap up without further "
                                "action.\n"
                                "\n",
                            }
                        }
                    },
                    {
                        "text": "The bug is that `retries_remaining` returns `limit + attempts` when it should return `max(limit - "
                        "attempts, 0)`. The only other caller is `src/worker.py` line 12 (`retries_remaining(3, "
                        "attempts)`), and `tests/test_retry.py::test_retries_remaining` fails with `assert 4 == 2`; I "
                        "haven't changed any files."
                    },
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1452,
            "outputTokens": 142,
            "totalTokens": 1594,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-opus-5-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` returns `limit + attempts` when it should "
                    "return `max(limit - attempts, 0)`. The only other caller is `src/worker.py` line "
                    "12 (`retries_remaining(3, attempts)`), and "
                    "`tests/test_retry.py::test_retries_remaining` fails with `assert 4 == 2`; I "
                    "haven't changed any files.",
                    "role": "assistant",
                    "reasoning_content": "The user is asking me to stop and not use any more tools, so I should "
                    "respect that and wrap up without further action.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user is asking me to stop and not use any more tools, so "
                            "I should respect that and wrap up without further action.\n"
                            "\n",
                            "signature": "synthetic-signature",
                        }
                    ],
                    "provider_specific_fields": {
                        "reasoningContentBlocks": [
                            {
                                "reasoningText": {
                                    "signature": "synthetic-signature",
                                    "text": "The user "
                                    "is asking "
                                    "me to stop "
                                    "and not "
                                    "use any "
                                    "more "
                                    "tools, so "
                                    "I should "
                                    "respect "
                                    "that and "
                                    "wrap up "
                                    "without "
                                    "further "
                                    "action.\n"
                                    "\n",
                                }
                            }
                        ]
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 142,
            "prompt_tokens": 1452,
            "total_tokens": 1594,
            "completion_tokens_details": {"reasoning_tokens": 19, "text_tokens": 123},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1452,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
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
    expected_provider_endpoint="/model/global.anthropic.claude-opus-5-5/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdef", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR",
                "start": {
                    "toolUse": {
                        "name": "search_code",
                        "toolUseId": "tooluse_KDUbO8FtozzPrRkLBLseP8",
                        "type": "tool_use",
                    }
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": ""}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"qu'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": 'ery"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEF",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": ': "r'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "etries_rem"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "ainin"}}, "p": "abcdefghijklmnopqrstuvwxyzA"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": 'g"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abc"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR",
                "start": {
                    "toolUse": {"name": "run_tests", "toolUseId": "tooluse_6Y9Zbb8KWmGmufNdN2AMiT", "type": "tool_use"}
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ""}}, "p": "abcdefghijkl"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"path'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '":'}}, "p": "abcdefghijklmnopqrstuvwxyzABCD"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ' "tests/tes'}}, "p": "abcdefghijklmnopqrst"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": 't_retry.py"}'}}, "p": "abcd"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGH",
                "stopReason": "tool_use",
            },
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 1944},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ",
                "usage": {"inputTokens": 1144, "outputTokens": 105, "serverToolUsage": {}, "totalTokens": 1249},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("tooluse_KDUbO8FtozzPrRkLBLseP8", "tooluse_6Y9Zbb8KWmGmufNdN2AMiT"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1144,
        "completion_tokens": 105,
        "total_tokens": 1249,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 1144,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
    },
)


CLAUDE_OPUS_5_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    CLAUDE_OPUS_5_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        CLAUDE_OPUS_5_TEST_CASE.litellm_request, CLAUDE_OPUS_5_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.anthropic.claude-opus-5/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_J2NI1YJRdC0KDZibAcgZhF",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_64n9QInquvlU5vhkAleMpJ",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "tooluse_J2NI1YJRdC0KDZibAcgZhF",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "tooluse_64n9QInquvlU5vhkAleMpJ",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
        "metrics": {"latencyMs": 2845},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "**Bug:** `retries_remaining` adds `limit + attempts` instead of subtracting, so "
                        "`retries_remaining(3, 1)` returns 4 instead of 2, and it also lacks a floor at 0 for `attempts > "
                        "limit`; the fix would be `max(0, limit - attempts)`.\n"
                        "\n"
                        "**Caller & test:** The only caller is `src/worker.py:12` (`remaining = retries_remaining(3, "
                        "attempts)`), and `tests/test_retry.py::test_retries_remaining` currently fails with `assert 4 == "
                        "2`. No files were changed."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1446,
            "outputTokens": 190,
            "totalTokens": 1636,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-opus-5",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "**Bug:** `retries_remaining` adds `limit + attempts` instead of subtracting, so "
                    "`retries_remaining(3, 1)` returns 4 instead of 2, and it also lacks a floor at 0 "
                    "for `attempts > limit`; the fix would be `max(0, limit - attempts)`.\n"
                    "\n"
                    "**Caller & test:** The only caller is `src/worker.py:12` (`remaining = "
                    "retries_remaining(3, attempts)`), and "
                    "`tests/test_retry.py::test_retries_remaining` currently fails with `assert 4 == "
                    "2`. No files were changed.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 190,
            "prompt_tokens": 1446,
            "total_tokens": 1636,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 190},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1446,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
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
    expected_provider_endpoint="/model/global.anthropic.claude-opus-5/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "tool_choice": {"disable_parallel_tool_use": False},
            "output_config": {"effort": "low"},
        },
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzA", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEF",
                "start": {
                    "toolUse": {
                        "name": "search_code",
                        "toolUseId": "tooluse_xkzgC2ZDeF1Pt8XMhFk9zc",
                        "type": "tool_use",
                    }
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": ""}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"que'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": 'ry"'}}, "p": "abcdefghijklmnopqrstuv"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": ': "re'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "tries_re"}}, "p": "abcdefghij"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "maining"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuvw",
                "start": {
                    "toolUse": {"name": "run_tests", "toolUseId": "tooluse_H5ziLCR1MOfF30pTWKLjGx", "type": "tool_use"}
                },
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": ""}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"path": "t'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "ests/t"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "est_retry."}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": 'py"}'}}, "p": "abcdefghijklmnopqrs"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefg"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
                "p": "abcde",
                "stopReason": "tool_use",
            },
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 1654},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012",
                "usage": {"inputTokens": 1140, "outputTokens": 105, "serverToolUsage": {}, "totalTokens": 1245},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("tooluse_xkzgC2ZDeF1Pt8XMhFk9zc", "tooluse_H5ziLCR1MOfF30pTWKLjGx"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 1140,
        "completion_tokens": 105,
        "total_tokens": 1245,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 1140,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
    },
)


GPT_5_6_LUNA_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_6_LUNA_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_6_LUNA_TEST_CASE.litellm_request, GPT_5_6_LUNA_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.openai.gpt-5.6-luna/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "call_ddecd7a963c55f68a8f60049c783bd95",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "call_63fb22f6ea7459e1988a9e41a01c01c4",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "call_ddecd7a963c55f68a8f60049c783bd95",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "call_63fb22f6ea7459e1988a9e41a01c01c4",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 927},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "`retries_remaining` incorrectly adds `limit` and `attempts`, returning 4 instead of 2 and "
                        "potentially negative values are not clamped. It is called by `src/worker.py` line 12, and "
                        "`tests/test_retry.py` failed with `assert 4 == 2`."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 619,
            "outputTokens": 66,
            "totalTokens": 685,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-5.6-luna",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds `limit` and `attempts`, returning 4 instead "
                    "of 2 and potentially negative values are not clamped. It is called by "
                    "`src/worker.py` line 12, and `tests/test_retry.py` failed with `assert 4 == 2`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 66,
            "prompt_tokens": 619,
            "total_tokens": 685,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 66},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 619,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_5_6_LUNA_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_5_6_LUNA_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_5_6_LUNA_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_5_6_LUNA_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/model/global.openai.gpt-5.6-luna/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABC", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"redactedContent": "synthetic-redacted-content"}},
                "p": "abcd",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEF",
                "start": {"toolUse": {"name": "search_code", "toolUseId": "call_56e00189845b5c3e9db58f1c533f8ffa"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{"}}, "p": "abcdefghijklmnopqrst"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"query'}}, "p": "abcdef"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '":"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "re"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "tries"}}, "p": "abcdefghijklmno"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "_remaining"}}, "p": "abcdefghijklmnopq"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"}'}}, "p": "abcdefghijklmnopqrst"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghij"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "start": {"toolUse": {"name": "run_tests", "toolUseId": "call_5bc228109f6b50aab59208a1bb87a278"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "delta": {"toolUse": {"input": "{"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": '"path'}}, "p": "abcdefghij"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "delta": {"toolUse": {"input": '":"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCD",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": "tests"}}, "p": "abcdefghijklmnopqrs"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "delta": {"toolUse": {"input": "/test"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": "_retry"}}, "p": "abcdefg"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 2,
                "delta": {"toolUse": {"input": ".py"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "delta": {"toolUse": {"input": '"}'}}, "p": "abcdef"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 2, "p": "abcdefghijklmnopqrstuvwxyz"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"p": "abcdefghijkl", "stopReason": "tool_use"},
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 1012},
                "p": "abcdefghijklmnopq",
                "usage": {"inputTokens": 448, "outputTokens": 65, "serverToolUsage": {}, "totalTokens": 513},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_56e00189845b5c3e9db58f1c533f8ffa", "call_5bc228109f6b50aab59208a1bb87a278"),
        thinking_blocks=({"type": "redacted_thinking", "data": "synthetic-redacted-content"},),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 65,
        "total_tokens": 513,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 448,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
        "completion_tokens_details": {},
    },
)


GPT_5_6_SOL_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_6_SOL_TEST_CASE.litellm_request, GPT_5_6_SOL_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.openai.gpt-5.6-sol/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "call_c1270df77e20598fb15ecb00e80d568f",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "call_809ae84cbd005e8284ec3168c6e1e29f",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "call_c1270df77e20598fb15ecb00e80d568f",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "call_809ae84cbd005e8284ec3168c6e1e29f",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 1465},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "The bug is that `retries_remaining` adds `attempts` to `limit` instead of returning `max(limit - "
                        "attempts, 0)`. The caller is `src/worker.py:12`, and `tests/test_retry.py` failed because "
                        "`retries_remaining(3, 1)` returned `4` instead of `2`."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 619,
            "outputTokens": 74,
            "totalTokens": 693,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-5.6-sol",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The bug is that `retries_remaining` adds `attempts` to `limit` instead of "
                    "returning `max(limit - attempts, 0)`. The caller is `src/worker.py:12`, and "
                    "`tests/test_retry.py` failed because `retries_remaining(3, 1)` returned `4` "
                    "instead of `2`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 74,
            "prompt_tokens": 619,
            "total_tokens": 693,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 74},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 619,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_5_6_SOL_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_5_6_SOL_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_5_6_SOL_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_5_6_SOL_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/model/global.openai.gpt-5.6-sol/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234",
                "start": {"toolUse": {"name": "search_code", "toolUseId": "call_1f91223677b85208a50c2e468f13bec6"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{"}}, "p": "abcdefghijklm"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"query'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEF",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '":"'}}, "p": "abcde"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "re"}}, "p": "abcdefghijklmnopqrstuvwxyzABCD"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "tries"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "_remaining"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijkl"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR",
                "start": {"toolUse": {"name": "run_tests", "toolUseId": "call_22c1b20e86fc5591a38991235c16a738"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "{"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"path'}}, "p": "abcdef"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '":"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "tests"}}, "p": "abcdefghijklmn"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "/test"}}, "p": "abcdefghijklmno"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "_retry"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ".py"}}, "p": "abcdefghijklmnopqrstuvwxy"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"p": "abcdefghijklm", "stopReason": "tool_use"},
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 922},
                "p": "abcdefghijkl",
                "usage": {"inputTokens": 448, "outputTokens": 53, "serverToolUsage": {}, "totalTokens": 501},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_1f91223677b85208a50c2e468f13bec6", "call_22c1b20e86fc5591a38991235c16a738"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 448,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
    },
)


GPT_5_6_TERRA_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_5_6_TERRA_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_5_6_TERRA_TEST_CASE.litellm_request, GPT_5_6_TERRA_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.openai.gpt-5.6-terra/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "call_4de9aceae32a5b3788b78f6cba38a447",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "call_423b772b9bd65ff7ac64ee75cb54f8a8",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "call_4de9aceae32a5b3788b78f6cba38a447",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "call_423b772b9bd65ff7ac64ee75cb54f8a8",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 5208},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "`retries_remaining` incorrectly adds `limit + attempts`; it should subtract attempts from the "
                        "limit and clamp the result at zero. The only caller found is `src/worker.py`, and "
                        "`tests/test_retry.py` fails because `retries_remaining(3, 1)` returns `4` instead of `2`."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 619,
            "outputTokens": 69,
            "totalTokens": 688,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-5.6-terra",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds `limit + attempts`; it should subtract "
                    "attempts from the limit and clamp the result at zero. The only caller found is "
                    "`src/worker.py`, and `tests/test_retry.py` fails because `retries_remaining(3, 1)` "
                    "returns `4` instead of `2`.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 69,
            "prompt_tokens": 619,
            "total_tokens": 688,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 69},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 619,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_5_6_TERRA_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_5_6_TERRA_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_5_6_TERRA_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_5_6_TERRA_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/model/global.openai.gpt-5.6-terra/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
                "start": {"toolUse": {"name": "search_code", "toolUseId": "call_0134ef3fc4655cf1bce3682ad64e2402"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "{"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '"query'}}, "p": "abcdefghijklmnopqrstuvwxy"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '":"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEF",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "re"}}, "p": "abcdefghijklmnopqrstuvwxyz"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "tries"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "_remaining"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '"}'}}, "p": "abcdefghijklm"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01",
                "start": {"toolUse": {"name": "run_tests", "toolUseId": "call_158da03b42155f8698a5e305286b575a"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "{"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '"path'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '":"'}}, "p": "ab"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "tests"}}, "p": "abcdefgh"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "/test"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "_retry"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ".py"}}, "p": "abcdefghijklmn"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"}'}}, "p": "abcdefghijklmnopqrstuvwxyz"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvwxyzAB"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012", "stopReason": "tool_use"},
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 2187},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ",
                "usage": {"inputTokens": 448, "outputTokens": 53, "serverToolUsage": {}, "totalTokens": 501},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_0134ef3fc4655cf1bce3682ad64e2402", "call_158da03b42155f8698a5e305286b575a"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 448,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
    },
)


GPT_6_ASTRA_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_6_ASTRA_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_6_ASTRA_TEST_CASE.litellm_request, GPT_6_ASTRA_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.openai.gpt-6-astra/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "call_008a8775094550c79dbbc084eb264469",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "call_da04451fac2455238ac6a0baeb96040b",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "call_008a8775094550c79dbbc084eb264469",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "call_da04451fac2455238ac6a0baeb96040b",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 1686},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "`retries_remaining` adds attempts to the limit instead of subtracting and clamping at zero; a "
                        "caller was found in `src/worker.py:12`. `tests/test_retry.py` failed because "
                        "`retries_remaining(3, 1)` returned 4 instead of 2; no files were changed."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 619,
            "outputTokens": 68,
            "totalTokens": 687,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-6-astra",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds attempts to the limit instead of subtracting and clamping "
                    "at zero; a caller was found in `src/worker.py:12`. `tests/test_retry.py` failed "
                    "because `retries_remaining(3, 1)` returned 4 instead of 2; no files were changed.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 68,
            "prompt_tokens": 619,
            "total_tokens": 687,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 68},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 619,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_6_ASTRA_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_6_ASTRA_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_6_ASTRA_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_6_ASTRA_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/model/global.openai.gpt-6-astra/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX",
                "start": {"toolUse": {"name": "search_code", "toolUseId": "call_8d938f5c31c25860ac7afbfabc6842e7"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "{"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"query'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '":"'}}, "p": "abcdefghijklmnopqrstu"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "re"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDE",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "tries"}}, "p": "abcdefghijklmnopqrstuvwxyz"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "_remaining"}}, "p": "abcdefghijklmnopqrs"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrst"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
                "start": {"toolUse": {"name": "run_tests", "toolUseId": "call_50e1bcf30f0f530999ba77cb7a9af6e1"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{"}}, "p": "abcdefghijklmnop"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '"path'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '":"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGH",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "tests"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "/test"}},
                "p": "abcdefghijklmnopqrstuvwxyzABC",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "_retry"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ".py"}}, "p": "abcdefghijklmnopq"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"}'}}, "p": "ab"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"p": "abcdefghijkl", "stopReason": "tool_use"},
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 1338},
                "p": "abcdefghijklmno",
                "usage": {"inputTokens": 448, "outputTokens": 53, "serverToolUsage": {}, "totalTokens": 501},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_8d938f5c31c25860ac7afbfabc6842e7", "call_50e1bcf30f0f530999ba77cb7a9af6e1"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 448,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
    },
)


GPT_6_LUNA_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_6_LUNA_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_6_LUNA_TEST_CASE.litellm_request, GPT_6_LUNA_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.openai.gpt-6-luna/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "call_0abaf8f8430151c49b5347a36cea178c",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "call_928195190d5b5d5b80a1448958d7f6b0",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "call_0abaf8f8430151c49b5347a36cea178c",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "call_928195190d5b5d5b80a1448958d7f6b0",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 837},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "`retries_remaining` adds the attempt count to the limit, so it returns incorrect values and can "
                        "be negative only if given negative inputs. The other caller is `src/worker.py`; "
                        "`tests/test_retry.py` failed because it expected 2 but got 4."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 619,
            "outputTokens": 59,
            "totalTokens": 678,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-6-luna",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds the attempt count to the limit, so it returns incorrect "
                    "values and can be negative only if given negative inputs. The other caller is "
                    "`src/worker.py`; `tests/test_retry.py` failed because it expected 2 but got 4.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 59,
            "prompt_tokens": 619,
            "total_tokens": 678,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 59},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 619,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_6_LUNA_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_6_LUNA_TEST_CASE.litellm_endpoint,
    litellm_request={**GPT_6_LUNA_TEST_CASE.litellm_request, "stream": True, "stream_options": {"include_usage": True}},
    expected_provider_headers=GPT_6_LUNA_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/model/global.openai.gpt-6-luna/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvwxy",
                "start": {"toolUse": {"name": "search_code", "toolUseId": "call_79fad04d4dc15725a47b94036fad121d"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{"}}, "p": "abcdefghijklmnopqr"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"query'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGH",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '":"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "re"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "tries"}}, "p": "abcdefgh"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "_remaining"}},
                "p": "abcdefghijklmnopqrstuvwxy",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '"}'}}, "p": "abcdefghijklmnopqrstu"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmnopq",
                "start": {"toolUse": {"name": "run_tests", "toolUseId": "call_5998b5b7890a5492b44686f49fdce50a"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "{"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '"path'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '":"'}}, "p": "abcdefghijklmnopqrst"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "tests"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "/test"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "_retry"}}, "p": "abcdefghijklmnopqr"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": ".py"}}, "p": "abcd"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"}'}}, "p": "abcdefghijklmnopqrstuvwxy"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY", "stopReason": "tool_use"},
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 696},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEF",
                "usage": {"inputTokens": 448, "outputTokens": 53, "serverToolUsage": {}, "totalTokens": 501},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_79fad04d4dc15725a47b94036fad121d", "call_5998b5b7890a5492b44686f49fdce50a"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 448,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
    },
)


GPT_6_SOL_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_6_SOL_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_6_SOL_TEST_CASE.litellm_request, GPT_6_SOL_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.openai.gpt-6-sol/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "call_432bebf6455e56e9aca3ca7d332b3b72",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "call_dff8cbdda3cf5ab6b65587a730cb39a1",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "call_432bebf6455e56e9aca3ca7d332b3b72",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "call_dff8cbdda3cf5ab6b65587a730cb39a1",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 954},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "`retries_remaining` adds attempts to the limit instead of subtracting them and clamping the "
                        "result to zero. `src/worker.py` calls it; `tests/test_retry.py` failed because "
                        "`retries_remaining(3, 1)` returned 4 instead of 2."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 619,
            "outputTokens": 62,
            "totalTokens": 681,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-6-sol",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` adds attempts to the limit instead of subtracting them and "
                    "clamping the result to zero. `src/worker.py` calls it; `tests/test_retry.py` "
                    "failed because `retries_remaining(3, 1)` returned 4 instead of 2.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 62,
            "prompt_tokens": 619,
            "total_tokens": 681,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 62},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 619,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_6_SOL_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_6_SOL_TEST_CASE.litellm_endpoint,
    litellm_request={**GPT_6_SOL_TEST_CASE.litellm_request, "stream": True, "stream_options": {"include_usage": True}},
    expected_provider_headers=GPT_6_SOL_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/model/global.openai.gpt-6-sol/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghi", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmn",
                "start": {"toolUse": {"name": "search_code", "toolUseId": "call_27bae639df025a45b45f9d6f390cf640"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{"}}, "p": "abcdefghijklmnop"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"query'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '":"'}}, "p": "abcdefghijklmnopq"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "re"}}, "p": "abcd"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "tries"}}, "p": "abcde"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "_remaining"}}, "p": "ab"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJK",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijkl"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmno",
                "start": {"toolUse": {"name": "run_tests", "toolUseId": "call_e1ff6a882dec5541999379a8e5ce6b4d"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "{"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"path'}}, "p": "abcdefg"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '":"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "tests"}}, "p": "abcdefghijklmnopqrstuvwx"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "/test"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "_retry"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVW",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": ".py"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVW",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"}'}}, "p": "abcdefghij"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuv"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"p": "abcdef", "stopReason": "tool_use"},
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 876},
                "p": "abcdefghijklmnopqrstuvwxyzABCDE",
                "usage": {"inputTokens": 448, "outputTokens": 53, "serverToolUsage": {}, "totalTokens": 501},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_27bae639df025a45b45f9d6f390cf640", "call_e1ff6a882dec5541999379a8e5ce6b4d"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 448,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
    },
)


GPT_6_1_SOL_AGENT_FOLLOWUP_TEST_CASE: Final = replace(
    GPT_6_1_SOL_TEST_CASE,
    scenario="agent_followup",
    litellm_request=followup_request(
        GPT_6_1_SOL_TEST_CASE.litellm_request, GPT_6_1_SOL_TEST_CASE.expected_litellm_response
    ),
    expected_provider_endpoint="/model/global.openai.gpt-6.1-sol/converse",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "messages": [
            *list_value(BEDROCK_CONVERSE_AGENT_REQUEST["messages"]),
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "call_e119dbd374765056bb1886401168fcdd",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "call_26ef40995a505036ac6917a11eb0eff4",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"matches": [{"path": "src/worker.py", "line": 12, '
                                    '"text": "remaining = retries_remaining(3, attempts)"}]}'
                                }
                            ],
                            "toolUseId": "call_e119dbd374765056bb1886401168fcdd",
                        }
                    },
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": '{"exit_code": 1, "output": "FAILED '
                                    "tests/test_retry.py::test_retries_remaining - assert 4 == "
                                    '2"}'
                                }
                            ],
                            "toolUseId": "call_26ef40995a505036ac6917a11eb0eff4",
                        }
                    },
                    {
                        "text": "Stop after these diagnostics. Do not call more tools or change files. Summarize the "
                        "bug, the caller found and the failed test in two short sentences."
                    },
                ],
            },
        ],
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 2347},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "text": "`retries_remaining` incorrectly adds attempts to the limit; it should subtract attempts and clamp "
                        "the result to zero, and a caller was found in `src/worker.py:12`. `tests/test_retry.py` failed "
                        "because `retries_remaining(3, 1)` returned 4 instead of 2; no files were changed."
                    }
                ],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 619,
            "outputTokens": 72,
            "totalTokens": 691,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-6.1-sol",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "`retries_remaining` incorrectly adds attempts to the limit; it should subtract "
                    "attempts and clamp the result to zero, and a caller was found in "
                    "`src/worker.py:12`. `tests/test_retry.py` failed because `retries_remaining(3, 1)` "
                    "returned 4 instead of 2; no files were changed.",
                    "role": "assistant",
                },
            }
        ],
        "usage": {
            "completion_tokens": 72,
            "prompt_tokens": 619,
            "total_tokens": 691,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 72},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 619,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_6_1_SOL_AGENT_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="agent_streaming",
    litellm_endpoint=GPT_6_1_SOL_TEST_CASE.litellm_endpoint,
    litellm_request={
        **GPT_6_1_SOL_TEST_CASE.litellm_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    expected_provider_headers=GPT_6_1_SOL_TEST_CASE.expected_provider_headers,
    expected_provider_endpoint="/model/global.openai.gpt-6.1-sol/converse-stream",
    expected_provider_request={
        **BEDROCK_CONVERSE_AGENT_REQUEST,
        "additionalModelRequestFields": {"reasoning": {"effort": "low"}},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "read_file",
                        "description": "Read a UTF-8 file in the repository.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            }
                        },
                        "name": "search_code",
                        "description": "Search repository source for a literal query.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                        },
                        "name": "run_tests",
                        "description": "Run a specific pytest test path and return the exit code and output.",
                    }
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {"patch": {"type": "string"}},
                                "required": ["patch"],
                            }
                        },
                        "name": "apply_patch",
                        "description": "Apply a unified diff to repository files.",
                    }
                },
            ],
            "toolChoice": {"auto": {}},
        },
    },
    mock_provider_frames=(
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDE", "role": "assistant"},
            headers={":event-type": "messageStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "p": "abcdefghijklmnopqrstuvw",
                "start": {"toolUse": {"name": "search_code", "toolUseId": "call_cfd44aa11ede5a6c99cf18ae774b7085"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{"}}, "p": "abcdefghijklmnopqrstu"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '"query'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDE",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '":"'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "re"}}, "p": "abcdefg"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": "tries"}}, "p": "abcdefghijklmnopqrst"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "_remaining"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "delta": {"toolUse": {"input": '"}'}}, "p": "abcdefghijklmno"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 0, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "p": "abcdefghijklmno",
                "start": {"toolUse": {"name": "run_tests", "toolUseId": "call_027b2f22e5e959a38f6861130f2dc7df"}},
            },
            headers={":event-type": "contentBlockStart", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{"}}, "p": "abcdefgh"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '"path'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": '":"'}}, "p": "abcdefghijklmnopqrstuvwxyzA"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "tests"}}, "p": "abcdefghijklmnopqrstuv"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "/test"}}, "p": "abcdefghij"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "delta": {"toolUse": {"input": "_retry"}}, "p": "abcdefghijklmnopq"},
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": ".py"}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '"}'}},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            },
            headers={":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"contentBlockIndex": 1, "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO"},
            headers={":event-type": "contentBlockStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={"p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", "stopReason": "tool_use"},
            headers={":event-type": "messageStop", ":content-type": "application/json", ":message-type": "event"},
        ),
        EventStreamFrame(
            data={
                "metrics": {"latencyMs": 1909},
                "p": "abcdefghijklmnopqrstuvwxyzABCDEFGH",
                "usage": {"inputTokens": 448, "outputTokens": 53, "serverToolUsage": {}, "totalTokens": 501},
            },
            headers={":event-type": "metadata", ":content-type": "application/json", ":message-type": "event"},
        ),
    ),
    expected_litellm_message=coding_agent_stream_message(
        ("call_cfd44aa11ede5a6c99cf18ae774b7085", "call_027b2f22e5e959a38f6861130f2dc7df"),
    ),
    expected_litellm_usage={
        "prompt_tokens": 448,
        "completion_tokens": 53,
        "total_tokens": 501,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": 448,
            "cache_write_tokens": 0,
            "cache_creation_tokens": 0,
        },
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
    },
)
