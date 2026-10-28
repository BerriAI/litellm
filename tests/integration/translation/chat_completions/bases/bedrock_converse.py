from typing import Final
from unittest.mock import ANY

from integration._support.client import object_value
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import (
    CHAT_COMPLETIONS_REQUEST,
    CHAT_COMPLETIONS_RESPONSE,
    CODING_AGENT_REQUEST,
    CODING_AGENT_RESPONSE,
)
from integration.translation.templates.bedrock_converse import (
    BEDROCK_CONVERSE_AGENT_REQUEST,
    BEDROCK_CONVERSE_AGENT_RESPONSE,
    BEDROCK_CONVERSE_REQUEST,
    BEDROCK_CONVERSE_RESPONSE,
    CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
)

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 771},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 17,
            "outputTokens": 5,
            "serverToolUsage": {},
            "totalTokens": 22,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 17,
            "total_tokens": 22,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 17,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


CLAUDE_FABLE_5_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.anthropic.claude-fable-5"},
    expected_provider_endpoint="/model/global.anthropic.claude-fable-5/converse",
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
        "metrics": {"latencyMs": 2256},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_8F5NPCLa3Auzb12CgAn28Q",
                            "type": "tool_use",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_0GdPjDJbx5w7EzbCQi49uO",
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
        "model": "bedrock/converse/global.anthropic.claude-fable-5",
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
                            "id": "tooluse_8F5NPCLa3Auzb12CgAn28Q",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "tooluse_0GdPjDJbx5w7EzbCQi49uO",
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


CLAUDE_OPUS_5_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.anthropic.claude-opus-5"},
    expected_provider_endpoint="/model/global.anthropic.claude-opus-5/converse",
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
        "metrics": {"latencyMs": 1929},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
                "content": [
                    {
                        "toolUse": {
                            "input": {"query": "retries_remaining"},
                            "name": "search_code",
                            "toolUseId": "tooluse_J2NI1YJRdC0KDZibAcgZhF",
                            "type": "tool_use",
                        }
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_retry.py"},
                            "name": "run_tests",
                            "toolUseId": "tooluse_64n9QInquvlU5vhkAleMpJ",
                            "type": "tool_use",
                        }
                    },
                ],
            },
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 1140,
            "outputTokens": 105,
            "totalTokens": 1245,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.anthropic.claude-opus-5",
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
                            "id": "tooluse_J2NI1YJRdC0KDZibAcgZhF",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "tooluse_64n9QInquvlU5vhkAleMpJ",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1140,
            "total_tokens": 1245,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1140,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.openai.gpt-5.6-luna"},
    expected_provider_endpoint="/model/global.openai.gpt-5.6-luna/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
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
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 911},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
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
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 448,
            "outputTokens": 53,
            "totalTokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-5.6-luna",
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
                            "id": "call_ddecd7a963c55f68a8f60049c783bd95",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_63fb22f6ea7459e1988a9e41a01c01c4",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 448,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.openai.gpt-5.6-sol"},
    expected_provider_endpoint="/model/global.openai.gpt-5.6-sol/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
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
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 1065},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
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
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 448,
            "outputTokens": 53,
            "totalTokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-5.6-sol",
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
                            "id": "call_c1270df77e20598fb15ecb00e80d568f",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_809ae84cbd005e8284ec3168c6e1e29f",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 448,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_5_6_TERRA_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.openai.gpt-5.6-terra"},
    expected_provider_endpoint="/model/global.openai.gpt-5.6-terra/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
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
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 2192},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
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
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 448,
            "outputTokens": 53,
            "totalTokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-5.6-terra",
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
                            "id": "call_4de9aceae32a5b3788b78f6cba38a447",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_423b772b9bd65ff7ac64ee75cb54f8a8",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 448,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_6_ASTRA_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.openai.gpt-6-astra"},
    expected_provider_endpoint="/model/global.openai.gpt-6-astra/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
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
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 1399},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
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
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 448,
            "outputTokens": 53,
            "totalTokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-6-astra",
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
                            "id": "call_008a8775094550c79dbbc084eb264469",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_da04451fac2455238ac6a0baeb96040b",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 448,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.openai.gpt-6-luna"},
    expected_provider_endpoint="/model/global.openai.gpt-6-luna/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
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
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 789},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
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
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 448,
            "outputTokens": 53,
            "totalTokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-6-luna",
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
                            "id": "call_0abaf8f8430151c49b5347a36cea178c",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_928195190d5b5d5b80a1448958d7f6b0",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 448,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.openai.gpt-6-sol"},
    expected_provider_endpoint="/model/global.openai.gpt-6-sol/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
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
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 841},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
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
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 448,
            "outputTokens": 53,
            "totalTokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-6-sol",
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
                            "id": "call_432bebf6455e56e9aca3ca7d332b3b72",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_dff8cbdda3cf5ab6b65587a730cb39a1",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 448,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "bedrock/converse/global.openai.gpt-6.1-sol"},
    expected_provider_endpoint="/model/global.openai.gpt-6.1-sol/converse",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-agent-key"},
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
    mock_provider_response={
        **BEDROCK_CONVERSE_AGENT_RESPONSE,
        "metrics": {"latencyMs": 1981},
        "output": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"]),
            "message": {
                **object_value(object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["output"])["message"]),
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
        },
        "stopReason": "tool_use",
        "usage": {
            **object_value(BEDROCK_CONVERSE_AGENT_RESPONSE["usage"]),
            "inputTokens": 448,
            "outputTokens": 53,
            "totalTokens": 501,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "bedrock/converse/global.openai.gpt-6.1-sol",
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
                            "id": "call_e119dbd374765056bb1886401168fcdd",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_26ef40995a505036ac6917a11eb0eff4",
                            "type": "function",
                        },
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 53,
            "prompt_tokens": 448,
            "total_tokens": 501,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 53},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 448,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-4-6/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 997},
        "output": {"message": {"content": [{"text": "Hello!"}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 18,
            "outputTokens": 5,
            "serverToolUsage": {},
            "totalTokens": 23,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello!", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 18,
            "total_tokens": 23,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 18,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 1201},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 21,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 27,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 21,
            "total_tokens": 27,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 21,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-opus-4-8/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 827},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 21,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 27,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 21,
            "total_tokens": 27,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 21,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock/converse/us.anthropic.claude-haiku-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={**BEDROCK_CONVERSE_REQUEST},
    mock_provider_response={**BEDROCK_CONVERSE_RESPONSE},
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock/converse/us.anthropic.claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` next, starting with the `validate_quantity` function "
                    "found at line 18.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "tooluse_qWzJMc6h58NlspiEa01FUC",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "I need to figure out which file to read for validation, then use a read-only tool to "
                    "look at it—I'll check the first 60 lines or so since I don't know its exact "
                    "length.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to figure out which file to read for validation, then use a read-only "
                            "tool to look at it—I'll check the first 60 lines or so since I don't know "
                            "its exact length.\n\n",
                            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "reasoningContentBlocks": [
                            {
                                "reasoningText": {
                                    "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                                    "text": "I need to figure out which file to read for validation, then use a "
                                    "read-only tool to look at it—I'll check the first 60 lines or so "
                                    "since I don't know its exact length.\n\n",
                                },
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 230,
            "prompt_tokens": 860,
            "total_tokens": 1090,
            "completion_tokens_details": {"reasoning_tokens": 96, "text_tokens": 134},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 856,
                "cache_creation_tokens": 856,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 856, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 856,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQS9QUKEggSEAEYAjgBQgh0aGlua2luZxIMHeQumEXUSngowYp2Ggw6RETkY6n11eC2W98iMLxQa5+C/eZvHir2PVIhfArk/wZFsCSOH7e3"
    "dqBUFaBA8tvcE0LO+g3l1B51aj/2bCqQBbsVVKGduphL2uGxAmAfucVEigmAzobLXjnxMizAS4+QFSE4iS/vbgj9oH/JgaZu1izkc77Y8vtC"
    "IzTWqlSeyu8ThzVgljSrPMO76qbBKJPM3N/3VYKrV2iK1OljKXeJMZXuEFTtUJm562vm1sihud8t1EnG+Q849jjXts3Y4sB5tfDewbFG5LsY"
    "tclTSq468Kv0fw7vTEGQ1silrl1hMvMc65ZD1o7YAwYurdDaTc1JJbEYLVNtHgwFK4G1G4vdWIHuk9DKNvCeOxFoGhfyYwGbDMX9SqYaHQh3"
    "rcX5I/b5JHf2bE2l/5pyiBirAXjlG9EhBOYCdypDGXzlDqr4Oc4xaRWYjv8ixXhov9FHEUj0sdB3YTMxnEaUn4KgvdBSTS+O89wOMcoZBz8T"
    "1mR3ig3IxUV27YS6ZSYImaoie9VzuWLNovu1x3rFyrM2cP222Z9NahtcOIXysUVSoe1nYYAYa4K5A7+BXf2xWhUZjfyzG/mGP/LsQCZeXxwY"
    "1e0m3ivyUanPUxVrnK1nvpB9FcY7T8UPw2LH2W8VWDXG1MPdjPwZcQJsdc/LxwSmWUv4jwa/jEBOvyKC9GTJRn2TDiUNhzFoLn/wSzdfk5aV"
    "rwJd0z5ALiaR9yXm5qgGmK0v1slPo8VmgzoFzLGmkU4UE0uT3etr9IfiR9wAyctSMuKuZ+ApETINbZWOybLaJynLYdaoGYrCwTDB2p3JTQV5"
    "HP8aQm6l11YgxxymNnSMHEXmYibBp2B1YH/2TkysI01LMnxBGo79aa1KnzYHSZQpbmqSDmjn6SR3KhjwYxDz7drr8S49a5QPxsYoaffWTrwj"
    "0BbxGgQuwjB0pbIQFWiK+eOA+lShN3Sm+8WrdtcwPUbmGAE="
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={**BEDROCK_CONVERSE_REQUEST},
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 86}}},
        "metrics": {"latencyMs": 3449},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                                "text": "The user asked me to read the validation file first, so I'll do that "
                                "before looking at the tests, even though reading both in parallel would be "
                                "fine too. I'll pull in the full validation file since it's short, roughly "
                                "18 lines.\n\n",
                            },
                        },
                    },
                    {
                        "text": "I'll read `shop/validation.py` first. `validate_quantity` is defined at line 18, "
                        "so I'm reading the top of the file through the function and what follows it.",
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                            "name": "read_file",
                            "toolUseId": "tooluse_rN6y4cQORe8ESxDQcIl9SR",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 856, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 856,
            "cacheWriteInputTokens": 856,
            "inputTokens": 4,
            "outputTokens": 237,
            "serverToolUsage": {},
            "totalTokens": 1097,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` first. `validate_quantity` is defined at line 18, so "
                    "I'm reading the top of the file through the function and what follows it.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "tooluse_rN6y4cQORe8ESxDQcIl9SR",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "The user asked me to read the validation file first, so I'll do that before looking at "
                    "the tests, even though reading both in parallel would be fine too. I'll pull in the "
                    "full validation file since it's short, roughly 18 lines.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user asked me to read the validation file first, so I'll do that before "
                            "looking at the tests, even though reading both in parallel would be fine too. "
                            "I'll pull in the full validation file since it's short, roughly 18 lines.\n\n",
                            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "reasoningContentBlocks": [
                            {
                                "reasoningText": {
                                    "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                                    "text": "The user asked me to read the validation file first, so I'll do that "
                                    "before looking at the tests, even though reading both in parallel "
                                    "would be fine too. I'll pull in the full validation file since it's "
                                    "short, roughly 18 lines.\n\n",
                                },
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 237,
            "prompt_tokens": 860,
            "total_tokens": 1097,
            "completion_tokens_details": {"reasoning_tokens": 86, "text_tokens": 151},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 856,
                "cache_creation_tokens": 856,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 856, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 856,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSwwUKEggSEAEYAjgBQgh0aGlua2luZxIMeqMdLtmn33y9cIexGgym8fsZQt5P/lSqgpQiMAC92xJoxWLNXHHgGhS+0mdadsD4aDidh2B3"
    "XDq8VlyD6g3Heynf1faRQ72efXv4zCreBGbwmU6DTP7IlfHZXPnTXAo979PeGnhoChdOtJ6GKSs/w9rjsAee+Eys02cjJerLDxhf44lHr2iW"
    "bxhJQNft2DnosDk3OCMtWx77ksDERl4mcnJplzLJSMTCPO9HSbTkJzWJ5Jj7x0DjznblKBes8q1NPFVV7pqXMB7yo2mPlledoEzt+rYXXiFa"
    "NBpuSQ8wvflyON6rwl0UoToTGpUb8rjvjqVkInhX82Y3hIbYgixYzhA9OCh+xMuDQzldRLdM5NgT+9dxHrklL1ZaHuWrpBP12q7qo0j3A7dT"
    "/2/JAvGiJqfCcGptfc2RPDK/NgnTxBAbqvUW0VUm8zhg7SDHMC+hunwy4phUc8yBpaJFI5vmthSU++vvyVwNYChs6i0DVdMe6c7Lxzzk3W0Z"
    "iAMrgsalHyjK6IzsnwMusiMW64foeMN0OKx3lYSZT9oqp6nDVecBwgH4npD2Pq4PPe7C5S5KVCzvq4guVTCcQM7XUXdck6QzgZrDm/kFp3lz"
    "splN7GdOh+PUYAvGvmhnBvPkWKFRPSEcX6vPlbNfhC+ADY8C/oVjjVFjvyBv/aF4BDlY8s4ZV76kmiiyfXQ8P+BpO2QtId1GvlmOq8SmwcAe"
    "cMgtz4o0nWWNAaTV8CH+YjswCSe6lscs5bTZzf8724OKBx0lOs+Axn0Lwo5gHekzCimCUpA8leG0dIg6t6FOVRmjcmIoTlh39Xu2LmjrTjNG"
    "A8BBS3u+O+R2158XIfII393jsgpVwg8Dn5WIAK8U1LzQDAj7IrhSuN1q2UJG8ZEsi6JmFmWdGXdCi1FVWNkyDBgB"
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock/converse/us.anthropic.claude-opus-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-opus-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={**BEDROCK_CONVERSE_REQUEST},
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 65}}},
        "metrics": {"latencyMs": 3552},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                                "text": "Since the user asked to look at the validation file first, I'll focus on "
                                "that around line 18 before touching the tests, which can wait since "
                                "they're independent anyway.\n\n",
                            },
                        },
                    },
                    {
                        "text": "Next I'll open `shop/validation.py` around `validate_quantity` (line 18). That "
                        "should show how zero, negative and non-integer values are handled. After that I'll "
                        "read the tests.",
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
                            "name": "read_file",
                            "toolUseId": "tooluse_zGYCbMhgYTTAQl5dBHNkKs",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 856, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 856,
            "cacheWriteInputTokens": 856,
            "inputTokens": 4,
            "outputTokens": 223,
            "serverToolUsage": {},
            "totalTokens": 1083,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock/converse/us.anthropic.claude-opus-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "Next I'll open `shop/validation.py` around `validate_quantity` (line 18). That should "
                    "show how zero, negative and non-integer values are handled. After that I'll read the "
                    "tests.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                                "name": "read_file",
                            },
                            "id": "tooluse_zGYCbMhgYTTAQl5dBHNkKs",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "Since the user asked to look at the validation file first, I'll focus on that around "
                    "line 18 before touching the tests, which can wait since they're independent anyway.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "Since the user asked to look at the validation file first, I'll focus on that "
                            "around line 18 before touching the tests, which can wait since they're "
                            "independent anyway.\n\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "reasoningContentBlocks": [
                            {
                                "reasoningText": {
                                    "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                                    "text": "Since the user asked to look at the validation file first, I'll focus "
                                    "on that around line 18 before touching the tests, which can wait since "
                                    "they're independent anyway.\n\n",
                                },
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 223,
            "prompt_tokens": 860,
            "total_tokens": 1083,
            "completion_tokens_details": {"reasoning_tokens": 65, "text_tokens": 158},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 856,
                "cache_creation_tokens": 856,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 856, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 856,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0: Final = (
    "CAQSrwUKEwgSEAEYAjgBQgluYXJyYXRpb24SDNCGJ8kKyGwJSh36gxoM2/14CeUIPZHnkx/iIjAYx/Gy8bDJhXdDJVKLzNi0l/VKFEq/9uTz"
    "mI1Vx1GUXRI6Yj7imDIn3cnUzf5dj6AqyQQhWjFSAqAjMK8gOQ2PqoyPonbTrxwsld+dZM44c+ffNp2YgBSEQyNUlXF+nCYX6kAGwmzSZkKq"
    "W8CGWrIH2dB+TzWfz3fsx/GglprifZ0YFvyoMq+Ri6LQnS/GFuqtaJrYhwngV34u1M1jFJv3qw+HvTGmiums0KACTSP8F+vI3OfedHfhaiyz"
    "pQnbFie2gqVJOvRNst521omILp8ds5QrBoq2VYnwt7TZEO1IfNfan8OAcetSMUvAYCrW2I3IiZ7lTpMTV2K2Dzyw5NLY7LNrwA/bJuKNbyWI"
    "JvFi8zrgIxLAsLNXV0gCLib7qdywHg8k/otUlOSOfHzLaDuwO0BR8OpSjJcm38lUpCg+6twP49FC943vhOJ3t9qR3PSeCEkLdzFEZGUak1Xr"
    "sqwK/dfU8R22lGelQK7xwVNGKB6JOQ5pbIA8gUviT8DDQ9nQAaj+HKSYiN3ss3+qD3qPMuBK/XYojp5rbIrpqfoyWogvX9hwWCBFAYC9OUHR"
    "199q8fAFQxSRVUKSjCPopCQC9AKouLmeeotr0AI6KKB0deg1MudLkgy6KcKvEgptDqOqJJMpyO+3WxJ5serIbwNgIAbs3Kl9rLsJrmZ3qzfW"
    "qw0/yxJGjwLFRLAy2o6Q+6y5kAQA0CsURIjbdvIF7hpHkvlAYdzfX+ApwQY115nAUTkYyliu8myjDBCRXb2mEnAKk7+rQ/tr67/szVV/8B8S"
    "3CmxaoFlhHHbwCAHpGZNWPFAOa8hvOPxsewJ+Hr5z34kC4g2QdDdiyPRANAYAQ=="
)

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock/converse/us.anthropic.claude-fable-5-1"},
    expected_provider_endpoint="/model/us.anthropic.claude-fable-5-1/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "paths": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["query", "paths"],
                                "additionalProperties": False,
                            },
                        },
                        "name": "search_files",
                        "description": "Search source and test files for a literal term.",
                    },
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "start_line": {"type": "integer", "minimum": 1},
                                    "end_line": {"type": "integer", "minimum": 1},
                                },
                                "required": ["path", "start_line", "end_line"],
                                "additionalProperties": False,
                            },
                        },
                        "name": "read_file",
                        "description": "Read a range of lines from a project file.",
                    },
                },
            ],
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
        "metrics": {"latencyMs": 5476},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                                "text": "Checking `shop/validation.py` to see how `validate_quantity` handles zero, "
                                "negative, and non-integer inputs, and pulling in the test file at the same "
                                "time.\n\n",
                            },
                        },
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                            "name": "read_file",
                            "toolUseId": "tooluse_u1AisurVMo0RZh6mAjHMl2",
                            "type": "tool_use",
                        },
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
                            "name": "read_file",
                            "toolUseId": "tooluse_M4eCGwAM39nEOXlNrXyTx5",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 876, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 876,
            "cacheWriteInputTokens": 876,
            "inputTokens": 4,
            "outputTokens": 267,
            "serverToolUsage": {},
            "totalTokens": 1147,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock/converse/us.anthropic.claude-fable-5-1",
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
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "tooluse_u1AisurVMo0RZh6mAjHMl2",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 40}',
                                "name": "read_file",
                            },
                            "id": "tooluse_M4eCGwAM39nEOXlNrXyTx5",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "Checking `shop/validation.py` to see how `validate_quantity` handles zero, negative, "
                    "and non-integer inputs, and pulling in the test file at the same time.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "Checking `shop/validation.py` to see how `validate_quantity` handles zero, "
                            "negative, and non-integer inputs, and pulling in the test file at the same "
                            "time.\n\n",
                            "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "reasoningContentBlocks": [
                            {
                                "reasoningText": {
                                    "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                                    "text": "Checking `shop/validation.py` to see how `validate_quantity` handles "
                                    "zero, negative, and non-integer inputs, and pulling in the test file "
                                    "at the same time.\n\n",
                                },
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 267,
            "prompt_tokens": 880,
            "total_tokens": 1147,
            "completion_tokens_details": {},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
        },
    },
)
