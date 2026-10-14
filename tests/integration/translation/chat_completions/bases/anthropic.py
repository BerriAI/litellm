from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "anthropic/claude-haiku-4-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_011CfgDzBnuPYc7TdysZFRpo",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 17,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "anthropic/claude-haiku-4-5",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "not_available",
            "service_tier": "standard",
        },
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "anthropic/claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_011CfgDzH3AcaSNXtvurLgAG",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "anthropic/claude-sonnet-4-6",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "anthropic/claude-sonnet-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_011CfgDzPkj2HxBq8qGJsGSi",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "anthropic/claude-sonnet-5",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "anthropic/claude-opus-4-8",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_011CfgDzV9vZCkuQJt15n3Z1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "anthropic/claude-opus-4-8",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQStwcKEAgSGAI4AUIIdGhpbmtpbmcSDEA+6GPyhIL+cHBM6hoMpwYXoehOPK1mBGhtIjDguiYzlMx1jPECHgGFpqkHbtNtffMogyxpYwsE"
    "AbRP/nNhq8gn/9IQtkz5VyDzUpYq1AZOf4t+yfOs73ydOUCSs4Kr9htKGVZzuBYckMoNxXxAoGXJWWyYlnDK9Fn2o1L/ds8fJIEzLXnTeaSc"
    "+IxU50dASha+OYgkyDAW7nzO3kr17gd/uqgaLylQqgiyUkkHksVWGqtYoP8jrzuS6Oip/p1GPR8AtATJv61sDowDgtcFm0ZS504aJpma6bdj"
    "+8zNN7DGRrHmbZAcaii0lxH2vAigoE3kp7vCfRamt6p2Fhss7b7LbJSkMVZk6icUVLuQ6dNod7ZAtlys9yMdxAR99+NLtzptZrPIGIYFo+eb"
    "qVPtEvpvqQHRUqfBSthrLQbszj1X3EEN1Mu3IawYAht2qi1uE+jr/pjH7dYetQvdc6uFFdVTwwqK9jFz47hN4Ms6u28x4GZppQxcCdkn9KKO"
    "m5FBzaZULJbnRmp7h5Ag+VytuPCrbGAG5pZyOm1QXh+FpI53ncKc/g+rPlqxUhmenObVHchfO4tjBySyHHKQi8IVApZHwoA7UPqavTOCGk69"
    "AVDFCczNPCiNfZHOHlrzMYAQj88oCLfuCYoNFyfhYt1Yz5RucQDaFeAInIRpcg2GQim4ikJQ+/x7bNJF6H1ugJcqensOZt2FuQKHu5FHWFkq"
    "kYY4Gcx4mIzwgPELWOIIgsBeBBqCwkNeiBSreXj5Fr9rBs5qBIK0DVZFLUph+bv974EKzuc5Gx/ymgK2Ib77kUVmaGX84r5orIbk7Bpvi306"
    "CltdCjIiZ1vRZaFamZJ2bKi1B87uP9D8IbCA0ZqE8hhpBgEMOP95Rs8I+1a4PococAG5pEoY7SVbw2ebZJZwHLGZlJj8eIg/5Bpn7K39tswy"
    "dTvkACOA+ZWbNFtw7sveDy5ZqnjUSyp9a88l0FsmDxoHnY//0s7Fw6mJTHgIzP5ACUjAXxvZWlKDRzv+igwbRqgv4r4qAi0YqQwyLuzjIZTH"
    "9xtk7GayN7rBH+5P6Z3c9e9eveEnFQ45td1floAAS1tQR/LU3ClTF4J8+nN9Z226dQy8H4jC0w4Inqkog1BbIWopB11v2lwLh0nQMmKJXw5a"
    "vODORLlugGHR6FIjeF2BWObz53UL3jA8mBPJLNP8ANqp/fHfcnDyI8zGJ0yNb0Irc2LziceECjlC0uEhPUp4b7gYAQ=="
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "anthropic/claude-sonnet-5-5",
        "cache": {"no-cache": True},
        "max_tokens": 1024,
        "messages": [
            {
                "role": "system",
                "content": "You help maintain a synthetic Python project. Read the relevant files "
                "before suggesting a fix. Explain which file you need next, then call a "
                "read-only tool. Never change files.",
            },
            {
                "role": "user",
                "content": "The checkout rejects quantities of zero. Find the validation code and its tests.",
            },
            {
                "role": "assistant",
                "content": "I will locate the quantity validation and its tests.",
                "tool_calls": [
                    {
                        "id": "toolu_synthetic_search_01",
                        "type": "function",
                        "function": {
                            "name": "search_files",
                            "arguments": '{"query": "quantity", "paths": ["shop/", "tests/"]}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_synthetic_search_01",
                "content": '{"matches": [{"path": "shop/validation.py", "line": 18, "text": "def '
                'validate_quantity(quantity):"}, {"path": "tests/test_validation.py", '
                '"line": 7, "text": "def test_zero_quantity():"}]}',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Read the validation file first so we can understand how zero and "
                        "negative quantities are handled.",
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search source and test files for a literal term.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "paths": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["query", "paths"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a range of lines from a project file.",
                    "parameters": {
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
            },
        ],
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-5-5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The checkout rejects quantities of zero. Find the validation code and its tests.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will locate the quantity validation and its tests."},
                    {
                        "type": "tool_use",
                        "id": "toolu_synthetic_search_01",
                        "name": "search_files",
                        "input": {"query": "quantity", "paths": ["shop/", "tests/"]},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_synthetic_search_01",
                        "content": '{"matches": [{"path": "shop/validation.py", "line": 18, '
                        '"text": "def validate_quantity(quantity):"}, {"path": '
                        '"tests/test_validation.py", "line": 7, "text": "def '
                        'test_zero_quantity():"}]}',
                    },
                    {
                        "type": "text",
                        "text": "Read the validation file first so we can understand how zero and "
                        "negative quantities are handled.",
                    },
                ],
            },
        ],
        "max_tokens": 1024,
        "tools": [
            {
                "name": "search_files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "type": "custom",
                "description": "Search source and test files for a literal term.",
            },
            {
                "name": "read_file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "type": "custom",
                "description": "Read a range of lines from a project file.",
            },
        ],
        "system": [
            {
                "type": "text",
                "text": "You help maintain a synthetic Python project. Read the relevant files before "
                "suggesting a fix. Explain which file you need next, then call a read-only tool. "
                "Never change files.",
            }
        ],
    },
    mock_provider_response={
        "model": "claude-sonnet-5-5",
        "id": "msg_011Cg2zmsQqPndxqGrGgYWL8",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0},
            {
                "type": "text",
                "text": "I need `shop/validation.py` first. `validate_quantity` is defined at line 18, "
                "so I'll read the area around it to see how zero and negative values are "
                "handled.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01TJriSK9oYsS8yhSm9ThjDE",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
                "caller": {"type": "direct"},
            },
        ],
        "container": None,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 838,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 250,
            "output_tokens_details": {"thinking_tokens": 99},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "anthropic/claude-sonnet-5-5",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I need `shop/validation.py` first. `validate_quantity` is "
                    "defined at line 18, so I'll read the area around it to see how "
                    "zero and negative values are handled.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                                "name": "read_file",
                            },
                            "id": "toolu_01TJriSK9oYsS8yhSm9ThjDE",
                            "type": "function",
                        }
                    ],
                    "reasoning_content": "",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0}
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0}
                        ],
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 250,
            "prompt_tokens": 838,
            "total_tokens": 1088,
            "completion_tokens_details": {"reasoning_tokens": 99, "text_tokens": 151},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 838,
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

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "anthropic/claude-opus-5-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-opus-5-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-opus-5-5",
        "id": "msg_011CfgDzp3URQ38dAAc8F3RP",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 23,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "anthropic/claude-opus-5-5",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 23,
            "total_tokens": 29,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 23,
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
