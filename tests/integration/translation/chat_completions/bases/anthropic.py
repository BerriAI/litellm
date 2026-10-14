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
    "CAQS6AYKEAgSGAI4AUIIdGhpbmtpbmcSDMgHDEwvhCWxzhI8xhoMRhO9WFMe66gmXinaIjCsd5QyXV2yye1/gHsrjZY0DUgMwyexaUDOaW82"
    "n43MT0YfpNtKB9eUXLsKedHUC7wqhQaR1+SDqqxoRnT1IqD+0uCRc5ynqXyo5VzPWKVKK6XgabxnRy/oGPw1MPT2cMJbSE+nwGu31Dr8rv58"
    "f+ewsNYvFmZ+Pw1H151zoMnrAiSsww7G4XVaMRty5AEQKccViiHRxzNhTlpQhEOZeupQi2c3qfeBBD2zWoww4FQwhQg2SyKZdTuN7vuTnAVW"
    "1h9exG2ahsBVEbnJf8Ukp9lEi1YTtays1AzAlnHNj/TTwQvuA0QNRR+652loFhbpkySm1GRlXcgD812rRf1qQqLNGVLzefRQt3k1L3eE2+WB"
    "q8vqUBdJkZDxGlK0qSuA4kfacnIihyHO7vfuY/Gg7KAlTX7oDpvV9YbA8ZiFk4C5I/uaNJpeN69S93gutvcJ07YcwtPtHIpCTATQJLY9AQoc"
    "gAYacjAGGNGhnzk0v4sOvCcutx1RnXcozOQ6ukG5NNivEK7pW7c+wUpVsTAjH1GPNZHvDmt4FRvLFgwYP9mxrjoD7WTJhd1VdiMZjXxAQVER"
    "zFmumgOYZ7/VT+yXTgslC7dgy8fzw3AqgRI+3438BuUNpV7tVf33r7GvnvdTHdK1KGxPaDtacmcxvX8MtX94poRHUsyIAggvH7fu0rCuhzBd"
    "/plM9iTizSn3y2mKIGfRm7D9w/B7LwkaPJCHDUSKyOThOGvk4X6XypC45BZbi10mc/nMuLZIDiJ9WWlnxzxON+VLiANxxa3neQ7FSHKPNv45"
    "KWW7UzZep+BG3NSp+KPmx2bt4DBj+76HGyrQZs/f6uPBApsoXvnORXGYl945OQbiueIR/Grf2hDKVBijncal/bley82JMptO0N7UI+4aO2Bu"
    "OyDuYDy9ysFvd6XahDm6X5/W/tVjCG2TidsKRRSo2hQ9aXRHOjye/8k0SuMdNOJdSg7KQcMqGk2WLxCYFO2BMSftkKW1HmeKQHytB940klJH"
    "d9fiq+Yel82rctzObh6pfS3GX7Wax4u//VhERceNh7lrSrailAQVPb3OOu0GR5IcCbiPlPA9/eyVpHyZdwNjiClpSxgB"
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
                "read-only tool. Never change files. Use the chat_completions client for "
                "this project review. Consider zero, negative and non-integer quantities "
                "before choosing the next file to inspect.",
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
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
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
        "reasoning_effort": "high",
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
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
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
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
        "system": [
            {
                "type": "text",
                "text": "You help maintain a synthetic Python project. Read the relevant files before "
                "suggesting a fix. Explain which file you need next, then call a read-only tool. "
                "Never change files. Use the chat_completions client for this project review. "
                "Consider zero, negative and non-integer quantities before choosing the next file "
                "to inspect.",
            }
        ],
    },
    mock_provider_response={
        "model": "claude-sonnet-5-5",
        "id": "msg_011Cg32pxq64L8VQGo1JKTnH",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll follow the user's request and read the validation file first, focusing "
                "on a reasonable range around line 18.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I need `shop/validation.py` first. It holds `validate_quantity`, which decides "
                "how zero, negative and non-integer values are treated. I'll read the area "
                "around line 18.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01NdDgeoKnYQe6YXxzc3XDmF",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
        ],
        "container": None,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 235,
            "output_tokens_details": {"thinking_tokens": 80},
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
                    "content": "I need `shop/validation.py` first. It holds `validate_quantity`, "
                    "which decides how zero, negative and non-integer values are "
                    "treated. I'll read the area around line 18.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_01NdDgeoKnYQe6YXxzc3XDmF",
                            "type": "function",
                        }
                    ],
                    "reasoning_content": "I'll follow the user's request and read the validation "
                    "file first, focusing on a reasonable range around line "
                    "18.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I'll follow the user's request and read "
                            "the validation file first, focusing on a "
                            "reasonable range around line 18.\n"
                            "\n",
                            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I'll follow "
                                "the user's "
                                "request and "
                                "read the "
                                "validation "
                                "file first, "
                                "focusing on a "
                                "reasonable "
                                "range around "
                                "line 18.\n"
                                "\n",
                                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                            }
                        ],
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 235,
            "prompt_tokens": 880,
            "total_tokens": 1115,
            "completion_tokens_details": {"reasoning_tokens": 80, "text_tokens": 155},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
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
