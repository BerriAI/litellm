from collections.abc import Mapping
from typing import Final
from unittest.mock import ANY

from integration.translation.templates.anthropic import CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0
from pydantic import JsonValue

CHAT_COMPLETIONS_REQUEST: Final[Mapping[str, JsonValue]] = {
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
}

CHAT_COMPLETIONS_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "id": ANY,
    "created": ANY,
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
}
