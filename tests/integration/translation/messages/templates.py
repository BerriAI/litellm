from collections.abc import Mapping
from typing import Final

from integration.translation.templates.anthropic import ANTHROPIC_MESSAGES_RESPONSE
from pydantic import JsonValue

MESSAGES_REQUEST: Final[Mapping[str, JsonValue]] = {
    "cache": {"no-cache": True},
    "max_tokens": 1024,
    "system": "You help maintain a synthetic Python project. Read the relevant files before suggesting a "
    "fix. Explain which file you need next, then call a read-only tool. Never change files. Use "
    "the messages client for this project review. Consider zero, negative and non-integer "
    "quantities before choosing the next file to inspect.",
    "messages": [
        {
            "role": "user",
            "content": "The checkout rejects quantities of zero. Find the validation code and its tests.",
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
    "tools": [
        {
            "name": "search_files",
            "description": "Search source and test files for a literal term.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query", "paths"],
                "additionalProperties": False,
            },
        },
        {
            "name": "read_file",
            "description": "Read a range of lines from a project file.",
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
        },
    ],
    "thinking": {"type": "adaptive", "display": "summarized"},
    "output_config": {"effort": "high"},
}

MESSAGES_RESPONSE: Final[Mapping[str, JsonValue]] = ANTHROPIC_MESSAGES_RESPONSE
