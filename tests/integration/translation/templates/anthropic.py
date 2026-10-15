from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

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

ANTHROPIC_MESSAGES_REQUEST: Final[Mapping[str, JsonValue]] = {
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
}

ANTHROPIC_MESSAGES_RESPONSE: Final[Mapping[str, JsonValue]] = {
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
}
