from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQShwcKEggSEAEYAjgBQgh0aGlua2luZxIMtEbP1UTgPIIXYE/jGgzmQCWS7ivK08RdGfwiMLQLZcccUrbxiBMQBmSloh21MAnvornAsyEF"
    "qdKjYweOzk9gu7R5cyTNk5NT0z00kSqiBmieg1Sbhy23Upls1PPy2fC8794rus5qb4NKBmFvnYPn3255F5xzBh4tucc9xoUcjHkA9Q9SwGEK"
    "VlBdgSfoIK5gvUMVz/VRPsvoBPfUJbbvEDQ9b2qhauTLOn/zxOwYfUAfKPPQQ/pQGf1Z5+n3GjbvzqcQll8uHra3BZyaJMmWZ2nzUbJTFP6F"
    "Jkw95ArvOiyKeBZ65EVLskXAe2ZICsTd1pLESIfxuyeV4Yogt7drR5ytyOn/u1cM67Q0EWuZUQoU3OerB+K3PNxUp74ckJeK4M07IqAf3Bro"
    "O7+oN5zWv7w/JfbtX9kDTSLRIu3eT/fmyGPrpgho2YIqcD4CO5pckkcLqhdi58RQ+Jnto9xZecoEdloi7sY6s4NYsa07X1qnOEQW0loS8Vs5"
    "kcTpR+kl+KQJpk8pTf9GEqmtQGRo83KE4HKsHnd9pkdpex7thJIpxtgqR8OFY1Urjbj5YKY5mIO3FrsR2PtYK2nxgEzo3t0Ut74xj8ssIR7b"
    "PoYxD66+mAZ10WODU4+o7uCm12FbRK/fPeV1uIfLHC5yUC4hw5Wi7mVdaEXjCPc0cwVnUFKpTKzVfH1fL3zNUermhufeqnCoLGhEOzQ5KJdX"
    "Be9jAmOPd/Si5O5aZURC+dp8NXrVR1yDKhsWFg3CRi8DZ4wvgKNkmW+XYVB6G99gjHk7GmzcMSy5sxCWFs3qdKRbFyJ97WleW5ljuCz2YyFW"
    "uDu3QYqWPOu0UlgzQcf+TDsLML4T8O6ddMqrErhmk52N1HW8rDHlEedB6+LlN2C+fcSEdLzFAdmmmQ8Y/HBmAzSzvsS9CYldAe9MmMKAr+J2"
    "XBQBjJfbTvCYptNDuvqIChTBan1FII1w6Xq+NqaPR4f+16LM3HYKtIvwULH05YBRuZxF7B5J3NIKQBb39Rkhv2D/PTaMmdNIrDJqd57AKxSd"
    "EKuGc9BFdyzwFz+X/3k/kO0Z6VZeLLIN5vdiz/9Aj1eYa3CnjVLiGH6zVUnhkx6eetUuHRZnrBx0ZXzRhZWf88bAQHgH9LwoTo61cf2Zqv4a"
    "H5GvCCyi2gyhwWmBQ/7DeTUYAQ=="
)

BEDROCK_INVOKE_REQUEST: Final[Mapping[str, JsonValue]] = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The checkout rejects quantities of zero. Find the validation code and its tests.",
                },
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
                    "content": '{"matches": [{"path": "shop/validation.py", "line": 18, "text": "def '
                    'validate_quantity(quantity):"}, {"path": "tests/test_validation.py", "line": 7, '
                    '"text": "def test_zero_quantity():"}]}',
                },
                {
                    "type": "text",
                    "text": "Read the validation file first so we can understand how zero and negative quantities "
                    "are handled.",
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
                "properties": {"query": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
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
            "text": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "chat_completions client for this project review. Consider zero, negative and non-integer "
            "quantities before choosing the next file to inspect.",
        },
    ],
    "anthropic_version": "bedrock-2023-05-31",
}

BEDROCK_INVOKE_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "model": "claude-haiku-5-5",
    "id": "msg_bdrk_pobluozqaykrnd3exk5nnrtwrxipiquzg7hvvispjxn6qi4sz5vq",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "thinking": "I should look at shop/validation.py, focusing on how it handles zero, negative, and "
            "non-integer inputs before deciding what to read next. I'll check the first 60 lines or so "
            "since I don't know the file's full length yet.\n\n",
            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
        },
        {
            "type": "text",
            "text": "I'll read `shop/validation.py` around the `validate_quantity` definition at line 18. I'm "
            "checking how it treats zero, negative, and non-integer values.",
        },
        {
            "type": "tool_use",
            "id": "toolu_bdrk_01Hfy4YfUU7hZUxEmpp4oDYV",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
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
        "output_tokens": 295,
        "output_tokens_details": {"thinking_tokens": 143},
        "service_tier": "standard",
    },
}
