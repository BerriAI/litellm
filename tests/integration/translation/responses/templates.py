import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import ANY

from pydantic import JsonValue

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSxQoKEAgSGAI4AUIIdGhpbmtpbmcSDJdM5+jcpq2kA3N39xoMeIIhaZ+bur3WodGvIjBSU6GZokT+noSpw7KkJ/YVl9U2kcKaRVBk89gg"
    "A9sOJ3s5dwOaPRzfGM9eBlPjMloq4gm8QOkNSzqF4i54b4JhCzRhnQcadRp9p+fVIKrJA2rwqWdSlKKK2kkAkzugndx0wBEhdkxrsbkTDTS2"
    "JbXw8xCTdWwP5rJ7Gwd2jVY2GmI60plZW+WcJbLJx4nL/B2rpNWb6xdriK8kKSIovInuhya0hdr8K1/l4+5M72LnFO/S0zFRcBjL3EF7kApi"
    "Hr7BwXSsSUJJn6BFsqyN9AkDgiWrdvFU6UzsrkIEEPy+/Mi0a5utQyP5PDGqZfHt3cOeULtfR71DNdt6H7/56g1hRWVig0rrhzRM7N0a3ebP"
    "uWtEAbPynqr9OL2IuEwYHFD+UIgUdSyMi8nGwP9vBsMu5UpJ8/79XD3CTth9tzLFRngwCPynkDPsCLqy+iiSRMBFEH5qEZja4PI1Xa1swLzb"
    "QRwEFbURPUcGt2P3KSJxMFqpithxIFUlKNawx5DGLOjt6IH4nB4fpSrR/sjynF5RsjD2fWmioB10f2qqWlgqO0nuzYxxO7ES0HFIcutLd9X7"
    "u8cQMHZ+FSz3M/IeZhwQloLg4SYi3adMgzLKX/81Re/ZmSgQCrNFPppSAWERaXlzcnreD9FoHHiHVPyRgq1kZffq2YOq7bJfNWfjIBpQsW6S"
    "2lmFwH3a2iiA3Jphq5Xvkshpsek+A+6+MrgfPOONA+Hz8WygBDZIEteGRYFTtwNJWTAUlo8fwZnbNWFXq+TyNotKVz1fYrRQSjtR6pKulO9I"
    "yzeG3d2UQalbgGpaZQsl2LumVFdXLz9r6DhVUnyzp9cTcMDahLjf0NSAewmp0WNU3vo4j6c7YXsV/Ddje9rXwkKXO8/cmxrDQRJowyNfRtkM"
    "XiEMRHLikS+Wdd+T0UbEpBQMF0qjn8aWMfV3adRt3TrMU6xsaKAP0nKXvzz3gXt4UwRTqVuOA25a/dQBMvuvtifuQI0iyC+rl7tSV6vh2Qrj"
    "96N7KyNdniQUNKKN/s8QXuPZ+nsxFDnC7hy5vZguZnc97LOFXVltUmUyIa7rHZCFu3Dx16m6t2BdCZZCaL3Z2VglRrYAsUj+0klVoYfJAzsr"
    "Fz9VR1y1froAdvB5K+waOAdcSyIG4NxArvXqlNHChqLgSMStGj3eLrtiddboZwNl7TFLd5BJrFBLiEwXjckhsi4nYqLoyrn/F5i3UwqzHV+O"
    "xJpKIoLjI+FT0ZappHmIHRHFl5cqJHeR0lv/m2A5jgVD8Pvy6toTLkubrpUw1AwVmwmyQCC/xmT1CNYqSiZFdNU0EfyD+sy09t9ORy7fr+2c"
    "aPZ3qEQcHLCWB5P2vowrKDLzQZt8WL8uCqhMh/XTkw24VGmLaoGyY5uCew6L8eu80x0DKEfaRJRiB64QyDCC2Qkor1yCypqiNNMLI/TenuLI"
    "ke83mM7YpHLteXjyjKnxhn7MrfnKO5U590/SH/Kr8mYciXsK51LDcbQYjVo/A2adHAO6d0+H4Vgn6I3fKlnQnQOsPxQApRtqDlZs5hneb4Qb"
    "OApYtKzJtZQSyS8yUmAASSpR4Jo/yo6KAtw9TTtmhVs6dWlVidny/yd78X7J0u2ZZYJ4DipDz5WcDeDKc1jasWtFStL3Z39K/X3avQ2X9N6V"
    "df+93hUKRBxCvCjT3PDiMv4vcfIjFzIjGA03QfDsRFIrZD3ujCv68CtEiDLkp2Vn29NZdPHTubLn7xgB"
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQSlgcKEQgSGAI4AUIJbmFycmF0aW9uEgxfkIhrUoVB6AUel5waDEMowcGwIVAmTM40HyIwQifedfhooWeUFo82sSwMDtiDs0UGGwhdjnUL"
    "h9QWzjOFDnKlV4106rtLiSqSSLPBKrIG50QLGXmeEfFW/6AT/p1GMwMsw2qCRIMPu6fs23OJ8bT5326+OKflQ9Sl1M5r4EnINcSl5K6h+0vA"
    "aikzMcw1vJI16ExuQDc/4J4FXZjGuoKTZN8+VqeAUcJQNAGyZwiQkmV3YUa5kY6rB8+ls7G8/07LWSt/d0OzXR/ax4B5JSHjEiG+kOxPA7IA"
    "Z4JDKxtrPYXp5zQ1JrdRXlFEQSe03k6C/HBzpg0njw0cqO23fJIXASmwHA5EV1okl9qqIpIfH2hOQ/v1ufuUkZFzE/jX1nHvzoE0zmvkwGJV"
    "11GB6vXzft9CpU6C5PUGYpJUi7+bTObpBSmQrvIVjPaCLZt2fi0U++c7BL8GcNY7B2thHBOhRxns1xz+Psw7z2z4H2+mZmYDTTnSOaq9a3L6"
    "FtSdDYcik4QqJg0t7HRnXURvEm7T1r5E5gdVNe28XeBFn+hpKRBY/B4sYlWg2XV/E5EPEX5EfH/noSwSwm7t5olR2bMcynRsjWaeVQC0Mcfq"
    "YbtleTJxdm3p1lmBUFZypZfWONK5ZZeSrkU3gZ4LifBta87LHuzl15gWlSPTYlzG6SYIdXytPFI2K7oGyLh/yRs2vjFyZVBhCm5gZ+rkv8JA"
    "cAMoTHUoPYgrAotw+dusfSuTr91lULs0wRjTC35K1MOYrb2dUzf0Bh57AXCKE3pCIoXKS8Mq58zHrYXgqzxANFpJQ5P4KzvvQRZa6XbhVgpv"
    "kBOkGEemcX57ay9tGYo5fxmHYYihgfJXyEn5dg+xUcwNV7qxprK8f/5piK6TfkvUxAU/Gspg6VUP8f/uHV4JOzZW9s/+l+ImBFbojmKSc5Zz"
    "y4HAL3F1E4YlsxkwIkTCs278wCgQnDVIG+UYzny202dxDpdyLaMbLjTWuWzQtRbsg3isYR98ndOLTXgtiJ4N5i5zeqyrbAPsp1xwO4PpmEzg"
    "hqNVSJ8vbhL1gBGsVIzSviCPOpL/Z622zhljm5ZhFMzwMqFzzh2PKioNPaVtZgpgn8J6cFRPzI0hCz93pCwRGN25SRcE7VqXdNfGR1k7JxsU"
    "ECZAHPt41McAZSpWT63h39oQB54reKKlwhqgw+cQQ0AYAQ=="
)

RESPONSES_REQUEST: Final[Mapping[str, JsonValue]] = {
    "cache": {"no-cache": True},
    "max_output_tokens": 1024,
    "instructions": "You help maintain a synthetic Python project. Read the relevant files before "
    "suggesting a fix. Explain which file you need next, then call a read-only tool. "
    "Never change files. Use the responses client for this project review. Consider zero, "
    "negative and non-integer quantities before choosing the next file to inspect.",
    "input": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "The checkout rejects quantities of zero. Find the validation code and its tests.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I will locate the quantity validation and its tests."}],
        },
        {
            "type": "function_call",
            "call_id": "toolu_synthetic_search_01",
            "name": "search_files",
            "arguments": '{"query": "quantity", "paths": ["shop/", "tests/"]}',
        },
        {
            "type": "function_call_output",
            "call_id": "toolu_synthetic_search_01",
            "output": '{"matches": [{"path": "shop/validation.py", "line": 18, "text": "def '
            'validate_quantity(quantity):"}, {"path": "tests/test_validation.py", "line": 7, '
            '"text": "def test_zero_quantity():"}]}',
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
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
        {
            "type": "function",
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
    ],
    "reasoning": {"effort": "high"},
}

RESPONSES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "id": ANY,
    "created_at": ANY,
    "error": None,
    "incomplete_details": None,
    "instructions": "You help maintain a synthetic Python project. Read the relevant files before "
    "suggesting a fix. Explain which file you need next, then call a read-only tool. "
    "Never change files. Use the responses client for this project review. Consider zero, "
    "negative and non-integer quantities before choosing the next file to inspect.",
    "metadata": {},
    "object": "response",
    "output": [
        {
            "type": "reasoning",
            "id": ANY,
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "The user wants the validation file read first, but since the test "
                    "reads are independent, I can just include both in the same batch "
                    "since it's effectively still prioritizing validation.\n"
                    "\n"
                    "I'll check `shop/validation.py` to see how `validate_quantity` "
                    "handles zero, negative, and non-integer values, and read "
                    "`tests/test_validation.py` alongside it to confirm the expected "
                    "behavior.\n"
                    "\n",
                    "annotations": [],
                }
            ],
            "phase": None,
            "encrypted_content": json.dumps(
                [
                    {
                        "type": "thinking",
                        "thinking": "The user wants the validation file read first, but since the test reads are "
                        "independent, I can just include both in the same batch since it's effectively still "
                        "prioritizing validation.\n"
                        "\n",
                        "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                    },
                    {
                        "type": "thinking",
                        "thinking": "I'll check `shop/validation.py` to see how `validate_quantity` handles zero, negative, "
                        "and non-integer values, and read `tests/test_validation.py` alongside it to confirm the "
                        "expected behavior.\n"
                        "\n",
                        "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
                    },
                ],
                separators=(",", ":"),
            ),
        },
        {
            "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
            "call_id": "toolu_01GYseiK8hh9QxfWJ6HbiAt3",
            "name": "read_file",
            "type": "function_call",
            "id": "fc_toolu_01GYseiK8hh9QxfWJ6HbiAt3",
            "namespace": None,
            "status": "completed",
        },
        {
            "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 40}',
            "call_id": "toolu_01U3TuUTifkSBLVzyTFnbMN1",
            "name": "read_file",
            "type": "function_call",
            "id": "fc_toolu_01U3TuUTifkSBLVzyTFnbMN1",
            "namespace": None,
            "status": "completed",
        },
    ],
    "parallel_tool_calls": False,
    "temperature": None,
    "tool_choice": "auto",
    "tools": [
        {
            "name": "search_files",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query", "paths"],
                "additionalProperties": False,
            },
            "strict": None,
            "type": "function",
            "defer_loading": None,
            "description": "Search source and test files for a literal term.",
        },
        {
            "name": "read_file",
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
            "strict": None,
            "type": "function",
            "defer_loading": None,
            "description": "Read a range of lines from a project file.",
        },
    ],
    "top_p": None,
    "max_output_tokens": 1024,
    "previous_response_id": None,
    "reasoning": {"effort": "high"},
    "status": "completed",
    "text": {},
    "truncation": None,
    "usage": {
        "input_tokens": 877,
        "input_tokens_details": {
            "audio_tokens": None,
            "cached_tokens": 0,
            "cached_tokens_details": None,
            "image_tokens": None,
            "text_tokens": 4,
            "video_tokens": None,
            "cache_write_tokens": 873,
        },
        "output_tokens": 509,
        "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 313, "text_tokens": 196},
        "total_tokens": 1386,
        "cost": None,
    },
    "user": None,
    "store": None,
    "provider_specific_fields": {
        "citations": None,
        "thinking_blocks": [
            {
                "type": "thinking",
                "thinking": "The user wants the validation file "
                "read first, but since the test reads "
                "are independent, I can just include "
                "both in the same batch since it's "
                "effectively still prioritizing "
                "validation.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'll check `shop/validation.py` to see "
                "how `validate_quantity` handles zero, "
                "negative, and non-integer values, and "
                "read `tests/test_validation.py` "
                "alongside it to confirm the expected "
                "behavior.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
            },
        ],
    },
}
