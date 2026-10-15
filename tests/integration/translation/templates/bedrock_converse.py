from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQS9AUKEggSEAEYAjgBQgh0aGlua2luZxIMQHB9X3K/kC6GNasiGgywUP8bZriTtiro/2MiMAi/+5c3bmdPrprOsDzKxvo/J9NAn2cBfonQ"
    "awxkQnqB3W5W5gKuBzPky9UZ+zTK6iqPBQhsFXO/q0Ky2I4cK8AZg1lsxNO6YTzWgD7fyCaEhn4HqY4hm4vK2V88DlytrsSGUTIQ/iEUSGVd"
    "83vAHJy9AnRW9pPrgF/nW1D3pSD/Z3xeuMmxjMLuhGTF1JclSqe1onncl6i1Hu3a4fdTxXCKwAkgXjSPGMcYEANs9ziU1Ep+g/lPs4yIddit"
    "mXaDnDzcCAq/6bFBnIe5IuN5nPeM4MqnXIXAhpDXa2R7HxUAg0v2dbzEzssSPuvn/PPF5VmusAXMQxM8zrLHdm5zUW6pss9p/OzxvgF6OlSj"
    "8iF8YlOQXOa8avjTVUXiOLG4L67euf2YehPWDuWhrv1RsFJ3eDFvrlSESRnXeXIp7U4qqav5G7dJwxnFszrQzN/Oc/VgFvwErqp39goRlZiT"
    "T2WpvuCOO8i8XsEg3NZUpTaypy2YfgMWJwVm/V06erqg3VKsVvIhXhp/XVkzu9AIHuPd+pJ1TgmybB4OGMuqYaj1Cs4OuNsH66YKEXSGexjB"
    "wbjgPFHBXLbG/Ek/ATXpUShR8Uxu6japXknWqT19rusAv/7KRiqDZR2WiYf/nFLF7zMM6Zjoi/SV1AVC4RS5alTEsom8Pp8w8ZoptXCalOXI"
    "fCh6kfqOZHxF9FCQ1i5hcPkvG2d1yzMkcV+i3worR+gXt/xZGT/Rdr28xceL9rOCEQawYT9fugI+BLHGEe/+X+jSAoFZxgcJ8Lqjz2kyoLqk"
    "ZTRgYzIEBZgqa+dFrCVuXO9m1B3MSv0+f8gBeZVZKt5ItBWzIfOs1q8KyfqNNRYg3yQAmiWJsF0SkbJZl/MDQXzSXqkzriwclWGnRv0hBC7W"
    "u75PM3a8YuBlc7Ww8Vdrw0AaG1s5FX7nhkHiV6DRII0YAQ=="
)

BEDROCK_CONVERSE_REQUEST: Final[Mapping[str, JsonValue]] = {
    "messages": [
        {
            "role": "user",
            "content": [{"text": "The checkout rejects quantities of zero. Find the validation code and its tests."}],
        },
        {
            "role": "assistant",
            "content": [
                {"text": "I will locate the quantity validation and its tests."},
                {
                    "toolUse": {
                        "input": {"query": "quantity", "paths": ["shop/", "tests/"]},
                        "name": "search_files",
                        "toolUseId": "toolu_synthetic_search_01",
                    },
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
                                "text": '{"matches": [{"path": "shop/validation.py", "line": 18, "text": '
                                '"def validate_quantity(quantity):"}, {"path": '
                                '"tests/test_validation.py", "line": 7, "text": "def '
                                'test_zero_quantity():"}]}',
                            },
                        ],
                        "toolUseId": "toolu_synthetic_search_01",
                    },
                },
                {
                    "text": "Read the validation file first so we can understand how zero and negative quantities "
                    "are handled.",
                },
                {"cachePoint": {"type": "default", "ttl": "5m"}},
            ],
        },
    ],
    "inferenceConfig": {"maxTokens": 1024},
    "additionalModelRequestFields": {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    },
    "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
    "system": [
        {
            "text": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "chat_completions client for this project review. Consider zero, negative and non-integer "
            "quantities before choosing the next file to inspect.",
        },
    ],
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
                        },
                    },
                    "name": "read_file",
                    "description": "Read a range of lines from a project file.",
                },
            },
        ],
    },
}

BEDROCK_CONVERSE_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 96}}},
    "metrics": {"latencyMs": 3022},
    "output": {
        "message": {
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                            "text": "I need to figure out which file to read for validation, then use a read-only "
                            "tool to look at it—I'll check the first 60 lines or so since I don't know "
                            "its exact length.\n\n",
                        },
                    },
                },
                {
                    "text": "I'll read `shop/validation.py` next, starting with the `validate_quantity` function "
                    "found at line 18.",
                },
                {
                    "toolUse": {
                        "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                        "name": "read_file",
                        "toolUseId": "tooluse_qWzJMc6h58NlspiEa01FUC",
                        "type": "tool_use",
                    },
                },
            ],
            "role": "assistant",
        },
    },
    "stopReason": "tool_use",
    "usage": {
        "cacheDetails": [{"inputTokens": 856, "ttl": "5m"}],
        "cacheReadInputTokenCount": 0,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokenCount": 856,
        "cacheWriteInputTokens": 856,
        "inputTokens": 4,
        "outputTokens": 230,
        "serverToolUsage": {},
        "totalTokens": 1090,
    },
}
