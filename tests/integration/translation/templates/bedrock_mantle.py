from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSkAQKEggSEAEYAjgBQgh0aGlua2luZxIMAi8mku8BopClY7wPGgwlWfmp5AbEIPA1xuAiMIbNUAxSlU56KwYLJ1P+/QAOb0fy2iNuUIvN"
    "uu0NeTLet5Bf5yDruo7S0IwCi/HD+SqrA68zipbpxKWBeB+abRfM6xMZbYGcvRzePKzfZ6CYii1l08tO3fgsqbBitmj/6jvyogZ/a4Zf138x"
    "ReoliJfLLudbpQB3rol3BtDrm8YrYqWW/vBn/Ioe3v3wHYRA3r4TllD0tyJwB4eNXENsq8vmxHrlTjQohAVsIyEjNy6xqs/YPW2zrCV85ZTE"
    "UQQ2MDoCmAJa/wPgR/i68XIQ3kSWP3SllvhwlGcEUFa+lNUofPIUL7YBPyweBgrBSoS6hGr7LT3XZWNddtvjx6n5n1DF42FpapUdsK2iwerN"
    "Gg/c6AHxr44urpLJ3XWjc/jgGs3KrLtfSFJ4pV3oRlZDn9upJKjTl97f6aIJqzxqJXlNIJ8VAWEV6kH/TVkRq5AdlwGQROLEA2taVBP01wAO"
    "nxfwTfxj3UG49lVrwgi/+m3kEXOGsP7V876Ieb7wH/2zu+XxCz+k99d3bcc3MZr79VLxXif5QWbt70cm+2NjtCMPDYf0hm5o0KAtz9S9ZBUj"
    "528KMDGsVR/rqwWcEV3mq/hWKjZmmeMSbhinrNVng5uM9cBe9YwaulbpS3s2lvQYAQ=="
)

BEDROCK_MANTLE_ANTHROPIC_MESSAGES_REQUEST: Final[Mapping[str, JsonValue]] = {
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
    "model": "anthropic.claude-opus-5-5",
}

BEDROCK_MANTLE_ANTHROPIC_MESSAGES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "model": "claude-opus-5-5",
    "id": "msg_bdrk_cs3sdsy2tpe6vxiv7ipubz5hhgknryzkga3mhragbuieyoskse2q",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "thinking": "I should check the validation file first, starting around line 18.\n\n",
            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
        },
        {
            "type": "text",
            "text": "Next I'll read `shop/validation.py` around `validate_quantity` (line 18) to see how it handles "
            "zero, negative and non-integer values.",
        },
        {
            "type": "tool_use",
            "id": "toolu_bdrk_018J8SJwAgjhRkkmtm3xdJx2",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
        },
    ],
    "container": None,
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "stop_details": None,
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 876,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 166,
        "output_tokens_details": {"thinking_tokens": 19},
        "service_tier": "standard",
    },
}


BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST: Final[Mapping[str, JsonValue]] = {
    "model": "openai.gpt-6.1-sol",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "The checkout rejects quantities of zero. Find the validation code and its tests.",
                },
            ],
        },
        {
            "type": "message",
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
            "output": [
                {
                    "type": "input_text",
                    "text": '{"matches": [{"path": "shop/validation.py", "line": 18, "text": "def '
                    'validate_quantity(quantity):"}, {"path": "tests/test_validation.py", "line": 7, '
                    '"text": "def test_zero_quantity():"}]}',
                },
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Read the validation file first so we can understand how zero and negative quantities "
                    "are handled.",
                },
            ],
        },
    ],
    "instructions": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. Explain "
    "which file you need next, then call a read-only tool. Never change files. Use the chat_completions "
    "client for this project review. Consider zero, negative and non-integer quantities before choosing the "
    "next file to inspect.",
    "max_output_tokens": 1024,
    "reasoning": {"effort": "high"},
    "tools": [
        {
            "name": "search_files",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
                "required": ["query", "paths"],
                "additionalProperties": False,
            },
            "strict": None,
            "type": "function",
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
            "description": "Read a range of lines from a project file.",
        },
    ],
}

BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "background": False,
    "billing": {"payer": "developer"},
    "completed_at": 1792027271,
    "created_at": 1792027269,
    "error": None,
    "frequency_penalty": 0.0,
    "id": "resp_izakuft4jrxwmodefqee2jjt6gkmayywwuuc242pduoydywb3v4a",
    "incomplete_details": None,
    "instructions": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. Explain "
    "which file you need next, then call a read-only tool. Never change files. Use the chat_completions "
    "client for this project review. Consider zero, negative and non-integer quantities before choosing the "
    "next file to inspect.",
    "max_output_tokens": 1024,
    "max_tool_calls": None,
    "metadata": {},
    "model": "openai.gpt-6.1-sol",
    "object": "response",
    "output": [
        {
            "content": [],
            "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeJ7rFK0vbfOLCW3lZUKZX7nZu5wsyKm4u2DeoLRbILW0AF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBN1N2LzZrWDd1QWV1WkpjQ1dEOFQvSDR2Y0ZaY2ZWemtpR2ZWNGRncEgwZmJqMVBNd3FIV2NnbFlTcHVOcVMyZWc9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcmpoCBv5IPHctwun5UTXrXrEmFVDe7MUULd6PZATyexPQO0PepR8jRQ3zVYSjkLkI8eYSjtPsT9kCKrIpsJlAuid+IimJvnTBHGplsaTC/53IpOSzB87nJlx+m4kCAAAQANtzzELZF5mKNBTvW/ca/WZ6iJI42OwsDigg6J5ZOIdJaC35TDfpQsssoDFQIHtI/v////8AAAABAAAAAAAAAAAAAAABAAABSyq/hcdJ8rImbQvE2c8qmEFM3skJI4sMoBBIdIZm8ljKX6zuu9kBGRln5StGWzlUOVYemoWeHBoDDuZkGDWAeTcQYdNeiCuTOQBg1kfIvhSGpGZHyKIfg+XnWIjrExKbZhmAC/GMebMNNuJ1/ZUFnq0UtD8X1r2xEkIWCUsQHysv9E6D6AstjrAgpo5arvZG0Gi5gryi/eE9Q7IGwilIvzeslUlM7sVcmD5RWe83nMeXmCUR41N50TK11F1j/H314CtKbJbMQWNWGLUUSRiKjqDFiC+o4bioQCRzvka6NUkIBitgjjkJ+etpL5ibB065xADWxaRc6WoUe1VkNVKXlYq4kOBNhnkJSHuasIJiD8/suYWXshD9XMF+SffKyKk3zscZwAuZ7N/ZOayYqGrhOMryCFYXvSo7v8PJ1hvg64z1MknImFG/6rSEVox9yeyfSEQmeDX/ABwO8PAsAGcwZQIwIpngWR/1wARV01RTUI+B4dV7XaR1AC0aOv5KfuQw5UJOnfGVp8E7MS379SPiA+SEAjEA0TMJ1LJ3cXitGqWvPATglnscU7UW9Mhny4elQ+cw7pFm5kRlYA/lkAyDdNyjZFVU",
            "id": "rs_7a90cfc98ff653aeb165c2b845bc1e4b",
            "summary": [],
            "type": "reasoning",
        },
        {
            "content": [
                {
                    "annotations": [],
                    "logprobs": [],
                    "text": "I’ll read `shop/validation.py` first to see how it handles zero, negative, and "
                    "non-integer quantities before inspecting the tests.",
                    "type": "output_text",
                },
            ],
            "id": "msg_4aeb32a9101256fe89e762125d4ffeee",
            "phase": "commentary",
            "role": "assistant",
            "status": "completed",
            "type": "message",
        },
        {
            "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
            "call_id": "call_399ba0e7ac865a798b46a6cabe6b6bde",
            "id": "fc_db8b798957b65c379754d08da376417f",
            "name": "read_file",
            "status": "completed",
            "type": "function_call",
        },
    ],
    "parallel_tool_calls": True,
    "presence_penalty": 0.0,
    "previous_response_id": None,
    "prompt_cache_key": None,
    "prompt_cache_retention": "in_memory",
    "reasoning": {"summary": None, "effort": "high", "context": "all_turns"},
    "safety_identifier": None,
    "service_tier": "default",
    "status": "completed",
    "store": True,
    "temperature": 1.0,
    "text": {"format": {"type": "text"}, "verbosity": "medium"},
    "tool_choice": "auto",
    "tools": [
        {
            "description": "Search source and test files for a literal term.",
            "name": "search_files",
            "output_schema": None,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
                "required": ["query", "paths"],
                "additionalProperties": False,
            },
            "strict": True,
            "type": "function",
        },
        {
            "description": "Read a range of lines from a project file.",
            "name": "read_file",
            "output_schema": None,
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
            "strict": True,
            "type": "function",
        },
    ],
    "top_logprobs": 0,
    "top_p": 0.98,
    "truncation": "disabled",
    "usage": {
        "input_tokens": 312,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": 108,
        "output_tokens_details": {"reasoning_tokens": 39},
        "total_tokens": 420,
    },
    "user": None,
    "moderation": None,
}
