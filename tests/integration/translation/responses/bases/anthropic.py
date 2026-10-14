import json
from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-haiku-4-5",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-haiku-4-5",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 17,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 17,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 5,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 5},
            "total_tokens": 22,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-sonnet-4-6",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-sonnet-4-6",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 18,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 18,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 5,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 5},
            "total_tokens": 23,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-sonnet-5",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-sonnet-5",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 21,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 21,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 27,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-opus-4-8",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-opus-4-8",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 21,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 21,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 27,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSjQkKEAgSGAI4AUIIdGhpbmtpbmcSDEDc474Gky7oluMeexoMoWHq7QNph0AD/iryIjBcOfp/vVmnNwx2Ql5XrpBpTFGDsjSPUsiduXpS"
    "BqNqJGL1LhcVECZZLpUCtioz8mgqqggFRJFK+jNCTu3z+DdHydzsOklKi5bz3JpKFrInRTsmq4DS1mvT+R1GR0kJsaUoUdObf0QOayaATZ4k"
    "b929l0av8s21YHFS9BYWzQiCjF5OdsUNbn6Z/a1ezA+Py4bd1DaqnvgCBXJisayy2L2LedYWaNPHqOZVco5RDJ2N0Z2uV7khaU4WqAaOT6TI"
    "+5jPBsyYsdAMBb6M2zFJxg3ViPCbFfAW/VJi0+vhEfSKBE4mf+ySZnQtZv8I+B8lTZUZio3Brt6n+8r3UhOu4CwxmoQSH/ud0gXyqHwXdKtD"
    "mzyPb4MoE+Pcr9EHCzuQ2fQwM4wt9+7XXgwN170M/KqDcXqI2JTF4spF6ONAk8q4VTVlkctZiCsSERghb1sqWKgByBCT2zmQmAUICDR8p0qR"
    "G4FXXOpOcyTdfELu1YQIN89ezFQ8haIhqCZmODdr44uBjB+bDL86HkGYBcU4808/P+ZzKvbQOr0pXRyAAOSr74l/BsB3Fj5eZuRDOcie/apc"
    "DiOKrh1ZyyAizWA1y2pyd0/FKNfHnyHNzTs/3wovmpEVfOJZTOwj9wkNfQvyzZP3fc+g6MwjRE6Mac6C9NealUatZ4tfAP1o+4SiWnIee48o"
    "6K+RHXsv5gal/a72q04YNr50WNMEY6huBYZRrwZkVYNc+jGXpxE9QSRz3w4iRfsZ/Kbm9VN2LcMyTYB+m9fedzJmXogMpvfv7YrspL1aZCWP"
    "J7GD0+3mCKPewkFF/tWe5mY1i5ui1siMDhbiCB8EqtX7RR+3ELRTAr6v0lERED+wpca0drdEDxGH22758Yq5gF8QzFWUJrvpaPhLsy9PplrD"
    "M6tjqA5Z31cgKXVWRQzBZ7NV2/QURQDYRfIsr+SKv8Ojl6wf5Nf/D9V65C1yswYWRT7+98OVbBd3gYuecyiJ73m560yDXpmjng1wW/K0mxls"
    "XJBzIs2O4lIKkOlz96A6cPVyR1WaPTuSKLfzgBqcNFQrX/C4SczTi9RIoeaJwSmRbaKUEZSotv9+MQwx6QmzQegQIS8UJ5O8fZ7GOK6bqyKE"
    "61MZt0rbyX992cvwDqNWWXDxT6mJv+SMqSnujcqoy/RaoG6HjD1NrOCF0qyYDpY6kWAuttOovtnPBuQQ68hBtmVTc3a/uISxBBqfz6/9ZjsM"
    "V6QxGAZf4B9u/3YPQ7L6MBNuQvsCZe0ZdqTHh+O+XtpMvKvQaDCpyrhNIV5jybHoqSv07IkzK8z6mJyA3LrR3Upcm5r9IS/5U7UIP9ariYVD"
    "Np96mHFW+ST4IBA89zfV1LKavOZUNBS9wyZJur/ufxG9yOEf2zIU5E27xwwKK5RuZEiohvN47ilE95wzvhl6E7t1zLhNwGTc8EN1/8y0RFdj"
    "544ZKR09dIbDYbPKtsF2Uiwj91NZogAXpbKza12/jIVPZnqRGAE="
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQS7gYKEQgSGAI4AUIJbmFycmF0aW9uEgyrfyuCAMwXBi476R8aDNm9DpKl4kUQ9DgSxSIw57Jh079XdSlUgcZ6cFLzlWbAMZX1L2zVUXnW"
    "NCFG35L1Z54gfMM/EFOINc/csBmBKooGaFN5bEc3aHWkBxWJBxceD9czOfIPvB+V53z+mRbPnM80f84y565AIJ8mPUCIdzJRecWlJYNhz3f3"
    "xJQz4PPaB8IUI6N/Wqk75BFcgyObj2AWXpStHys+gySbbfyYBKniWg3awuBiA9KiAHvr7JHcYYnCMKimN75+yrwzEMNGNzdgnJpzm+uFgZ8A"
    "riQGtmBCnHAZyo4guSxocDgvwHFOKkio85wVDHHp5pZjA2lJIWL5ypYWgsExkPu2wT5AJGOV/GEoLPSYVnZp2NUqc1nBRVxoxO1cVAwa8tjV"
    "ew9ouDGF4+anwtHQJtQ2/Jdzr6C277m8M03IKwq+sY8MpvHoI6AOhKmDyTSDMQ2IDKY0a1lyf49aFCc1pexkSbf2Xo6nfwAl7ae0shjuCoe6"
    "EV/StOthGPaxDiLniUlWLSNYqViwv7lNQLz1uD5cl6+RpppbbqRpL9ueKQc6qElYI18Ei5i4gYj7nxhWZadwQGd560a34m9NPpq0qCpEd78A"
    "gGLEKrsSBaKvZNxmDdV1siklMt5QSwtzeCAcDtPAuY4C+2MerCXVmPJiJSkENVIBicpcSYl/l/+ZCPNDItkfQKWcdrCwXKiKLuEJ2OXWDUuK"
    "CV5/2+RY2TdfEU17ILHQy++/1n9j5c3ZvKgUlF3JqzVofxQ4dytWYMsjh/iQKi+qavh8tixMbyzE/nz+zFcCxEQywbLpWIODxcXiQVIz1lA6"
    "V70LoSLSyZ/+1kuPkRfS86YgmBFXOXvAyqffyaoe2/cA9qq5PBbYw96Jki5AcirYxBTZUAnImgvJULOp6C/U7dK7KflRChbmO75CALUfXLIC"
    "tixfAmee44siyjDj5FVUzoT0VLLrn3LhUiTDUyQYDxfDLTs7rE+M723MPJZpERmPMXncs0F3M8gi+o9BoqeT6sH7lH02hk/+QXg8JFcFWoxR"
    "YSfG/aSxsQ04cv9qFceZDRcn9h4ptZ3SpmSIXviSbmrZD0/4wSDFHZaDK4ntWOof+4Zi64xoRdEKcr5woJ7WYeeUb8aSeLZ42BgB"
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-sonnet-5-5",
        "cache": {"no-cache": True},
        "max_output_tokens": 1024,
        "instructions": "You help maintain a synthetic Python project. Read the relevant files before "
        "suggesting a fix. Explain which file you need next, then call a read-only tool. "
        "Never change files.",
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
        "id": "msg_011Cg2znCfy1nTYHVguQMAxy",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0},
            {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1},
            {
                "type": "tool_use",
                "id": "toolu_01RkzHcq1TFKkZbJkDcKaByh",
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
            "input_tokens": 838,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 326,
            "output_tokens_details": {"thinking_tokens": 228},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You help maintain a synthetic Python project. Read the relevant files before "
        "suggesting a fix. Explain which file you need next, then call a read-only tool. "
        "Never change files.",
        "metadata": {},
        "model": "anthropic/claude-sonnet-5-5",
        "object": "response",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0},
                        {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1},
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_01RkzHcq1TFKkZbJkDcKaByh",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01RkzHcq1TFKkZbJkDcKaByh",
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
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 838,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 838,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 326,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 228, "text_tokens": 98},
            "total_tokens": 1164,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0},
                {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1},
            ],
        },
    },
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-opus-5-5",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
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
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-opus-5-5",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 23,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 23,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 29,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)
