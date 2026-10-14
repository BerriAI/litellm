from typing import Final

from integration.translation.case import TranslationTestCase

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-opus-5-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-opus-5-5",
        "id": "msg_011CfgC5HKvyve78CTFAw97f",
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
        "model": "anthropic/claude-opus-5-5",
        "id": "msg_011CfgC5HKvyve78CTFAw97f",
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
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_011CffzUNHaEfzVxCh5hskBG",
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
        "model": "anthropic/claude-sonnet-4-6",
        "id": "msg_011CffzUNHaEfzVxCh5hskBG",
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
)

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-haiku-4-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_011CfgDz9Vj8wXvKzhoM4MQA",
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
        "model": "anthropic/claude-haiku-4-5",
        "id": "msg_011CfgDz9Vj8wXvKzhoM4MQA",
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
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-sonnet-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_011CfgDzLXmJmnHRTpeBoDmZ",
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
        "model": "anthropic/claude-sonnet-5",
        "id": "msg_011CfgDzLXmJmnHRTpeBoDmZ",
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
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-opus-4-8",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_011CfgDzSFYYPC3aYE9rH98J",
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
        "model": "anthropic/claude-opus-4-8",
        "id": "msg_011CfgDzSFYYPC3aYE9rH98J",
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
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSxQYKEAgSGAI4AUIIdGhpbmtpbmcSDJrX5VjkbmdGUFJ+sxoMnu8RpbLlm7ssOeuRIjCKooYO6VuwjYwaz9vs2D11iWFsNehT86MXOUq6"
    "VmzAS0us82XJepr/PrUmluVsHa0q4gVaPWwfRFl+gRbl5+9M5bv5JE6dcT0eIEfCUP6Hdxbrt2zzIKhOKHhzEKtNrtZXlvg+XYNYA2n2GeFM"
    "tJMCafG2dKLSd78zNL4FN2QFxExwzchESzzNNtGDKELIOBnvee281ooCzJtYcloExzCb3URdsyGGAMN/s9PZNy6ix2MuKI2pvUrKT47CXNHf"
    "h+M7ogfC/+RxMboo4OIDMqWw1igyXu0tlx0vgwc+Szxq8iATI22CT+ADtPybfx6uia79G9UZdqCGOhHdhoU9u606Q7eKavuwe7QD5T7KuzVY"
    "Src6SMFSnxhPDJXQqRSYRXqEA3futkmQYT6tK84otQy9jkYxbqZ5PFoLBWy1V391ren4poE/MdO8ng7bMsNJYN2vKCZOExUkMdzDROQBZgsB"
    "INJkThkjl/JxqNYJASN9x7x4cx91Pyj573pjXP0mkCbt0kV2qM0FdxGdvrIL6aa2T6kXWbymm0HgIQtN/Funm22EZ9MxV8CONZ77vbiE8ew6"
    "SWpFQQQkv00x7RNse57DCszhOa8MqWdNqMRUKmVBCjP/O+YeT1SxuY44ZQi4lVvCI7l/R3NM2V4vp9WwPber7jT3ybRVEUmWfXSKfUKxhRlJ"
    "7b/Lo7ixG5XhYQEoI3UaN1nGW+HPliYaFVwXnqOKp0zWxeflbSMrmVrFUVJqrykUJWSQ0UYusEkxR85OoXF0k/Zzboriie6FENuDHzrUNc7M"
    "A/ad9LAcNUwuq4Tnr64loSvn/aQPlzawTlv3QiuRWK6LcGlgIYpRWkRRtVVAqpInlRKJMtuuM9oWALr7UHRTvTDOIMumdrmcKZNWD6WtLGYK"
    "wyGvC7iXcNI256w6hUClMBzv9P64XxgPAsPRu3SIUWY8deFSfkxMAImYtwEej2KRUNevuPUhpeE3lTvNsGx0HgOI30xPfSH48F1Dqcyz8tv7"
    "gm6ZSeqNxScJkXPKSb8V2P2vcBPMDhBwFc8iQOYtMJEYAQ=="
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-sonnet-5-5",
        "cache": {"no-cache": True},
        "max_tokens": 1024,
        "system": "You help maintain a synthetic Python project. Read the relevant files before suggesting a "
        "fix. Explain which file you need next, then call a read-only tool. Never change files.",
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
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
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
                    },
                ],
            },
        ],
        "max_tokens": 1024,
        "model": "claude-sonnet-5-5",
        "stream": False,
        "system": "You help maintain a synthetic Python project. Read the relevant files before suggesting a "
        "fix. Explain which file you need next, then call a read-only tool. Never change files.",
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
    },
    mock_provider_response={
        "model": "claude-sonnet-5-5",
        "id": "msg_011Cg2zn3d1CskESUyuoxwUm",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0},
            {
                "type": "text",
                "text": "I need `shop/validation.py` first. It defines `validate_quantity` at line 18, "
                "and that function decides how zero and negative values are handled. I'll read "
                "the function and the code around it.",
            },
            {
                "type": "tool_use",
                "id": "toolu_018f5Ht15nH2o12Bxi39Gi4V",
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
            "output_tokens": 226,
            "output_tokens_details": {"thinking_tokens": 69},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "model": "anthropic/claude-sonnet-5-5",
        "id": "msg_011Cg2zn3d1CskESUyuoxwUm",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "", "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0},
            {
                "type": "text",
                "text": "I need `shop/validation.py` first. It defines `validate_quantity` at line 18, "
                "and that function decides how zero and negative values are handled. I'll read "
                "the function and the code around it.",
            },
            {
                "type": "tool_use",
                "id": "toolu_018f5Ht15nH2o12Bxi39Gi4V",
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
            "output_tokens": 226,
            "output_tokens_details": {"thinking_tokens": 69},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
)
