from typing import Final
from unittest.mock import ANY

from integration._support.client import object_value
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import (
    CHAT_COMPLETIONS_REQUEST,
    CHAT_COMPLETIONS_RESPONSE,
    CODING_AGENT_REQUEST,
    CODING_AGENT_RESPONSE,
)
from integration.translation.templates.anthropic import (
    ANTHROPIC_AGENT_REQUEST,
    ANTHROPIC_AGENT_RESPONSE,
    ANTHROPIC_MESSAGES_REQUEST,
    ANTHROPIC_MESSAGES_RESPONSE,
)

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


CLAUDE_FABLE_5_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "anthropic/claude-fable-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-agent-key",
        "anthropic-version": "2023-06-01",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-fable-5",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-fable-5",
        "id": "msg_011Cg3Kre4kW64wgdnJZesno",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_017kyzUjeWYmUvPe1JuA3sP9",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01LJUyCenUD37DbG5waJEG3r",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
                "caller": {"type": "direct"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1184,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-fable-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_017kyzUjeWYmUvPe1JuA3sP9",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_01LJUyCenUD37DbG5waJEG3r",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1184,
            "total_tokens": 1289,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1184,
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


CLAUDE_OPUS_5_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "anthropic/claude-opus-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-agent-key",
        "anthropic-version": "2023-06-01",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "model": "claude-opus-5",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-opus-5",
        "id": "msg_011Cg3Kpga2JLhw4zru2itKu",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01NFinjr4AyFZJJ1g44QYJJJ",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_018ZYT8DgJn6QPmdWqASvZnS",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
                "caller": {"type": "direct"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1180,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "anthropic/claude-opus-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_01NFinjr4AyFZJJ1g44QYJJJ",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_018ZYT8DgJn6QPmdWqASvZnS",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1180,
            "total_tokens": 1285,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1180,
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

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "anthropic/claude-sonnet-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={**ANTHROPIC_MESSAGES_REQUEST, "model": "claude-sonnet-5-5"},
    mock_provider_response={**ANTHROPIC_MESSAGES_RESPONSE, "model": "claude-sonnet-5-5"},
    expected_litellm_response={**CHAT_COMPLETIONS_RESPONSE, "model": "anthropic/claude-sonnet-5-5"},
)


CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSoggKEAgSGAI4AUIIdGhpbmtpbmcSDHAtt0k5pW+O+4p0yhoMaSOEZS5bpcu05veLIjDPL1g9WsQtCQvCTT+d7CtZXdj8Ky/U+qJXk5YK"
    "V+PIuWEZQeNKcaPTleGe9utaR6QqvwdoSvSgT9fVmK4SrR/XSTbPcPpCcRV84Y8g7ZDohrCzpb826nqiRF1A9tQ1K9OrlqiBEmG9ODXI3oy2"
    "ag915CwW8oJBjCWAmjmRvkQf64/0IOhqWEunbBPwdfCqqiRtbv1qnqaT2v+WfJeVd7Q72GMGVPjDpI8sZw+k5BoaczYZtJ+T0PbMMNXEwDG8"
    "q780JLCmcw5fByE+mX3cUX7SSoAehpwvvmd1/BkzZ7EQaWic7G759KgxzYRtBOiF8Mg9N/holW5GfDF3eBEhAH7JgnV1t+yY+oPy1LpzYGJb"
    "GP8gJryTIDRfJmEIX9LwrGjcO30rfGemRv/kzKy1PAY1oLB1jHY6jbO/uGYxZRJy51Sm2Byqj4BHUhZKEEV2GmKwxvr4CBYObD3dBF32FlxD"
    "pF4Xz6KuGGUx88ftC51Azp6ySLT/kOsVfqHcXYpH/RbO3BozdCg1AK3pceCljkGmlvZDEVgXZh66O41ZGi7Ku+KbvHADhfJ1dBYodZzNJg8F"
    "+MVcQLy4ZhAyfcxp91g6l46vjyEhLEoRNKwrg/dh3cavCsu0EBFQtUgd61U6Eovzyh1JOWDd5mnRE/rN4ISXWcOMKd8o05LZpzmoLkbnub8R"
    "DPV0LnPzMd+HHrDpMGoO79qDJU1MdiKhPKGacRP2GnA9Bv5T74r3Y6Zz3KO1tj9TtQ93aPQuiU5Te45T1aIzkNAemkDwuDN/8h/hd0CRCklk"
    "2XLb+7ZhQAINBc+yrwoDMq4+vB6y5qjsBBeOMGgWzyPvimWOt4VZczacLJRTKSocioZSYD+tJycCVbaeny41gTGPKicjI4CAji6MEcloNIVt"
    "G0bcQwPamtpDokxo/XKQL6W7sL37DQRU5UD+mNG9ntVMgQPb6EmC4boMq/EoEAEqbok0WwUwWhxPfN0EE87gtYUxavNx2/VQJuAVyM8tdKeK"
    "hvO8fy0G9d28tRO6yCFvbpKBYDuscZexVT8y5PLJOqVyqIdTYOG6pDDp8MyQ95mDtPHOpTLHo8XAsYtQlHi7qLKmkFx6BjuHu/r97FH2pdgt"
    "F2KT5CQmELeTJ1uXD/PsGN60qwONfsWD3fhrrP+rXo32utPzSNM6ndXNfSHHWwKUQ2YyJ3Z/aZCcDFW3e2FkjtcXuhzVL19h3tONpUDyXYOt"
    "BGecFGhVULxeQLMjhdK+6gibmoyjgQk/B/F7SO6SJjdvrtgwuKbOEYjSYJUC/49n4CE1gzNdtKdeI2qvYMyADizfeL15vU8mhdoFlOdlPCKd"
    "KY+uwXEHfI4/mRgB"
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "anthropic/claude-haiku-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={**ANTHROPIC_MESSAGES_REQUEST, "model": "claude-haiku-5-5"},
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-haiku-5-5",
        "id": "msg_011Cg36zdxXCpWpnwgbTX1Ss",
        "content": [
            {
                "type": "thinking",
                "thinking": "There's a mention of a chat_completions client I don't actually have access to, "
                "so I'll ignore that and stick to my available tools. I'll proceed by reading "
                "validation.py, keeping in mind edge cases like zero, negative, and non-integer "
                "values before deciding what to examine next.\n"
                "\n",
                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` to see how `validate_quantity` treats zero, "
                "negative, and non-integer values.",
            },
            {
                "type": "tool_use",
                "id": "toolu_015ndrvU4VqqoMBzK35jt4TX",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 288,
            "output_tokens_details": {"thinking_tokens": 149},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "anthropic/claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` to see how `validate_quantity` treats zero, "
                    "negative, and non-integer values.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_015ndrvU4VqqoMBzK35jt4TX",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "There's a mention of a chat_completions client I don't actually have access "
                    "to, so I'll ignore that and stick to my available tools. I'll proceed by "
                    "reading validation.py, keeping in mind edge cases like zero, negative, and "
                    "non-integer values before deciding what to examine next.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "There's a mention of a chat_completions client I don't actually have "
                            "access to, so I'll ignore that and stick to my available tools. I'll "
                            "proceed by reading validation.py, keeping in mind edge cases like "
                            "zero, negative, and non-integer values before deciding what to "
                            "examine next.\n"
                            "\n",
                            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "There's a mention of a chat_completions client I don't actually "
                                "have access to, so I'll ignore that and stick to my available "
                                "tools. I'll proceed by reading validation.py, keeping in mind "
                                "edge cases like zero, negative, and non-integer values before "
                                "deciding what to examine next.\n"
                                "\n",
                                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 288,
            "prompt_tokens": 880,
            "total_tokens": 1168,
            "completion_tokens_details": {"reasoning_tokens": 149, "text_tokens": 139},
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

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSqAYKEAgSGAI4AUIIdGhpbmtpbmcSDBSDqizSHGEEWmpWOxoM4DGvp9vRY4pul7dSIjB07rrpoEV7n5wqYphj7CiaeVIq2s4AlZ2SwEWL"
    "xmrdDsZmUBGnz3qCIj9GC21mC5QqxQWhWTlPxM+tJt925MxVve4HmM6e+Gydxpty4kFx7Pk8nqlfDTooB3vyVOyto6ACj/sooxJ1BgEGJgOX"
    "z1cH1Y9TG8Y1ca6DiiVm5Gw+6W5/U2FSWZyejNWbykYLxCdMLiLgH3MlCi5SvfIiQR0wgyT/0/1JpM6nrSmPp+zODEqWpbwrCTcwW3g9jSbe"
    "Jvzfhr9pi6RZQj37dXkWzBDstK1La/HQoUSn/Zp8EMa8fjQOclN/VzQbAtbcfnhO+jJpQfmfWWC3UVoXFfSw86v+sX4/yBdZ+r7eXLB+dF8B"
    "2mEcsWQuCtWa/EBgFR+1Xwyfaqno9/LpFfmiN58IwiAvTwYVkiwhpoJBY6035QDcXQGZURB2eNEMzg4Ay3lY/bp/ElMQPvjKy58ZihDXGysB"
    "PPOinfZYNEeCyctJ9VC2957KTV2GlgNVxJ0No9KnZumvvbad4BdU9Bt9z2VsNahvwPVO+sHb2tNLJUsHpznIIoUfrdaEfdPI7r/UoCRt2KRA"
    "OO+ejBYkS/D7otJHkjt/1/wOhEFuSNGLZ9yIsSPtI/qFvklkpptEbVQWd8Hs3cwyhZXBoXkFLr7Lku86jFMj8JuErAerc0zgMPLVsUC1s48L"
    "mB5lt2xHuYhoqSYbAtRhLBS47t1YMpPf11bUFjRX72wS4jOb0ZMzXHw2G+yQfFhb+xHkUO4OKrDB3+x3GIbjOraTpmeYN6e9MydUuHJiZFzp"
    "wTmrNzfYqHN6nNugAjMBJiPDHOmHn4AzeMJEoNu+PU5B73q5Xi22x0AqOF/D0xFIeDQHhDwnqBfvrVIq+Z0JVuoAwVroAx8y9MiIz5YUPlnn"
    "N2njDJs7VfN6JIh/J7Ak+mfOgYt92Ofi7HcKAyg4INNwAp3a9lts21O6sBDugOfVr9VFWmASX/TQ/IelHuHHxNtDIgtOFRVLj1QY1iFAyXfT"
    "dzYcGAE="
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "anthropic/claude-opus-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={**ANTHROPIC_MESSAGES_REQUEST, "model": "claude-opus-5-5"},
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_011Cg371jh5oo3SEfLpT5Eep",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll focus on reading the validation file first, around line 18, since that's "
                "what was specifically requested before looking at the tests.\n"
                "\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "Next I'll open `shop/validation.py` around `validate_quantity` (line 18) to see "
                "how it handles zero, negative and non-integer quantities.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01Ew2u9E9j2mkT6RbwSw13uR",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 215,
            "output_tokens_details": {"thinking_tokens": 66},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "anthropic/claude-opus-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "Next I'll open `shop/validation.py` around `validate_quantity` (line 18) to "
                    "see how it handles zero, negative and non-integer quantities.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                                "name": "read_file",
                            },
                            "id": "toolu_01Ew2u9E9j2mkT6RbwSw13uR",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "I'll focus on reading the validation file first, around line 18, since that's "
                    "what was specifically requested before looking at the tests.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I'll focus on reading the validation file first, around line 18, "
                            "since that's what was specifically requested before looking at the "
                            "tests.\n"
                            "\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I'll focus on reading the validation file first, around line 18, "
                                "since that's what was specifically requested before looking at "
                                "the tests.\n"
                                "\n",
                                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 215,
            "prompt_tokens": 880,
            "total_tokens": 1095,
            "completion_tokens_details": {"reasoning_tokens": 66, "text_tokens": 149},
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

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "anthropic/claude-fable-5-1"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={**ANTHROPIC_MESSAGES_REQUEST, "model": "claude-fable-5-1"},
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_011Cg372bGMBoXaSKUXjtQcr",
        "content": [
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` around `validate_quantity` to see how zero, "
                "negative, and non-integer quantities are handled.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01XMSjzRvHCE753Wx4dU2Vxo",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 143,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "anthropic/claude-fable-5-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` around `validate_quantity` to see how zero, "
                    "negative, and non-integer quantities are handled.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                                "name": "read_file",
                            },
                            "id": "toolu_01XMSjzRvHCE753Wx4dU2Vxo",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            },
        ],
        "usage": {
            "completion_tokens": 143,
            "prompt_tokens": 880,
            "total_tokens": 1023,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 143},
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
