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
        "model": "azure_ai/claude-haiku-4-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
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
        "id": "msg_011Cg7uyAiPY51HN4rSYzQW5",
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
            "speed": "standard",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "azure_ai/claude-haiku-4-5",
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
            "speed": "standard",
        },
    },
)


CLAUDE_FABLE_5_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure_ai/claude-fable-5"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "synthetic-agent-key",
        "anthropic-version": "2023-06-01",
        "api-key": "synthetic-agent-key",
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
        "id": "msg_011Cg3KumdWax5bBqBtNE1nh",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_013SS65unrdz97WLKGBouZc1",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01TVCw5m95kexg5C2n728prK",
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
            "inference_geo": "not_available",
            "speed": "standard",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "azure_ai/claude-fable-5",
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
                            "id": "toolu_013SS65unrdz97WLKGBouZc1",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_01TVCw5m95kexg5C2n728prK",
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
            "inference_geo": "not_available",
            "service_tier": "standard",
            "speed": "standard",
        },
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure_ai/claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
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
        "id": "msg_011Cg7uzQ1rtYRwCd9aeTyoF",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 11,
            "service_tier": "standard",
            "inference_geo": "not_available",
            "speed": "standard",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "azure_ai/claude-sonnet-4-6",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello! How can I help you?",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 11,
            "prompt_tokens": 18,
            "total_tokens": 29,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 11},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 18,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "inference_geo": "not_available",
            "service_tier": "standard",
            "speed": "standard",
        },
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure_ai/claude-sonnet-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
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
        "id": "msg_011CfjaxuLcxiEd4m9YZqXaA",
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
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "azure_ai/claude-sonnet-5",
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
            "inference_geo": "not_available",
            "service_tier": "standard",
        },
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure_ai/claude-opus-4-8",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
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
        "id": "msg_011Cg7uyf9RNvQWkpJobt6JU",
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
            "speed": "standard",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "azure_ai/claude-opus-4-8",
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
            "speed": "standard",
        },
    },
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure_ai/claude-sonnet-5-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5-5",
        "id": "msg_011CfjayBjr8GfdBPi3hbif2",
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
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "azure_ai/claude-sonnet-5-5",
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
            "prompt_tokens": 23,
            "total_tokens": 29,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 23,
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

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure_ai/claude-opus-5-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
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
        "id": "msg_011CfjayXnc5jLujSNNoJpS2",
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
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "azure_ai/claude-opus-5-5",
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
            "prompt_tokens": 23,
            "total_tokens": 29,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 23,
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

CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSwwgKEAgSGAI4AUIIdGhpbmtpbmcSDKfqwDliolwjkEk4/xoMbtopFWwu6m8MsKPLIjBYY7k/jbVLmSR7KHeAvyUV9jUyWBV6qdv+M9Uq"
    "JG7K7iHhxyzCRoLI+DsRvaqtq3gq4AdEnhxf+3y6bp2Gfgm19WyxAVxsc3BDwyrIOL2frop5j8DtwrGL7KwweKZcQ9Zm6R/VtGDH2r88PsxM"
    "g2oSgmYTBR0jEyaOxl4EwRllE3RfE8vkQP66TAx4RyCpbIGVB1lBXxdxVnFgzS8JQMaJnQ4jdXUdR8sH57heOUMuE2trGBVaKoDSWli2FtTw"
    "kDNHd99zv/b2cSZtIN7W/d/FuxRVzxlvuuFQO27y7pmaczsSldqKmCRtcGOb21RFF9u3I9eznIN8394JyQWIXEdWmnXuQL/7BKai2nD+R7No"
    "5gLT9gY5y1stqQZ8FPkbHxKnt9/WciUc5lyGz7uGJbUiEUYAFR0j41l8+3VXWpF4OV1aEQa2eHNATLpYJ8I3Wctlu2mcOm+096aak+BW1XXt"
    "vnkMSvBbM3JUElx5k5chZmfMUHFA32KuAj3VZ58lfnOBxGDE7gMNMVvxdAC/h7nRSrO6ICgMYYXy4kFkpVLZz6kH6K32cEAwl1pCPjxYIZVL"
    "OwrPXzjsQUqRpjekBFr1lx23oxR2fUId0EsFpPbJSZHjaN/amBsMY/g3iGFiPU3uQW7+Llvhb/OGpNwI3YtWxi3cTZkNdLaItCnl0QsJ6aFS"
    "D0oq0HcrKsIvcbMp05kz/rNFAGRBd5QrUthuh/CvYJUHosauz5OYrvRlzzKlsEVomzIkwVQw0ZrYhq0xZvwIRa7JbRoDadHL6uTNybKFCISX"
    "33IimEMVpkaMBEvygWya25H7pSK9OREbzV17rpS5cGBebOLKYjgZms7WhAERsyNm+gYIDcr+X5AGrOrz3qM/diknhnC5rNgbaO8w+JQl1i+D"
    "wrNXsCW61u+/KRI2Qf3ekaHG4fcETWz+Y1+JisK7WskmyihKAkexlLaxqyn5RisYGj9tKs4OpjCd1w4lhS6346ypK0ufsfyHjqfn7TVjIsVt"
    "posdTv+Wc3fTr7WDcmiJ+hBD/YxouCYYxS+hssBS/ut0QFfWH2dh0/2qGeNd70QqXEt+tbsopsrqpV2OZy9YE8VvLqZicNJHqbZTivXKsEFX"
    "YN4pnKv4X7ykWtCILbpbTZNg1iBa4z+T2fSFdAUJVgRFherqjCcD209AA4wj3UJewZX8nwX2Js1W/MRFrAIWzwwBI3x6eCq6OaaeXzSwUS9M"
    "I24Sk4o5rm9/czgj2Ottsq0Rmvsng/r3ZOpu8uKfjhxKmCSSQEE4ZZ7NBwLTOUVOcmBoeRfQbyjDKr7NYiaVuN7NlJw2NQbsyhAgqlD8IBpz"
    "Z0XczPP0u+uEKTIsVXcjAtHIjyx361hlCa1waGh5DqmWg7zhv2QenFwjBBgB"
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "azure_ai/claude-haiku-5-5"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={**ANTHROPIC_MESSAGES_REQUEST, "model": "claude-haiku-5-5"},
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-haiku-5-5",
        "id": "msg_011Cg37MxTLdwH6PHKZXCEey",
        "content": [
            {
                "type": "thinking",
                "thinking": "I need to read shop/validation.py next, keeping in mind edge cases like zero, "
                "negative, or non-integer quantities since the validation logic likely handles those. "
                "I'll look at the function starting around line 18 to understand its scope.\n\n",
                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` first. The search hit `validate_quantity` at line 18, "
                "so I'll start there and check how it treats zero, negative, and non-integer values.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01M68cqREYA4WqXBUK8rqcEq",
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
            "output_tokens": 324,
            "output_tokens_details": {"thinking_tokens": 167},
            "service_tier": "standard",
            "inference_geo": "global",
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "azure_ai/claude-haiku-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` first. The search hit `validate_quantity` at line 18, "
                    "so I'll start there and check how it treats zero, negative, and non-integer values.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_01M68cqREYA4WqXBUK8rqcEq",
                            "type": "function",
                        }
                    ],
                    "reasoning_content": "I need to read shop/validation.py next, keeping in mind edge cases like zero, "
                    "negative, or non-integer quantities since the validation logic likely handles those. "
                    "I'll look at the function starting around line 18 to understand its scope.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I need to read shop/validation.py next, keeping in mind edge cases like zero, "
                            "negative, or non-integer quantities since the validation logic likely handles those. "
                            "I'll look at the function starting around line 18 to understand its scope.\n\n",
                            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I need to read shop/validation.py next, keeping in mind edge cases like zero, "
                                "negative, or non-integer quantities since the validation logic likely handles those. "
                                "I'll look at the function starting around line 18 to understand its scope.\n\n",
                                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                            }
                        ],
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 324,
            "prompt_tokens": 880,
            "total_tokens": 1204,
            "completion_tokens_details": {"reasoning_tokens": 167, "text_tokens": 157},
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
            "speed": "standard",
        },
    },
)

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "azure_ai/claude-fable-5-1"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={**ANTHROPIC_MESSAGES_REQUEST, "model": "claude-fable-5-1"},
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_011Cg37NcfzdXibtC3dWhVWi",
        "content": [
            {"type": "text", "text": "I'll read the validation code and the test together since they are independent."},
            {
                "type": "tool_use",
                "id": "toolu_01LxkeKf9WX9BVpiWrvfKUyc",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_0139NDXUipoVov5pyoTErFZZ",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 30},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 216,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "not_available",
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "azure_ai/claude-fable-5-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read the validation code and the test together since they are independent.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                                "name": "read_file",
                            },
                            "id": "toolu_01LxkeKf9WX9BVpiWrvfKUyc",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 30}',
                                "name": "read_file",
                            },
                            "id": "toolu_0139NDXUipoVov5pyoTErFZZ",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 216,
            "prompt_tokens": 880,
            "total_tokens": 1096,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 216},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "inference_geo": "not_available",
            "service_tier": "standard",
            "speed": "standard",
        },
    },
)


CLAUDE_FABLE_5_2_THINKING_SIGNATURE_0: Final = (
    "CAQStwcKEAgUGAI4AUIIdGhpbmtpbmcSDMp41h6wTQ02m2K+OBoMthyKQtnkX80sa9LGIjCEb57cIXjs/sQO2/6OtWYJFkqLoMbS0oeP6g"
    "lueceYXK8PZgFUuVEB6jcdOZ450OYq1AbtG5xVJSIAMNMa2wP6RTuC9T0fT85tK6DUG7ywWg19IrOyUV6x4OD1ul9e4pdoy3UMLvR2zjHS"
    "WHiD+0eDgWOdY2P2ZbotXHa/qfYRvRM3emMit+cOxshS/zo55oiqc/PypGLOtB/YMSNMk3U3IqK1W2Xrj9jGzuR0GUL0njExNE1G5Viwwd"
    "85aP8wV4oIjdPuo4gSjlZFGUjAsEViHBTu7KK1z4BrFvlJJx0B7IqR3WGfJ+q32uhK3mZlw957Fr2X4qiexTRCgz1OXhJezRgrqmtTbdRr"
    "UoyzSB2jTEyylZ2dMYv23zcvfhBadoXvwETPvTcSeteXKmCHTtOfwp/+6YQvhBOXZO6HIMFM5itq3jAvsWTvmAykjg5viF0LP7smMiGaKm"
    "UtH44q65UWfK70MzEWpwpAFeslKxuWtzs/2mGnmnRy7Bv+uhSfbHwXO5v2bWsSJJSaeQ0FCxu85V9DtVmLWFdKWBW+WDaXuroJRgubpChu"
    "+nqiGLV1L9oj+ofdT2rUYU5US2SL+fqCYH8MZK8lFjxNyR7u54GuDd1nciJMpJSu/9VdyuZHU0z8BoxgfmTE1aDoPlpZTCZl2ODJk258EX"
    "iqZ2slpXkL+KdhthW32xVvwGw5HLpw22dXYzEhCNrmUyFi/cN8S5dEvkz+7Ee5OGDeHC1UyeIIIUj0HhiYVGTYFmfxqMdpS6Xfw63WYqBV"
    "JsPO9KsQn6/R9MqsHQfF1eT277xAlV0VIeJ9/Gvqrj+a4o5Dqxtm6OEOiubNZkMN9w6dKe3dkK67OOBUQ/YCneAoFPLdvb85gcrAxxupRT"
    "4LDOqT03ZUufdeczKCwir0FU5lKcKh1FTHnVrQ9QEt0+z/+CS1p7/Rcm87fzWx5N++MZpLJHzzFkYGwJddCfqDzBPBCPMri+d1JiDdtio4"
    "iYz/k+VeMMV9zwUSaBXPzSs0D0JYAursgtP5E9yKI6GDRSFsif0Tp9/G0L/VvjvZ31j4r+7bU+K6g/qL8VIObmae5QwBvUxLDCz393kHAp"
    "/ksK8QdhLfA3dQ82ALxDljwB1+HHGh6yMvR2rmatRHteXzJRy633iANSpTHbOFp6QGiE/NlCDaTKTTlWeLpYixd2z4Xlu+rLiTbOaFW/T8"
    "0AsYAQ=="
)

CLAUDE_FABLE_5_2_THINKING_SIGNATURE_1: Final = (
    "CAQSkwcKEQgUGAI4AUIJbmFycmF0aW9uEgwORwymPwVMSgXBmnIaDLAWjOw4kkZEhDe7uyIwGx7TqV64PXZi7rSmO0hBUzG+aPy33juU3R"
    "ml6kp6FAxg5urpZMMPwdNdlknZmgwIKq8Gs1KPtp/yG3buLiyMh74Fjhr7JncoUlVbmwh3DMQDxLoYrGhQ6qnnvpZV21WyEzULHCKg/Srs"
    "DN6SOCkJE4jGF5HyrQgVrZvOjdVf4t1WGnSnDhCxAYPShBKZDHDAGNiJ5zYmSR2+O4sgppiE3svTP8qRPO3MjWcRvmJM96tXGcPqk2WpTk"
    "iNg2o1gQFfpd3PDO/aVPmV/n1Uwy78+fLST4zfZ+TsF6i82Jm/w15MTOtZiY18mnDe+OlKrPkUrT7tblUWni6v9d5TxKVYhbXl9N73MRvA"
    "vOKATFgp3e/POxEVdOUuAT7op9oq3KRPZ70Z6LiE28xgO/zl+gww0ms4On5yI2fG9+RUm48qTVrzCwFozy4gtVVi+1NZzFlm7Z5NnWQ0fH"
    "L3xKt1JJMGbQvm0uO1/Jr3a4eq+GgdsST3ay/wTaTN+9xFQJf8vrDAMKGlObdbrtGWNaMvRGBw525bWlt0DY6pD+vd2Fja7ZiAhJznXQn4"
    "eaAve62VBM1VeSklGlYaTZUPr8SJswuOAI0hfQoM20Ozm1Y5qm82Fo73zdxtgMm2h87c5Bkp94A8iH7uqKiVV5aTXjzWp9YQIxSsgJDz1U"
    "+1nRmCfnXSHwUIo6x3j7E+TpmqwDgemJZzYJog62KKUNNfNJG/11vY/f6nzidEEh0cyGLvrw247Qne5r0RJ2mdMuZ10xW0DBj3wdDSyyPi"
    "G8Wbhj0YR9FLkGXlyXgfYPxoJhFoOkmhO49pGXbjWJR2WhAff/UvgedcNsffS5wEDormHLzHUCZhUagdl88J9CjjJvrxjGGgE7P88+xCkC"
    "swGJD5JdUvOHmmspEeTG361DD1ydua67YYxES128wfX7UkqH9Vn7ERI57RSDJP1o4rLYPx2h1LgKtBkforDUtOpXy/Ys/t+KjrvVZQO1IP"
    "RWHI8gVYKdF/KQBdmFUzsWqCb0MarBW4JVqNRUgE0AUS2LTkJuderc1hyXj4Fux9KDjcOtZoUVl84tK6YTtbbzXkyWC3Qjws/zHe5XQ8J0"
    "rOktcW0/EfLKC4rNf1iBu6YQ6VTeUkuUuOgpOzCm6nHvBP5ViSkU6stNp6FusYAQ=="
)


CLAUDE_FABLE_5_2_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "azure_ai/claude-fable-5-2"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={**ANTHROPIC_MESSAGES_REQUEST, "model": "claude-fable-5-2"},
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-fable-5-2",
        "id": "msg_011ChTeBsnCJAphow7NAB77X",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'm going to read shop/validation.py first, checking roughly lines 1 to 60, and "
                "I'll leave the tests for later since the user asked for the validation file first.\n"
                "\n",
                "signature": CLAUDE_FABLE_5_2_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'm opening shop/validation.py to check the imports and helper checks that affect "
                "how zero, negative and non-integer quantities are handled, since validate_quantity "
                "is defined there at line 18.\n"
                "\n",
                "signature": CLAUDE_FABLE_5_2_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_01QzZVTbzLq6Gkyna6rooMr5",
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
            "output_tokens": 281,
            "output_tokens_details": {"thinking_tokens": 102},
            "service_tier": "standard",
            "inference_geo": "global",
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "azure_ai/claude-fable-5-2",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_01QzZVTbzLq6Gkyna6rooMr5",
                            "type": "function",
                        }
                    ],
                    "reasoning_content": "I'm going to read shop/validation.py first, checking roughly "
                    "lines 1 to 60, and I'll leave the tests for later since the "
                    "user asked for the validation file first.\n"
                    "\n"
                    "I'm opening shop/validation.py to check the imports and helper "
                    "checks that affect how zero, negative and non-integer "
                    "quantities are handled, since validate_quantity is defined "
                    "there at line 18.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I'm going to read shop/validation.py first, "
                            "checking roughly lines 1 to 60, and I'll leave the "
                            "tests for later since the user asked for the "
                            "validation file first.\n"
                            "\n",
                            "signature": CLAUDE_FABLE_5_2_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": "I'm opening shop/validation.py to check the "
                            "imports and helper checks that affect how zero, "
                            "negative and non-integer quantities are handled, "
                            "since validate_quantity is defined there at line "
                            "18.\n"
                            "\n",
                            "signature": CLAUDE_FABLE_5_2_THINKING_SIGNATURE_1,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I'm going to read "
                                "shop/validation.py "
                                "first, checking "
                                "roughly lines 1 to "
                                "60, and I'll leave "
                                "the tests for later "
                                "since the user asked "
                                "for the validation "
                                "file first.\n"
                                "\n",
                                "signature": CLAUDE_FABLE_5_2_THINKING_SIGNATURE_0,
                            },
                            {
                                "type": "thinking",
                                "thinking": "I'm opening "
                                "shop/validation.py to "
                                "check the imports and "
                                "helper checks that "
                                "affect how zero, "
                                "negative and "
                                "non-integer "
                                "quantities are "
                                "handled, since "
                                "validate_quantity is "
                                "defined there at line "
                                "18.\n"
                                "\n",
                                "signature": CLAUDE_FABLE_5_2_THINKING_SIGNATURE_1,
                            },
                        ],
                    },
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 281,
            "prompt_tokens": 880,
            "total_tokens": 1161,
            "completion_tokens_details": {"reasoning_tokens": 102, "text_tokens": 179},
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
            "speed": "standard",
            "service_tier": "standard",
        },
    },
)
