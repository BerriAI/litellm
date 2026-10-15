from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import CHAT_COMPLETIONS_REQUEST, CHAT_COMPLETIONS_RESPONSE
from integration.translation.templates.anthropic import ANTHROPIC_MESSAGES_REQUEST, ANTHROPIC_MESSAGES_RESPONSE

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
        "id": "msg_011Cfjax6mBrNVhFGQopmVCp",
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
        "id": "msg_011CfjaxGKedCbjtb7HuAUBC",
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
            "inference_geo": "not_available",
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
            "inference_geo": "not_available",
            "service_tier": "standard",
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
        "id": "msg_011CfjaxUiYCUsqMC4iqfc2U",
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
                        },
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
                        },
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
                            },
                        ],
                    },
                },
            },
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
            {
                "type": "text",
                "text": "I'll read the validation code and the test together since they are independent.",
            },
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
            },
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
        },
    },
)
