from collections.abc import Mapping
from typing import Final

from integration._support.client import object_value
from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST, MESSAGES_RESPONSE
from integration.translation.templates.anthropic import ANTHROPIC_MESSAGES_REQUEST, ANTHROPIC_MESSAGES_RESPONSE
from pydantic import JsonValue

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure_ai/claude-haiku-4-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-haiku-4-5",
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_011Cg7vB62SdiZVj3M7xJXpr",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
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
        "model": "azure_ai/claude-haiku-4-5",
        "id": "msg_011Cg7vB62SdiZVj3M7xJXpr",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
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
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure_ai/claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_011Cg7vCCFGaPEhwGwoS6VPy",
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
            "speed": "standard",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "model": "azure_ai/claude-sonnet-4-6",
        "id": "msg_011Cg7vCCFGaPEhwGwoS6VPy",
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
            "speed": "standard",
        },
        "diagnostics": None,
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure_ai/claude-sonnet-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_011Cfjaxm5HXLX7FBu38pDBJ",
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
        "model": "azure_ai/claude-sonnet-5",
        "id": "msg_011Cfjaxm5HXLX7FBu38pDBJ",
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
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure_ai/claude-opus-4-8",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_011Cg7vBVpRK6kd1kkYXJXyb",
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
        "model": "azure_ai/claude-opus-4-8",
        "id": "msg_011Cg7vBVpRK6kd1kkYXJXyb",
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
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure_ai/claude-sonnet-5-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-5-5",
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5-5",
        "id": "msg_011Cfjay3Wzxnz5vkMFngMey",
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
        "model": "azure_ai/claude-sonnet-5-5",
        "id": "msg_011Cfjay3Wzxnz5vkMFngMey",
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
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure_ai/claude-opus-5-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-opus-5-5",
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-opus-5-5",
        "id": "msg_011CfjayNQ3rbTgxNGMRusPa",
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
        "model": "azure_ai/claude-opus-5-5",
        "id": "msg_011CfjayNQ3rbTgxNGMRusPa",
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
)

CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSqgcKEAgSGAI4AUIIdGhpbmtpbmcSDJhRZ6mJjwWk7Qp33hoMdnwv42chh3ronr/mIjA/ti2oyTG92CZPcqNiCgjiqzwhodkEUejDgKoH"
    "fbOEwwLS6iSCqSKrLMCu9PQAp5gqxwZJ+Z/ZX5JUifYdc1pNfnYmlWpJJoBDLv30nDIpOQq73DNraWtbsW0Wm1ZWdrZAapLtVP4pPFrDlen3"
    "3gj1ZOfbLh/3tM1yfQb3reW/p8c9DMc54gQTlDqVEm0wSuqtmk0epMxueMh9Vu97JxR2SbeG/kU3M22oqt2HqsDXIxC4z8GQfuhJVa4CFvoV"
    "kPLudM9VrYwLzPQ9iOEVvjjw5fYFLf5KOBFI95+z0LHqBopKpPLeQFT9J4jO4h6jsRMkmeCK0GIPKAdvIl6XlFKEC8z1sUi8aIsU216w/8yS"
    "HDQdw6p5pe2jwG9XZqs98xRfELsxn29J7WaxzaWI3YQvMxaLJG0ze1InomeGzBZi46QCqlW7dK0XwnDSNbkcnN/1I4JIJlJhnSVZsbirzgK2"
    "B8EZXaloWxRSRzC4Aw4tJ3/hKw2R966UgQzp8vkr/Ezms3v0cnRjayopPCUHNcuHkuCuww2X69J22HcULSUcvEaQSLlVHxTw2cS4hOjhvXc5"
    "ySJTZPYlKNQhaioYvELPq5ULD/k2hj9JZp+Go/fvIt3F9ysuC5PRnvgN8V357N0L9a79S8lbw2eituEWmjRZzOQujGdVfz+d4NU0v0Btihuc"
    "/am2liAB+hY4JatYsXyRRx/tV9HcV5ZTjh5I8bV5NHC/MsP0dFWM9XmsRcxQ+X/SyNHWvZUaSzYd+oKtv4RLDvanfNdG8kIcSzrSJe6DdqY9"
    "7d6DWU2/mcbIIBZm94Wv40Snxg/CEnaVB3SdK/E4cTmlUfvaRZvaUMWUSk4wPrIA5De3htk7d2AP97fv2QayaS0PlvdnwbAO+kOSmEEvBKq/"
    "yI/0LoSrPCi08CRJ8rO2KNXv5spyCJAfLTG8im6KaeENj0M7+e/BWYVOlZfgqNqaXUu5a2EUlO/0QycPe0NEIZ70SdG54C1zkE9MTjzYoYXM"
    "EFXaawPQ61dheOHClEyEbLlfKSa9IUX19hhht9GD6wJIjAGQ3A+YqS1HIqZEvI7gnkQlk19BWPHs4UCF8JhnTsJQ+KZLTYFO7kfX5shhPYff"
    "DTlrCRGUrArdsM9ooNXDAqgEA6O0xMuX8G2eg13+4tmcZGv407LHLWtreo8fgrBmS1yWMRgB"
)

CLAUDE_HAIKU_5_5_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011Cg37NEs5KS227ncBai4Sg",
    "content": [
        {
            "type": "thinking",
            "thinking": "I should read shop/validation.py to check how it handles zero, negative, and "
            "non-integer quantities before moving to the next file. I'll grab a reasonable line "
            "range since I don't know the exact file length.\n\n",
            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
        },
        {
            "type": "tool_use",
            "id": "toolu_012CKWjp8JY4UGL1TC5Lj79N",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 873,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 220,
        "output_tokens_details": {"thinking_tokens": 120},
        "service_tier": "standard",
        "inference_geo": "global",
    },
}

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "azure_ai/claude-haiku-5-5"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-haiku-5-5",
        "messages": MESSAGES_REQUEST["messages"],
        "system": [{"type": "text", "text": MESSAGES_REQUEST["system"]}],
        "tools": MESSAGES_REQUEST["tools"],
        "stream": False,
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        **CLAUDE_HAIKU_5_5_RESPONSE_FIELDS,
        "model": "claude-haiku-5-5",
        "usage": {**object_value(CLAUDE_HAIKU_5_5_RESPONSE_FIELDS["usage"]), "speed": "standard"},
    },
    expected_litellm_response={
        **MESSAGES_RESPONSE,
        **CLAUDE_HAIKU_5_5_RESPONSE_FIELDS,
        "model": "azure_ai/claude-haiku-5-5",
        "usage": {**object_value(CLAUDE_HAIKU_5_5_RESPONSE_FIELDS["usage"]), "speed": "standard"},
    },
)

CLAUDE_FABLE_5_1_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011Cg37Nrg7YQTVeZkczeqqj",
    "content": [
        {
            "type": "text",
            "text": "I'll read `shop/validation.py` around `validate_quantity` to see how zero, negative, "
            "and non-integer quantities are handled.",
        },
        {
            "type": "tool_use",
            "id": "toolu_01TCWoDbheT9e18mtVDH3xnQ",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 873,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 143,
        "output_tokens_details": {"thinking_tokens": 0},
        "service_tier": "standard",
        "inference_geo": "not_available",
    },
}

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "azure_ai/claude-fable-5-1"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-fable-5-1",
        "messages": MESSAGES_REQUEST["messages"],
        "system": [{"type": "text", "text": MESSAGES_REQUEST["system"]}],
        "tools": MESSAGES_REQUEST["tools"],
        "stream": False,
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        **CLAUDE_FABLE_5_1_RESPONSE_FIELDS,
        "model": "claude-fable-5-1",
        "usage": {**object_value(CLAUDE_FABLE_5_1_RESPONSE_FIELDS["usage"]), "speed": "standard"},
    },
    expected_litellm_response={
        **MESSAGES_RESPONSE,
        **CLAUDE_FABLE_5_1_RESPONSE_FIELDS,
        "model": "azure_ai/claude-fable-5-1",
        "usage": {**object_value(CLAUDE_FABLE_5_1_RESPONSE_FIELDS["usage"]), "speed": "standard"},
    },
)
