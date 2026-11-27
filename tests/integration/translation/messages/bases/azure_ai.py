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


CLAUDE_FABLE_5_2_THINKING_SIGNATURE_0: Final = (
    "CAQS3AcKEAgUGAI4AUIIdGhpbmtpbmcSDGSKgLiI46UrWnk2cBoMANyfQbILJXKYeAMRIjCZmyTsgvQhIqUfH1Zq8kVKqWqLNSYV9oo97I"
    "vPHVhJF6GyiWHBeqmRRowqmzNaf80q+QamgjsAkBYcMnGuS4RXGfhWjqxZWSVRMomYaeyalaLwWDbHG3ibcvpLhDk4B5RdJUGauGTmuUbW"
    "i6tDW8y/uIXQpTywtVGnq01ltv553FaegZYYcBaYr1cZtC5+9oBhROsGtun7/8vdrHb/otFEFolkX5H1AAdQ5Xo1kwBQN1gv8Tj0JdBVfy"
    "K/jwWVTIN1dP3S6xiTG/yvSf7LP4V81ifCIaZvybijVvItAsUFq4Jk5/lN5B4tiPhIkoePBSheV+hVMbCQoaCxhiYjfHEqRYBbnlEs4JlP"
    "guRgwOthbF6v2JzMLzQe2xOK6lDl9B4Qx7oZANBW9feBp+HQr1FQnE8xMlwmSCUipbMb9/c04qt9JMYLiFrDU2rzFuGeiiO1epJWvROzLC"
    "vSqj8dIDH52okj2IKnU7uTmI03zbWG+xfcevK2ze5QWpyalYmet32cQV7ef5SGLW+IHufMS1LFoJlRqtY/bPyqU0Un+/OLnZ0lnl9fjOoa"
    "jy1OiaRG0hDAaXvz7LlX3BZGxpUsk1X25SwHuSV5x7jetQB69eRGc3IUwaPiqfjVHxtaol8R90kJ2yVdIfvQxiCJNWYf9/vhLDj6QggN1u"
    "JkTqnxIXzQLPSw+S17QuYGbZ78YoIEVrEMX7S0mqeiVaFbN2h87d1PBaLSdVSgBRusqj+aFBvrDDz7Ez0t4bqYIlSI5YpkTMp5cXOByIK5"
    "JkPuTt2bbddJoB+cJV9XGpC8T4226EPEjCnlPUxFXJNYn8Yx4E0zUnPg4Zx0vsG4ps+lLQrl6dWqh5wMW/EeNf2wQyXqpy108z5bOelUHH"
    "vi9BuwH7n9VZUC7+3dmy4id7dweiOvsN4y2IgEqNlL+KfRh2AXnv1ZaKrU4YJfXrKOQC+m++XLLD4TaGJGGDniWlUF+eZN2DIjyCwvgoL2"
    "M2pLqCqLaxvEns9e8MFioFZiZskdnuEXSB6bq2C1FaqbTDLjuD3ThavE6MAwbJku0ufOxaZ0WdECtSibkV7RTadfx/vTdN3ShyHvuRPINO"
    "04nUPcPJDln1E3QJ9swfPXbV10qtu/NVlUBYBbZ3JH/RfQnZs3bIQ8P5DZcjQMuKt5abit9KiBmJe6cEkSU3XT6gFIiMkUsDwWuR7tCmK2"
    "XmgdWZtThjXBsxP8VC+R2QNQDPe/u52jdbrcA8IOoKH/ik0MhozaGAE="
)

CLAUDE_FABLE_5_2_THINKING_SIGNATURE_1: Final = (
    "CAQSjQcKEQgUGAI4AUIJbmFycmF0aW9uEgy2aIB1yN2FYwtLeR0aDORMeWfefRMIzmtHYiIwZRPcWgfiEy8c7zA5kxu0BobsCNMcCKmGUr"
    "E8W37Cgfg6Q8lnpZqjK5K6+IX+s6btKqkGsJhcfIkYxt5CKStoQYlmqv4dSm1UlPLRdN9R0m+kTlQAQw+5WbSJG2qeCscUFQgoj+OQQcdX"
    "5Tl/c/E4ZqchhLVl2uW3KZ6HtQdlgo23u6S0uXUt7xIS7p6zhCHWaA0HyChUvgaw+YvD3VJsDhxjds8yr8iqB3xBB+LA1u3tKmhioBaKxc"
    "gPjZDp6hOZJ8E0SrOyQmvWQixw4zGLsr8bzn5CGjMsYmw7UwIIQzqBgJS7JuaRm2gHL3s5NVulBRNjO1wJm/StK/KQ5hPI58FKuIaOsEcO"
    "zSGVfXyCfTkT6SdfLFE5ztGEPo5I/kgIlPCRoGkZJWE7ch8qHtaLnIFaLDCZZX+qMvkgXe0A356rMYEeAEsLGFHWsECkE0lhWqksXu7L0u"
    "Te2ouErNuglyT3y9fJsM8+btC4MwWb59R/dIgyklPXGnsLyLUikETD68LMSCgOpZoFdKkfSrFxAmWG6qOwu+KhLysew+04cOOvBtmai8RU"
    "o4xi1AwGxN1uenx2f7+b81C892+/syX/4v8g3pI+PRR+nSuk0Sm6994kF9vAgoHykdnzWaQMBFPHFgqtmUflJZCpwf0wGl+SDRlIQOfb5G"
    "3N7jCsGrGCLpUaR4OKr8AtfklnxXdPYEvV9emBgOTtZDjta/uoNmPMJO8PAr1WSoEYJLbjk0bP5Nbjn0R5fbyHluvZiKHKYzSkH3bSRsZo"
    "jd+lgbJcmeDvl84/paf1qHntkSFtkldPEh9aO21JbCjmLEWAp0hyO8FFF3gC3Bd+ZhdSXTCNAbmJLIS9RZif7TGLLTkNQ3ODhuwfYxWEIo"
    "6HT3g2lmr9kiW7gjskkqWhrNqqOYBaBW1IwQlUYjp4DaVXoHfT5DKVtAXqUJ0EJvAtq8Sobzg4+qKaj80CvIQsplwDj8wQq+v+ehHDHLUm"
    "7E8gOPURj5UYJvpAsdtqfT2RRQqhUuwmSVGaz8nmw48FkvG388KwpvaupVCxYviV2A/ucVGE4v832uEMULVwBldBckUyiPQ7Fd8Zl96rt8"
    "dlAkHgTQNLuWe0LiIqRDKEGru6ZP4AGwXYhwO9BFknW6hubPxW4ucYAQ=="
)


CLAUDE_FABLE_5_2_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011ChTeC96CM9Pfgp9m5miZJ",
    "content": [
        {
            "type": "thinking",
            "thinking": "I want to read the validation file first, as the user asked, so I'll name it as the "
            "next file I need and then make a read-only call on its opening lines, roughly the "
            "first 60. I'm leaving the tests file for now, since the user specifically asked for "
            "validation first.\n"
            "\n",
            "signature": CLAUDE_FABLE_5_2_THINKING_SIGNATURE_0,
        },
        {
            "type": "thinking",
            "thinking": "I'm opening shop/validation.py to check the imports, constants and helpers that "
            "affect how validate_quantity handles zero, negative and non-integer values, since "
            "it's defined at line 18.\n"
            "\n",
            "signature": CLAUDE_FABLE_5_2_THINKING_SIGNATURE_1,
        },
        {
            "type": "tool_use",
            "id": "toolu_01SD8Zr45JTNz3a7wyTTHMYG",
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
        "output_tokens": 285,
        "output_tokens_details": {"thinking_tokens": 107},
        "service_tier": "standard",
        "inference_geo": "global",
        "speed": "standard",
    },
}


CLAUDE_FABLE_5_2_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "azure_ai/claude-fable-5-2"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-azure-ai-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-fable-5-2",
        "messages": MESSAGES_REQUEST["messages"],
        "system": [{"type": "text", "text": MESSAGES_REQUEST["system"]}],
        "tools": MESSAGES_REQUEST["tools"],
        "stream": False,
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        **CLAUDE_FABLE_5_2_RESPONSE_FIELDS,
        "model": "claude-fable-5-2",
    },
    expected_litellm_response={
        **MESSAGES_RESPONSE,
        **CLAUDE_FABLE_5_2_RESPONSE_FIELDS,
        "model": "azure_ai/claude-fable-5-2",
    },
)
