from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.azure_ai import (
    CLAUDE_FABLE_5_1_TEST_CASE,
    CLAUDE_HAIKU_5_5_TEST_CASE,
)
from integration.translation.runner import assert_translation

CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQS3gYKEAgSGAI4AUIIdGhpbmtpbmcSDEoUbRA8/sNqRoVoJRoMjpP0glsrmS7QMeraIjDxIxhKjyeY5Y65IZo5YTWEVqymcjH8k7a0mAjW"
    "bQoNTG9gw0R+WqIckDwWmnUrlFgq+wUl+/iw8hZ5IAryY71AHQuvdYaGuEgmN8jqzGujRvAeW4tw3zLG0td8x1lyoJk02iTU5aQV0wO5uO87"
    "wJqAEJ6bbheWT9m9l+M3tW9DOFsQMHYpj95wz6crrlZxzhBgMN7dZMzy+l+ATwMSxpQHX4FXRNAsz63J2tOADTeHSFXeP6Evn5abAhSiznHf"
    "ezSB2bRv6eBRVi7hDR2MgVJ1CEo5b2YtGroJPzVJxZRZpyHKg04dRkAUn0uQb+N5FxJ5I4v2phqqMLqpvDR8sU14Q91v7SstrT7NyRlAFhv9"
    "pzocddLhJkyHJpFctQr0D1k1oiyPCWpWdRVMEPrAybvkh6vVij2l6CP/kBfD8JUUulRCPO+xoyfAglQJsA4zc1gRCggBKbim+26oazBYat8K"
    "+1RO5iRzQe/gvs8DwQSPML/YWqzGK/qd1G7yJujaMnkBt9Jew44DQ/eUgKP5030zKd/ft+jmj8uu2PNP6zZrvZA5grdTSY6pSnu4X++1wE5r"
    "NExLgC/TJgDejVzCazAP9MsDnXz7nv+6yVytEsL//xOVq7s5nF0IYpNdawt1bAzNaHAnIMH8meeksDnnRJuBKt4mhg0nPgueOXgo7RLEUHWm"
    "MYakXV5wFNLY1oAUV4RpQJ+zF+/mJ/f7AHmghcxoPnPPPlISar3f9w00CQRVQjJDkm8ozxWyLN6AXhDQfQEDvYWPl33VaOcv7TpLAJ7HkWyk"
    "AJA8oB3LflgBJCSOXL67LGiGeSw8eR0bp89+Yux2C9MzORRvE8asjfl1W6NFWJzsvpNYcRYm0QREuNDRabg+mkcfwpDJyeaopSsNZLOMS7zq"
    "mwFMPrmW9CY+MJTxG0pQX1G8Ln59hhPu242z8l6wP1eqBbRzV54RMjE0wBGV7vfoPkqBzxGAj0Cr+/BeNbM4ne9k4pL9s5JYRsWhhhF1mkN7"
    "pFlOxz8cswlKkAaq/KnCWp2ptVgzxJazuokShk4hnnbwcdxaDVX2NynBGnmbcpfG8zso2GC52Ke+GAE="
)

CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.mock_provider_response,
        "id": "msg_011Cg37bzPpF6nmTd4cxR6zH",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll start by reading shop/validation.py to understand its contents, focusing on the "
                "first portion of the file before deciding if more context is needed.\n\n",
                "signature": CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` around the `validate_quantity` definition at line 18 "
                "to see how it treats zero, negative, and non-integer values.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01L2rdexX7pthHtxd9WR8UbE",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 876,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 252,
            "output_tokens_details": {"thinking_tokens": 103},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read `shop/validation.py` around the `validate_quantity` definition at line 18 "
                    "to see how it treats zero, negative, and non-integer values.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_01L2rdexX7pthHtxd9WR8UbE",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "I'll start by reading shop/validation.py to understand its contents, focusing on the "
                    "first portion of the file before deciding if more context is needed.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I'll start by reading shop/validation.py to understand its contents, focusing on the "
                            "first portion of the file before deciding if more context is needed.\n\n",
                            "signature": CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I'll start by reading shop/validation.py to understand its contents, focusing on the "
                                "first portion of the file before deciding if more context is needed.\n\n",
                                "signature": CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 252,
            "prompt_tokens": 880,
            "total_tokens": 1132,
            "completion_tokens_details": {"reasoning_tokens": 103, "text_tokens": 149},
            "prompt_tokens_details": {
                "cached_tokens": 876,
                "text_tokens": 4,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 876,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)

CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_FABLE_5_1_TEST_CASE.mock_provider_response,
        "id": "msg_011Cg37V6cvTmK19kQhSJPvQ",
        "content": [
            {
                "type": "text",
                "text": "I'll read the validation file to see how zero and negative quantities are handled.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01Gk8RXusGMKa2tw97hQjuzk",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 876,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 123,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
    },
    expected_litellm_response={
        **CLAUDE_FABLE_5_1_TEST_CASE.expected_litellm_response,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I'll read the validation file to see how zero and negative quantities are handled.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                                "name": "read_file",
                            },
                            "id": "toolu_01Gk8RXusGMKa2tw97hQjuzk",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            },
        ],
        "usage": {
            "completion_tokens": 123,
            "prompt_tokens": 880,
            "total_tokens": 1003,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 123},
            "prompt_tokens_details": {
                "cached_tokens": 876,
                "text_tokens": 4,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 876,
            "inference_geo": "not_available",
            "service_tier": "standard",
        },
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE, CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE],
    ids=[CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE.id, CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE.id],
)
def test_chat_completions_prompt_cache_hit_azure_ai(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
