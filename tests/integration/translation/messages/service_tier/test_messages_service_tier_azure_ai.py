from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.azure_ai import CLAUDE_SONNET_4_6_TEST_CASE
from integration.translation.runner import assert_translation

CLAUDE_SONNET_4_6_SERVICE_TIER_RESPONSE_FIELDS: Final = {
    "content": [{"type": "text", "text": "Hello!"}],
    "usage": {
        "input_tokens": 18,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 5,
        "service_tier": "standard",
        "inference_geo": "not_available",
    },
}

CLAUDE_SONNET_4_6_SERVICE_TIER_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="service_tier",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "service_tier": "standard_only"},
    expected_provider_request={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request,
        "service_tier": "standard_only",
    },
    mock_provider_response={
        **CLAUDE_SONNET_4_6_TEST_CASE.mock_provider_response,
        **CLAUDE_SONNET_4_6_SERVICE_TIER_RESPONSE_FIELDS,
        "id": "msg_011Cg1SnrstCdzZPwmTk2qQa",
    },
    expected_litellm_response={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_litellm_response,
        **CLAUDE_SONNET_4_6_SERVICE_TIER_RESPONSE_FIELDS,
        "id": "msg_011Cg1SnrstCdzZPwmTk2qQa",
    },
)


@pytest.mark.parametrize("case", [CLAUDE_SONNET_4_6_SERVICE_TIER_TEST_CASE], ids=lambda case: case.id)
def test_messages_service_tier_azure_ai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
