from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.azure_ai import CLAUDE_SONNET_4_6_TEST_CASE
from integration.translation.runner import assert_translation, translation_params

CLAUDE_SONNET_4_6_SERVICE_TIER_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="service_tier",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "service_tier": "standard_only"},
    expected_provider_request={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request,
        "service_tier": "standard_only",
    },
    mock_provider_response={**CLAUDE_SONNET_4_6_TEST_CASE.mock_provider_response, "id": "msg_011Cg1SnrstCdzZPwmTk2qQa"},
    expected_litellm_response={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_litellm_response,
        "id": "msg_011Cg1SnrstCdzZPwmTk2qQa",
    },
    known_bug="/v1/messages drops service_tier before the Azure AI Foundry Anthropic request. get_requested_anthropic_messages_optional_param keeps only AnthropicMessagesRequestOptionalParams keys and service_tier is not one of them",
)


@pytest.mark.parametrize("case", translation_params(CLAUDE_SONNET_4_6_SERVICE_TIER_TEST_CASE))
def test_messages_service_tier_azure_ai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
