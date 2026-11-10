from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_4_6_TEST_CASE
from integration.translation.runner import assert_translation

DIAGNOSTICS: Final = {"previous_message_id": "msg_011Cg7v1w1xZexMrWpf5GU75"}

CLAUDE_SONNET_4_6_DIAGNOSTICS_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="diagnostics",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "diagnostics": DIAGNOSTICS},
    expected_provider_request={**CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, "diagnostics": DIAGNOSTICS},
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_SONNET_4_6_DIAGNOSTICS_TEST_CASE],
    ids=lambda case: case.id,
)
def test_messages_diagnostics_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
