import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import CLAUDE_OPUS_5_5_TEST_CASE, CLAUDE_SONNET_4_6_TEST_CASE
from integration.translation.runner import run


@pytest.mark.parametrize("case", [CLAUDE_OPUS_5_5_TEST_CASE, CLAUDE_SONNET_4_6_TEST_CASE], ids=lambda case: case.id)
def test_messages_basic_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    run(case, gateway, provider)
