import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.bedrock_chat_completions import GROK_4_7_TEST_CASE
from integration.translation.runner import assert_translation


@pytest.mark.parametrize("case", [GROK_4_7_TEST_CASE], ids=lambda case: case.id)
def test_messages_basic_bedrock_chat_completions(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
