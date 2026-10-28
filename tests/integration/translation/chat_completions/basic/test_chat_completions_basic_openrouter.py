import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.openrouter import (
    CLAUDE_SONNET_4_TEST_CASE,
    GEMINI_3_1_FLASH_IMAGE_TEST_CASE,
)
from integration.translation.runner import assert_translation


@pytest.mark.parametrize(
    "case",
    [CLAUDE_SONNET_4_TEST_CASE, GEMINI_3_1_FLASH_IMAGE_TEST_CASE],
    ids=lambda case: case.id,
)
def test_chat_completions_basic_openrouter(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
