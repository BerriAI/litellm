import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.gemini import (
    GEMINI_3_1_PRO_PREVIEW_TEST_CASE,
    GEMINI_3_5_FLASH_TEST_CASE,
    GEMINI_3_8_FLASH_TEST_CASE,
)
from integration.translation.runner import assert_translation


@pytest.mark.parametrize(
    "case",
    [GEMINI_3_5_FLASH_TEST_CASE, GEMINI_3_8_FLASH_TEST_CASE, GEMINI_3_1_PRO_PREVIEW_TEST_CASE],
    ids=lambda case: case.id,
)
def test_responses_basic_gemini(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
