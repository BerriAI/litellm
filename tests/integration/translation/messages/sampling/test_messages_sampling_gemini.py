from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.gemini import GEMINI_3_5_FLASH_TEST_CASE
from integration.translation.runner import assert_translation

GENERATION_CONFIG: Final = object_value(GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request["generationConfig"])

GEMINI_3_5_FLASH_TEMPERATURE_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="temperature",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "temperature": 0.4},
    expected_provider_request={
        **GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {**GENERATION_CONFIG, "temperature": 0.4},
    },
)


@pytest.mark.parametrize("case", [GEMINI_3_5_FLASH_TEMPERATURE_TEST_CASE], ids=lambda case: case.id)
def test_messages_sampling_gemini(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
