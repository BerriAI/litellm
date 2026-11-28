from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.gemini import GEMINI_3_5_FLASH_TEST_CASE
from integration.translation.params import with_field
from integration.translation.runner import assert_translation

GEMINI_3_5_FLASH_TEMPERATURE_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="temperature",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "temperature": 0.37},
    expected_provider_request=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("generationConfig", "temperature"), 0.37
    ),
)
GEMINI_3_5_FLASH_TOP_P_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="top_p",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "top_p": 0.83},
    expected_provider_request=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("generationConfig", "top_p"), 0.83
    ),
)
GEMINI_3_5_FLASH_TOP_K_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="top_k",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "top_k": 7},
    expected_provider_request=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("generationConfig", "top_k"), 7
    ),
)
GEMINI_3_5_FLASH_STOP_SEQUENCES_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="stop_sequences",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "stop_sequences": ["<<HALT>>"]},
    expected_provider_request=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("generationConfig", "stop_sequences"), ["<<HALT>>"]
    ),
)

CASES: Final = (
    GEMINI_3_5_FLASH_TEMPERATURE_TEST_CASE,
    GEMINI_3_5_FLASH_TOP_P_TEST_CASE,
    GEMINI_3_5_FLASH_TOP_K_TEST_CASE,
    GEMINI_3_5_FLASH_STOP_SEQUENCES_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_messages_params_gemini(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
