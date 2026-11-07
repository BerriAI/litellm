from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.gemini import GEMINI_3_5_FLASH_TEST_CASE
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
GEMINI_3_5_FLASH_N_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="n",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "n": 2},
    expected_provider_request=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("generationConfig", "candidate_count"), 2
    ),
)
GEMINI_3_5_FLASH_STOP_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="stop",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "stop": ["<<HALT>>"]},
    expected_provider_request=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("generationConfig", "stop_sequences"), ["<<HALT>>"]
    ),
)
GEMINI_3_5_FLASH_SEED_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="seed",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "seed": 4242},
    expected_provider_request=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("generationConfig", "seed"), 4242
    ),
)
GEMINI_3_5_FLASH_SERVICE_TIER_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="service_tier",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "service_tier": "flex"},
    expected_provider_request=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("serviceTier",), "flex"
    ),
)

CASES: Final = (
    GEMINI_3_5_FLASH_TEMPERATURE_TEST_CASE,
    GEMINI_3_5_FLASH_TOP_P_TEST_CASE,
    GEMINI_3_5_FLASH_N_TEST_CASE,
    GEMINI_3_5_FLASH_STOP_TEST_CASE,
    GEMINI_3_5_FLASH_SEED_TEST_CASE,
    GEMINI_3_5_FLASH_SERVICE_TIER_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_params_gemini(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
