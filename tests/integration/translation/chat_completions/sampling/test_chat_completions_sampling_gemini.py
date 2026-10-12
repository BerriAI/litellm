from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.gemini import GEMINI_3_8_FLASH_TEST_CASE
from integration.translation.chat_completions.bases.vertex_ai import (
    GEMINI_3_8_FLASH_TEST_CASE as VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE,
)
from integration.translation.runner import assert_translation

GEMINI_GENERATION_CONFIG: Final = object_value(GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request["generationConfig"])
VERTEX_AI_GENERATION_CONFIG: Final = object_value(
    VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request["generationConfig"]
)

GEMINI_3_8_FLASH_TEMPERATURE_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="temperature",
    litellm_request={**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "temperature": 0.4},
    expected_provider_request={
        **GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {**GEMINI_GENERATION_CONFIG, "temperature": 0.4},
    },
)

GEMINI_3_8_FLASH_ZERO_TEMPERATURE_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="zero_temperature",
    litellm_request={**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "temperature": 0},
    expected_provider_request={
        **GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {**GEMINI_GENERATION_CONFIG, "temperature": 0},
    },
)

GEMINI_3_8_FLASH_REASONING_EFFORT_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="reasoning_effort",
    litellm_request={**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "reasoning_effort": "medium"},
    expected_provider_request={
        **GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {
            **GEMINI_GENERATION_CONFIG,
            "thinkingConfig": {"thinkingLevel": "medium", "includeThoughts": True},
        },
    },
)

VERTEX_AI_GEMINI_3_8_FLASH_TEMPERATURE_TEST_CASE: Final = replace(
    VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE,
    scenario="temperature",
    litellm_request={**VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "temperature": 0.4},
    expected_provider_request={
        **VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {**VERTEX_AI_GENERATION_CONFIG, "temperature": 0.4},
    },
)


@pytest.mark.parametrize(
    "case",
    [
        GEMINI_3_8_FLASH_TEMPERATURE_TEST_CASE,
        GEMINI_3_8_FLASH_ZERO_TEMPERATURE_TEST_CASE,
        GEMINI_3_8_FLASH_REASONING_EFFORT_TEST_CASE,
        VERTEX_AI_GEMINI_3_8_FLASH_TEMPERATURE_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_chat_completions_sampling_gemini(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
