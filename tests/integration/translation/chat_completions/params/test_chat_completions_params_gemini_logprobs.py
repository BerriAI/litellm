from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.gemini import GEMINI_3_5_FLASH_TEST_CASE
from integration.translation.params import with_field
from integration.translation.runner import assert_translation

GEMINI_3_5_FLASH_TOP_LOGPROBS_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario="logprobs_top_logprobs",
    litellm_request={**GEMINI_3_5_FLASH_TEST_CASE.litellm_request, "logprobs": True, "top_logprobs": 3},
    expected_provider_request=with_field(
        with_field(GEMINI_3_5_FLASH_TEST_CASE.expected_provider_request, ("generationConfig", "responseLogprobs"), True),
        ("generationConfig", "logprobs"),
        3,
    ),
)

CASES: Final = (GEMINI_3_5_FLASH_TOP_LOGPROBS_TEST_CASE,)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_params_gemini_logprobs(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
