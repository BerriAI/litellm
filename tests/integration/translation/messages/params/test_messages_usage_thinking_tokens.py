from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.azure import GPT_5_4_TEST_CASE as AZURE_GPT_5_4_TEST_CASE
from integration.translation.messages.bases.gemini import GEMINI_3_5_FLASH_TEST_CASE
from integration.translation.messages.bases.openai import GPT_5_4_TEST_CASE as OPENAI_GPT_5_4_TEST_CASE
from integration.translation.params import with_field
from integration.translation.runner import assert_translation

SCENARIO: Final = "usage_thinking_tokens"
THINKING_TOKENS: Final = 4
EXPECTED_OUTPUT_TOKENS_DETAILS: Final = {"thinking_tokens": THINKING_TOKENS}

OPENAI_GPT_5_4_THINKING_TOKENS_TEST_CASE: Final = replace(
    OPENAI_GPT_5_4_TEST_CASE,
    scenario=SCENARIO,
    mock_provider_response=with_field(
        OPENAI_GPT_5_4_TEST_CASE.mock_provider_response,
        ("usage", "output_tokens_details", "reasoning_tokens"),
        THINKING_TOKENS,
    ),
    expected_litellm_response=with_field(
        OPENAI_GPT_5_4_TEST_CASE.expected_litellm_response,
        ("usage", "output_tokens_details"),
        EXPECTED_OUTPUT_TOKENS_DETAILS,
    ),
)

AZURE_GPT_5_4_THINKING_TOKENS_TEST_CASE: Final = replace(
    AZURE_GPT_5_4_TEST_CASE,
    scenario=SCENARIO,
    mock_provider_response=with_field(
        AZURE_GPT_5_4_TEST_CASE.mock_provider_response,
        ("usage", "completion_tokens_details", "reasoning_tokens"),
        THINKING_TOKENS,
    ),
    expected_litellm_response=with_field(
        AZURE_GPT_5_4_TEST_CASE.expected_litellm_response,
        ("usage", "output_tokens_details"),
        EXPECTED_OUTPUT_TOKENS_DETAILS,
    ),
)

GEMINI_3_5_FLASH_THINKING_TOKENS_TEST_CASE: Final = replace(
    GEMINI_3_5_FLASH_TEST_CASE,
    scenario=SCENARIO,
    expected_litellm_response=with_field(
        GEMINI_3_5_FLASH_TEST_CASE.expected_litellm_response,
        ("usage", "output_tokens_details"),
        {"thinking_tokens": 125},
    ),
)

CASES: Final = (
    OPENAI_GPT_5_4_THINKING_TOKENS_TEST_CASE,
    AZURE_GPT_5_4_THINKING_TOKENS_TEST_CASE,
    GEMINI_3_5_FLASH_THINKING_TOKENS_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_messages_usage_thinking_tokens(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
