from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.anthropic import CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_translation

CLAUDE_HAIKU_5_5_REASONING_EFFORT_LOW_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="reasoning_effort_low",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "reasoning": {"effort": "low"}},
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "output_config": {"effort": "low"},
    },
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "reasoning": {"effort": "low"}},
)

CLAUDE_HAIKU_5_5_REASONING_SUMMARY_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="reasoning_summary",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "reasoning": {"effort": "high", "summary": "auto"}},
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "reasoning": {"effort": "high", "summary": "auto"},
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_REASONING_EFFORT_LOW_TEST_CASE, CLAUDE_HAIKU_5_5_REASONING_SUMMARY_TEST_CASE],
    ids=lambda case: case.id,
)
def test_responses_reasoning_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
