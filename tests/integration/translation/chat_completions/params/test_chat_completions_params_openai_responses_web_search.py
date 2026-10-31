from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.openai_responses import GPT_5_6_SOL_TEST_CASE
from integration.translation.runner import assert_translation

GPT_5_6_SOL_WEB_SEARCH_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="web_search_options",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "web_search_options": {"search_context_size": "low"}},
    expected_provider_request={
        **GPT_5_6_SOL_TEST_CASE.expected_provider_request,
        "tools": [{"type": "web_search", "search_context_size": "low"}],
    },
)

CASES: Final = (GPT_5_6_SOL_WEB_SEARCH_TEST_CASE,)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_params_openai_responses_web_search(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
