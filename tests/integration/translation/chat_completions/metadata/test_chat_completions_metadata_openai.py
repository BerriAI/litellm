from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.openai import GPT_5_4_TEST_CASE
from integration.translation.runner import assert_translation, translation_params

GPT_5_4_METADATA_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="metadata",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "store": True, "metadata": {"ticket": "123"}},
    expected_provider_request={
        **GPT_5_4_TEST_CASE.expected_provider_request,
        "store": True,
        "metadata": {"ticket": "123"},
    },
    mock_provider_response={
        **GPT_5_4_TEST_CASE.mock_provider_response,
        "id": "chatcmpl-EYjxC57J1e6kY6aHbQqnnxKRp00Xp",
        "created": 1791948506,
    },
    expected_litellm_response={
        **GPT_5_4_TEST_CASE.expected_litellm_response,
        "id": "chatcmpl-EYjxC57J1e6kY6aHbQqnnxKRp00Xp",
        "created": 1791948506,
    },
    known_bug="/v1/chat/completions treats metadata as LiteLLM logging metadata and only forwards it to OpenAI when litellm.enable_preview_features is on, so the provider never receives the caller's metadata",
)


@pytest.mark.parametrize("case", translation_params(GPT_5_4_METADATA_TEST_CASE))
def test_chat_completions_metadata_openai(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
