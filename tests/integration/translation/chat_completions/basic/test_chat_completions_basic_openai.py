import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.openai import (
    GPT_5_4_TEST_CASE,
    GPT_5_6_LUNA_TEST_CASE,
    GPT_5_6_SOL_TEST_CASE,
    GPT_6_1_SOL_TEST_CASE,
    GPT_6_LUNA_TEST_CASE,
)
from integration.translation.runner import assert_translation


@pytest.mark.parametrize(
    "case",
    [
        GPT_5_4_TEST_CASE,
        GPT_5_6_SOL_TEST_CASE,
        GPT_5_6_LUNA_TEST_CASE,
        GPT_6_LUNA_TEST_CASE,
        GPT_6_1_SOL_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_chat_completions_basic_openai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: LIT-9235 chat completions moves message.refusal into provider_specific_fields and drops null system_fingerprint and logprobs"
    )
    assert_translation(case, gateway, provider)
