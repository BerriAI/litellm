import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.openai_responses import (
    GPT_5_6_SOL_TEST_CASE,
    GPT_6_1_SOL_TEST_CASE,
)
from integration.translation.runner import assert_translation


@pytest.mark.parametrize(
    "case",
    [
        GPT_5_6_SOL_TEST_CASE,
        GPT_6_1_SOL_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_responses_basic_openai_responses(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
