import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.bedrock_mantle import (
    CLAUDE_HAIKU_4_5_TEST_CASE,
    CLAUDE_OPUS_5_5_TEST_CASE,
    CLAUDE_SONNET_5_TEST_CASE,
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
        CLAUDE_HAIKU_4_5_TEST_CASE,
        CLAUDE_SONNET_5_TEST_CASE,
        CLAUDE_OPUS_5_5_TEST_CASE,
        GPT_5_4_TEST_CASE,
        GPT_5_6_SOL_TEST_CASE,
        GPT_5_6_LUNA_TEST_CASE,
        GPT_6_LUNA_TEST_CASE,
        GPT_6_1_SOL_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_responses_basic_bedrock_mantle(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
