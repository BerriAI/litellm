import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.anthropic import (
    CLAUDE_HAIKU_4_5_TEST_CASE,
    CLAUDE_OPUS_4_8_TEST_CASE,
    CLAUDE_OPUS_5_5_TEST_CASE,
    CLAUDE_SONNET_4_6_TEST_CASE,
    CLAUDE_SONNET_5_5_TEST_CASE,
    CLAUDE_SONNET_5_TEST_CASE,
)
from integration.translation.runner import assert_translation


@pytest.mark.parametrize(
    "case",
    [
        CLAUDE_HAIKU_4_5_TEST_CASE,
        CLAUDE_SONNET_4_6_TEST_CASE,
        CLAUDE_SONNET_5_TEST_CASE,
        CLAUDE_OPUS_4_8_TEST_CASE,
        CLAUDE_SONNET_5_5_TEST_CASE,
        CLAUDE_OPUS_5_5_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_responses_basic_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
