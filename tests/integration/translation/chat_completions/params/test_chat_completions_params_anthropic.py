from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.anthropic import CLAUDE_SONNET_4_6_TEST_CASE
from integration.translation.params import with_field
from integration.translation.runner import assert_translation

CLAUDE_SONNET_4_6_TEMPERATURE_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="temperature",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "temperature": 0.37},
    expected_provider_request=with_field(CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("temperature",), 0.37),
)
CLAUDE_SONNET_4_6_TOP_P_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="top_p",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "top_p": 0.83},
    expected_provider_request=with_field(CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("top_p",), 0.83),
)
CLAUDE_SONNET_4_6_STOP_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="stop",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "stop": ["<<HALT>>"]},
    expected_provider_request=with_field(
        CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("stop_sequences",), ["<<HALT>>"]
    ),
)
CLAUDE_SONNET_4_6_USER_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="user",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "user": "user-sentinel-01"},
    expected_provider_request=with_field(
        CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("metadata", "user_id"), "user-sentinel-01"
    ),
)

CASES: Final = (
    CLAUDE_SONNET_4_6_TEMPERATURE_TEST_CASE,
    CLAUDE_SONNET_4_6_TOP_P_TEST_CASE,
    CLAUDE_SONNET_4_6_STOP_TEST_CASE,
    CLAUDE_SONNET_4_6_USER_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_params_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
