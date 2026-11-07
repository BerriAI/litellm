from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.azure import GPT_5_4_TEST_CASE
from integration.translation.runner import assert_translation

GPT_5_4_ANTHROPIC_ONLY_PARAMS_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="anthropic_only_params_dropped",
    litellm_request={
        **GPT_5_4_TEST_CASE.litellm_request,
        "inference_geo": "us",
        "mcp_servers": [{"type": "url", "url": "https://mcp.example.com/sse", "name": "example"}],
        "speed": "fast",
    },
)

CASES: Final = (GPT_5_4_ANTHROPIC_ONLY_PARAMS_TEST_CASE,)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_messages_params_anthropic_only(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
