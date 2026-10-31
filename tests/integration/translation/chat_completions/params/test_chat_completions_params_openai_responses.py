from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.openai_responses import GPT_5_6_SOL_TEST_CASE
from integration.translation.params import with_field
from integration.translation.runner import assert_translation

GPT_5_6_SOL_USER_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="user",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "user": "user-sentinel-01"},
    expected_provider_request=with_field(
        GPT_5_6_SOL_TEST_CASE.expected_provider_request, ("user",), "user-sentinel-01"
    ),
)
GPT_5_6_SOL_SERVICE_TIER_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="service_tier",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "service_tier": "flex"},
    expected_provider_request=with_field(GPT_5_6_SOL_TEST_CASE.expected_provider_request, ("service_tier",), "flex"),
)
GPT_5_6_SOL_PROMPT_CACHE_KEY_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="prompt_cache_key",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "prompt_cache_key": "pck-sentinel-01"},
    expected_provider_request=with_field(
        GPT_5_6_SOL_TEST_CASE.expected_provider_request, ("prompt_cache_key",), "pck-sentinel-01"
    ),
)
GPT_5_6_SOL_SAFETY_IDENTIFIER_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="safety_identifier",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "safety_identifier": "safety-sentinel-01"},
    expected_provider_request=with_field(
        GPT_5_6_SOL_TEST_CASE.expected_provider_request, ("safety_identifier",), "safety-sentinel-01"
    ),
)
GPT_5_6_SOL_STORE_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="store",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "store": False},
    expected_provider_request=with_field(GPT_5_6_SOL_TEST_CASE.expected_provider_request, ("store",), False),
)
GPT_5_6_SOL_VERBOSITY_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="verbosity",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "verbosity": "low"},
    expected_provider_request=with_field(GPT_5_6_SOL_TEST_CASE.expected_provider_request, ("text", "verbosity"), "low"),
)

CASES: Final = (
    GPT_5_6_SOL_USER_TEST_CASE,
    GPT_5_6_SOL_SERVICE_TIER_TEST_CASE,
    GPT_5_6_SOL_PROMPT_CACHE_KEY_TEST_CASE,
    GPT_5_6_SOL_SAFETY_IDENTIFIER_TEST_CASE,
    GPT_5_6_SOL_STORE_TEST_CASE,
    GPT_5_6_SOL_VERBOSITY_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_params_openai_responses(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
