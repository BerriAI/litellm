from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.openai import GPT_5_4_TEST_CASE
from integration.translation.params import with_field
from integration.translation.runner import assert_translation

GPT_5_4_TEMPERATURE_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="temperature",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "temperature": 0.37},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("temperature",), 0.37),
)
GPT_5_4_TOP_P_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="top_p",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "top_p": 0.83},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("top_p",), 0.83),
)
GPT_5_4_N_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="n",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "n": 2},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("n",), 2),
)
GPT_5_4_SEED_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="seed",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "seed": 4242},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("seed",), 4242),
)
GPT_5_4_USER_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="user",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "user": "user-sentinel-01"},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("user",), "user-sentinel-01"),
)
GPT_5_4_SERVICE_TIER_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="service_tier",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "service_tier": "flex"},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("service_tier",), "flex"),
)
GPT_5_4_PROMPT_CACHE_KEY_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="prompt_cache_key",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "prompt_cache_key": "pck-sentinel-01"},
    expected_provider_request=with_field(
        GPT_5_4_TEST_CASE.expected_provider_request, ("prompt_cache_key",), "pck-sentinel-01"
    ),
)
GPT_5_4_SAFETY_IDENTIFIER_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="safety_identifier",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "safety_identifier": "safety-sentinel-01"},
    expected_provider_request=with_field(
        GPT_5_4_TEST_CASE.expected_provider_request, ("safety_identifier",), "safety-sentinel-01"
    ),
)
GPT_5_4_STORE_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="store",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "store": False},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("store",), False),
)
GPT_5_4_VERBOSITY_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="verbosity",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "verbosity": "low"},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("verbosity",), "low"),
)

CASES: Final = (
    GPT_5_4_TEMPERATURE_TEST_CASE,
    GPT_5_4_TOP_P_TEST_CASE,
    GPT_5_4_N_TEST_CASE,
    GPT_5_4_SEED_TEST_CASE,
    GPT_5_4_USER_TEST_CASE,
    GPT_5_4_SERVICE_TIER_TEST_CASE,
    GPT_5_4_PROMPT_CACHE_KEY_TEST_CASE,
    GPT_5_4_SAFETY_IDENTIFIER_TEST_CASE,
    GPT_5_4_STORE_TEST_CASE,
    GPT_5_4_VERBOSITY_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_params_openai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
