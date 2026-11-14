from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.openai import GPT_5_4_TEST_CASE
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
GPT_5_4_METADATA_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="metadata",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "metadata": {"trace_id": "meta-sentinel-01"}},
    expected_provider_request=with_field(
        GPT_5_4_TEST_CASE.expected_provider_request, ("metadata", "trace_id"), "meta-sentinel-01"
    ),
)
GPT_5_4_USER_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="user",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "user": "user-sentinel-01"},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("user",), "user-sentinel-01"),
)
GPT_5_4_SAFETY_IDENTIFIER_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="safety_identifier",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "safety_identifier": "safety-sentinel-01"},
    expected_provider_request=with_field(
        GPT_5_4_TEST_CASE.expected_provider_request, ("safety_identifier",), "safety-sentinel-01"
    ),
)
GPT_5_4_PROMPT_CACHE_KEY_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="prompt_cache_key",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "prompt_cache_key": "pck-sentinel-01"},
    expected_provider_request=with_field(
        GPT_5_4_TEST_CASE.expected_provider_request, ("prompt_cache_key",), "pck-sentinel-01"
    ),
)
GPT_5_4_TEXT_VERBOSITY_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="text_verbosity",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "text": {"verbosity": "low"}},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("text", "verbosity"), "low"),
)
GPT_5_4_STORE_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="store",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "store": False},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("store",), False),
)
GPT_5_4_INCLUDE_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="include",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "include": ["reasoning.encrypted_content"]},
    expected_provider_request=with_field(
        GPT_5_4_TEST_CASE.expected_provider_request, ("include",), ["reasoning.encrypted_content"]
    ),
)
GPT_5_4_TRUNCATION_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="truncation",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "truncation": "auto"},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("truncation",), "auto"),
)
GPT_5_4_SERVICE_TIER_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="service_tier",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "service_tier": "flex"},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("service_tier",), "flex"),
)
GPT_5_4_TOP_LOGPROBS_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="top_logprobs",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "top_logprobs": 3},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("top_logprobs",), 3),
)
GPT_5_4_PROMPT_CACHE_RETENTION_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="prompt_cache_retention",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "prompt_cache_retention": "24h"},
    expected_provider_request=with_field(
        GPT_5_4_TEST_CASE.expected_provider_request, ("prompt_cache_retention",), "24h"
    ),
)

CASES: Final = (
    GPT_5_4_TEMPERATURE_TEST_CASE,
    GPT_5_4_TOP_P_TEST_CASE,
    GPT_5_4_METADATA_TEST_CASE,
    GPT_5_4_USER_TEST_CASE,
    GPT_5_4_SAFETY_IDENTIFIER_TEST_CASE,
    GPT_5_4_PROMPT_CACHE_KEY_TEST_CASE,
    GPT_5_4_TEXT_VERBOSITY_TEST_CASE,
    GPT_5_4_STORE_TEST_CASE,
    GPT_5_4_INCLUDE_TEST_CASE,
    GPT_5_4_TRUNCATION_TEST_CASE,
    GPT_5_4_SERVICE_TIER_TEST_CASE,
    GPT_5_4_TOP_LOGPROBS_TEST_CASE,
    GPT_5_4_PROMPT_CACHE_RETENTION_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_responses_params_openai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
