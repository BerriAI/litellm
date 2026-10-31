from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.azure import GPT_5_4_TEST_CASE
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
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "metadata": {"user_id": "meta-sentinel-01"}},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("user",), "meta-sentinel-01"),
)
GPT_5_4_SERVICE_TIER_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="service_tier",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "service_tier": "auto"},
    expected_provider_request=with_field(GPT_5_4_TEST_CASE.expected_provider_request, ("service_tier",), "auto"),
)

CASES: Final = (
    GPT_5_4_TEMPERATURE_TEST_CASE,
    GPT_5_4_TOP_P_TEST_CASE,
    GPT_5_4_METADATA_TEST_CASE,
    GPT_5_4_SERVICE_TIER_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_messages_params_azure(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
