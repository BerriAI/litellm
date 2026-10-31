from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_4_6_TEST_CASE
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
CLAUDE_SONNET_4_6_TOP_K_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="top_k",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "top_k": 7},
    expected_provider_request=with_field(CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("top_k",), 7),
)
CLAUDE_SONNET_4_6_STOP_SEQUENCES_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="stop_sequences",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "stop_sequences": ["<<HALT>>"]},
    expected_provider_request=with_field(
        CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("stop_sequences",), ["<<HALT>>"]
    ),
)
CLAUDE_SONNET_4_6_METADATA_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="metadata",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "metadata": {"user_id": "meta-sentinel-01"}},
    expected_provider_request=with_field(
        CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("metadata", "user_id"), "meta-sentinel-01"
    ),
)
CLAUDE_SONNET_4_6_SERVICE_TIER_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="service_tier",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "service_tier": "auto"},
    expected_provider_request=with_field(
        CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("service_tier",), "auto"
    ),
)
CLAUDE_SONNET_4_6_INFERENCE_GEO_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="inference_geo",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "inference_geo": "us"},
    expected_provider_request=with_field(
        CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("inference_geo",), "us"
    ),
)
CLAUDE_SONNET_4_6_CONTAINER_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="container",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "container": "container_sentinel01"},
    expected_provider_request=with_field(
        CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request, ("container",), "container_sentinel01"
    ),
)

CASES: Final = (
    CLAUDE_SONNET_4_6_TEMPERATURE_TEST_CASE,
    CLAUDE_SONNET_4_6_TOP_P_TEST_CASE,
    CLAUDE_SONNET_4_6_TOP_K_TEST_CASE,
    CLAUDE_SONNET_4_6_STOP_SEQUENCES_TEST_CASE,
    CLAUDE_SONNET_4_6_METADATA_TEST_CASE,
    CLAUDE_SONNET_4_6_SERVICE_TIER_TEST_CASE,
    CLAUDE_SONNET_4_6_INFERENCE_GEO_TEST_CASE,
    CLAUDE_SONNET_4_6_CONTAINER_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_messages_params_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
