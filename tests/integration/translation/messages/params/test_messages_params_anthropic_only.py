from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.azure import GPT_5_4_TEST_CASE
from integration.translation.runner import assert_translation
from pydantic import JsonValue

ANTHROPIC_ONLY_PARAMS: Final = {"inference_geo": "us", "speed": "fast"}
AZURE_UNRECOGNIZED_ARGUMENT: Final[dict[str, JsonValue]] = {
    "error": {
        "message": "Unrecognized request argument supplied: inference_geo",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }
}

GPT_5_4_ANTHROPIC_ONLY_PARAMS_DROPPED_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="anthropic_only_params_dropped_under_drop_params",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, **ANTHROPIC_ONLY_PARAMS, "drop_params": True},
)

GPT_5_4_ANTHROPIC_ONLY_PARAMS_REFUSED_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="anthropic_only_params_refused_without_drop_params",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, **ANTHROPIC_ONLY_PARAMS},
    expected_provider_request={**GPT_5_4_TEST_CASE.expected_provider_request, **ANTHROPIC_ONLY_PARAMS},
    mock_provider_status_code=400,
    mock_provider_response=AZURE_UNRECOGNIZED_ARGUMENT,
    expected_litellm_status_code=400,
    expected_litellm_response={
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                "litellm.BadRequestError: AzureException BadRequestError - "
                "Unrecognized request argument supplied: inference_geo\n\n"
                "LiteLLM: model group 'azure/gpt-5.4' failed with the error above. No fallback was attempted."
            ),
        },
    },
)

CASES: Final = (
    GPT_5_4_ANTHROPIC_ONLY_PARAMS_DROPPED_TEST_CASE,
    GPT_5_4_ANTHROPIC_ONLY_PARAMS_REFUSED_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_messages_params_anthropic_only(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
