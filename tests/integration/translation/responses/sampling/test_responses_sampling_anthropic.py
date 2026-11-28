from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import RejectionTestCase, TranslationTestCase
from integration.translation.responses.bases.anthropic import CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_rejection, assert_translation
from pydantic import JsonValue


def unsupported_params_error(message: str) -> Mapping[str, JsonValue]:
    return {
        "error": {
            "message": f"litellm.UnsupportedParamsError: {message}\n\nLiteLLM: model group "
            "'anthropic/claude-haiku-5-5' failed with the error above. No fallback was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    }


def chat_layer_unsupported_params_message(fields: str) -> str:
    return (
        f"anthropic does not support parameters: {fields}, for model=claude-haiku-5-5. To drop these, set "
        "`litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n. \n If you want to "
        f"use these params dynamically send allowed_openai_params={fields} in your request."
    )


def unsupported_service_tier_message(tier: str) -> str:
    return (
        f"claude-haiku-5-5 does not support service_tier='{tier}'. Anthropic accepts 'auto', 'standard_only', or "
        "OpenAI's 'default' (sent as 'standard_only'). To drop unsupported params, set `litellm.drop_params = True`."
    )


CLAUDE_HAIKU_5_5_TEMPERATURE_ONE_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="temperature_one",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "temperature": 1},
    expected_provider_request={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request, "temperature": 1},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "temperature": 1},
)

CLAUDE_HAIKU_5_5_TEMPERATURE_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="temperature_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "temperature": 0.4},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error(chat_layer_unsupported_params_message("['temperature']")),
)

CLAUDE_HAIKU_5_5_TEMPERATURE_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="temperature_dropped",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "temperature": 0.4, "drop_params": True},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "temperature": 0.4},
)

CLAUDE_HAIKU_5_5_TOP_P_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="top_p_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "top_p": 0.9},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error(chat_layer_unsupported_params_message("['top_p']")),
)

CLAUDE_HAIKU_5_5_TOP_P_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="top_p_dropped",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "top_p": 0.9, "drop_params": True},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "top_p": 0.9},
)

CLAUDE_HAIKU_5_5_MAX_OUTPUT_TOKENS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="max_output_tokens",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "max_output_tokens": 2048},
    expected_provider_request={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request, "max_tokens": 2048},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "max_output_tokens": 2048},
)

CLAUDE_HAIKU_5_5_TOP_LOGPROBS_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="top_logprobs_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "top_logprobs": 3},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error(
        chat_layer_unsupported_params_message("['logprobs', 'top_logprobs']")
    ),
)

CLAUDE_HAIKU_5_5_TOP_LOGPROBS_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="top_logprobs_dropped",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "top_logprobs": 3, "drop_params": True},
)

CLAUDE_HAIKU_5_5_USER_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="user",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "user": "synthetic-user-01"},
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "metadata": {"user_id": "synthetic-user-01"},
    },
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "user": "synthetic-user-01"},
)

CLAUDE_HAIKU_5_5_USER_EMAIL_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="user_email_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "user": "synthetic.user@example.com"},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error(chat_layer_unsupported_params_message("['user']")),
)

CLAUDE_HAIKU_5_5_USER_EMAIL_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="user_email_dropped",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "user": "synthetic.user@example.com",
        "drop_params": True,
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "user": "synthetic.user@example.com",
    },
)

CLAUDE_HAIKU_5_5_SAFETY_IDENTIFIER_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="safety_identifier_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "safety_identifier": "synthetic-safety-01"},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error(chat_layer_unsupported_params_message("['safety_identifier']")),
)

CLAUDE_HAIKU_5_5_SAFETY_IDENTIFIER_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="safety_identifier_dropped",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "safety_identifier": "synthetic-safety-01",
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_SERVICE_TIER_AUTO_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="service_tier_auto",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "service_tier": "auto"},
    expected_provider_request={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request, "service_tier": "auto"},
)

CLAUDE_HAIKU_5_5_SERVICE_TIER_DEFAULT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="service_tier_default",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "service_tier": "default"},
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "service_tier": "standard_only",
    },
)

CLAUDE_HAIKU_5_5_SERVICE_TIER_FLEX_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="service_tier_flex_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "service_tier": "flex"},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error(unsupported_service_tier_message("flex")),
)

CLAUDE_HAIKU_5_5_SERVICE_TIER_FLEX_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="service_tier_flex_dropped",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "service_tier": "flex", "drop_params": True},
)

GREEN_CASES: Final = (
    CLAUDE_HAIKU_5_5_TEMPERATURE_UNSUPPORTED_TEST_CASE,
    CLAUDE_HAIKU_5_5_TEMPERATURE_DROPPED_TEST_CASE,
    CLAUDE_HAIKU_5_5_TOP_P_UNSUPPORTED_TEST_CASE,
    CLAUDE_HAIKU_5_5_TOP_P_DROPPED_TEST_CASE,
    CLAUDE_HAIKU_5_5_MAX_OUTPUT_TOKENS_TEST_CASE,
    CLAUDE_HAIKU_5_5_TOP_LOGPROBS_DROPPED_TEST_CASE,
    CLAUDE_HAIKU_5_5_USER_TEST_CASE,
    CLAUDE_HAIKU_5_5_USER_EMAIL_DROPPED_TEST_CASE,
    CLAUDE_HAIKU_5_5_SAFETY_IDENTIFIER_UNSUPPORTED_TEST_CASE,
    CLAUDE_HAIKU_5_5_SAFETY_IDENTIFIER_DROPPED_TEST_CASE,
    CLAUDE_HAIKU_5_5_SERVICE_TIER_AUTO_TEST_CASE,
    CLAUDE_HAIKU_5_5_SERVICE_TIER_DEFAULT_TEST_CASE,
    CLAUDE_HAIKU_5_5_SERVICE_TIER_FLEX_UNSUPPORTED_TEST_CASE,
    CLAUDE_HAIKU_5_5_SERVICE_TIER_FLEX_DROPPED_TEST_CASE,
)


@pytest.mark.parametrize("case", GREEN_CASES, ids=lambda case: case.id)
def test_responses_sampling_anthropic(
    case: TranslationTestCase | RejectionTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    match case:
        case RejectionTestCase():
            assert_rejection(case, gateway, provider)
        case TranslationTestCase():
            assert_translation(case, gateway, provider)


def test_responses_temperature_one_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip("BUG: temperature=1 is rejected for Claude Haiku 5.5 although it is the one value the model accepts")
    assert_translation(CLAUDE_HAIKU_5_5_TEMPERATURE_ONE_TEST_CASE, gateway, provider)


def test_responses_top_logprobs_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: top_logprobs reaches chat completions without logprobs=true, so the error names only top_logprobs"
    )
    assert_rejection(CLAUDE_HAIKU_5_5_TOP_LOGPROBS_UNSUPPORTED_TEST_CASE, gateway, provider)


def test_responses_user_email_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip("BUG: an email user is silently dropped instead of a 400 that mentions drop_params")
    assert_rejection(CLAUDE_HAIKU_5_5_USER_EMAIL_UNSUPPORTED_TEST_CASE, gateway, provider)
