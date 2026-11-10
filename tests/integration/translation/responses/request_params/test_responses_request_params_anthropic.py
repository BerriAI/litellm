from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import RejectionTestCase, TranslationTestCase
from integration.translation.responses.bases.anthropic import CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_rejection, assert_translation
from pydantic import JsonValue


def _bridge_rejection_message(field: str) -> str:
    return (
        f"litellm.UnsupportedParamsError: anthropic does not support parameters: ['{field}'], "
        "for model=claude-haiku-5-5. The Responses API to chat completions bridge cannot translate them. "
        "To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n. "
        f"\n If you want to use these params dynamically send allowed_openai_params=['{field}'] in your request."
        "\n\nLiteLLM: model group 'anthropic/claude-haiku-5-5' failed with the error above. "
        "No fallback was attempted."
    )


def _bridge_rejection_response(field: str) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": _bridge_rejection_message(field),
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    }


ACCESS_PROGRAMS: Final[dict[str, JsonValue]] = {"cyber": "daybreak_blue"}

MODERATION: Final[dict[str, JsonValue]] = {
    "model": "omni-moderation-latest",
    "policy": {"input": {"mode": "score"}, "output": {"mode": "block"}},
}

CLAUDE_HAIKU_5_5_INSTRUCTIONS: Final = CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request["instructions"]

CLAUDE_HAIKU_5_5_DEVELOPER_INPUT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="developer_input",
    litellm_request={
        **{key: value for key, value in CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request.items() if key != "instructions"},
        "input": [
            {"role": "developer", "content": CLAUDE_HAIKU_5_5_INSTRUCTIONS},
            *CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request["input"],
        ],
    },
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "instructions": None},
)

CLAUDE_HAIKU_5_5_ACCESS_PROGRAMS_TEST_CASE: Final = RejectionTestCase(
    scenario="access_programs",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "access_programs": ACCESS_PROGRAMS},
    expected_litellm_status_code=400,
    expected_litellm_response=_bridge_rejection_response("access_programs"),
)

CLAUDE_HAIKU_5_5_ACCESS_PROGRAMS_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="access_programs_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "access_programs": ACCESS_PROGRAMS,
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_MODERATION_TEST_CASE: Final = RejectionTestCase(
    scenario="moderation",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "moderation": MODERATION},
    expected_litellm_status_code=400,
    expected_litellm_response=_bridge_rejection_response("moderation"),
)

CLAUDE_HAIKU_5_5_MODERATION_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="moderation_drop_params",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "moderation": MODERATION, "drop_params": True},
)


@pytest.mark.parametrize(
    "case",
    [
        CLAUDE_HAIKU_5_5_DEVELOPER_INPUT_TEST_CASE,
        CLAUDE_HAIKU_5_5_MODERATION_TEST_CASE,
        CLAUDE_HAIKU_5_5_MODERATION_DROP_PARAMS_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_responses_request_params_anthropic(
    case: TranslationTestCase | RejectionTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    match case:
        case RejectionTestCase():
            assert_rejection(case, gateway, provider)
        case TranslationTestCase():
            assert_translation(case, gateway, provider)


def test_responses_request_params_anthropic_access_programs_rejected(
    gateway: Gateway, provider: SharedProvider
) -> None:
    pytest.skip(
        "BUG: access_programs is sent to Anthropic inside the request body instead of being rejected with a 400"
    )
    assert_rejection(CLAUDE_HAIKU_5_5_ACCESS_PROGRAMS_TEST_CASE, gateway, provider)


def test_responses_request_params_anthropic_access_programs_dropped_with_drop_params(
    gateway: Gateway, provider: SharedProvider
) -> None:
    pytest.skip("BUG: access_programs still reaches the Anthropic request body when drop_params is true")
    assert_translation(CLAUDE_HAIKU_5_5_ACCESS_PROGRAMS_DROP_PARAMS_TEST_CASE, gateway, provider)
