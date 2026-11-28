from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import RejectionTestCase, TranslationTestCase
from integration.translation.responses.bases.anthropic import CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_rejection, assert_translation

CLAUDE_HAIKU_5_5_INCLUDE_REASONING_ENCRYPTED_CONTENT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="include_reasoning_encrypted_content",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "include": ["reasoning.encrypted_content"]},
)

CLAUDE_HAIKU_5_5_INCLUDE_LOGPROBS_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="include_logprobs_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "include": ["message.output_text.logprobs"],
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_INCLUDE_CODE_INTERPRETER_OUTPUTS_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="include_code_interpreter_outputs_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "include": ["code_interpreter_call.outputs"],
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_INCLUDE_LOGPROBS_TEST_CASE: Final = RejectionTestCase(
    scenario="include_logprobs",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "include": ["message.output_text.logprobs"]},
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": "litellm.UnsupportedParamsError: anthropic does not support parameters: ['logprobs'], for "
            "model=claude-haiku-5-5. To drop these, set `litellm.drop_params=True` or for proxy:\n\n"
            "`litellm_settings:\n drop_params: true`\n. \n If you want to use these params dynamically send "
            "allowed_openai_params=['logprobs'] in your request.\n\nLiteLLM: model group "
            "'anthropic/claude-haiku-5-5' failed with the error above. No fallback was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)

CLAUDE_HAIKU_5_5_INCLUDE_CODE_INTERPRETER_OUTPUTS_TEST_CASE: Final = RejectionTestCase(
    scenario="include_code_interpreter_outputs",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "include": ["code_interpreter_call.outputs"]},
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": "litellm.UnsupportedParamsError: anthropic does not support parameters: ['include'], "
            "for model=claude-haiku-5-5. The Responses API to chat completions bridge cannot translate them. To "
            "drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n. "
            "\n If you want to use these params dynamically send allowed_openai_params=['include'] in your "
            "request.\n\nLiteLLM: model group 'anthropic/claude-haiku-5-5' failed with the error above. No fallback "
            "was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)


@pytest.mark.parametrize(
    "case",
    [
        CLAUDE_HAIKU_5_5_INCLUDE_REASONING_ENCRYPTED_CONTENT_TEST_CASE,
        CLAUDE_HAIKU_5_5_INCLUDE_LOGPROBS_DROP_PARAMS_TEST_CASE,
        CLAUDE_HAIKU_5_5_INCLUDE_CODE_INTERPRETER_OUTPUTS_DROP_PARAMS_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_responses_include_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)


def test_responses_include_logprobs_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: include message.output_text.logprobs is silently dropped instead of asking chat completions for logprobs, which Anthropic rejects"
    )
    assert_rejection(CLAUDE_HAIKU_5_5_INCLUDE_LOGPROBS_TEST_CASE, gateway, provider)


def test_responses_include_code_interpreter_outputs_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: include code_interpreter_call.outputs is silently dropped instead of a 400 that names drop_params"
    )
    assert_rejection(CLAUDE_HAIKU_5_5_INCLUDE_CODE_INTERPRETER_OUTPUTS_TEST_CASE, gateway, provider)
