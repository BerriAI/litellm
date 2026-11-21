from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import RejectionTestCase, TranslationTestCase
from integration.translation.responses.bases.anthropic import CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_rejection, assert_translation

NEXT_FILE_SCHEMA: Final = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}

CLAUDE_HAIKU_5_5_TEXT_FORMAT_JSON_SCHEMA_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="text_format_json_schema",
    expected_provider_headers={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_headers,
        "anthropic-beta": "structured-outputs-2025-11-13",
    },
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "text": {"format": {"type": "json_schema", "name": "next_file", "schema": NEXT_FILE_SCHEMA, "strict": True}},
    },
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "output_format": {"type": "json_schema", "schema": NEXT_FILE_SCHEMA},
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "text": {"format": {"name": "next_file", "schema": NEXT_FILE_SCHEMA, "type": "json_schema", "strict": True}},
    },
)

CLAUDE_HAIKU_5_5_TEXT_VERBOSITY_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="text_verbosity_drop_params",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "text": {"verbosity": "low"}, "drop_params": True},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "text": {"verbosity": "low"}},
)

CLAUDE_HAIKU_5_5_TEXT_VERBOSITY_TEST_CASE: Final = RejectionTestCase(
    scenario="text_verbosity",
    litellm_endpoint="/v1/responses",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "text": {"verbosity": "low"}},
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": "litellm.UnsupportedParamsError: anthropic does not support parameters: ['verbosity'], for "
            "model=claude-haiku-5-5. To drop these, set `litellm.drop_params=True` or for proxy:\n\n"
            "`litellm_settings:\n drop_params: true`\n. \n If you want to use these params dynamically send "
            "allowed_openai_params=['verbosity'] in your request.\n\nLiteLLM: model group "
            "'anthropic/claude-haiku-5-5' failed with the error above. No fallback was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_TEXT_FORMAT_JSON_SCHEMA_TEST_CASE, CLAUDE_HAIKU_5_5_TEXT_VERBOSITY_DROP_PARAMS_TEST_CASE],
    ids=lambda case: case.id,
)
def test_responses_text_format_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)


def test_responses_text_verbosity_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    assert_rejection(CLAUDE_HAIKU_5_5_TEXT_VERBOSITY_TEST_CASE, gateway, provider)
