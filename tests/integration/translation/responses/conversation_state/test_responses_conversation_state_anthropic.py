import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import RejectionTestCase, StreamingTranslationTestCase, TranslationTestCase
from integration.translation.chat_completions.cross_model.anthropic import CLAUDE_HAIKU_5_5_AGENT_STREAMING_TEST_CASE
from integration.translation.responses.bases.anthropic import (
    CLAUDE_HAIKU_5_5_TEST_CASE,
    CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
)
from integration.translation.runner import (
    assert_provider_request,
    assert_rejection,
    assert_translation,
    parse_sse_frames,
    streaming_reply,
)
from pydantic import JsonValue, TypeAdapter

MODEL: Final = "anthropic/claude-haiku-5-5"
READ_FILE_CALL_ID: Final = "toolu_01H53znxFvBbuHPXNePjEhW5"
READ_FILE_OUTPUT: Final = "def validate_quantity(quantity):\n    return quantity > 0\n"
UNKNOWN_RESPONSE_ID: Final = "resp_0d8c51f3a6e24b7f9e1a2c3d4e5f6a7b"
BASE_PROVIDER_MESSAGES: Final = TypeAdapter(list[JsonValue]).validate_python(
    CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request["messages"]
)


def _unsupported_param_response(param: str) -> Mapping[str, JsonValue]:
    return {
        "error": {
            "message": f"litellm.UnsupportedParamsError: anthropic does not support parameters: ['{param}'], for "
            "model=claude-haiku-5-5. The Responses API to chat completions bridge cannot translate them. To drop "
            "these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n. \n "
            f"If you want to use these params dynamically send allowed_openai_params=['{param}'] in your request."
            f"\n\nLiteLLM: model group '{MODEL}' failed with the error above. No fallback was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    }


def _unsupported_param_rejection(scenario: str, overrides: Mapping[str, JsonValue], param: str) -> RejectionTestCase:
    return RejectionTestCase(
        scenario=scenario,
        litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
        litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, **overrides},
        expected_litellm_status_code=400,
        expected_litellm_response=_unsupported_param_response(param),
    )


CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="previous_response_id",
    litellm_request={
        "model": MODEL,
        "cache": {"no-cache": True},
        "max_output_tokens": CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request["max_output_tokens"],
        "instructions": CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request["instructions"],
        "input": [{"type": "function_call_output", "call_id": READ_FILE_CALL_ID, "output": READ_FILE_OUTPUT}],
        "tools": CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request["tools"],
        "reasoning": CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request["reasoning"],
    },
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "messages": [
            *BASE_PROVIDER_MESSAGES,
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I'll just read the validation.py file directly, starting with a reasonable "
                        "range near the top of the file since no specific lines were requested.\n"
                        "\n",
                        "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                    },
                    {
                        "type": "tool_use",
                        "id": READ_FILE_CALL_ID,
                        "name": "read_file",
                        "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": READ_FILE_CALL_ID, "content": READ_FILE_OUTPUT}],
            },
        ],
    },
)

CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_TEST_CASE,
    scenario="previous_response_id_drop_params",
    litellm_request={**CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_TEST_CASE.litellm_request, "drop_params": True},
)

CLAUDE_HAIKU_5_5_CONVERSATION_TEST_CASE: Final = _unsupported_param_rejection(
    "conversation", {"conversation": "conv_synthetic_checkout_review"}, "conversation"
)

CLAUDE_HAIKU_5_5_CONVERSATION_OBJECT_TEST_CASE: Final = _unsupported_param_rejection(
    "conversation_object", {"conversation": {"id": "conv_synthetic_checkout_review"}}, "conversation"
)

CLAUDE_HAIKU_5_5_CONVERSATION_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="conversation_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "conversation": "conv_synthetic_checkout_review",
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_CONVERSATION_OBJECT_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="conversation_object_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "conversation": {"id": "conv_synthetic_checkout_review"},
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_STORE_TRUE_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="store_true",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "store": True},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "store": True},
)

CLAUDE_HAIKU_5_5_STORE_FALSE_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="store_false",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "store": False},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "store": False},
)

CLAUDE_HAIKU_5_5_STORE_FALSE_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="store_false_streaming",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "store": False, "stream": True},
    expected_provider_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_endpoint,
    expected_provider_headers=CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_headers,
    expected_provider_request={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request, "stream": True},
    mock_provider_frames=CLAUDE_HAIKU_5_5_AGENT_STREAMING_TEST_CASE.mock_provider_frames,
)

CLAUDE_HAIKU_5_5_BACKGROUND_TEST_CASE: Final = _unsupported_param_rejection(
    "background", {"background": True}, "background"
)

CLAUDE_HAIKU_5_5_BACKGROUND_FALSE_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="background_false",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "background": False},
)

CLAUDE_HAIKU_5_5_BACKGROUND_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="background_drop_params",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "background": True, "drop_params": True},
)

CLAUDE_HAIKU_5_5_TRUNCATION_DISABLED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="truncation_disabled",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "truncation": "disabled"},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "truncation": "disabled"},
)

CLAUDE_HAIKU_5_5_TRUNCATION_AUTO_TEST_CASE: Final = _unsupported_param_rejection(
    "truncation_auto", {"truncation": "auto"}, "truncation"
)

CLAUDE_HAIKU_5_5_TRUNCATION_AUTO_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="truncation_auto_drop_params",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "truncation": "auto", "drop_params": True},
    expected_litellm_response={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response, "truncation": "auto"},
)

CLAUDE_HAIKU_5_5_PROMPT_TEST_CASE: Final = _unsupported_param_rejection(
    "prompt",
    {"prompt": {"id": "pmpt_synthetic_checkout_review", "version": "2", "variables": {"project": "shop"}}},
    "prompt",
)

CLAUDE_HAIKU_5_5_PROMPT_DROP_PARAMS_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="prompt_drop_params",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "prompt": {"id": "pmpt_synthetic_checkout_review", "version": "2", "variables": {"project": "shop"}},
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_CONTEXT_MANAGEMENT_COMPACTION_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="context_management_compaction",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "context_management": [{"type": "compaction", "compact_threshold": 200000}],
    },
    expected_provider_headers={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_headers,
        "anthropic-beta": "compact-2026-01-12",
    },
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "context_management": {
            "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 200000}}]
        },
    },
)

CLAUDE_HAIKU_5_5_CONTEXT_MANAGEMENT_COMPACTION_DEFAULT_THRESHOLD_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_CONTEXT_MANAGEMENT_COMPACTION_TEST_CASE,
    scenario="context_management_compaction_default_threshold",
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "context_management": [{"type": "compaction"}]},
    expected_provider_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request,
        "context_management": {"edits": [{"type": "compact_20260112"}]},
    },
)

CLAUDE_HAIKU_5_5_METADATA_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="metadata",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "metadata": {"ticket": "CHECKOUT-142", "team": "checkout"},
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "metadata": {"ticket": "CHECKOUT-142", "team": "checkout"},
    },
)


def _first_turn_response_id(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> str:
    provider.expect(Reply(body=json.dumps(case.mock_provider_response).encode()))
    response: Final = gateway.request("POST", case.litellm_endpoint, case.litellm_request)
    assert_provider_request(
        provider.received(),
        case.expected_provider_endpoint,
        case.expected_provider_headers,
        case.expected_provider_request,
    )
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body == case.expected_litellm_response
    call_id: Final = response.headers["x-litellm-call-id"]
    eventually(
        lambda: gateway.request("GET", f"/spend/logs/ui/{call_id}").json(),
        lambda logged: logged is not None,
        seconds=70,
    )
    return body["id"]


def _assert_previous_response_not_found(
    case: TranslationTestCase, previous_response_id: str, gateway: Gateway, provider: SharedProvider
) -> None:
    response: Final = gateway.request(
        "POST", case.litellm_endpoint, {**case.litellm_request, "previous_response_id": previous_response_id}
    )
    assert provider.received() == ()
    assert response.status_code in (400, 404), response.text
    assert previous_response_id in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "case",
    [
        CLAUDE_HAIKU_5_5_STORE_TRUE_TEST_CASE,
        CLAUDE_HAIKU_5_5_STORE_FALSE_TEST_CASE,
        CLAUDE_HAIKU_5_5_BACKGROUND_FALSE_TEST_CASE,
        CLAUDE_HAIKU_5_5_BACKGROUND_DROP_PARAMS_TEST_CASE,
        CLAUDE_HAIKU_5_5_TRUNCATION_DISABLED_TEST_CASE,
        CLAUDE_HAIKU_5_5_TRUNCATION_AUTO_DROP_PARAMS_TEST_CASE,
        CLAUDE_HAIKU_5_5_PROMPT_DROP_PARAMS_TEST_CASE,
        CLAUDE_HAIKU_5_5_CONTEXT_MANAGEMENT_COMPACTION_TEST_CASE,
        CLAUDE_HAIKU_5_5_CONTEXT_MANAGEMENT_COMPACTION_DEFAULT_THRESHOLD_TEST_CASE,
        CLAUDE_HAIKU_5_5_METADATA_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_responses_conversation_state_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)


def test_previous_response_id_replays_the_stored_turn(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        "BUG: a previous_response_id follow-up drops every earlier turn unless prompts are stored in spend logs, "
        "and echoes the internal chatcmpl id as previous_response_id"
    )
    previous_response_id: Final = _first_turn_response_id(CLAUDE_HAIKU_5_5_TEST_CASE, gateway, provider)
    follow_up: Final = replace(
        CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_TEST_CASE,
        litellm_request={
            **CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_TEST_CASE.litellm_request,
            "previous_response_id": previous_response_id,
        },
        expected_litellm_response={
            **CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_TEST_CASE.expected_litellm_response,
            "previous_response_id": previous_response_id,
        },
    )
    assert_translation(follow_up, gateway, provider)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_TEST_CASE, CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_DROP_PARAMS_TEST_CASE],
    ids=lambda case: case.id,
)
def test_unknown_previous_response_id_is_rejected(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    pytest.skip("BUG: an unknown previous_response_id silently starts a fresh conversation instead of a 404")
    _assert_previous_response_not_found(case, UNKNOWN_RESPONSE_ID, gateway, provider)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_CONVERSATION_TEST_CASE, CLAUDE_HAIKU_5_5_CONVERSATION_OBJECT_TEST_CASE],
    ids=lambda case: case.id,
)
def test_conversation_is_rejected(case: RejectionTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip("BUG: conversation is passed raw into the Anthropic request body instead of a 400")
    assert_rejection(case, gateway, provider)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_CONVERSATION_DROP_PARAMS_TEST_CASE, CLAUDE_HAIKU_5_5_CONVERSATION_OBJECT_DROP_PARAMS_TEST_CASE],
    ids=lambda case: case.id,
)
def test_conversation_is_dropped_with_drop_params(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    pytest.skip("BUG: conversation still reaches the Anthropic request body with drop_params on")
    assert_translation(case, gateway, provider)


def test_store_false_turn_cannot_be_continued(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip("BUG: a store=false turn is still saved and a previous_response_id follow-up to it goes through")
    previous_response_id: Final = _first_turn_response_id(CLAUDE_HAIKU_5_5_STORE_FALSE_TEST_CASE, gateway, provider)
    _assert_previous_response_not_found(
        CLAUDE_HAIKU_5_5_PREVIOUS_RESPONSE_ID_TEST_CASE, previous_response_id, gateway, provider
    )


def test_store_false_streaming_reports_store_false(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip("BUG: a store=false stream says store=true in response.created and response.in_progress")
    case: Final = CLAUDE_HAIKU_5_5_STORE_FALSE_STREAMING_TEST_CASE
    provider.expect(streaming_reply(case))
    response: Final = gateway.request("POST", case.litellm_endpoint, case.litellm_request)
    assert_provider_request(
        provider.received(),
        case.expected_provider_endpoint,
        case.expected_provider_headers,
        case.expected_provider_request,
    )
    assert response.status_code == 200, response.text
    reported: Final = tuple(
        (frame.event, frame.data["response"].get("store"))
        for frame in parse_sse_frames(response.text)
        if isinstance(frame.data, Mapping) and isinstance(frame.data.get("response"), Mapping)
    )
    assert reported == (
        ("response.created", False),
        ("response.in_progress", False),
        ("response.completed", False),
    )


def test_background_is_rejected(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip("BUG: background=true silently runs in the foreground instead of a 400 naming drop_params")
    assert_rejection(CLAUDE_HAIKU_5_5_BACKGROUND_TEST_CASE, gateway, provider)


def test_truncation_auto_is_rejected(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip("BUG: truncation=auto is silently ignored instead of trimming or a 400 naming drop_params")
    assert_rejection(CLAUDE_HAIKU_5_5_TRUNCATION_AUTO_TEST_CASE, gateway, provider)


def test_prompt_is_rejected(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip("BUG: a prompt template reference is silently dropped instead of a 400 naming drop_params")
    assert_rejection(CLAUDE_HAIKU_5_5_PROMPT_TEST_CASE, gateway, provider)
