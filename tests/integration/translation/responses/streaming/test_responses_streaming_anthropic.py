from collections.abc import Mapping
from dataclasses import replace
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import SSEFrame, StreamingTranslationTestCase, TranslationTestCase
from integration.translation.chat_completions.cross_model.anthropic import CLAUDE_HAIKU_5_5_AGENT_STREAMING_TEST_CASE
from integration.translation.responses.bases.anthropic import CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_streaming_translation, assert_translation
from pydantic import JsonValue

SEARCH_CALL_ID: Final = "toolu_0177HUb25ArysRAu8eF3u4DJ"
RUN_TESTS_CALL_ID: Final = "toolu_011gegGWvYoxPwxWY7LSK4Uk"
SEARCH_ARGUMENT_DELTAS: Final = ('{"query', '": "ret', "ries_re", "maining", '"}')
RUN_TESTS_ARGUMENT_DELTAS: Final = ('{"path"', ': "test', "s/test_", "retry.p", 'y"}')
SEARCH_ARGUMENTS: Final = '{"query": "retries_remaining"}'
RUN_TESTS_ARGUMENTS: Final = '{"path": "tests/test_retry.py"}'


def _function_call(call_id: str, name: str, arguments: str, status: str) -> Mapping[str, JsonValue]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "namespace": None,
        "status": status,
        "arguments": arguments,
    }


def _event(sequence_number: int, data: Mapping[str, JsonValue]) -> SSEFrame:
    event_type: Final = data["type"]
    assert isinstance(event_type, str)
    return SSEFrame(event=event_type, data={**data, "sequence_number": sequence_number})


def _function_call_events(
    first_sequence_number: int,
    output_index: int,
    call: tuple[str, str, tuple[str, ...]],
    obfuscation: Mapping[str, JsonValue],
) -> tuple[SSEFrame, ...]:
    call_id, name, deltas = call
    arguments: Final = "".join(deltas)
    return (
        _event(
            first_sequence_number,
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": _function_call(call_id, name, "", "in_progress"),
            },
        ),
        *(
            _event(
                first_sequence_number + 1 + offset,
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": f"fc_{call_id}",
                    "output_index": output_index,
                    "delta": delta,
                    **obfuscation,
                },
            )
            for offset, delta in enumerate(deltas)
        ),
        _event(
            first_sequence_number + 1 + len(deltas),
            {
                "type": "response.function_call_arguments.done",
                "item_id": f"fc_{call_id}",
                "name": name,
                "output_index": output_index,
                "arguments": arguments,
            },
        ),
        _event(
            first_sequence_number + 2 + len(deltas),
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": _function_call(call_id, name, arguments, "completed"),
            },
        ),
    )


COMPLETED_RESPONSE: Final[Mapping[str, JsonValue]] = {
    **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
    "output": [
        _function_call(SEARCH_CALL_ID, "search_code", SEARCH_ARGUMENTS, "completed"),
        _function_call(RUN_TESTS_CALL_ID, "run_tests", RUN_TESTS_ARGUMENTS, "completed"),
    ],
    "usage": {
        "input_tokens": 1184,
        "input_tokens_details": {
            "audio_tokens": None,
            "cached_tokens": 0,
            "cached_tokens_details": None,
            "image_tokens": None,
            "text_tokens": 1184,
            "video_tokens": None,
            "cache_write_tokens": 0,
        },
        "output_tokens": 105,
        "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 105},
        "total_tokens": 1289,
        "cost": ANY,
    },
    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
}

IN_PROGRESS_RESPONSE: Final[Mapping[str, JsonValue]] = {
    **COMPLETED_RESPONSE,
    "status": "in_progress",
    "output": [],
    "usage": None,
}


def _expected_frames(obfuscation: Mapping[str, JsonValue]) -> tuple[SSEFrame, ...]:
    return (
        _event(0, {"type": "response.created", "response": IN_PROGRESS_RESPONSE}),
        _event(1, {"type": "response.in_progress", "response": IN_PROGRESS_RESPONSE}),
        *_function_call_events(2, 0, (SEARCH_CALL_ID, "search_code", SEARCH_ARGUMENT_DELTAS), obfuscation),
        *_function_call_events(10, 1, (RUN_TESTS_CALL_ID, "run_tests", RUN_TESTS_ARGUMENT_DELTAS), obfuscation),
        _event(18, {"type": "response.completed", "response": COMPLETED_RESPONSE}),
    )


CLAUDE_HAIKU_5_5_STREAMING_TEST_CASE: Final = StreamingTranslationTestCase(
    scenario="streaming",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "stream": True},
    expected_provider_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_endpoint,
    expected_provider_headers=CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_headers,
    expected_provider_request={**CLAUDE_HAIKU_5_5_TEST_CASE.expected_provider_request, "stream": True},
    mock_provider_frames=CLAUDE_HAIKU_5_5_AGENT_STREAMING_TEST_CASE.mock_provider_frames,
    expected_litellm_frames=_expected_frames({}),
    expected_litellm_response=COMPLETED_RESPONSE,
)

CLAUDE_HAIKU_5_5_STREAM_OPTIONS_NO_OBFUSCATION_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_STREAMING_TEST_CASE,
    scenario="stream_options_no_obfuscation",
    litellm_request={
        **CLAUDE_HAIKU_5_5_STREAMING_TEST_CASE.litellm_request,
        "stream_options": {"include_obfuscation": False},
    },
)

CLAUDE_HAIKU_5_5_STREAM_OPTIONS_OBFUSCATION_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_STREAMING_TEST_CASE,
    scenario="stream_options_obfuscation",
    litellm_request={
        **CLAUDE_HAIKU_5_5_STREAMING_TEST_CASE.litellm_request,
        "stream_options": {"include_obfuscation": True},
    },
    expected_litellm_frames=_expected_frames({"obfuscation": ANY}),
)

CLAUDE_HAIKU_5_5_STREAM_OPTIONS_WITHOUT_STREAM_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="stream_options_without_stream",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "stream_options": {"include_obfuscation": True, "include_usage": True},
    },
)


STREAM_SHAPE_BUG: Final = (
    "Responses stream events drift from the OpenAI shape: every event carries an extra top-level model key, "
    "created and in_progress echo defaults instead of the request (reasoning.effort null, parallel_tool_calls true, "
    "top_p 1.0, store true, no max_output_tokens), function_call_arguments.done has no name, item 0 is only "
    "finished after item 1 streams, and response.completed drops fields the non-stream response has"
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_STREAMING_TEST_CASE, CLAUDE_HAIKU_5_5_STREAM_OPTIONS_NO_OBFUSCATION_TEST_CASE],
    ids=lambda case: case.id,
)
def test_responses_streaming_anthropic(
    case: StreamingTranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    pytest.skip(f"BUG: {STREAM_SHAPE_BUG}")
    assert_streaming_translation(case, gateway, provider)


@pytest.mark.parametrize("case", [CLAUDE_HAIKU_5_5_STREAM_OPTIONS_WITHOUT_STREAM_TEST_CASE], ids=lambda case: case.id)
def test_responses_stream_options_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)


def test_responses_streaming_include_obfuscation_anthropic(gateway: Gateway, provider: SharedProvider) -> None:
    pytest.skip(
        f"BUG: include_obfuscation=true never adds an obfuscation field to streamed delta events. Also: {STREAM_SHAPE_BUG}"
    )
    assert_streaming_translation(CLAUDE_HAIKU_5_5_STREAM_OPTIONS_OBFUSCATION_TEST_CASE, gateway, provider)
