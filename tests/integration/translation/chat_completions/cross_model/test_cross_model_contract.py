from collections.abc import Mapping
from dataclasses import replace
from itertools import chain
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.provider import SharedProvider
from integration.translation.case import (
    StreamingTranslationTestCase,
    TranslationTestCase,
)
from integration.translation.chat_completions.cross_model.case import (
    CASES,
    reference_for,
)
from integration.translation.chat_completions.cross_model.contracts import (
    assembled_message,
    assert_completion,
    assert_final_text,
    assert_stream_preserved,
    followup_request,
    response_structure,
    tool_calls_shape,
    tool_shape,
)
from integration.translation.runner import (
    assert_provider_request,
    assert_translation,
    parse_sse_frames,
    streaming_reply,
)
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import JsonValue, TypeAdapter

MESSAGES: Final = TypeAdapter(list[ChatCompletionMessageParam])
CASE_IDS: Final = tuple(str(case.litellm_request["model"]) for case, _, _ in CASES)


def assert_upstream(provider: SharedProvider, case: StreamingTranslationTestCase) -> None:
    assert_provider_request(
        provider.received(),
        case.expected_provider_endpoint,
        case.expected_provider_headers,
        case.expected_provider_request,
    )


@pytest.mark.parametrize("case", tuple(case for case, _, _ in CASES), ids=CASE_IDS)
def test_same_coding_request_has_the_same_response_structure_within_tier(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    reference: Final = assert_translation(reference_for(case), gateway, provider)
    response: Final = assert_translation(case, gateway, provider)
    assert response_structure(response) == response_structure(reference)
    assert tool_shape(response) == tool_shape(reference)


@pytest.mark.parametrize("case,followup", tuple((case, followup) for case, followup, _ in CASES), ids=CASE_IDS)
def test_returned_assistant_message_and_tool_ids_replay_into_a_final_answer(
    case: TranslationTestCase, followup: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    response: Final = assert_translation(case, gateway, provider)
    tool_shape(response)
    replay: Final = replace(followup, litellm_request=followup_request(case.litellm_request, response))
    final: Final = assert_translation(replay, gateway, provider)
    assert_final_text(final)
    reference: Final = assert_translation(reference_for(followup), gateway, provider)
    assert response_structure(final) == response_structure(reference)


@pytest.mark.parametrize("case", tuple(stream for _, _, stream in CASES), ids=CASE_IDS)
def test_raw_agent_stream_preserves_tool_indexes_finish_reason_and_usage(
    case: StreamingTranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    provider.expect(streaming_reply(case))
    response: Final = gateway.request("POST", case.litellm_endpoint, case.litellm_request)
    assert_upstream(provider, case)
    assert response.status_code == 200, response.text
    frames: Final = parse_sse_frames(response.text)
    assert frames and tuple(frame.data == "[DONE]" for frame in frames) == (False,) * (len(frames) - 1) + (True,)
    chunks: Final = tuple(frame.data for frame in frames[:-1])
    deltas: Final = tuple(chain.from_iterable(assert_chunk(chunk) for chunk in chunks))
    roles: Final = tuple(delta["role"] for delta in deltas if delta.get("role"))
    assert roles and all(role == "assistant" for role in roles)
    message: Final = assembled_message(deltas)
    assert message.tool_calls is not None
    tool_calls_shape(tuple(message.tool_calls))
    usage: Final = tuple(chunk["usage"] for chunk in chunks if isinstance(chunk, dict) and chunk.get("usage"))
    assert usage, response.text
    assert isinstance(chunks[-1], dict) and chunks[-1].get("usage"), response.text
    assert_stream_preserved(message, object_value(usage[-1]), case)
    terminal: Final = tuple(_finish_reasons(chunk) for chunk in chunks if isinstance(chunk, dict))
    assert tuple(chain.from_iterable(terminal)) == ("tool_calls",)
    identifiers: Final = tuple(chunk["id"] for chunk in chunks if isinstance(chunk, Mapping))
    assert len(set(str(identifier) for identifier in identifiers)) == 1, identifiers


def assert_chunk(chunk: Mapping[str, JsonValue] | str) -> tuple[dict[str, JsonValue], ...]:
    assert isinstance(chunk, dict), chunk
    assert chunk["object"] == "chat.completion.chunk"
    assert isinstance(chunk["id"], str) and chunk["id"]
    assert type(chunk["created"]) is int and chunk["created"] >= 0
    assert isinstance(chunk["model"], str) and chunk["model"]
    choices: Final = chunk["choices"]
    assert isinstance(choices, list) and len(choices) <= 1
    return tuple(assert_stream_choice(choice) for choice in choices)


def assert_stream_choice(choice: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(choice, dict) and type(choice["index"]) is int and choice["index"] == 0
    delta: Final = choice["delta"]
    assert isinstance(delta, dict)
    assert choice.get("finish_reason") in (None, "tool_calls")
    assert delta.get("role") in (None, "assistant")
    calls: Final = delta.get("tool_calls") or []
    assert isinstance(calls, list)
    for call in calls:
        assert isinstance(call, dict) and type(call["index"]) is int
    return delta


def _finish_reasons(chunk: dict[str, JsonValue]) -> tuple[JsonValue, ...]:
    choices: Final = chunk["choices"]
    assert isinstance(choices, list)
    return tuple(
        choice["finish_reason"] for choice in choices if isinstance(choice, dict) and choice.get("finish_reason")
    )


@pytest.mark.parametrize("case", tuple(stream for _, _, stream in CASES), ids=CASE_IDS)
def test_openai_sdk_stream_helper_assembles_the_agent_tool_calls(
    case: StreamingTranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    model: Final = case.litellm_request["model"]
    assert isinstance(model, str)
    provider.expect(streaming_reply(case))
    with OpenAI(
        api_key=gateway.key, base_url=str(gateway.client.base_url).rstrip("/") + "/v1", max_retries=0
    ) as client:
        with client.chat.completions.stream(
            model=model,
            messages=MESSAGES.validate_python(case.litellm_request["messages"]),
            extra_body={
                **{
                    key: value
                    for key, value in case.litellm_request.items()
                    if key not in ("model", "messages", "stream")
                },
            },
        ) as stream:
            for event in stream:
                assert event.type
            final: Final = stream.get_final_completion()
    assert_upstream(provider, case)
    body: Final = JSON_OBJECT.validate_python(final.model_dump(mode="json"))
    assert tool_shape(body)
    completion: Final = assert_completion(body)
    assert_stream_preserved(completion.choices[0].message, object_value(body["usage"]), case)
