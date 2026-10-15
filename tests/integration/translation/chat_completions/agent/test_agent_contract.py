import json
from collections.abc import Mapping
from itertools import chain
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.provider import SharedProvider
from integration.translation.chat_completions.agent.case import (
    CASES,
    AgentCase,
    Exchange,
    assert_extensions,
    reference_for,
)
from integration.translation.chat_completions.agent.contracts import (
    FOLLOWUP,
    Usage,
    assembled_tools,
    assert_completion,
    assert_final_text,
    tool_calls_shape,
    tool_results,
    tool_shape,
)
from integration.translation.chat_completions.templates import CODING_AGENT_REQUEST
from integration.translation.runner import parse_sse_frames
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import JsonValue, TypeAdapter

ENDPOINT: Final = "/v1/chat/completions"
MESSAGES: Final = TypeAdapter(list[ChatCompletionMessageParam])
CASE_IDS: Final = tuple(f"{case.tier}-{case.model}" for case in CASES)


def assert_upstream(provider: SharedProvider, exchange: Exchange) -> None:
    received: Final = provider.received()
    assert len(received) == 1
    assert received[0].method == "POST"
    assert received[0].target == exchange.target
    assert json.loads(received[0].body) == exchange.request


def tools_response(case: AgentCase, gateway: Gateway, provider: SharedProvider) -> dict[str, JsonValue]:
    provider.expect(case.tools.reply())
    response: Final = gateway.request("POST", ENDPOINT, {**CODING_AGENT_REQUEST, "model": case.model})
    assert_upstream(provider, case.tools)
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert_extensions(body, case)
    return body


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_same_coding_request_has_the_same_tool_shape_within_tier(
    case: AgentCase, gateway: Gateway, provider: SharedProvider
) -> None:
    reference: Final = tools_response(reference_for(case), gateway, provider)
    response: Final = tools_response(case, gateway, provider)
    assert tool_shape(response) == tool_shape(reference)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_returned_assistant_message_and_tool_ids_replay_into_a_final_answer(
    case: AgentCase, gateway: Gateway, provider: SharedProvider
) -> None:
    response: Final = tools_response(case, gateway, provider)
    tool_shape(response)
    completion: Final = assert_completion(response)
    messages: Final = CODING_AGENT_REQUEST["messages"]
    assert isinstance(messages, list)
    message: Final = JSON_OBJECT.validate_python(
        completion.choices[0].message.model_dump(mode="json", exclude_none=True)
    )
    followup: Final[dict[str, JsonValue]] = {
        **CODING_AGENT_REQUEST,
        "model": case.model,
        "messages": [*messages, message, *tool_results(completion.choices[0].message), FOLLOWUP],
    }
    provider.expect(case.followup.reply())
    answer: Final = gateway.request("POST", ENDPOINT, followup)
    assert_upstream(provider, case.followup)
    assert answer.status_code == 200, answer.text
    assert_final_text(JSON_OBJECT.validate_json(answer.content))


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_raw_agent_stream_preserves_tool_indexes_finish_reason_and_usage(
    case: AgentCase, gateway: Gateway, provider: SharedProvider
) -> None:
    provider.expect(case.stream.reply())
    response: Final = gateway.request(
        "POST",
        ENDPOINT,
        {**CODING_AGENT_REQUEST, "model": case.model, "stream": True, "stream_options": {"include_usage": True}},
    )
    assert_upstream(provider, case.stream)
    assert response.status_code == 200, response.text
    frames: Final = parse_sse_frames(response.text)
    assert frames and frames[-1].data == "[DONE]", response.text
    chunks: Final = tuple(frame.data for frame in frames[:-1])
    deltas: Final = tuple(chain.from_iterable(assert_chunk(chunk) for chunk in chunks))
    roles: Final = tuple(delta["role"] for delta in deltas if delta.get("role"))
    assert roles and all(role == "assistant" for role in roles)
    tool_calls_shape(assembled_tools(deltas))
    usage: Final = tuple(chunk["usage"] for chunk in chunks if isinstance(chunk, dict) and chunk.get("usage"))
    assert usage, response.text
    assert isinstance(chunks[-1], dict) and chunks[-1].get("usage"), response.text
    tokens: Final = Usage.model_validate(usage[-1])
    assert tokens.total_tokens == tokens.prompt_tokens + tokens.completion_tokens
    terminal: Final = tuple(_finish_reasons(chunk) for chunk in chunks if isinstance(chunk, dict))
    assert tuple(chain.from_iterable(terminal)) == ("tool_calls",)


def assert_chunk(chunk: Mapping[str, JsonValue] | str) -> tuple[dict[str, JsonValue], ...]:
    assert isinstance(chunk, dict), chunk
    assert chunk["object"] == "chat.completion.chunk"
    assert isinstance(chunk["id"], str) and chunk["id"]
    assert type(chunk["created"]) is int
    assert isinstance(chunk["model"], str) and chunk["model"]
    choices: Final = chunk["choices"]
    assert isinstance(choices, list)
    return tuple(assert_stream_choice(choice) for choice in choices)


def assert_stream_choice(choice: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(choice, dict) and type(choice["index"]) is int and choice["index"] == 0
    delta: Final = choice["delta"]
    assert isinstance(delta, dict)
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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_openai_sdk_stream_helper_assembles_the_agent_tool_calls(
    case: AgentCase, gateway: Gateway, provider: SharedProvider
) -> None:
    if "haiku" in case.model:
        pytest.skip("BUG: LIT-9653, reasoning thinking_blocks lack index and crash OpenAI SDK stream assembly")
    provider.expect(case.stream.reply())
    with OpenAI(
        api_key=gateway.key, base_url=str(gateway.client.base_url).rstrip("/") + "/v1", max_retries=0
    ) as client:
        with client.chat.completions.stream(
            model=case.model,
            messages=MESSAGES.validate_python(CODING_AGENT_REQUEST["messages"]),
            extra_body={
                **{key: value for key, value in CODING_AGENT_REQUEST.items() if key != "messages"},
                "stream_options": {"include_usage": True},
            },
        ) as stream:
            for event in stream:
                assert event.type
            final: Final = stream.get_final_completion()
    assert_upstream(provider, case.stream)
    assert tool_shape(JSON_OBJECT.validate_python(final.model_dump(mode="json")))
