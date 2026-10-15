import json
import uuid
from collections.abc import Sequence
from typing import Final, cast

import httpx
import litellm
import openai
import pytest
from integration._support.anthropic_thinking import JSON_LIST, JSON_OBJECT
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers._cache_control_marks_support import (
    ASK_LABEL,
    EPHEMERAL,
    POINTS,
    SYSTEM,
    SYSTEM_LABEL,
    anthropic_labels,
    anthropic_peer,
    ask,
    call_id,
    final_text,
    gateway_injected,
    post_chat,
    tool_call,
    tool_use_label,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolMessage
from litellm.types.utils import ChatCompletionMessageToolCall
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import JsonValue

from integration.providers._tool_search_support import (
    KEPT_FIELDS,
    WEATHER_ANSWER,
    SEARCH_CALL,
    SEARCH_ID,
    SEARCH_INPUT,
    SEARCH_RESULT,
    SEARCH_TOOL,
    SEARCH_TURN,
    SEARCH_USE,
    TOOLS,
    WEATHER_CALL,
    WEATHER_ID,
    WEATHER_USE,
    Event,
    Provider,
    answer,
    answer_text,
    answered_ids,
    assistant_from,
    chat,
    chat_body,
    deployment,
    first_messages,
    folded_assistant,
    hand_built_assistant,
    kept_results,
    marker_of,
    provider_fields,
    question,
    replay_messages,
    request_messages,
    response_id,
    scripted_peer,
    search_result,
    spend_row,
    target,
    target_path,
    tool_answer,
    tool_call_ids,
    turn_reply,
    user_texts,
    wire_assistant,
)

_REPLAYING_PROVIDERS: Final[tuple[Provider, ...]] = (
    "anthropic",
    "bedrock_invoke",
    "claude_platform",
    "bedrock_mantle",
    "vertex_ai",
    "azure_ai",
)
_STREAM_MODES: Final = pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
_PLAIN_SEARCH_USE: Final[Event] = {"type": "tool_use", "id": SEARCH_ID, "name": SEARCH_TOOL, "input": SEARCH_INPUT}
_BASH_ID: Final = "srvtoolu_01ScriptedBash"
_BASH_USE: Final[Event] = {
    "type": "server_tool_use",
    "id": _BASH_ID,
    "name": "bash_code_execution",
    "input": {"command": "echo sunny"},
}
_BASH_RESULT: Final[Event] = {
    "type": "bash_code_execution_tool_result",
    "tool_use_id": _BASH_ID,
    "content": {
        "type": "bash_code_execution_result",
        "stdout": "sunny\n",
        "stderr": "",
        "return_code": 0,
        "content": [],
    },
}
_SIGNED_THINKING: Final[Event] = {
    "type": "thinking",
    "thinking": "search for the weather tool first",
    "signature": "scripted-signature-" + "s" * 32,
}


def _round_trip(
    gateway: Gateway, provider: Provider, stream: bool, first_turn: Sequence[Event] = SEARCH_TURN
) -> tuple[Event, httpx.Response, httpx.Response, tuple[Request, ...], str]:
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer(first_turn)) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, wire.url, gateway.upstream_url)
        first: Final = chat(gateway, chat_body(model, first_messages(marker), stream))
        assert first.status_code == 200, first.text
        assistant: Final = assistant_from(first, stream)
        second: Final = chat(gateway, chat_body(model, replay_messages(marker, assistant), stream))
        assert second.status_code == 200, second.text
        assert answer_text(second, stream) == answer(marker), second.text
        received: Final = wire.drain()
    return assistant, first, second, received, model


@pytest.mark.parametrize("provider", _REPLAYING_PROVIDERS)
@_STREAM_MODES
def test_tool_search_result_round_trips_on_every_anthropic_wire_provider(
    gateway: Gateway, provider: Provider, stream: bool
) -> None:
    assistant, first, second, received, model = _round_trip(gateway, provider, stream)
    assert [target_path(request) for request in received] == [target(provider, stream)] * 2, received
    assert kept_results(assistant) == [SEARCH_RESULT], assistant
    assert "code_interpreter_results" not in provider_fields(assistant), assistant
    assert tool_call_ids(assistant) == [SEARCH_ID, WEATHER_ID], assistant
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body
    for identity in (response_id(first, stream), response_id(second, stream)):
        row = spend_row(identity)
        assert (row["model_group"], row["status"]) == (model, "success"), row


def test_snowflake_claude_stream_keeps_the_tool_search_result_on_the_response(gateway: Gateway) -> None:
    stream: Final = True
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "snowflake", wire.url, gateway.upstream_url)
        first: Final = chat(gateway, chat_body(model, first_messages(marker), stream))
        assert first.status_code == 200, first.text
        received: Final = wire.drain()
    assistant: Final = assistant_from(first, stream)
    assert [target_path(request) for request in received] == [target("snowflake", stream)], received
    assert kept_results(assistant) == [SEARCH_RESULT], assistant
    assert tool_call_ids(assistant) == [SEARCH_ID, WEATHER_ID], assistant
    assert spend_row(response_id(first, stream))["model_group"] == model


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _sdk_messages(messages: Sequence[JsonValue]) -> list[ChatCompletionMessageParam]:
    return cast(list[ChatCompletionMessageParam], list(messages))


_SDK_TOOLS: Final = cast(list[ChatCompletionToolParam], TOOLS)
_NO_CACHE: Final[dict[str, object]] = {"cache": {"no-cache": True}}


def test_openai_sdk_sync_round_trip_replays_the_tool_search_result(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario, _openai_client(gateway) as client:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        first: Final = client.chat.completions.create(
            model=model,
            messages=_sdk_messages(first_messages(marker)),
            tools=_SDK_TOOLS,
            max_tokens=64,
            extra_body=_NO_CACHE,
        )
        assistant: Final = JSON_OBJECT.validate_python(first.choices[0].message.model_dump(exclude_none=True))
        second: Final = client.chat.completions.create(
            model=model,
            messages=_sdk_messages(replay_messages(marker, assistant)),
            tools=_SDK_TOOLS,
            max_tokens=64,
            extra_body=_NO_CACHE,
        )
        received: Final = wire.drain()
    assert second.choices[0].message.content == answer(marker), second.model_dump()
    assert kept_results(assistant) == [SEARCH_RESULT], assistant
    assert len(received) == 2, received
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body
    assert spend_row(second.id)["model_group"] == model


async def test_openai_sdk_async_stream_round_trip_replays_the_tool_search_result(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        async with _async_openai_client(gateway) as client:
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=_sdk_messages(first_messages(marker)),
                tools=_SDK_TOOLS,
                max_tokens=64,
                stream=True,
                extra_body=_NO_CACHE,
            )
            chunks: Final = [chunk async for chunk in stream]
            deltas: Final = tuple(
                JSON_OBJECT.validate_python(chunk.choices[0].delta.model_dump(exclude_none=True))
                for chunk in chunks
                if chunk.choices
            )
            assistant: Final = folded_assistant(deltas)
            second: Final = await client.chat.completions.create(
                model=model,
                messages=_sdk_messages(replay_messages(marker, assistant)),
                tools=_SDK_TOOLS,
                max_tokens=64,
                stream=True,
                extra_body=_NO_CACHE,
            )
            answered: Final = "".join([chunk.choices[0].delta.content or "" async for chunk in second if chunk.choices])
        received: Final = wire.drain()
    assert answered == answer(marker), answered
    assert kept_results(assistant) == [SEARCH_RESULT], assistant
    assert len(received) == 2, received
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body


@_STREAM_MODES
def test_code_execution_next_to_a_tool_search_keeps_both_results_and_only_the_code_run(
    gateway: Gateway, stream: bool
) -> None:
    turn: Final = (_BASH_USE, _BASH_RESULT, SEARCH_USE, SEARCH_RESULT, WEATHER_USE)
    assistant, _, _, received, _ = _round_trip(gateway, "anthropic", stream, turn)
    fields: Final = provider_fields(assistant)
    assert fields.get("tool_results") == [_BASH_RESULT, SEARCH_RESULT], assistant
    interpreted: Final = fields.get("code_interpreter_results")
    assert isinstance(interpreted, list) and len(interpreted) == 1, assistant
    run: Final = JSON_OBJECT.validate_python(interpreted[0])
    assert (run["id"], run["code"], run["status"]) == (_BASH_ID, "echo sunny", "completed"), run
    assert wire_assistant(received[1]) == list(turn), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body


def _replayed(gateway: Gateway, messages: Sequence[JsonValue]) -> tuple[httpx.Response, tuple[Request, ...]]:
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = chat(gateway, chat_body(model, messages, stream=False))
        received: Final = wire.drain()
    return response, received


def _kept_search(*calls: tuple[str, str, Event]) -> Event:
    return hand_built_assistant(calls or (SEARCH_CALL, WEATHER_CALL), KEPT_FIELDS)


@pytest.mark.parametrize(
    ("answers", "assistant_turn", "answered", "texts"),
    (
        pytest.param(
            (tool_answer(SEARCH_ID, "client search"), tool_answer(WEATHER_ID)),
            (SEARCH_CALL, WEATHER_CALL),
            [WEATHER_ID],
            1,
            id="search-and-client-tool-answered",
        ),
        pytest.param(
            (tool_answer(WEATHER_ID), tool_answer(SEARCH_ID, "client search")),
            (SEARCH_CALL, WEATHER_CALL),
            [WEATHER_ID],
            1,
            id="client-tool-answered-first",
        ),
        pytest.param(
            (tool_answer(SEARCH_ID, "one"), tool_answer(SEARCH_ID, "two"), tool_answer(WEATHER_ID)),
            (SEARCH_CALL, WEATHER_CALL),
            [WEATHER_ID],
            1,
            id="search-answered-twice",
        ),
        pytest.param(
            (tool_answer(SEARCH_ID, "client search"), {"role": "user", "content": "go on"}),
            (SEARCH_CALL,),
            [],
            2,
            id="search-answer-then-user-text",
        ),
    ),
)
def test_client_answer_to_a_kept_search_call_is_dropped_when_another_user_turn_follows(
    gateway: Gateway,
    answers: Sequence[JsonValue],
    assistant_turn: tuple[tuple[str, str, Event], ...],
    answered: list[JsonValue],
    texts: int,
) -> None:
    marker: Final = uuid.uuid4().hex
    response, received = _replayed(gateway, replay_messages(marker, _kept_search(*assistant_turn), answers))
    assert response.status_code == 200, response.text
    assert answer_text(response, stream=False) == answer(marker), response.text
    assert len(received) == 1, received
    expected_turn: Final = [SEARCH_USE, SEARCH_RESULT, *([WEATHER_USE] if len(assistant_turn) == 2 else [])]
    assert wire_assistant(received[0]) == expected_turn, received[0].body
    assert answered_ids(received[0]) == answered, received[0].body
    assert len(user_texts(received[0])) == texts, received[0].body


def test_client_answer_alone_replays_the_search_call_the_way_the_client_sent_it(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final = replay_messages(marker, _kept_search(SEARCH_CALL), (tool_answer(SEARCH_ID, "client search"),))
    response, received = _replayed(gateway, messages)
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    assert wire_assistant(received[0]) == [_PLAIN_SEARCH_USE], received[0].body
    assert answered_ids(received[0]) == [SEARCH_ID], received[0].body


def test_interleaved_thinking_replay_drops_the_client_answer_to_the_kept_search(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    assistant: Final = hand_built_assistant(
        (SEARCH_CALL, WEATHER_CALL), KEPT_FIELDS, thinking_blocks=[_SIGNED_THINKING]
    )
    answers: Final = (tool_answer(SEARCH_ID, "client search"), tool_answer(WEATHER_ID))
    response, received = _replayed(gateway, replay_messages(marker, assistant, answers))
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    assert wire_assistant(received[0]) == [_SIGNED_THINKING, SEARCH_USE, SEARCH_RESULT, WEATHER_USE], received[0].body
    assert answered_ids(received[0]) == [WEATHER_ID], received[0].body


@pytest.mark.parametrize(
    "fields",
    (
        pytest.param({"provider_specific_fields": None}, id="null"),
        pytest.param({"provider_specific_fields": {}}, id="empty"),
        pytest.param({}, id="missing"),
        pytest.param({"provider_specific_fields": {"tool_results": []}}, id="no-results"),
    ),
)
def test_search_call_without_a_kept_result_replays_plainly_with_both_answers(
    gateway: Gateway, fields: dict[str, JsonValue]
) -> None:
    marker: Final = uuid.uuid4().hex
    assistant: Final = {**hand_built_assistant((SEARCH_CALL, WEATHER_CALL)), **fields}
    answers: Final = (tool_answer(SEARCH_ID, "client search"), tool_answer(WEATHER_ID))
    response, received = _replayed(gateway, replay_messages(marker, assistant, answers))
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    assert wire_assistant(received[0]) == [_PLAIN_SEARCH_USE, WEATHER_USE], received[0].body
    assert answered_ids(received[0]) == [SEARCH_ID, WEATHER_ID], received[0].body


@pytest.mark.parametrize(
    "fields",
    (
        pytest.param("x" * 5120, id="5kb-string"),
        pytest.param([SEARCH_RESULT], id="list"),
        pytest.param({"tool_results": "x" * 5120}, id="results-5kb-string"),
        pytest.param({"tool_results": [1, "a", None]}, id="results-without-objects"),
        pytest.param({"tool_results": SEARCH_RESULT}, id="results-object"),
        pytest.param({"tool_results": [search_result("srvtoolu_01Other")]}, id="result-for-another-call"),
    ),
)
def test_malformed_kept_results_replay_the_search_call_plainly(gateway: Gateway, fields: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    assistant: Final = hand_built_assistant((SEARCH_CALL, WEATHER_CALL), fields)
    response, received = _replayed(gateway, replay_messages(marker, assistant))
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    assert wire_assistant(received[0]) == [_PLAIN_SEARCH_USE, WEATHER_USE], received[0].body
    assert answered_ids(received[0]) == [WEATHER_ID], received[0].body


@pytest.mark.parametrize(
    "tool_call_id",
    (
        pytest.param(123, id="int"),
        pytest.param([SEARCH_ID], id="list"),
        pytest.param({"id": SEARCH_ID}, id="object"),
    ),
)
def test_non_string_tool_answer_ids_never_crash_the_replay(gateway: Gateway, tool_call_id: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    answers: Final = (tool_answer(tool_call_id, "odd"), tool_answer(WEATHER_ID))
    response, received = _replayed(gateway, replay_messages(marker, _kept_search(), answers))
    assert response.status_code != 500, response.text
    assert '"error"' in response.text or response.status_code == 200, response.text
    assert gateway.client.get("/health/liveliness").status_code == 200
    assert len(received) <= 1, received


@pytest.mark.parametrize(
    "tool_call_id", (pytest.param("", id="empty-string"), pytest.param("x" * 5120, id="5kb-string"))
)
def test_odd_string_tool_answer_ids_keep_the_search_rebuilt(gateway: Gateway, tool_call_id: str) -> None:
    marker: Final = uuid.uuid4().hex
    answers: Final = (tool_answer(tool_call_id, "odd"), tool_answer(WEATHER_ID))
    response, received = _replayed(gateway, replay_messages(marker, _kept_search(), answers))
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    assert wire_assistant(received[0]) == list(SEARCH_TURN), received[0].body
    forwarded: Final = answered_ids(received[0])
    assert len(forwarded) == 2, received[0].body
    assert forwarded[1] == WEATHER_ID, received[0].body
    assert SEARCH_ID not in forwarded, received[0].body


_ODD_RESULTS: Final = (
    pytest.param(
        {
            "type": "tool_search_tool_result",
            "tool_use_id": SEARCH_ID,
            "content": {"type": "tool_search_tool_result_error", "error_code": "unavailable"},
        },
        id="error-content",
    ),
    pytest.param(
        {
            "type": "tool_search_tool_result",
            "tool_use_id": SEARCH_ID,
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [{"type": "tool_reference", "tool_name": "w" * 5120}],
            },
        },
        id="5kb-reference",
    ),
)


@pytest.mark.parametrize("odd", _ODD_RESULTS)
@_STREAM_MODES
def test_unusual_search_results_from_the_upstream_are_kept_and_replayed_verbatim(
    gateway: Gateway, odd: Event, stream: bool
) -> None:
    assistant, _, _, received, _ = _round_trip(gateway, "anthropic", stream, (SEARCH_USE, odd, WEATHER_USE))
    assert kept_results(assistant) == [odd], assistant
    assert wire_assistant(received[1]) == [SEARCH_USE, odd, WEATHER_USE], received[1].body


@_STREAM_MODES
def test_search_result_without_its_call_id_is_kept_but_not_rebuilt(gateway: Gateway, stream: bool) -> None:
    orphan: Final = {key: value for key, value in SEARCH_RESULT.items() if key != "tool_use_id"}
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer((SEARCH_USE, orphan, WEATHER_USE))) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        first: Final = chat(gateway, chat_body(model, first_messages(marker), stream))
        assert first.status_code == 200, first.text
        assistant: Final = assistant_from(first, stream)
        answers: Final = (tool_answer(SEARCH_ID, "client search"), tool_answer(WEATHER_ID))
        second: Final = chat(gateway, chat_body(model, replay_messages(marker, assistant, answers), stream=False))
        assert second.status_code == 200, second.text
        received: Final = wire.drain()
    assert kept_results(assistant) == [orphan], assistant
    assert wire_assistant(received[1]) == [_PLAIN_SEARCH_USE, WEATHER_USE], received[1].body
    assert answered_ids(received[1]) == [SEARCH_ID, WEATHER_ID], received[1].body


@_STREAM_MODES
def test_duplicated_search_result_blocks_rebuild_the_call_once(gateway: Gateway, stream: bool) -> None:
    assistant, _, _, received, _ = _round_trip(
        gateway, "anthropic", stream, (SEARCH_USE, SEARCH_RESULT, SEARCH_RESULT, WEATHER_USE)
    )
    assert kept_results(assistant) == [SEARCH_RESULT, SEARCH_RESULT], assistant
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body


@pytest.mark.parametrize(
    ("status", "kind", "caller_status"),
    (
        pytest.param(400, "invalid_request_error", 400, id="400"),
        pytest.param(401, "authentication_error", 401, id="401"),
        pytest.param(529, "overloaded_error", 500, id="529"),
    ),
)
def test_provider_error_on_the_replay_reaches_the_caller(
    gateway: Gateway, status: int, kind: str, caller_status: int
) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if len(request_messages(request)) == 1:
            return scripted_peer()(request)
        body: Final = {"type": "error", "error": {"type": kind, "message": f"scripted {kind} {marker_of(request)}"}}
        return Reply(status=status, body=json.dumps(body).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        first: Final = chat(gateway, chat_body(model, first_messages(marker), stream=False))
        assert first.status_code == 200, first.text
        second: Final = chat(gateway, chat_body(model, replay_messages(marker, assistant_from(first, False)), False))
        received: Final = wire.drain()
    assert second.status_code == caller_status, second.text
    assert f"scripted {kind} {marker}" in second.text, second.text
    assert len(received) >= 2, received
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert gateway.client.get("/health/liveliness").status_code == 200


def test_unauthenticated_replay_never_reaches_the_provider(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final = chat_body(model, replay_messages(marker, _kept_search()), stream=False)
        response: Final = chat(gateway, body, key=f"sk-not-a-key-{marker}")
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


def _cut_stream_text(gateway: Gateway, body: Event) -> str:
    try:
        with gateway.client.stream(
            "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {gateway.key}"}
        ) as response:
            return response.read().decode()
    except httpx.TransportError as error:
        return repr(error)


def test_stream_cut_right_after_the_search_result_leaves_the_proxy_serving(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if marker_of(request) != marker:
            return scripted_peer()(request)
        return turn_reply(request, f"msg_{marker}_cut", SEARCH_TURN, "tool_use", abort_after=6)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        text: Final = _cut_stream_text(gateway, chat_body(model, first_messages(marker), stream=True))
        assert "get_weather" not in text, text
        assert gateway.client.get("/health/liveliness").status_code == 200
        follow_up: Final = uuid.uuid4().hex
        recovered: Final = chat(gateway, chat_body(model, first_messages(follow_up), stream=True))
        assert recovered.status_code == 200, recovered.text
        assert kept_results(assistant_from(recovered, stream=True)) == [SEARCH_RESULT], recovered.text
        assert len(wire.drain()) == 2


def test_identical_replays_each_reach_the_provider_and_land_their_own_spend_row(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final = chat_body(model, replay_messages(marker, _kept_search()), stream=False)
        responses: Final = tuple(chat(gateway, body) for _ in range(2))
        received: Final = wire.drain()
    for response in responses:
        assert response.status_code == 200, response.text
    assert len(received) == 2, received
    assert [wire_assistant(request) for request in received] == [list(SEARCH_TURN)] * 2
    ids: Final = tuple(response_id(response, stream=False) for response in responses)
    assert len(set(ids)) == 2, ids
    for identity in ids:
        assert spend_row(identity)["model_group"] == model


def test_response_cache_hit_still_carries_the_search_result_for_the_replay(gateway: Gateway) -> None:
    stream: Final = False
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        cached_body: Final = {
            key: value for key, value in chat_body(model, first_messages(marker), stream).items() if key != "cache"
        }
        first: Final = chat(gateway, cached_body)
        assert first.status_code == 200, first.text
        hit: Final = chat(gateway, cached_body)
        assert hit.status_code == 200, hit.text
        assistant: Final = assistant_from(hit, stream)
        second: Final = chat(gateway, chat_body(model, replay_messages(marker, assistant), stream=False))
        assert second.status_code == 200, second.text
        received: Final = wire.drain()
    assert kept_results(assistant_from(first, stream)) == [SEARCH_RESULT], first.text
    assert kept_results(assistant) == [SEARCH_RESULT], hit.text
    assert len(received) == 2, [request.body for request in received]
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body


def _messages_turn(model: str, marker: str, stream: bool, replay: bool) -> Event:
    turns: Final[list[JsonValue]] = [{"role": "user", "content": question(marker)}]
    replayed: Final[list[JsonValue]] = [
        {"role": "assistant", "content": list(SEARCH_TURN)},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": WEATHER_ID, "content": "18C"}]},
    ]
    return {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "tools": [
            {"type": "tool_search_tool_bm25_20251119", "name": SEARCH_TOOL},
            {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                "defer_loading": True,
            },
        ],
        "messages": [*turns, *(replayed if replay else [])],
    }


def _streamed_message_blocks(text: str) -> list[JsonValue]:
    events: Final = tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    )
    return [event["content_block"] for event in events if event["type"] == "content_block_start"]


@_STREAM_MODES
def test_messages_endpoint_passes_the_search_result_and_its_replay_through(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        first: Final = gateway.request("POST", "/v1/messages", _messages_turn(model, marker, stream, replay=False))
        second: Final = gateway.request("POST", "/v1/messages", _messages_turn(model, marker, stream, replay=True))
        received: Final = wire.drain()
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    blocks: Final = (
        _streamed_message_blocks(first.text)
        if stream
        else JSON_LIST.validate_python(JSON_OBJECT.validate_json(first.content)["content"])
    )
    assert SEARCH_RESULT in blocks, first.text
    assert answer(marker) in second.text, second.text
    assert len(received) == 2, received
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body


_RESPONSES_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]


def test_responses_bridge_reports_the_search_turn_the_same_way(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": question(marker),
                "tools": _RESPONSES_TOOLS,
                "max_output_tokens": 64,
                **_NO_CACHE,
            },
        )
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    output: Final = JSON_OBJECT.validate_json(response.content)["output"]
    assert isinstance(output, list), response.text
    calls: Final = [
        JSON_OBJECT.validate_python(item)["name"]
        for item in output
        if JSON_OBJECT.validate_python(item)["type"] == "function_call"
    ]
    assert "get_weather" in calls, response.text


def test_run_tool_loop_through_the_proxy_executes_only_the_client_tool(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    executed: Final[list[str]] = []  # mutable-ok: the executor records each tool it ran for the assertion

    def execute(call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executed.append(str(call.function.name))  # mutable-ok: see above
        return {"role": "tool", "tool_call_id": call.id, "content": WEATHER_ANSWER}

    with wire_server(scripted_peer()) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        content: Final = litellm.run_tool_loop(
            model=f"litellm_proxy/{model}",
            messages=cast(list[AllMessageValues], first_messages(marker)),
            tools=cast(list[ChatCompletionToolParam], TOOLS),
            execute_tool=execute,
            api_base=str(gateway.client.base_url),
            api_key=gateway.key,
            max_tokens=64,
        )
        received: Final = wire.drain()
    assert content == answer(marker), content
    assert executed == ["get_weather"], executed
    assert len(received) == 2, received
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body


def _marked_search_call() -> Event:
    return {
        "id": SEARCH_ID,
        "type": "function",
        "function": {"name": SEARCH_TOOL, "arguments": json.dumps(SEARCH_INPUT)},
        "cache_control": EPHEMERAL,
    }


def _kept_marked_search() -> Event:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [_marked_search_call()],
        "provider_specific_fields": KEPT_FIELDS,
    }


_SEARCH_LABEL: Final = f"assistant:tool_use:{SEARCH_ID}"
_CITY_CALLS: Final = ("London", "Paris")


def _answered_cities() -> list[JsonValue]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call(city, cache_control=EPHEMERAL) for city in _CITY_CALLS],
        },
        *({"role": "tool", "tool_call_id": call_id(city), "content": "sunny."} for city in _CITY_CALLS),
    ]


def test_client_answered_search_call_mark_counts_toward_the_cap(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        ask(marked=True),
        *_answered_cities(),
        _kept_marked_search(),
        tool_answer(SEARCH_ID, f"client search marker-{marker}"),
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        status, _, text = post_chat(
            gateway, chat_body(model, messages, stream=False, cache_control_injection_points=POINTS)
        )
        received: Final = wire.drain()
    assert status == 200, text
    assert len(received) == 1, received
    assert anthropic_labels(received[0]) == [
        ASK_LABEL,
        *(tool_use_label(city) for city in _CITY_CALLS),
        _SEARCH_LABEL,
    ]
    assert answered_ids(received[0]) == [*(call_id(city) for city in _CITY_CALLS), SEARCH_ID]


def test_search_call_mark_is_not_counted_when_its_client_answer_is_dropped(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        ask(marked=True),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [*(tool_call(city, cache_control=EPHEMERAL) for city in _CITY_CALLS), _marked_search_call()],
            "provider_specific_fields": KEPT_FIELDS,
        },
        *({"role": "tool", "tool_call_id": call_id(city), "content": "sunny."} for city in _CITY_CALLS),
        tool_answer(SEARCH_ID, "client search"),
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        status, _, text = post_chat(
            gateway, chat_body(model, messages, stream=False, cache_control_injection_points=POINTS)
        )
        received: Final = wire.drain()
    assert status == 200, text
    assert len(received) == 1, received
    assert anthropic_labels(received[0]) == [SYSTEM_LABEL, ASK_LABEL, *(tool_use_label(city) for city in _CITY_CALLS)]
    assert answered_ids(received[0]) == [call_id(city) for city in _CITY_CALLS]
    assert wire_assistant(received[0])[-2:] == [SEARCH_USE, SEARCH_RESULT], received[0].body


def test_response_cache_hit_serves_the_client_answered_search_request_once(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        ask(marked=True),
        _kept_marked_search(),
        tool_answer(SEARCH_ID, f"client search marker-{marker}"),
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, gateway.upstream_url)
        body: Final = {
            key: value
            for key, value in chat_body(model, messages, stream=False, cache_control_injection_points=POINTS).items()
            if key != "cache"
        }
        first: Final = post_chat(gateway, body)
        second: Final = post_chat(gateway, body)
        received: Final = wire.drain()
    assert (first[0], second[0]) == (200, 200), (first[2], second[2])
    assert first[1] == second[1], (first[2], second[2])
    assert [anthropic_labels(request) for request in received] == [
        [SYSTEM_LABEL, ASK_LABEL, _SEARCH_LABEL, f"user:tool_result:{SEARCH_ID}"]
    ]
    assert gateway_injected(first[1]) is True
