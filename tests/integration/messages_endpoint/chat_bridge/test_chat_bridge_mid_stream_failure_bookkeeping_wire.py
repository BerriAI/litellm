import asyncio
import json
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final, Literal

import anthropic
import httpx
import pytest
from anthropic.types import Message, MessageParam
from integration._support.anthropic_sse import (
    ANTHROPIC_ERROR_TYPES,
    SseEvent,
    delta_text,
    dropping_reply,
    error_type,
    event_types,
    parse_sse,
    stream_reply,
    user_prompt,
)
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_stream, openai_error, posted_targets
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-4o-mini"
_PROVIDER_MODEL: Final = f"hosted_vllm/{_BACKEND}"
_PROVIDER_KEY: Final = "integration-provider-key"
_UPSTREAM_TARGET: Final = "/v1/chat/completions"
_TEXT: Final = "Hello"
_CALL_ID_HEADER: Final = "x-litellm-call-id"
_ROWS: Final = (
    "SELECT metadata->>'litellm_call_id' AS call_id, request_id, status, cache_hit, metadata "
    'FROM "LiteLLM_SpendLogs" WHERE model_group=%s'
)
_ROW_SECONDS: Final = 60
_SLOW_PAUSE: Final = 1.0
_BURST_PER_OUTCOME: Final = 8

Outcome = Literal[
    "drop_after_content", "error_frame_after_content", "slow_drop_after_content", "succeeds", "rejected_before_any_body"
]
_OUTCOME: Final = TypeAdapter(Outcome)
_FAILING_STREAMS: Final[tuple[Outcome, ...]] = ("drop_after_content", "error_frame_after_content")
_BURST: Final[tuple[Outcome, ...]] = (
    "slow_drop_after_content",
    "error_frame_after_content",
    "succeeds",
) * _BURST_PER_OUTCOME


@dataclass(frozen=True, slots=True)
class _Call:
    outcome: Outcome
    marker: str
    prompt_tag: str
    call_id: str

    @property
    def prompt(self) -> str:
        return f"{self.outcome}:{self.marker}:{self.prompt_tag}"

    @property
    def streams(self) -> bool:
        return self.outcome != "rejected_before_any_body"


@dataclass(frozen=True, slots=True)
class _SdkFailure:
    text: str
    error: anthropic.APIStatusError


def _call(outcome: Outcome, marker: str) -> _Call:
    tag: Final = uuid.uuid4().hex
    return _Call(outcome, marker, tag, tag)


def _marker() -> str:
    return "bridge-mid-stream-" + uuid.uuid4().hex


def _error_frame(status: int) -> bytes:
    error: Final = {"message": f"scripted mid-stream {status}", "type": "server_error", "code": status}
    return b"data: " + json.dumps({"error": error}).encode() + b"\n\n"


def _reply(outcome: Outcome, chunks: tuple[bytes, bytes, bytes]) -> Reply:
    match outcome:
        case "drop_after_content":
            return dropping_reply(chunks, abort_after=2)
        case "error_frame_after_content":
            return stream_reply((chunks[0], chunks[1], _error_frame(500) + b"data: [DONE]\n\n"))
        case "slow_drop_after_content":
            return stream_reply(chunks, abort_after=2, pause=_SLOW_PAUSE)
        case "succeeds":
            return stream_reply(chunks)
        case "rejected_before_any_body":
            return openai_error(500)


def _upstream(marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _UPSTREAM_TARGET), request
        assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}", request.headers
        body: Final = object_value(json.loads(request.body))
        assert body["model"] == _BACKEND, body
        outcome, scripted_marker, prompt_tag = user_prompt(body).split(":")
        assert scripted_marker == marker, body
        scripted: Final = _OUTCOME.validate_python(outcome)
        assert body.get("stream", False) is (scripted != "rejected_before_any_body"), body
        return _reply(scripted, chat_stream(f"chunk-{prompt_tag}", _BACKEND, _TEXT))

    return answering_model_discovery(respond)


def _body(model: str, call: _Call) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "stream": call.streams,
        "messages": [{"role": "user", "content": call.prompt}],
    }


def _headers(call: _Call) -> dict[str, str]:
    return {_CALL_ID_HEADER: call.call_id}


def _stream(gateway: Gateway, model: str, call: _Call) -> tuple[SseEvent, ...]:
    response: Final = gateway.request("POST", "/v1/messages", _body(model, call), headers=_headers(call))
    assert response.status_code == 200, response.text
    assert response.headers[_CALL_ID_HEADER] == call.call_id, response.headers
    return parse_sse(response.text)


async def _stream_concurrently(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> tuple[SseEvent, ...]:
    response: Final = await client.post(
        "/v1/messages", json=_body(model, call), headers={"Authorization": f"Bearer {key}", **_headers(call)}
    )
    assert response.status_code == 200, response.text
    assert response.headers[_CALL_ID_HEADER] == call.call_id, response.headers
    return parse_sse(response.text)


def _leave_after_the_first_content_delta(gateway: Gateway, model: str, call: _Call) -> str:
    headers: Final = {"Authorization": f"Bearer {gateway.key}", **_headers(call)}
    with gateway.client.stream("POST", "/v1/messages", json=_body(model, call), headers=headers) as response:
        assert response.status_code == 200, response
        assert response.headers[_CALL_ID_HEADER] == call.call_id, response.headers
        return next(
            line for line in response.iter_lines() if line.startswith("data:") and '"content_block_delta"' in line
        )


def _call_id(row: dict[str, JsonValue]) -> str:
    return string_value(row["call_id"])


def _rows(model: str, call_ids: frozenset[str]) -> dict[str, dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: read_rows(_ROWS, (model,)),
        lambda rows: call_ids.issubset(_call_id(row) for row in rows),
        seconds=_ROW_SECONDS,
    )
    by_call_id: Final = {_call_id(row): row for row in rows}
    assert len(by_call_id) == len(rows), rows
    return by_call_id


def _assert_failed_after_content(events: tuple[SseEvent, ...]) -> None:
    types: Final = event_types(events)
    assert types[0] == "message_start", events
    assert delta_text(events) == _TEXT, events
    assert types[-1] == "error" and "message_stop" not in types, events


def _assert_completed(events: tuple[SseEvent, ...]) -> None:
    types: Final = event_types(events)
    assert types[0] == "message_start" and types[-1] == "message_stop", events
    assert "error" not in types, events
    assert delta_text(events) == _TEXT, events


def _assert_outcome(outcome: Outcome, events: tuple[SseEvent, ...]) -> None:
    if outcome == "succeeds":
        _assert_completed(events)
        return
    _assert_failed_after_content(events)


def _expected_status(outcome: Outcome) -> str:
    return "success" if outcome == "succeeds" else "failure"


def _assert_failure_row(row: Mapping[str, JsonValue], anthropic_error_type: str | None) -> None:
    assert row["status"] == "failure", row
    error: Final = object_value(object_value(row["metadata"])["error_information"])
    assert error["error_class"] != "MidStreamFallbackError", error
    assert ANTHROPIC_ERROR_TYPES[int(str(error["error_code"]))] == anthropic_error_type, (error, anthropic_error_type)


def _snapshot_text(snapshot: Message) -> str:
    return "".join(block.text for block in snapshot.content if block.type == "text")


def _sdk_messages(call: _Call) -> list[MessageParam]:
    return [{"role": "user", "content": call.prompt}]


def _sdk_error_type(error: anthropic.APIStatusError) -> str:
    return string_value(object_value(object_value(error.body)["error"])["type"])


def _sdk_sync_failure(client: anthropic.Anthropic, model: str, call: _Call) -> _SdkFailure:
    with client.messages.stream(
        model=model, max_tokens=16, messages=_sdk_messages(call), extra_headers=_headers(call)
    ) as stream:
        assert stream.response.headers[_CALL_ID_HEADER] == call.call_id, stream.response.headers
        try:
            deque(stream, maxlen=0)
        except anthropic.APIStatusError as error:
            return _SdkFailure(_snapshot_text(stream.current_message_snapshot), error)
    pytest.fail(f"{call.call_id}: the bridged stream ended without the scripted mid-stream failure")


async def _sdk_async_failure(client: anthropic.AsyncAnthropic, model: str, call: _Call) -> _SdkFailure:
    async with client.messages.stream(
        model=model, max_tokens=16, messages=_sdk_messages(call), extra_headers=_headers(call)
    ) as stream:
        assert stream.response.headers[_CALL_ID_HEADER] == call.call_id, stream.response.headers
        try:
            async for _ in stream:
                pass
        except anthropic.APIStatusError as error:
            return _SdkFailure(_snapshot_text(stream.current_message_snapshot), error)
    pytest.fail(f"{call.call_id}: the bridged stream ended without the scripted mid-stream failure")


def _assert_sdk_failure_bookkept(failure: _SdkFailure, model: str, call: _Call) -> None:
    assert failure.text == _TEXT, failure
    rows: Final = _rows(model, frozenset({call.call_id}))
    assert rows.keys() == {call.call_id}, rows
    _assert_failure_row(rows[call.call_id], _sdk_error_type(failure.error))


@pytest.mark.parametrize("outcome", _FAILING_STREAMS)
def test_bridged_stream_failing_after_content_writes_one_failure_row(gateway: Gateway, outcome: Outcome) -> None:
    marker: Final = _marker()
    with wire_server(_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_PROVIDER_MODEL, api_base=wire.url + "/v1")
        call: Final = _call(outcome, marker)
        events: Final = _stream(gateway, model, call)
        _assert_failed_after_content(events)
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        rows: Final = _rows(model, frozenset({call.call_id}))
        assert rows.keys() == {call.call_id}, rows
        _assert_failure_row(rows[call.call_id], error_type(events))


def test_anthropic_sdk_sync_stream_failing_after_content_raises_and_writes_one_failure_row(gateway: Gateway) -> None:
    marker: Final = _marker()
    with wire_server(_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_PROVIDER_MODEL, api_base=wire.url + "/v1")
        call: Final = _call("drop_after_content", marker)
        client: Final = anthropic.Anthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60
        )
        failure: Final = _sdk_sync_failure(client, model, call)
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_sdk_failure_bookkept(failure, model, call)


async def test_anthropic_sdk_async_stream_failing_after_content_raises_and_writes_one_failure_row(
    gateway: Gateway,
) -> None:
    marker: Final = _marker()
    with wire_server(_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_PROVIDER_MODEL, api_base=wire.url + "/v1")
        call: Final = _call("error_frame_after_content", marker)
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60
        )
        failure: Final = await _sdk_async_failure(client, model, call)
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        _assert_sdk_failure_bookkept(failure, model, call)


def test_bridged_request_rejected_before_any_body_keeps_its_error_and_failure_row(gateway: Gateway) -> None:
    marker: Final = _marker()
    with wire_server(_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_PROVIDER_MODEL, api_base=wire.url + "/v1")
        call: Final = _call("rejected_before_any_body", marker)
        response: Final = gateway.request("POST", "/v1/messages", _body(model, call), headers=_headers(call))
        assert response.status_code == 500, response.text
        error: Final = object_value(object_value(json.loads(response.text))["error"])
        assert error["type"] == ANTHROPIC_ERROR_TYPES[500], response.text
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        rows: Final = _rows(model, frozenset({call.call_id}))
        assert rows.keys() == {call.call_id}, rows
        _assert_failure_row(rows[call.call_id], string_value(error["type"]))


def test_bridged_stream_served_from_the_response_cache_keeps_its_success_bookkeeping(gateway: Gateway) -> None:
    marker: Final = _marker()
    with wire_server(_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_PROVIDER_MODEL, api_base=wire.url + "/v1")
        first: Final = _call("succeeds", marker)
        _assert_completed(_stream(gateway, model, first))
        miss: Final = _rows(model, frozenset({first.call_id}))[first.call_id]
        assert (miss["status"], miss["cache_hit"] != "True") == ("success", True), miss
        twin: Final = replace(first, call_id=uuid.uuid4().hex)
        _assert_completed(_stream(gateway, model, twin))
        assert posted_targets(wire) == (_UPSTREAM_TARGET,)
        rows: Final = _rows(model, frozenset({twin.call_id}))
        hit: Final = rows[twin.call_id]
        assert (hit["status"], hit["cache_hit"]) == ("success", "True"), hit
        assert string_value(hit["request_id"]) != string_value(miss["request_id"]), rows
        assert set(rows) == {first.call_id, twin.call_id}, rows


def test_client_leaving_a_bridged_stream_before_its_failure_leaves_the_proxy_serving(gateway: Gateway) -> None:
    marker: Final = _marker()
    with wire_server(_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_PROVIDER_MODEL, api_base=wire.url + "/v1")
        abandoned: Final = _call("slow_drop_after_content", marker)
        first_delta: Final = _leave_after_the_first_content_delta(gateway, model, abandoned)
        assert _TEXT in first_delta, first_delta
        follow_up: Final = _call("succeeds", marker)
        _assert_completed(_stream(gateway, model, follow_up))
        assert posted_targets(wire) == (_UPSTREAM_TARGET,) * 2
        rows: Final = _rows(model, frozenset({follow_up.call_id}))
        assert rows[follow_up.call_id]["status"] == "success", rows
        assert set(rows) <= {abandoned.call_id, follow_up.call_id}, rows
        assert gateway.request("GET", "/health/liveliness").status_code == 200


async def test_concurrent_bridged_streams_failing_after_content_each_land_one_row(gateway: Gateway) -> None:
    marker: Final = _marker()
    with wire_server(_upstream(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_PROVIDER_MODEL, api_base=wire.url + "/v1")
        calls: Final = tuple(_call(outcome, marker) for outcome in _BURST)
        async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
            streams: Final = await asyncio.gather(
                *(_stream_concurrently(client, gateway.key, model, call) for call in calls)
            )
        for call, events in zip(calls, streams, strict=True):
            _assert_outcome(call.outcome, events)
        assert posted_targets(wire) == (_UPSTREAM_TARGET,) * len(calls)
        rows: Final = _rows(model, frozenset(call.call_id for call in calls))
        assert {call_id: row["status"] for call_id, row in rows.items()} == {
            call.call_id: _expected_status(call.outcome) for call in calls
        }, rows
        for call, events in zip(calls, streams, strict=True):
            if call.outcome != "succeeds":
                _assert_failure_row(rows[call.call_id], error_type(events))
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        _assert_completed(_stream(gateway, model, _call("succeeds", marker)))
