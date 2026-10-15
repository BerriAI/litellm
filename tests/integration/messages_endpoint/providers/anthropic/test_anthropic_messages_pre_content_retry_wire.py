import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import anthropic
import httpx
import pytest
from integration._support.anthropic_sse import (
    LIFECYCLE,
    Attempts,
    SseEvent,
    delta_text,
    dropping_reply,
    error_frame,
    error_type,
    event_types,
    message_id,
    message_json,
    message_stream,
    parse_sse,
    status_reply,
    stream_reply,
    user_prompt,
)
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue
from redis import Redis

_MODEL: Final = "claude-sonnet-4-5-20250929"
_API_KEY: Final = "synthetic-anthropic-key"
_TEXT: Final = "Hello"

FirstAttempt = Literal[
    "drop_after_headers",
    "drop_after_message_start",
    "error_frame",
    "error_frame_after_message_start",
    "http_status",
    "drop_after_content",
]
BudgetSource = Literal["deployment", "request"]


@dataclass(frozen=True, slots=True)
class _Failure:
    kind: FirstAttempt
    status: int = 500

    def reply(self, served_id: str) -> Reply:
        chunks: Final = message_stream(served_id, _MODEL, _TEXT)
        match self.kind:
            case "drop_after_headers":
                return dropping_reply(chunks, abort_after=0)
            case "drop_after_message_start":
                return dropping_reply(chunks, abort_after=1)
            case "error_frame":
                return stream_reply((error_frame(self.status, f"scripted {self.status}"),))
            case "error_frame_after_message_start":
                return stream_reply((chunks[0], error_frame(self.status, f"scripted {self.status}")))
            case "http_status":
                return status_reply(self.status)
            case "drop_after_content":
                return dropping_reply(chunks, abort_after=3)


_MID_STREAM_FAILURES: Final = (
    _Failure("drop_after_headers"),
    _Failure("drop_after_message_start"),
    _Failure("error_frame", 529),
    _Failure("error_frame", 429),
    _Failure("error_frame_after_message_start", 500),
)
_PRE_STREAM_STATUSES: Final = (529, 500, 429, 408, 409)


def _served_id(prompt: str, attempt: int) -> str:
    return f"msg_{prompt}_a{attempt}"


def _prompt() -> str:
    return "pre-content-" + uuid.uuid4().hex


def _upstream(
    prompt: str, failure: _Failure, attempts: Attempts, *, failing_attempts: int = 1, stream: bool = True
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/messages"), request
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        body: Final = object_value(json.loads(request.body))
        assert body["model"] == _MODEL, body
        assert body.get("stream", False) is stream, body
        assert user_prompt(body) == prompt, body
        assert "num_retries" not in body, body
        attempt: Final = attempts.record(prompt)
        if attempt <= failing_attempts:
            return failure.reply(_served_id(prompt, attempt))
        if stream:
            return stream_reply(message_stream(_served_id(prompt, attempt), _MODEL, _TEXT))
        return Reply(body=message_json(_served_id(prompt, attempt), _MODEL, _TEXT))

    return respond


def _body(
    model: str, prompt: str, source: BudgetSource, *, stream: bool = True, budget: int = 1
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
        **({"num_retries": budget} if source == "request" else {}),
    }


def _stream(gateway: Gateway, body: Mapping[str, JsonValue]) -> tuple[int, tuple[SseEvent, ...]]:
    response: Final = gateway.request("POST", "/v1/messages", body)
    return response.status_code, parse_sse(response.text)


def _success_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows('SELECT status, model_group FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )


def _assert_completed_by_retry(
    status: int, events: tuple[SseEvent, ...], prompt: str, model: str, wire: Wire, *, attempts: int = 2
) -> None:
    assert status == 200, events
    assert event_types(events) == LIFECYCLE, events
    assert message_id(events) == _served_id(prompt, attempts), events
    assert delta_text(events) == _TEXT, events
    assert [request.target for request in wire.drain()] == ["/v1/messages"] * attempts
    assert _success_rows(_served_id(prompt, attempts)) == [{"status": "success", "model_group": model}]


def _assert_rejected_before_content(
    response: httpx.Response, wire: Wire, *, status: int, attempts: int, error: str
) -> None:
    assert response.status_code == status, response.text
    assert error in response.text, response.text
    assert "content_block_delta" not in response.text, response.text
    assert [request.target for request in wire.drain()] == ["/v1/messages"] * attempts


def _assert_stream_failed_after_message_start(
    status: int, events: tuple[SseEvent, ...], wire: Wire, *, attempts: int
) -> None:
    assert status == 200, events
    types: Final = event_types(events)
    assert types[0] == "message_start", events
    assert types[-1] == "error", events
    assert "content_block_delta" not in types, events
    assert [request.target for request in wire.drain()] == ["/v1/messages"] * attempts


@pytest.mark.parametrize("source", ["deployment", "request"])
@pytest.mark.parametrize("failure", _MID_STREAM_FAILURES, ids=lambda failure: f"{failure.kind}-{failure.status}")
def test_stream_failing_before_content_is_retried_on_the_same_group_and_completes(
    gateway: Gateway, failure: _Failure, source: BudgetSource
) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with wire_server(_upstream(prompt, failure, attempts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}",
            api_base=wire.url,
            api_key=_API_KEY,
            **({"num_retries": 1} if source == "deployment" else {}),
        )
        status, events = _stream(gateway, _body(model, prompt, source))
        _assert_completed_by_retry(status, events, prompt, model, wire)


@pytest.mark.parametrize("source", ["deployment", "request"])
@pytest.mark.parametrize("http_status", _PRE_STREAM_STATUSES)
def test_stream_rejected_before_it_opens_is_retried_and_completes(
    gateway: Gateway, http_status: int, source: BudgetSource
) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("http_status", http_status), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}",
            api_base=wire.url,
            api_key=_API_KEY,
            **({"num_retries": 1} if source == "deployment" else {}),
        )
        status, events = _stream(gateway, _body(model, prompt, source))
        _assert_completed_by_retry(status, events, prompt, model, wire)


def _sdk(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60)


def _async_sdk(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60
    )


def test_anthropic_sdk_sync_stream_completes_after_a_drop_following_message_start(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_message_start"), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        with _sdk(gateway).messages.stream(
            model=model, max_tokens=16, messages=[{"role": "user", "content": prompt}]
        ) as stream:
            final: Final = stream.get_final_message()
        assert final.id == _served_id(prompt, 2), final
        assert [block.text for block in final.content if block.type == "text"] == [_TEXT], final
        assert [request.target for request in wire.drain()] == ["/v1/messages"] * 2
        assert _success_rows(final.id) == [{"status": "success", "model_group": model}]


async def test_anthropic_sdk_async_stream_completes_after_a_drop_following_message_start(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_message_start"), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        async with _async_sdk(gateway).messages.stream(
            model=model, max_tokens=16, messages=[{"role": "user", "content": prompt}], extra_body={"num_retries": 1}
        ) as stream:
            final: Final = await stream.get_final_message()
        assert final.id == _served_id(prompt, 2), final
        assert [block.text for block in final.content if block.type == "text"] == [_TEXT], final
        assert [request.target for request in wire.drain()] == ["/v1/messages"] * 2


def test_anthropic_sdk_sync_stream_completes_after_an_overloaded_error_frame(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("error_frame", 529), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        with _sdk(gateway).messages.create(
            model=model, max_tokens=16, messages=[{"role": "user", "content": prompt}], stream=True
        ) as stream:
            events: Final = tuple(stream)
        starts: Final = [event.message.id for event in events if event.type == "message_start"]
        assert starts == [_served_id(prompt, 2)], events
        assert [
            event.delta.text
            for event in events
            if event.type == "content_block_delta" and event.delta.type == "text_delta"
        ] == [_TEXT]
        assert [request.target for request in wire.drain()] == ["/v1/messages"] * 2


async def test_anthropic_sdk_async_stream_completes_after_a_529_before_the_stream_opens(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("http_status", 529), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        async with _async_sdk(gateway).messages.stream(
            model=model, max_tokens=16, messages=[{"role": "user", "content": prompt}]
        ) as stream:
            final: Final = await stream.get_final_message()
        assert final.id == _served_id(prompt, 2), final
        assert [request.target for request in wire.drain()] == ["/v1/messages"] * 2


def test_stream_dropping_after_content_is_not_retried(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_content"), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        status, events = _stream(gateway, _body(model, prompt, "deployment"))
        assert status == 200, events
        assert event_types(events)[:3] == LIFECYCLE[:3], events
        assert event_types(events)[-1] == "error", events
        assert message_id(events) == _served_id(prompt, 1), events
        assert delta_text(events) == _TEXT, events
        assert [request.target for request in wire.drain()] == ["/v1/messages"]


def test_stream_rejected_with_401_before_it_opens_is_not_retried(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("http_status", 401), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        response: Final = gateway.request("POST", "/v1/messages", _body(model, prompt, "deployment"))
        assert response.status_code == 401, response.text
        assert [request.target for request in wire.drain()] == ["/v1/messages"]


def test_stream_invalid_request_error_frame_is_not_retried(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("error_frame", 400), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        status, events = _stream(gateway, _body(model, prompt, "deployment"))
        assert status == 200, events
        assert event_types(events) == ("error",), events
        assert error_type(events) == "invalid_request_error", events
        assert [request.target for request in wire.drain()] == ["/v1/messages"]


def test_request_num_retries_zero_turns_the_retry_off_for_a_deployment_with_a_budget(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_message_start"), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        status, events = _stream(gateway, {**_body(model, prompt, "deployment"), "num_retries": 0})
        _assert_stream_failed_after_message_start(status, events, wire, attempts=1)


def test_always_dropping_upstream_is_attempted_once_per_budget_unit_plus_the_first_call(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_message_start"), attempts, failing_attempts=99)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=2)
        status, events = _stream(gateway, _body(model, prompt, "deployment"))
        _assert_stream_failed_after_message_start(status, events, wire, attempts=3)
        assert message_id(events) == _served_id(prompt, 3), events


def test_retry_rejected_before_it_opens_counts_against_the_same_budget(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()

    def respond(request: Request) -> Reply:
        body: Final = object_value(json.loads(request.body))
        assert user_prompt(body) == prompt, body
        attempt: Final = attempts.record(prompt)
        if attempt == 1:
            return dropping_reply(message_stream(_served_id(prompt, attempt), _MODEL, _TEXT), abort_after=1)
        return status_reply(529)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        response: Final = gateway.request("POST", "/v1/messages", _body(model, prompt, "deployment"))
        _assert_rejected_before_content(response, wire, status=500, attempts=2, error="error")


def test_non_stream_messages_rejected_with_500_is_retried_per_the_deployment_budget(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("http_status", 500), attempts, stream=False)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        response: Final = gateway.request("POST", "/v1/messages", _body(model, prompt, "deployment", stream=False))
        assert response.status_code == 200, response.text
        payload: Final = object_value(json.loads(response.content))
        assert payload["id"] == _served_id(prompt, 2), response.text
        assert payload["content"] == [{"type": "text", "text": _TEXT}], response.text
        assert [request.target for request in wire.drain()] == ["/v1/messages"] * 2
        assert _success_rows(_served_id(prompt, 2)) == [{"status": "success", "model_group": model}]


def test_retried_stream_response_headers_name_the_attempt(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_message_start"), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        response: Final = gateway.request("POST", "/v1/messages", _body(model, prompt, "deployment"))
        events: Final = parse_sse(response.text)
        _assert_completed_by_retry(response.status_code, events, prompt, model, wire)
        assert response.headers.get("x-litellm-attempted-retries") == "1", dict(response.headers)
        assert response.headers.get("x-litellm-max-retries") == "1", dict(response.headers)


def test_two_drops_stamp_two_attempted_retries_on_the_response(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_message_start"), attempts, failing_attempts=2)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=2)
        response: Final = gateway.request("POST", "/v1/messages", _body(model, prompt, "deployment"))
        events: Final = parse_sse(response.text)
        assert response.status_code == 200, response.text
        assert event_types(events) == LIFECYCLE, events
        assert message_id(events) == _served_id(prompt, 3), events
        assert [request.target for request in wire.drain()] == ["/v1/messages"] * 3
        assert response.headers.get("x-litellm-attempted-retries") == "2", dict(response.headers)
        assert response.headers.get("x-litellm-max-retries") == "2", dict(response.headers)


def test_retried_stream_spend_row_records_the_attempt_count(gateway: Gateway) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_message_start"), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        status, events = _stream(gateway, _body(model, prompt, "deployment"))
        _assert_completed_by_retry(status, events, prompt, model, wire)
        rows: Final = read_rows(
            "SELECT metadata->>'attempted_retries' AS attempted, metadata->>'max_retries' AS budget "
            'FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (_served_id(prompt, 2),),
        )
        assert rows == [{"attempted": "1", "budget": "1"}], rows


def _string_values(cache: Redis) -> tuple[bytes, ...]:
    keys: Final = tuple(key for key in cache.scan_iter(count=1000) if cache.type(key) == b"string")
    return tuple(value for value in cache.mget(keys) if value is not None) if keys else ()


def _cached_somewhere(served_id: str) -> bool:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        return any(served_id.encode() in value for value in _string_values(cache))


def test_retried_stream_is_cached_and_the_identical_request_is_served_without_the_upstream(
    gateway: Gateway,
) -> None:
    prompt: Final = _prompt()
    attempts: Final = Attempts()
    with (
        wire_server(_upstream(prompt, _Failure("drop_after_message_start"), attempts)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, num_retries=1)
        body: Final = _body(model, prompt, "deployment")
        status, events = _stream(gateway, body)
        _assert_completed_by_retry(status, events, prompt, model, wire)
        eventually(lambda: _cached_somewhere(_served_id(prompt, 2)), bool, seconds=30)
        replay_status, replay = _stream(gateway, body)
        assert replay_status == 200, replay
        assert message_id(replay) == _served_id(prompt, 2), replay
        assert delta_text(replay) == _TEXT, replay
        assert event_types(replay)[-1] == "message_stop", replay
        assert wire.drain() == ()
