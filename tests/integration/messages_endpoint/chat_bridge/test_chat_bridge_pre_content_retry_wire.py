import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import anthropic
import pytest
from integration._support.anthropic_sse import (
    Attempts,
    SseEvent,
    delta_text,
    dropping_reply,
    event_types,
    parse_sse,
    stream_reply,
)
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.openai_wire import (
    answering_model_discovery,
    chat_stream,
    openai_error,
    posted_targets,
    responses_stream,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_BACKEND: Final = "gpt-4o-mini"
_PROVIDER_KEY: Final = "integration-provider-key"
_TEXT: Final = "Hello"

FirstAttempt = Literal["drop_after_headers", "drop_after_pre_content_frame", "http_500", "drop_after_content"]


@dataclass(frozen=True, slots=True)
class _Bridge:
    name: str
    provider_model: str
    target: str
    stream: Callable[[str, str, str], tuple[bytes, bytes, bytes]]


_CHAT_COMPLETIONS: Final = _Bridge("chat", f"hosted_vllm/{_BACKEND}", "/v1/chat/completions", chat_stream)
_RESPONSES_API: Final = _Bridge("responses", f"openai/{_BACKEND}", "/v1/responses", responses_stream)
_BRIDGES: Final = (_CHAT_COMPLETIONS, _RESPONSES_API)


def _bridge_id(bridge: _Bridge) -> str:
    return bridge.name


def _first_attempt_reply(kind: FirstAttempt, chunks: tuple[bytes, bytes, bytes]) -> Reply:
    match kind:
        case "drop_after_headers":
            return dropping_reply(chunks, abort_after=0)
        case "drop_after_pre_content_frame":
            return dropping_reply(chunks, abort_after=1)
        case "http_500":
            return openai_error(500)
        case "drop_after_content":
            return dropping_reply(chunks, abort_after=2)


def _upstream(bridge: _Bridge, marker: str, kind: FirstAttempt, attempts: Attempts) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", bridge.target), request
        assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}", request.headers
        body: Final = object_value(json.loads(request.body))
        assert body["model"] == _BACKEND, body
        assert body["stream"] is True, body
        assert marker in request.body.decode(), body
        assert "num_retries" not in body, body
        attempt: Final = attempts.record(marker)
        chunks: Final = bridge.stream(f"{bridge.name}-{marker}-a{attempt}", _BACKEND, _TEXT)
        if attempt > 1:
            return stream_reply(chunks)
        return _first_attempt_reply(kind, chunks)

    return answering_model_discovery(respond)


def _marker() -> str:
    return "bridge-pre-content-" + uuid.uuid4().hex


def _body(model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "stream": True,
        "messages": [{"role": "user", "content": marker}],
        **extra,
    }


def _stream(gateway: Gateway, body: dict[str, JsonValue]) -> tuple[int, tuple[SseEvent, ...]]:
    response: Final = gateway.request("POST", "/v1/messages", body)
    return response.status_code, parse_sse(response.text)


def _success_rows(model: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )


def _assert_completed_by_retry(
    status: int, events: tuple[SseEvent, ...], model: str, wire: Wire, bridge: _Bridge
) -> None:
    assert status == 200, events
    types: Final = event_types(events)
    assert types[0] == "message_start", events
    assert "content_block_delta" in types and "error" not in types, events
    assert types[-1] == "message_stop", events
    assert delta_text(events) == _TEXT, events
    assert posted_targets(wire) == (bridge.target,) * 2
    rows: Final = _success_rows(model)
    assert [row["status"] for row in rows] == ["success"], rows


@pytest.mark.parametrize("bridge", _BRIDGES, ids=_bridge_id)
@pytest.mark.parametrize("kind", ["drop_after_headers", "drop_after_pre_content_frame", "http_500"])
def test_bridge_stream_failing_before_content_is_retried_per_the_deployment_budget(
    gateway: Gateway, bridge: _Bridge, kind: FirstAttempt
) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    with wire_server(_upstream(bridge, marker, kind, attempts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=bridge.provider_model, api_base=wire.url + "/v1", num_retries=1)
        status, events = _stream(gateway, _body(model, marker))
        _assert_completed_by_retry(status, events, model, wire, bridge)


@pytest.mark.parametrize("bridge", _BRIDGES, ids=_bridge_id)
def test_bridge_stream_failing_before_content_is_retried_per_the_request_budget(
    gateway: Gateway, bridge: _Bridge
) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    with wire_server(_upstream(bridge, marker, "drop_after_headers", attempts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=bridge.provider_model, api_base=wire.url + "/v1")
        status, events = _stream(gateway, _body(model, marker, num_retries=1))
        _assert_completed_by_retry(status, events, model, wire, bridge)


@pytest.mark.parametrize("bridge", _BRIDGES, ids=_bridge_id)
async def test_anthropic_sdk_async_stream_over_the_bridge_completes_after_a_drop(
    gateway: Gateway, bridge: _Bridge
) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    with wire_server(_upstream(bridge, marker, "drop_after_headers", attempts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=bridge.provider_model, api_base=wire.url + "/v1", num_retries=1)
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=60
        )
        async with client.messages.stream(
            model=model, max_tokens=16, messages=[{"role": "user", "content": marker}]
        ) as stream:
            final: Final = await stream.get_final_message()
        assert [block.text for block in final.content if block.type == "text"] == [_TEXT], final
        assert posted_targets(wire) == (bridge.target,) * 2


@pytest.mark.parametrize("bridge", _BRIDGES, ids=_bridge_id)
def test_bridge_stream_dropping_after_content_is_not_retried(gateway: Gateway, bridge: _Bridge) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    with wire_server(_upstream(bridge, marker, "drop_after_content", attempts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=bridge.provider_model, api_base=wire.url + "/v1", num_retries=1)
        status, events = _stream(gateway, _body(model, marker))
        assert status == 200, events
        assert delta_text(events) == _TEXT, events
        assert event_types(events)[-1] == "error", events
        assert posted_targets(wire) == (bridge.target,)
