import asyncio
import json
import re
import socket
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import anthropic
import httpx
import openai
import psutil
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, configless_proxy_process, graceful_stop_seconds
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.responses import ResponseFailedEvent
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "gemini/gemini-3.8-flash"
_GEMINI_KEY: Final = "scripted-gemini-key"
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_HEALTHY_PREFIX: Final = "/healthy"
_ANTHROPIC_VERSION: Final[Mapping[str, str]] = MappingProxyType({"anthropic-version": "2023-06-01"})

Serving: TypeAlias = Literal["router", "configless"]
Kind: TypeAlias = Literal["chat_stream", "chat", "messages_stream", "responses_stream"]
PassThrough: TypeAlias = Literal["gemini", "anthropic"]
_SERVINGS: Final[tuple[Serving, ...]] = ("router", "configless")
_PASS_THROUGHS: Final[tuple[PassThrough, ...]] = ("gemini", "anthropic")
_FORWARDS_PROVIDER_HEADERS: Final[frozenset[tuple[Serving, Kind]]] = frozenset(
    {("router", "chat"), ("router", "chat_stream"), ("configless", "chat")}
)
_KINDS: Final[tuple[Kind, ...]] = ("chat_stream", "chat", "messages_stream", "responses_stream")


@dataclass(frozen=True, slots=True)
class _Rig:
    port: int
    proxy: OwnedProxy

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True, slots=True)
class _Call:
    kind: Kind
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    headers: Mapping[str, str]
    text: str
    call_id: str


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


@contextmanager
def _configless(gateway: Gateway, directory: Path, port: int) -> Iterator[_Rig]:
    url: Final = f"http://127.0.0.1:{port}"
    overrides: Final = {
        "GEMINI_API_KEY": _GEMINI_KEY,
        "GEMINI_API_BASE": url,
        "ANTHROPIC_API_BASE": url,
        "ANTHROPIC_API_KEY": "scripted-anthropic-key",
    }
    with configless_proxy_process(gateway, directory, overrides, model=_MODEL, api_base=url, workers=2) as owned:
        yield _Rig(port, owned)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    with (
        gateway_from_environment() as shared,
        _configless(shared, tmp_path_factory.mktemp("configless"), _free_port()) as owned,
    ):
        yield owned


def _newest_marker(text: str) -> str | None:
    found: Final = _MARKER.findall(text)
    return found[-1] if found else None


def _request_id(marker: str) -> str:
    return f"req-{marker}"


def _gemini_frame(text: str, finish: str | None) -> bytes:
    candidate: Final = {
        "content": {"role": "model", "parts": [{"text": text}]},
        **({"finishReason": finish} if finish else {}),
    }
    usage: Final = {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
    return b"data: " + json.dumps({"candidates": [candidate], "usageMetadata": usage}).encode() + b"\n\n"


def _respond(request: Request) -> Reply:
    assert request.method == "POST", request.method
    marker: Final = _newest_marker(request.body.decode())
    assert marker is not None, request.body
    if request.target.startswith(_HEALTHY_PREFIX):
        return Reply(
            content_type="text/event-stream",
            chunks=(_gemini_frame("healthy ", None), _gemini_frame(f"marker-{marker}", "STOP")),
        )
    headers: Final = {"retry-after": "7", "x-request-id": _request_id(marker)}
    if request.target.startswith("/v1/messages"):
        error: Final = {"type": "error", "error": {"type": "rate_limit_error", "message": "scripted rate limit"}}
        return Reply(status=429, body=json.dumps(error).encode(), headers=headers)
    body: Final = {"error": {"code": 429, "message": "scripted rate limit", "status": "RESOURCE_EXHAUSTED"}}
    return Reply(status=429, body=json.dumps(body).encode(), headers=headers)


@contextmanager
def _model_on(serving: Serving, gateway: Gateway, rig: _Rig, wire: Wire) -> Iterator[tuple[Gateway, str]]:
    match serving:
        case "router":
            with gateway.scenario() as scenario:
                yield gateway, scenario.model(model=_MODEL, api_base=wire.url, api_key=_GEMINI_KEY, num_retries=0)
        case "configless":
            yield rig.proxy.gateway, _MODEL


def _forwarded(headers: Mapping[str, str], name: str) -> str | None:
    return headers.get(f"llm_provider-{name}") or headers.get(name)


def _metadata(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    stored: Final = row["metadata"]
    return JSON_OBJECT.validate_json(stored) if isinstance(stored, str) else object_value(stored)


def _provider_request_id(row: Mapping[str, JsonValue]) -> JsonValue:
    return object_value(_metadata(row).get("error_information") or {}).get("error_provider_request_id")


def _rows(call_ids: Sequence[str]) -> Sequence[Mapping[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT litellm_call_id, status, metadata FROM "LiteLLM_SpendLogs" '
            "WHERE litellm_call_id = ANY(string_to_array(%s, %s))",
            (",".join(call_ids), ","),
        ),
        lambda found: len(found) >= len(call_ids),
        seconds=70,
    )


def _rows_by_call(call_ids: Sequence[str]) -> Mapping[str, Mapping[str, JsonValue]]:
    rows: Final = _rows(call_ids)
    by_call: Final = MappingProxyType({string_value(row["litellm_call_id"]): row for row in rows})
    assert len(by_call) == len(rows) == len(call_ids), rows
    return by_call


def _assert_failure_row_names_its_provider_request(call_id: str, marker: str) -> None:
    row: Final = _rows_by_call((call_id,))[call_id]
    assert (row["status"], _provider_request_id(row)) == ("failure", _request_id(marker)), _metadata(row)


def _assert_one_upstream_call(received: tuple[Request, ...], marker: str, target: str) -> None:
    assert [(request.target.split("?")[0], _newest_marker(request.body.decode())) for request in received] == [
        (target, marker)
    ], [request.target for request in received]


def _upstream_target(kind: Kind) -> str:
    match kind:
        case "chat":
            return "/models/gemini-3.8-flash:generateContent"
        case "chat_stream" | "messages_stream" | "responses_stream":
            return "/models/gemini-3.8-flash:streamGenerateContent"


def _route(kind: Kind) -> str:
    match kind:
        case "chat_stream" | "chat":
            return "/v1/chat/completions"
        case "messages_stream":
            return "/v1/messages"
        case "responses_stream":
            return "/v1/responses"


def _body(call: _Call, model: str) -> Mapping[str, JsonValue]:
    prompt: Final = f"{call.kind} marker-{call.marker}"
    match call.kind:
        case "chat_stream" | "chat":
            return {
                "model": model,
                "stream": call.kind == "chat_stream",
                "messages": [{"role": "user", "content": prompt}],
                "cache": {"no-cache": True},
            }
        case "messages_stream":
            return {"model": model, "stream": True, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]}
        case "responses_stream":
            return {"model": model, "stream": True, "input": prompt}


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _route(call.kind),
        json=_body(call, model),
        headers={"Authorization": f"Bearer {key}", **_ANTHROPIC_VERSION},
    ) as response:
        raw: Final = await response.aread()
    return _Served(
        call,
        response.status_code,
        MappingProxyType(dict(response.headers)),
        raw.decode(),
        response.headers["x-litellm-call-id"],
    )


async def _burst(
    gateway: Gateway, model: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _data_frames(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    )


def _delta_text(frame: Mapping[str, JsonValue]) -> str:
    choices: Final = frame.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    return string_value(object_value(object_value(choices[0])["delta"]).get("content") or "")


def _assert_provider_headers_forwarded(headers: httpx.Headers, marker: str) -> None:
    forwarded: Final = (_forwarded(headers, "x-request-id"), _forwarded(headers, "retry-after"))
    assert forwarded == (_request_id(marker), "7"), headers


def _assert_answered_in_its_own_shape(served: _Served, serving: Serving) -> None:
    match served.call.kind:
        case "chat_stream" | "chat":
            error: Final = object_value(JSON_OBJECT.validate_json(served.text)["error"])
            assert (served.status, str(error["code"])) == (429, "429"), served.text
            if (serving, served.call.kind) in _FORWARDS_PROVIDER_HEADERS:
                _assert_provider_headers_forwarded(served.headers, served.call.marker)
        case "messages_stream":
            assert served.status == 200, served.text
            frame: Final = _data_frames(served.text)[-1]
            assert frame["type"] == "error", served.text
            assert object_value(frame["error"])["type"] == "rate_limit_error", frame
        case "responses_stream":
            assert served.status == 200, served.text
            failed: Final = _data_frames(served.text)[-1]
            assert failed["type"] == "response.failed", served.text
            assert object_value(object_value(failed["response"])["error"])["code"] == "rate_limit_exceeded", failed


def _assert_forwarded_once_each(received: tuple[Request, ...], calls: Sequence[_Call]) -> None:
    upstream: Final = sorted(
        (request.target.split("?")[0], _newest_marker(request.body.decode()) or "") for request in received
    )
    assert upstream == sorted((_upstream_target(call.kind), call.marker) for call in calls), upstream


def _assert_each_row_names_its_own_provider_request(served: Sequence[_Served]) -> None:
    by_call: Final = _rows_by_call(tuple(item.call_id for item in served))
    observed: Final = {
        item.call.marker: (by_call[item.call_id]["status"], _provider_request_id(by_call[item.call_id]))
        for item in served
    }
    assert observed == {item.call.marker: ("failure", _request_id(item.call.marker)) for item in served}, observed


@pytest.mark.parametrize("serving", _SERVINGS)
async def test_chat_stream_open_rate_limit_answers_429_with_the_provider_request_id_like_its_non_stream_twin(
    gateway: Gateway, rig: _Rig, serving: Serving
) -> None:
    with wire_server(_respond, port=rig.port) as wire, _model_on(serving, gateway, rig, wire) as (target, model):
        (twin,) = await _burst(target, model, (_Call("chat", uuid.uuid4().hex),))
        (streamed,) = await _burst(target, model, (_Call("chat_stream", uuid.uuid4().hex),))
        _assert_answered_in_its_own_shape(twin, serving)
        _assert_answered_in_its_own_shape(streamed, serving)
        _assert_forwarded_once_each(wire.drain(), (twin.call, streamed.call))
    _assert_each_row_names_its_own_provider_request((twin, streamed))


@pytest.mark.parametrize("serving", _SERVINGS)
def test_openai_sdk_chat_stream_open_rate_limit_raises_with_the_provider_request_id(
    gateway: Gateway, rig: _Rig, serving: Serving
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_respond, port=rig.port) as wire, _model_on(serving, gateway, rig, wire) as (target, model):
        client: Final = openai.OpenAI(base_url=f"{target.client.base_url}/v1", api_key=target.key, max_retries=0)
        with pytest.raises(openai.RateLimitError) as raised:
            tuple(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"chat_stream marker-{marker}"}],
                    stream=True,
                    extra_body={"cache": {"no-cache": True}},
                )
            )
        _assert_one_upstream_call(wire.drain(), marker, _upstream_target("chat_stream"))
    headers: Final = raised.value.response.headers
    if (serving, "chat_stream") in _FORWARDS_PROVIDER_HEADERS:
        _assert_provider_headers_forwarded(headers, marker)
    _assert_failure_row_names_its_provider_request(headers["x-litellm-call-id"], marker)


@pytest.mark.parametrize("serving", _SERVINGS)
def test_anthropic_sdk_messages_stream_open_rate_limit_ends_with_a_rate_limit_error_event(
    gateway: Gateway, rig: _Rig, serving: Serving
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_respond, port=rig.port) as wire, _model_on(serving, gateway, rig, wire) as (target, model):
        client: Final = anthropic.Anthropic(base_url=str(target.client.base_url), api_key=target.key, max_retries=0)
        with pytest.raises(anthropic.APIStatusError) as raised:
            tuple(
                client.messages.create(
                    model=model,
                    max_tokens=16,
                    messages=[{"role": "user", "content": f"messages_stream marker-{marker}"}],
                    stream=True,
                )
            )
        _assert_one_upstream_call(wire.drain(), marker, _upstream_target("messages_stream"))
    assert raised.value.status_code == 200, raised.value
    error: Final = object_value(JSON_OBJECT.validate_python(raised.value.body)["error"])
    assert error["type"] == "rate_limit_error", raised.value.body
    _assert_failure_row_names_its_provider_request(raised.value.response.headers["x-litellm-call-id"], marker)


@pytest.mark.parametrize("serving", _SERVINGS)
def test_openai_sdk_responses_stream_open_rate_limit_ends_with_a_failed_response(
    gateway: Gateway, rig: _Rig, serving: Serving
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_respond, port=rig.port) as wire, _model_on(serving, gateway, rig, wire) as (target, model):
        client: Final = openai.OpenAI(base_url=f"{target.client.base_url}/v1", api_key=target.key, max_retries=0)
        raw: Final = client.responses.with_raw_response.create(
            model=model, input=f"responses_stream marker-{marker}", stream=True
        )
        events: Final = tuple(raw.parse())
        _assert_one_upstream_call(wire.drain(), marker, _upstream_target("responses_stream"))
    failed: Final = events[-1]
    assert isinstance(failed, ResponseFailedEvent), events
    assert failed.response.error is not None and failed.response.error.code == "rate_limit_exceeded", failed
    _assert_failure_row_names_its_provider_request(raw.headers["x-litellm-call-id"], marker)


def test_router_stream_fallback_answers_from_the_healthy_deployment_after_the_stream_open_rate_limit(
    gateway: Gateway, rig: _Rig
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_respond, port=rig.port) as wire, gateway.scenario() as scenario:
        limited: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_GEMINI_KEY, num_retries=0)
        healthy: Final = scenario.model(
            model=_MODEL, api_base=f"{wire.url}{_HEALTHY_PREFIX}", api_key=_GEMINI_KEY, num_retries=0
        )
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": limited,
                "stream": True,
                "messages": [{"role": "user", "content": f"fallback marker-{marker}"}],
                "fallbacks": [healthy],
                "cache": {"no-cache": True},
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=60,
        ) as response:
            text: Final = response.read().decode()
        received: Final = wire.drain()
    assert response.status_code == 200, text
    content: Final = "".join(_delta_text(frame) for frame in _data_frames(text))
    assert content == f"healthy marker-{marker}", text
    assert [(request.target.split("?")[0], _newest_marker(request.body.decode())) for request in received] == [
        (_upstream_target("chat_stream"), marker),
        (f"{_HEALTHY_PREFIX}{_upstream_target('chat_stream')}", marker),
    ], received


def _pass_through(gateway: Gateway, route: PassThrough, prompt: str) -> httpx.Response:
    match route:
        case "gemini":
            return gateway.client.post(
                "/gemini/v1beta/models/gemini-3.8-flash:streamGenerateContent",
                params={"alt": "sse", "key": gateway.key},
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
                timeout=60,
            )
        case "anthropic":
            return gateway.client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-opus-5-5",
                    "max_tokens": 16,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={"Authorization": f"Bearer {gateway.key}", **_ANTHROPIC_VERSION},
                timeout=60,
            )


def _pass_through_target(route: PassThrough) -> str:
    match route:
        case "gemini":
            return "/v1beta/models/gemini-3.8-flash:streamGenerateContent"
        case "anthropic":
            return "/v1/messages"


@pytest.mark.parametrize("route", _PASS_THROUGHS)
def test_pass_through_stream_rate_limit_relays_the_provider_status_and_headers(rig: _Rig, route: PassThrough) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_respond, port=rig.port) as wire:
        response: Final = _pass_through(rig.proxy.gateway, route, f"pass-through marker-{marker}")
        _assert_one_upstream_call(wire.drain(), marker, _pass_through_target(route))
    assert response.status_code == 429, response.text
    assert response.headers["x-request-id"] == _request_id(marker), response.headers
    assert response.headers["retry-after"] == "7", response.headers
    assert "scripted rate limit" in response.text, response.text
    call_id: Final = response.headers["x-litellm-call-id"]
    assert _rows_by_call((call_id,))[call_id]["status"] == "failure", call_id


@pytest.mark.parametrize("serving", _SERVINGS)
async def test_mixed_burst_answers_and_logs_each_stream_open_rate_limit_with_its_own_provider_request_id(
    gateway: Gateway, rig: _Rig, serving: Serving
) -> None:
    calls: Final = tuple(_Call(_KINDS[index % len(_KINDS)], uuid.uuid4().hex) for index in range(32))
    with wire_server(_respond, port=rig.port) as wire, _model_on(serving, gateway, rig, wire) as (target, model):
        served: Final = await _burst(target, model, calls)
        assert len(served) == len(calls)
        for item in served:
            _assert_answered_in_its_own_shape(item, serving)
        _assert_forwarded_once_each(wire.drain(), calls)
    _assert_each_row_names_its_own_provider_request(served)


@dataclass(frozen=True, slots=True)
class _Held:
    release: threading.Event
    markers: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        marker: Final = _newest_marker(request.body.decode())
        assert marker is not None, request.body
        self.markers.put(marker)
        assert self.release.wait(timeout=60), "The burst was never released"
        return _respond(request)


def _held() -> _Held:
    return _Held(threading.Event(), SimpleQueue())


def _stream_calls(count: int) -> tuple[_Call, ...]:
    return tuple(_Call("chat_stream", uuid.uuid4().hex) for _ in range(count))


async def _held_burst(gateway: Gateway, calls: tuple[_Call, ...], held: _Held) -> asyncio.Task[tuple[_Served, ...]]:
    burst: Final = asyncio.create_task(_burst(gateway, _MODEL, calls, tolerate_transport_errors=True))
    await asyncio.to_thread(eventually, held.markers.qsize, lambda size: size == len(calls), 60)
    return burst


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


async def test_worker_sigkill_mid_burst_leaves_the_sibling_answering_each_stream_open_rate_limit(rig: _Rig) -> None:
    calls: Final = _stream_calls(20)
    held: Final = _held()
    with wire_server(held.respond, port=rig.port) as wire:
        workers: Final = eventually(rig.proxy.worker_pids, lambda pids: len(pids) == 2, seconds=30)
        burst: Final = await _held_burst(rig.proxy.gateway, calls, held)
        held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
        assert sum(held_by.values()) == 20, held_by
        victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
        with rig.proxy.replacing_worker(victim_pid) as kill:
            kill.freeze()
            await asyncio.to_thread(kill)
            held.release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_answered_in_its_own_shape(item, "configless")
            (answered,) = await _burst(rig.proxy.gateway, _MODEL, _stream_calls(1))
            _assert_answered_in_its_own_shape(answered, "configless")
        _assert_forwarded_once_each(wire.drain(), (*calls, answered.call))
    _assert_each_row_names_its_own_provider_request((*served, answered))


@pytest.mark.timeout(4 * graceful_stop_seconds() + 240)
async def test_proxy_sigterm_mid_burst_answers_each_in_flight_stream_and_the_restarted_proxy_serves(
    gateway: Gateway, tmp_path: Path
) -> None:
    port: Final = _free_port()
    calls: Final = _stream_calls(20)
    held: Final = _held()
    with wire_server(held.respond, port=port) as wire:
        with _configless(gateway, tmp_path, port) as owned:
            burst: Final = await _held_burst(owned.proxy.gateway, calls, held)
            owned.proxy.begin_shutdown()
            held.release.set()
            served: Final = await burst
            await asyncio.to_thread(
                eventually, owned.proxy.process.poll, lambda code: code is not None, graceful_stop_seconds()
            )
        assert len(served) == len(calls), len(served)
        for item in served:
            _assert_answered_in_its_own_shape(item, "configless")
        _assert_forwarded_once_each(wire.drain(), calls)
        _assert_each_row_names_its_own_provider_request(served)
        with _configless(gateway, tmp_path, port) as restarted:
            (answered,) = await _burst(restarted.proxy.gateway, _MODEL, _stream_calls(1))
        _assert_answered_in_its_own_shape(answered, "configless")
        _assert_forwarded_once_each(wire.drain(), (answered.call,))
    _assert_each_row_names_its_own_provider_request((answered,))
