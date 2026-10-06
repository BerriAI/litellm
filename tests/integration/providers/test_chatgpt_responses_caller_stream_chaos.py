import asyncio
import re
import signal
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support import codex_vendor as cv
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, eventually, gateway_from_environment, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(240)

_MODEL: Final = "chatgpt/gpt-5.5"
_CONFIG_MODEL: Final = "chatgpt-caller-stream-chaos"
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_NO_CACHE: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
_ENDPOINTS: Final = ("responses", "chat", "messages")

Endpoint: TypeAlias = Literal["responses", "chat", "messages"]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str
    call_id: str


@dataclass(frozen=True, slots=True)
class _Rig:
    port: int
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway

    @property
    def vendor_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _free_port() -> int:
    with wire_server(cv.CodexVendor().respond) as probe:
        port: Final = urlsplit(probe.url).port
    assert port is not None, probe.url
    return port


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("chatgpt-caller-stream-chaos")
    port: Final = _free_port()
    overrides: Final = {"CHATGPT_TOKEN_DIR": str(cv.login(directory)), "CHATGPT_API_BASE": f"http://127.0.0.1:{port}"}
    config: Final = cv.proxy_config(directory, probe=False)
    with gateway_from_environment() as gateway:
        with owned_proxy_process(gateway, directory, overrides, config=config, workers=2) as owned:
            yield _Rig(port, owned)


@pytest.fixture
def model(rig: _Rig) -> Iterator[str]:
    with rig.gateway.scenario() as scenario:
        yield scenario.model(model=_MODEL, api_base=rig.vendor_url, api_key=None)


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "responses":
            return "/v1/responses"
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"


def _body(model: str, call: _Call) -> Mapping[str, JsonValue]:
    prompt: Final = f"Say marker-{call.marker}"
    common: Final[Mapping[str, JsonValue]] = {"model": model, "stream": call.stream, "num_retries": 0, **_NO_CACHE}
    match call.endpoint:
        case "responses":
            return {**common, "input": [{"role": "user", "content": prompt}]}
        case "chat":
            return {**common, "messages": [{"role": "user", "content": prompt}]}
        case "messages":
            return {**common, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]}


def _calls(count: int, endpoints: tuple[Endpoint, ...]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=index % 2 == 1, marker=uuid.uuid4().hex)
        for index in range(count)
    )


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(model, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode(), response.headers["x-litellm-call-id"])


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


def _frames(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(rv.JSON_OBJECT.validate_json(line[6:]) for line in text.splitlines() if line.startswith("data: {"))


def _upstream_id_shown_to_caller(served: _Served) -> str | None:
    if not served.call.stream:
        return string_value(rv.JSON_OBJECT.validate_json(served.text)["id"])
    frames: Final = _frames(served.text)
    match served.call.endpoint:
        case "responses":
            (completed,) = [frame for frame in frames if frame.get("type") == "response.completed"]
            return string_value(rv.JSON_OBJECT.validate_python(completed["response"])["id"])
        case "chat":
            return string_value(frames[0]["id"])
        case "messages":
            return None


def _assert_answered_in_its_own_shape(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {served.call.marker}, served.text
    assert served.text.startswith(("event:", "data:")) == served.call.stream, served.text
    assert served.text.startswith("{") != served.call.stream, served.text
    assert ("response.completed" in served.text) == (served.call.stream and served.call.endpoint == "responses")


def _marked(received: tuple[Request, ...]) -> Mapping[str, Request]:
    posts: Final = tuple(request for request in received if request.method == "POST")
    marked: Final = {marker: request for request in posts if (marker := rv.newest_marker(request.body.decode()))}
    assert len(marked) == len(posts), [request.body for request in posts]
    return marked


def _assert_forwarded(forwarded: Mapping[str, Request], calls: tuple[_Call, ...]) -> None:
    assert set(forwarded) == {call.marker for call in calls}, sorted(forwarded)
    for marker, request in forwarded.items():
        cv.forwarded(request, marker)


def _spend_rows(model: str, expected: int) -> Sequence[Mapping[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, litellm_call_id, status FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model,)
        ),
        lambda found: len(found) >= expected,
        seconds=70,
    )


def _assert_each_lands_once(
    rows: Sequence[Mapping[str, JsonValue]], failed: tuple[_Served, ...], served: tuple[_Served, ...]
) -> None:
    by_call: Final = {string_value(row["litellm_call_id"]): row for row in rows}
    assert len(by_call) == len(rows) == len(failed) + len(served), rows
    for item in failed:
        assert by_call[item.call_id]["status"] == "failure", (item.call_id, rows)
    for item in served:
        _assert_served_landed(by_call[item.call_id], item)


def _assert_served_landed(row: Mapping[str, JsonValue], item: _Served) -> None:
    assert row["status"] == "success", (item.call_id, row)
    shown: Final = _upstream_id_shown_to_caller(item)
    assert shown is None or rv.same_response(string_value(row["request_id"]), shown), (row, shown)


def _health(gateway: Gateway, model: str) -> Mapping[str, JsonValue]:
    response: Final = gateway.request("GET", f"/health?model={model}", None)
    assert response.status_code in (200, 503), response.text
    return rv.JSON_OBJECT.validate_json(response.text)


async def test_mixed_burst_across_the_three_endpoints_answers_each_in_its_own_shape(rig: _Rig, model: str) -> None:
    calls: Final = _calls(24, _ENDPOINTS)
    with wire_server(cv.CodexVendor().respond, port=rig.port) as wire:
        served: Final = await _burst(rig.gateway, model, calls)
        assert len(served) == 24
        for item in served:
            _assert_answered_in_its_own_shape(item)
        _assert_forwarded(_marked(wire.drain()), calls)
    _assert_each_lands_once(_spend_rows(model, 24), (), served)


async def test_vendor_outage_fails_each_call_cleanly_and_the_restarted_vendor_serves_the_next_burst(
    rig: _Rig, model: str
) -> None:
    while_down: Final = _calls(12, _ENDPOINTS)
    after: Final = _calls(12, _ENDPOINTS)
    failed: Final = await _burst(rig.gateway, model, while_down)
    assert len(failed) == 12
    for item in failed:
        assert item.status >= 500, (item.status, item.text)
        assert "answer marker" not in item.text and "event:" not in item.text, item.text
        assert item.call_id, item
    down: Final = _health(rig.gateway, model)
    assert (down["healthy_count"], down["unhealthy_count"]) == (0, 1), down
    with wire_server(cv.CodexVendor().respond, port=rig.port) as wire:
        _health(rig.gateway, model)
        probes: Final = wire.drain()
        assert [rv.newest_marker(request.body.decode()) for request in probes if request.method == "POST"] == [None]
        served: Final = await _burst(rig.gateway, model, after)
        assert len(served) == 12
        for item in served:
            _assert_answered_in_its_own_shape(item)
        _assert_forwarded(_marked(wire.drain()), after)
    _assert_each_lands_once(_spend_rows(model, 24), failed, served)


def _chaos_config(vendor_url: str, directory: Path) -> Path:
    config: Final = rv.JSON_OBJECT.validate_python(yaml.safe_load(cv.proxy_config(directory, probe=False).read_text()))
    path: Final = directory / "chatgpt-caller-stream-worker-chaos.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {"model_name": _CONFIG_MODEL, "litellm_params": {"model": _MODEL, "api_base": vendor_url}}
                ],
            }
        )
    )
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(300)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_answering_json(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final = tuple(_Call("responses", False, uuid.uuid4().hex) for _ in range(20))
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    vendor: Final = cv.CodexVendor()

    def held(request: Request) -> Reply:
        if request.method == "GET":
            return vendor.respond(request)
        marker: Final = rv.newest_marker(request.body.decode())
        assert marker is not None, request.body
        held_markers.put(marker)
        assert release.wait(timeout=60), "The burst was never released"
        return vendor.respond(request)

    with wire_server(held) as wire:
        overrides: Final = {"CHATGPT_TOKEN_DIR": str(cv.login(tmp_path)), "CHATGPT_API_BASE": wire.url}
        with owned_proxy_process(
            gateway, tmp_path, overrides, config=_chaos_config(wire.url, tmp_path), workers=2
        ) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(found.group(1)) for found in _STARTED_WORKER.finditer(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(_burst(candidate, _CONFIG_MODEL, calls, tolerate_transport_errors=True))
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_answered_in_its_own_shape(item)
            follow_up: Final = _Call("responses", False, uuid.uuid4().hex)
            (answered,) = await _burst(candidate, _CONFIG_MODEL, (follow_up,))
            _assert_answered_in_its_own_shape(answered)
            _assert_forwarded(_marked(wire.drain()), (*calls, follow_up))
