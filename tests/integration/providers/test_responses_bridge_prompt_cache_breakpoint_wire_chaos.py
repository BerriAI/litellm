import asyncio
import dataclasses
import signal
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
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
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, eventually, gateway_from_environment, string_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_GLOBAL_UNSET: Final = "bridge-breakpoint-global-unset"
_GLOBAL_FALSE: Final = "bridge-breakpoint-global-false"
_ENDPOINTS: Final = ("chat", "messages", "responses")

Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
_RecordProperty: TypeAlias = Callable[[str, object], None]


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
class _GlobalRig:
    wire: Wire
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _global_config(directory: Path, api_base: str) -> Path:
    stock: Final = rv.JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    deployment: Final[Mapping[str, JsonValue]] = {
        "model": pcb.MODEL,
        "api_base": api_base,
        "api_key": "integration-provider-key",
    }
    config: Final[Mapping[str, JsonValue]] = {
        **stock,
        "model_list": [
            {"model_name": _GLOBAL_UNSET, "litellm_params": dict(deployment)},
            {"model_name": _GLOBAL_FALSE, "litellm_params": {**deployment, "drop_params": False}},
        ],
        "litellm_settings": {**rv.JSON_OBJECT.validate_python(stock["litellm_settings"]), "drop_params": True},
        "router_settings": {**rv.JSON_OBJECT.validate_python(stock.get("router_settings") or {}), "num_retries": 0},
    }
    path: Final = directory / "bridge-breakpoint-global.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def global_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_GlobalRig]:
    directory: Final = tmp_path_factory.mktemp("bridge-breakpoint-global")
    with wire_server(pcb.respond) as wire, gateway_from_environment() as gateway:
        config: Final = _global_config(directory, f"{wire.url}/v1")
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2) as owned:
            yield _GlobalRig(wire, owned)


@pytest.fixture(scope="module")
def spend() -> Iterator[pcb.SpendLogs]:
    with pcb.spend_logs() as logs:
        yield logs


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(model: str, call: _Call, breakpoint: JsonValue) -> Mapping[str, JsonValue]:
    common: Final[Mapping[str, JsonValue]] = {"model": model, "stream": call.stream, **pcb.NO_CACHE}
    text: Final = pcb.marked(pcb.text(pcb.prompt(call.marker)), breakpoint)
    match call.endpoint:
        case "chat":
            return {**common, "messages": [{"role": "user", "content": [text]}]}
        case "messages":
            return {**common, "max_tokens": 64, "messages": [{"role": "user", "content": [text]}]}
        case "responses":
            return {
                **common,
                "input": [{"type": "message", "role": "user", "content": [{**text, "type": "input_text"}]}],
            }


def _calls(count: int, endpoints: tuple[Endpoint, ...], *, stream: bool | None = None) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoints[index % len(endpoints)], index % 2 == 1 if stream is None else stream, uuid.uuid4().hex)
        for index in range(count)
    )


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call, breakpoint: JsonValue) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(model, call, breakpoint),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode(), response.headers["x-litellm-call-id"])


async def _burst(
    gateway: Gateway,
    model: str,
    calls: tuple[_Call, ...],
    *,
    breakpoint: JsonValue = pcb.EXPLICIT,
    tolerate_transport_errors: bool = False,
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, model, call, breakpoint) for call in calls),
            return_exceptions=tolerate_transport_errors,
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _frames(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(rv.JSON_OBJECT.validate_json(line[6:]) for line in text.splitlines() if line.startswith("data: {"))


def _upstream_id_shown_to_caller(served: _Served) -> str | None:
    if served.call.endpoint == "messages":
        return None
    if not served.call.stream:
        return string_value(rv.JSON_OBJECT.validate_json(served.text)["id"])
    frames: Final = _frames(served.text)
    if served.call.endpoint == "responses":
        (completed,) = [frame for frame in frames if frame.get("type") == "response.completed"]
        return string_value(rv.JSON_OBJECT.validate_python(completed["response"])["id"])
    return string_value(frames[0]["id"])


def _assert_answered_in_its_own_shape(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {served.call.marker}, served.text
    assert served.text.startswith(("event:", "data:")) == served.call.stream, served.text
    assert served.text.startswith("{") != served.call.stream, served.text
    assert ("response.completed" in served.text) == (served.call.stream and served.call.endpoint == "responses")
    shown: Final = _upstream_id_shown_to_caller(served)
    assert shown is None or pcb.answers(shown, served.call.marker), served.text


def _marked_once(posts: Sequence[Request], calls: Sequence[_Call], expected: JsonValue) -> None:
    by_marker: Final = {marker: request for request in posts if (marker := rv.newest_marker(request.body.decode()))}
    assert len(by_marker) == len(posts), [request.body for request in posts]
    assert set(by_marker) == {call.marker for call in calls}, sorted(by_marker)
    for call in calls:
        block: Final = pcb.single_block(pcb.input_items(by_marker[call.marker]), "user")
        assert block["type"] == "input_text" and block["text"] == pcb.prompt(call.marker), block
        pcb.assert_marker(block, expected)


def _assert_each_lands_once(
    spend: pcb.SpendLogs, model: str, failed: Sequence[_Served], served: Sequence[_Served]
) -> None:
    expected: Final = len(failed) + len(served)
    rows: Final = eventually(lambda: spend.rows_for(model), lambda found: len(found) >= expected, seconds=70)
    by_call: Final = {string_value(row["litellm_call_id"]): row for row in rows}
    assert len(by_call) == len(rows) == expected, rows
    for item in failed:
        assert by_call[item.call_id]["status"] == "failure", (item.call_id, rows)
    for item in served:
        row: Final = by_call[item.call_id]
        assert row["status"] == "success", (item.call_id, row)
        shown: Final = _upstream_id_shown_to_caller(item)
        assert shown is None or rv.same_response(string_value(row["request_id"]), shown), (row, shown)


def _health(gateway: Gateway, model: str) -> Mapping[str, JsonValue]:
    response: Final = gateway.request("GET", f"/health?model={model}", None)
    assert response.status_code in (200, 503), response.text
    return rv.JSON_OBJECT.validate_json(response.text)


def _free_port() -> int:
    with wire_server(pcb.respond) as probe:
        port: Final = urlsplit(probe.url).port
    assert port is not None, probe.url
    return port


def _chat(gateway: Gateway, model: str, marker: str, breakpoint: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", dict(_body(model, _Call("chat", False, marker), breakpoint)))


def _completion(response: httpx.Response, marker: str) -> str:
    assert response.status_code == 200, response.text
    body: Final = rv.JSON_OBJECT.validate_json(response.text)
    assert pcb.answers(string_value(body["id"]), marker), body
    assert rv.answer(marker) in response.text, response.text
    return response.headers["x-litellm-call-id"]


def _user_block_on_wire(wire: Wire, marker: str) -> dict[str, JsonValue]:
    block: Final = pcb.single_block(pcb.input_items(pcb.posted(wire, marker)), "user")
    assert block["type"] == "input_text" and block["text"] == pcb.prompt(marker), block
    return block


@pytest.mark.parametrize("model", (_GLOBAL_UNSET, _GLOBAL_FALSE), ids=("deployment-unset", "deployment-false"))
def test_global_drop_params_drops_a_malformed_marker(global_rig: _GlobalRig, model: str, spend: pcb.SpendLogs) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _completion(_chat(global_rig.gateway, model, marker, "yes"), marker)
    pcb.assert_marker(_user_block_on_wire(global_rig.wire, marker), None)
    spend.landed(model, call_id, marker)
    control: Final = uuid.uuid4().hex
    control_id: Final = _completion(_chat(global_rig.gateway, model, control, pcb.EXPLICIT), control)
    pcb.assert_marker(_user_block_on_wire(global_rig.wire, control), pcb.EXPLICIT)
    spend.landed(model, control_id, control)


@pytest.mark.parametrize("model", (_GLOBAL_UNSET, _GLOBAL_FALSE), ids=("deployment-unset", "deployment-false"))
def test_global_drop_params_drops_the_audio_part_and_carries_its_marker(
    global_rig: _GlobalRig, model: str, spend: pcb.SpendLogs
) -> None:
    marker: Final = uuid.uuid4().hex
    content: Final[list[JsonValue]] = [pcb.text(pcb.prompt(marker)), pcb.marked(pcb.audio(), pcb.EXPLICIT)]
    response: Final = global_rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": content}], **pcb.NO_CACHE},
    )
    call_id: Final = _completion(response, marker)
    assert _user_block_on_wire(global_rig.wire, marker) == {
        "type": "input_text",
        "text": pcb.prompt(marker),
        "prompt_cache_breakpoint": pcb.EXPLICIT,
    }
    spend.landed(model, call_id, marker)


async def test_mixed_burst_carries_every_marker_once(gateway: Gateway, spend: pcb.SpendLogs) -> None:
    calls: Final = _calls(24, _ENDPOINTS)
    with wire_server(pcb.respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1", drop_params=True)
        served: Final = await _burst(gateway, model, calls)
        assert len(served) == 24
        for item in served:
            _assert_answered_in_its_own_shape(item)
        _marked_once(pcb.drained_posts(wire), calls, pcb.EXPLICIT)
        _assert_each_lands_once(spend, model, (), served)


async def test_upstream_outage_fails_cleanly_and_the_restarted_upstream_serves_marked_calls(
    gateway: Gateway, spend: pcb.SpendLogs
) -> None:
    port: Final = _free_port()
    while_down: Final = _calls(12, _ENDPOINTS)
    after: Final = _calls(12, _ENDPOINTS)
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"http://127.0.0.1:{port}/v1", drop_params=True)
        failed: Final = await _burst(gateway, model, while_down)
        assert len(failed) == 12
        for item in failed:
            assert item.status >= 500, (item.status, item.text)
            assert "answer marker" not in item.text and "event:" not in item.text, item.text
        down: Final = _health(gateway, model)
        assert (down["healthy_count"], down["unhealthy_count"]) == (0, 1), down
        with wire_server(pcb.respond, port=port) as wire:
            _health(gateway, model)
            probes: Final = pcb.drained_posts(wire)
            assert [rv.newest_marker(request.body.decode()) for request in probes] == [None], probes
            served: Final = await _burst(gateway, model, after)
            assert len(served) == 12
            for item in served:
                _assert_answered_in_its_own_shape(item)
            _marked_once(pcb.drained_posts(wire), after, pcb.EXPLICIT)
        _assert_each_lands_once(spend, model, failed, served)


def _slow(request: Request) -> Reply:
    reply: Final = pcb.respond(request)
    return dataclasses.replace(reply, pause_between_chunks=0.4) if reply.chunks else reply


async def test_concurrent_slow_streams_each_complete_with_one_upstream_call(
    gateway: Gateway, spend: pcb.SpendLogs
) -> None:
    calls: Final = _calls(6, ("chat",), stream=True)
    with wire_server(_slow) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1", drop_params=True)
        served: Final = await _burst(gateway, model, calls)
        assert len(served) == 6
        for item in served:
            _assert_answered_in_its_own_shape(item)
        _marked_once(pcb.drained_posts(wire), calls, pcb.EXPLICIT)
        _assert_each_lands_once(spend, model, (), served)


@dataclass(frozen=True, slots=True)
class _Held:
    release: threading.Event
    markers: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        marker: Final = rv.newest_marker(request.body.decode()) if request.method == "POST" else None
        if marker is None:
            return pcb.respond(request)
        self.markers.put(marker)
        if not self.release.wait(timeout=60):
            return rv.error(504, "the burst was never released", "held")
        return pcb.respond(request)


def _worker_pids(owned: OwnedProxy) -> tuple[int, ...]:
    return eventually(lambda: pcb.started_worker_pids(owned.log), lambda pids: len(pids) == 2, seconds=30)


async def _hold_burst(
    held: _Held, candidate: Gateway, model: str, calls: tuple[_Call, ...]
) -> asyncio.Task[tuple[_Served, ...]]:
    burst: Final = asyncio.create_task(_burst(candidate, model, calls, tolerate_transport_errors=True))
    await asyncio.to_thread(eventually, held.markers.qsize, lambda size: size == len(calls), 60)
    return burst


async def test_worker_sigkill_mid_burst_leaves_the_sibling_answering(
    gateway: Gateway, tmp_path: Path, spend: pcb.SpendLogs
) -> None:
    calls: Final = _calls(20, ("chat",), stream=False)
    held: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(held.respond) as wire:
        config: Final = _global_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = _worker_pids(owned)
            burst: Final = await _hold_burst(held, candidate, _GLOBAL_UNSET, calls)
            held_by: Final = MappingProxyType({pid: pcb.open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            held.release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_answered_in_its_own_shape(item)
            _marked_once(pcb.drained_posts(wire), calls, pcb.EXPLICIT)
            follow_up: Final = uuid.uuid4().hex
            call_id: Final = _completion(_chat(candidate, _GLOBAL_UNSET, follow_up, "yes"), follow_up)
            pcb.assert_marker(_user_block_on_wire(wire, follow_up), None)
            spend.landed(_GLOBAL_UNSET, call_id, follow_up)


async def test_proxy_restart_mid_burst_never_lands_a_served_call_twice(
    gateway: Gateway, tmp_path: Path, record_property: _RecordProperty, spend: pcb.SpendLogs
) -> None:
    calls: Final = _calls(20, ("chat",), stream=False)
    held: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(held.respond) as wire:
        config: Final = _global_config(tmp_path, f"{wire.url}/v1")
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as first:
            _worker_pids(first)
            burst: Final = await _hold_burst(held, first.gateway, _GLOBAL_UNSET, calls)
            first.process.terminate()
            held.release.set()
            served: Final = await burst
        for item in served:
            _assert_answered_in_its_own_shape(item)
        second_directory: Final = tmp_path / "second"
        second_directory.mkdir()
        with owned_proxy_process(gateway, second_directory, {}, config=config, workers=2) as second:
            follow_up: Final = uuid.uuid4().hex
            call_id: Final = _completion(_chat(second.gateway, _GLOBAL_UNSET, follow_up, pcb.EXPLICIT), follow_up)
            pcb.assert_marker(_user_block_on_wire(wire, follow_up), pcb.EXPLICIT)
            spend.landed(_GLOBAL_UNSET, call_id, follow_up)
    counts: Final = Counter(string_value(row["litellm_call_id"]) for row in spend.rows_for(_GLOBAL_UNSET))
    assert all(count == 1 for count in counts.values()), counts
    landed: Final = sum(1 for item in served if item.call_id in counts)
    record_property("served", len(served))
    record_property("landed", landed)
    record_property("lost_responses", len(calls) - len(served))
