import asyncio
import dataclasses
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, TypeAlias
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support import prompt_cache_breakpoint as pcb
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    ASK,
    ASK_LABEL,
    CITIES,
    EPHEMERAL,
    PROVIDER_KEY,
    SYSTEM,
    SYSTEM_LABEL,
    anthropic_deployment,
    anthropic_labels,
    anthropic_peer,
    bedrock_labels,
    bedrock_peer,
    call_id,
    final_label,
    final_text,
    marker_of,
    new_marker,
    owned_config,
    tool_use_label,
)
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OWNED: Final = "bridge-mark-owned"
_OWNED_DROP: Final = "bridge-mark-owned-drop"
_NOVA: Final = "bridge-mark-nova"
_CHAT: Final = "bridge-mark-chat"
_COMPARE: Final = "Compare the three cities."
_CHECK: Final = "I will check."
_TOOL: Final[dict[str, JsonValue]] = {"type": "function", "name": "lookup_weather", "parameters": {"type": "object"}}
_UNMARKED: Final = "unmarked"
_RecordProperty: TypeAlias = Callable[[str, object], None]


@dataclass(frozen=True, slots=True)
class _Call:
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str
    call_id: str


@dataclass(frozen=True, slots=True)
class _StoredRig:
    wire: Wire
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _owned_config(directory: Path, model_list: Sequence[JsonValue]) -> Path:
    stock: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final[Mapping[str, JsonValue]] = {
        **stock,
        "model_list": list(model_list),
        "general_settings": {**object_value(stock["general_settings"]), "store_prompts_in_spend_logs": True},
        "router_settings": {**object_value(stock.get("router_settings") or {}), "num_retries": 0},
    }
    path: Final = directory / "bridge-mark-owned.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _stored_peer(request: Request) -> Reply:
    if "/converse" in request.target:
        return bedrock_peer(request)
    if request.target.endswith("/chat/completions"):
        return chat_reply(f"chatcmpl-{marker_of(request)}", "gpt-5.4", "sunny", stream=False)
    return anthropic_peer(request)


def _stored_models(api_base: str) -> list[JsonValue]:
    return [
        anthropic_deployment(_OWNED, api_base),
        anthropic_deployment(_OWNED_DROP, api_base, drop_params=True),
        {
            "model_name": _NOVA,
            "litellm_params": {
                "model": "bedrock/converse/us.amazon.nova-2-lite-v1:0",
                "api_key": PROVIDER_KEY,
                "api_base": api_base,
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": api_base,
            },
        },
        {
            "model_name": _CHAT,
            "litellm_params": {
                "model": "openai/gpt-5.4",
                "api_key": PROVIDER_KEY,
                "api_base": f"{api_base}/v1",
                "use_chat_completions_api": True,
            },
        },
    ]


@pytest.fixture(scope="module")
def stored_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_StoredRig]:
    directory: Final = tmp_path_factory.mktemp("bridge-mark-stored")
    with wire_server(answering_model_discovery(_stored_peer)) as wire, gateway_from_environment() as gateway:
        config: Final = _owned_config(directory, _stored_models(wire.url))
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2) as owned:
            yield _StoredRig(wire, owned)


@pytest.fixture(scope="module")
def yaml_drop_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_StoredRig]:
    directory: Final = tmp_path_factory.mktemp("bridge-mark-yaml-drop")
    with wire_server(anthropic_peer) as wire, gateway_from_environment() as gateway:
        config: Final = owned_config(
            directory, [anthropic_deployment(_OWNED, wire.url)], litellm_settings={"drop_params": True}
        )
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2) as owned:
            yield _StoredRig(wire, owned)


@pytest.fixture(scope="module")
def spend() -> Iterator[pcb.SpendLogs]:
    with pcb.spend_logs() as logs:
        yield logs


def _body(model: str, call: _Call) -> Mapping[str, JsonValue]:
    return {
        "model": model,
        "stream": call.stream,
        "max_output_tokens": 64,
        "cache": {"no-cache": True},
        "input": [
            {"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL},
            {"role": "user", "content": final_text(call.marker)},
        ],
    }


def _calls(count: int, *, stream: bool | None = None) -> tuple[_Call, ...]:
    return tuple(_Call(index % 2 == 1 if stream is None else stream, new_marker()) for index in range(count))


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST", "/v1/responses", json=_body(model, call), headers={"Authorization": f"Bearer {key}"}
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


def _assert_answered(served: _Served) -> None:
    assert served.status == 200, served.text
    assert "sunny" in served.text, served.text
    assert served.text.startswith(("event:", "data:")) == served.call.stream, served.text
    assert ("response.completed" in served.text) == served.call.stream, served.text
    if not served.call.stream:
        assert _JSON_OBJECT.validate_json(served.text)["status"] == "completed", served.text


def _marked_posts(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if marker_of(request) != _UNMARKED)


def _marked_once(posts: Sequence[Request], calls: Sequence[_Call]) -> None:
    by_marker: Final = {marker_of(request): request for request in posts}
    assert len(by_marker) == len(posts), [request.body for request in posts]
    assert set(by_marker) == {call.marker for call in calls}, sorted(by_marker)
    assert [anthropic_labels(by_marker[call.marker]) for call in calls] == [[SYSTEM_LABEL]] * len(calls)


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
        assert by_call[item.call_id]["status"] == "success", (item.call_id, rows)


def _health(gateway: Gateway, model: str) -> Mapping[str, JsonValue]:
    response: Final = gateway.request("GET", f"/health?model={model}", None)
    assert response.status_code in (200, 503), response.text
    return _JSON_OBJECT.validate_json(response.text)


def _free_port() -> int:
    with wire_server(anthropic_peer) as probe:
        port: Final = urlsplit(probe.url).port
    assert port is not None, probe.url
    return port


def _deployment(scenario: Scenario, api_base: str) -> str:
    return scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=api_base, api_key=PROVIDER_KEY)


async def test_mixed_burst_carries_the_system_mark_once_per_call(gateway: Gateway, spend: pcb.SpendLogs) -> None:
    calls: Final = _calls(24)
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        served: Final = await _burst(gateway, model, calls)
        assert len(served) == 24
        for item in served:
            _assert_answered(item)
        _marked_once(_marked_posts(wire), calls)
        _assert_each_lands_once(spend, model, (), served)


async def test_peer_outage_fails_cleanly_and_the_restarted_peer_receives_the_marks(
    gateway: Gateway, spend: pcb.SpendLogs
) -> None:
    port: Final = _free_port()
    while_down: Final = _calls(12)
    after: Final = _calls(12)
    with gateway.scenario() as scenario:
        model: Final = _deployment(scenario, f"http://127.0.0.1:{port}")
        failed: Final = await _burst(gateway, model, while_down)
        assert len(failed) == 12
        for item in failed:
            assert item.status >= 500, (item.status, item.text)
            assert "sunny" not in item.text and "event:" not in item.text, item.text
        down: Final = _health(gateway, model)
        assert (down["healthy_count"], down["unhealthy_count"]) == (0, 1), down
        with wire_server(anthropic_peer, port=port) as wire:
            _health(gateway, model)
            assert _marked_posts(wire) == ()
            served: Final = await _burst(gateway, model, after)
            assert len(served) == 12
            for item in served:
                _assert_answered(item)
            _marked_once(_marked_posts(wire), after)
        _assert_each_lands_once(spend, model, failed, served)


def _slow(request: Request) -> Reply:
    reply: Final = anthropic_peer(request)
    return dataclasses.replace(reply, pause_between_chunks=0.4) if reply.chunks else reply


async def test_concurrent_slow_streams_each_carry_the_mark_once(gateway: Gateway, spend: pcb.SpendLogs) -> None:
    calls: Final = _calls(6, stream=True)
    with wire_server(_slow) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        served: Final = await _burst(gateway, model, calls)
        assert len(served) == 6
        for item in served:
            _assert_answered(item)
        _marked_once(_marked_posts(wire), calls)
        _assert_each_lands_once(spend, model, (), served)


@dataclass(frozen=True, slots=True)
class _Held:
    release: threading.Event
    markers: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        marker: Final = marker_of(request)
        if marker == _UNMARKED:
            return anthropic_peer(request)
        self.markers.put(marker)
        if not self.release.wait(timeout=60):
            return Reply(status=504, body=b'{"type":"error","error":{"type":"api_error","message":"never released"}}')
        return anthropic_peer(request)


def _worker_pids(owned: OwnedProxy) -> tuple[int, ...]:
    return eventually(lambda: pcb.started_worker_pids(owned.log), lambda pids: len(pids) == 2, seconds=30)


async def _hold_burst(
    held: _Held, candidate: Gateway, model: str, calls: tuple[_Call, ...]
) -> asyncio.Task[tuple[_Served, ...]]:
    burst: Final = asyncio.create_task(_burst(candidate, model, calls, tolerate_transport_errors=True))
    await asyncio.to_thread(eventually, held.markers.qsize, lambda size: size == len(calls), 60)
    return burst


def _follow_up(gateway: Gateway, wire: Wire, spend: pcb.SpendLogs) -> None:
    call: Final = _Call(False, new_marker())
    response: Final = gateway.request("POST", "/v1/responses", dict(_body(_OWNED, call)))
    served: Final = _Served(call, response.status_code, response.text, response.headers["x-litellm-call-id"])
    _assert_answered(served)
    _marked_once(_marked_posts(wire), (call,))
    spend.landed(_OWNED, served.call_id, None)


async def test_worker_sigkill_mid_burst_leaves_the_sibling_answering(
    gateway: Gateway, tmp_path: Path, spend: pcb.SpendLogs
) -> None:
    calls: Final = _calls(20, stream=False)
    held: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(held.respond) as wire:
        config: Final = _owned_config(tmp_path, [anthropic_deployment(_OWNED, wire.url)])
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = _worker_pids(owned)
            burst: Final = await _hold_burst(held, candidate, _OWNED, calls)
            held_by: Final = MappingProxyType({pid: pcb.open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill()
                held.release.set()
                served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_answered(item)
            posts: Final = _marked_posts(wire)
            assert {marker_of(request) for request in posts} == {call.marker for call in calls}, posts
            assert [anthropic_labels(request) for request in posts] == [[SYSTEM_LABEL]] * len(posts)
            _follow_up(candidate, wire, spend)


async def test_proxy_restart_mid_burst_never_lands_a_served_call_twice(
    gateway: Gateway, tmp_path: Path, record_property: _RecordProperty, spend: pcb.SpendLogs
) -> None:
    calls: Final = _calls(20, stream=False)
    held: Final = _Held(threading.Event(), SimpleQueue())
    with wire_server(held.respond) as wire:
        config: Final = _owned_config(tmp_path, [anthropic_deployment(_OWNED, wire.url)])
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as first:
            _worker_pids(first)
            burst: Final = await _hold_burst(held, first.gateway, _OWNED, calls)
            first.begin_shutdown()
            held.release.set()
            served: Final = await burst
        for item in served:
            _assert_answered(item)
        wire.drain()
        second_directory: Final = tmp_path / "second"
        second_directory.mkdir()
        with owned_proxy_process(gateway, second_directory, {}, config=config, workers=2) as second:
            _follow_up(second.gateway, wire, spend)
    counts: Final = Counter(string_value(row["litellm_call_id"]) for row in spend.rows_for(_OWNED))
    assert all(count == 1 for count in counts.values()), counts
    landed: Final = sum(1 for item in served if item.call_id in counts)
    record_property("served", len(served))
    record_property("landed", landed)
    record_property("lost_responses", len(calls) - len(served))


def test_previous_response_replay_keeps_the_system_mark(stored_rig: _StoredRig, spend: pcb.SpendLogs) -> None:
    first_call: Final = _Call(False, new_marker())
    first: Final = stored_rig.gateway.request(
        "POST",
        "/v1/responses",
        {
            **_body(_OWNED, first_call),
            "input": [
                {"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL},
                {"role": "user", "content": final_text(first_call.marker)},
            ],
        },
    )
    assert first.status_code == 200, first.text
    first_id: Final = string_value(_JSON_OBJECT.validate_json(first.text)["id"])
    spend.landed(_OWNED, first.headers["x-litellm-call-id"], None)
    assert [anthropic_labels(request) for request in _marked_posts(stored_rig.wire)] == [[SYSTEM_LABEL]]
    second: Final = stored_rig.gateway.request(
        "POST",
        "/v1/responses",
        {
            "model": _OWNED,
            "max_output_tokens": 64,
            "cache": {"no-cache": True},
            "previous_response_id": first_id,
            "input": [{"role": "user", "content": final_text(new_marker())}],
        },
    )
    assert second.status_code == 200, second.text
    assert _JSON_OBJECT.validate_json(second.text)["status"] == "completed", second.text
    replayed: Final = _marked_posts(stored_rig.wire)
    assert len(replayed) == 1, [request.target for request in replayed]
    body: Final = _JSON_OBJECT.validate_json(replayed[0].body)
    assert isinstance(body["messages"], list) and len(body["messages"]) == 3, body
    assert anthropic_labels(replayed[0]) == [SYSTEM_LABEL]


def _respond(rig: _StoredRig, body: Mapping[str, JsonValue]) -> httpx.Response:
    response: Final = rig.gateway.request("POST", "/v1/responses", dict(body))
    assert response.status_code == 200, response.text
    assert _JSON_OBJECT.validate_json(response.text)["status"] == "completed", response.text
    return response


def _stored_turn(rig: _StoredRig, spend: pcb.SpendLogs, model: str, items: Sequence[JsonValue]) -> tuple[str, Request]:
    rig.wire.drain()
    first: Final = _respond(
        rig, {"model": model, "max_output_tokens": 64, "cache": {"no-cache": True}, "tools": [_TOOL], "input": items}
    )
    spend.landed(model, first.headers["x-litellm-call-id"], None)
    (posted,) = _marked_posts(rig.wire)
    return string_value(_JSON_OBJECT.validate_json(first.text)["id"]), posted


def _replayed(rig: _StoredRig, model: str, previous: str, items: Sequence[JsonValue]) -> Request:
    _respond(
        rig,
        {
            "model": model,
            "max_output_tokens": 64,
            "cache": {"no-cache": True},
            "tools": [_TOOL],
            "previous_response_id": previous,
            "input": list(items),
        },
    )
    (posted,) = _marked_posts(rig.wire)
    return posted


def _marked(role: str, text: str, mark: JsonValue = EPHEMERAL) -> dict[str, JsonValue]:
    return {"role": role, "content": text, "cache_control": mark}


def _weather_call(**fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "type": "function_call",
        "call_id": call_id(CITIES[0]),
        "name": "lookup_weather",
        "arguments": '{"city": "London"}',
        **fields,
    }


def _weather_output(**fields: JsonValue) -> dict[str, JsonValue]:
    return {"type": "function_call_output", "call_id": call_id(CITIES[0]), "output": "sunny.", **fields}


def _marked_tool_round() -> list[JsonValue]:
    return [
        {"role": "user", "content": ASK},
        _weather_call(cache_control=EPHEMERAL),
        _weather_output(cache_control=EPHEMERAL),
    ]


def test_replay_gives_the_current_marks_their_slots_first(stored_rig: _StoredRig, spend: pcb.SpendLogs) -> None:
    first_marker: Final = new_marker()
    second_marker: Final = new_marker()
    previous, first = _stored_turn(
        stored_rig,
        spend,
        _OWNED,
        [_marked("system", SYSTEM), _marked("user", ASK), _marked("user", final_text(first_marker))],
    )
    assert anthropic_labels(first) == [SYSTEM_LABEL, ASK_LABEL, final_label(first_marker)]
    replayed: Final = _replayed(
        stored_rig, _OWNED, previous, [_marked("user", _COMPARE), _marked("user", final_text(second_marker))]
    )
    assert anthropic_labels(replayed) == [
        ASK_LABEL,
        final_label(first_marker),
        f"user:text:{_COMPARE}",
        final_label(second_marker),
    ]


def test_replay_never_forwards_a_mark_drop_params_dropped(stored_rig: _StoredRig, spend: pcb.SpendLogs) -> None:
    previous, first = _stored_turn(
        stored_rig,
        spend,
        _OWNED_DROP,
        [_marked("system", SYSTEM, "ephemeral"), {"role": "user", "content": final_text(new_marker())}],
    )
    assert b"cache_control" not in first.body, first.body
    replayed: Final = _replayed(stored_rig, _OWNED, previous, [{"role": "user", "content": final_text(new_marker())}])
    assert b"cache_control" not in replayed.body, replayed.body


def test_replay_into_nova_drops_the_tool_block_marks(stored_rig: _StoredRig, spend: pcb.SpendLogs) -> None:
    previous, first = _stored_turn(
        stored_rig, spend, _OWNED, [*_marked_tool_round(), {"role": "user", "content": final_text(new_marker())}]
    )
    assert anthropic_labels(first) == [tool_use_label(CITIES[0]), f"user:tool_result:{call_id(CITIES[0])}"]
    replayed: Final = _replayed(stored_rig, _NOVA, previous, [{"role": "user", "content": final_text(new_marker())}])
    assert "/converse" in replayed.target, replayed.target
    assert bedrock_labels(replayed) == []


def test_replay_into_a_chat_route_without_marks_sends_none(stored_rig: _StoredRig, spend: pcb.SpendLogs) -> None:
    previous, first = _stored_turn(
        stored_rig,
        spend,
        _OWNED,
        [_marked("system", SYSTEM), {"role": "user", "content": final_text(new_marker())}],
    )
    assert anthropic_labels(first) == [SYSTEM_LABEL]
    replayed: Final = _replayed(stored_rig, _CHAT, previous, [{"role": "user", "content": final_text(new_marker())}])
    assert replayed.target == "/v1/chat/completions", replayed.target
    assert b"cache_control" not in replayed.body, replayed.body


def test_yaml_drop_params_keeps_the_newest_four_marks(yaml_drop_rig: _StoredRig) -> None:
    marker: Final = new_marker()
    yaml_drop_rig.wire.drain()
    _respond(
        yaml_drop_rig,
        {
            "model": _OWNED,
            "max_output_tokens": 64,
            "cache": {"no-cache": True},
            "tools": [_TOOL],
            "input": [
                _marked("system", SYSTEM),
                _marked("user", ASK),
                _marked("assistant", _CHECK),
                _weather_call(),
                _weather_output(cache_control=EPHEMERAL),
                _marked("user", final_text(marker)),
            ],
        },
    )
    (posted,) = _marked_posts(yaml_drop_rig.wire)
    assert anthropic_labels(posted) == [
        ASK_LABEL,
        f"assistant:text:{_CHECK}",
        f"user:tool_result:{call_id(CITIES[0])}",
        final_label(marker),
    ]
