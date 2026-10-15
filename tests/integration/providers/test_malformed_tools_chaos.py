from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final, Literal, TypeAlias

import httpx
import psutil
import pytest
from integration._support.anthropic_sse import error_body, message_json, message_stream
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import write_rows
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.worker_readiness import wait_for_worker_readiness
from pydantic import JsonValue, TypeAdapter

_OPENAI_BACKEND: Final = "gpt-4o-mini"
_CLAUDE_BACKEND: Final = "claude-haiku-4-5"
_GEMINI_BACKEND: Final = "gemini-2.5-flash"
_PROVIDER_KEY: Final = "scripted-malformed-tools-chaos-key"
_OWNED_CHAT: Final = "malformed-tools-owned-chat"
_OWNED_STORE: Final = "vs_malformed_tools_owned_store"
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
_RESPAWN_SECONDS: Final = graceful_stop_seconds() + 60
_PROXY_WORKERS: Final = max(1, int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1")))
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_NOT_A_LIST: Final = "tools: Input should be a valid list"
_WEATHER_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_CHAT_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_weather", "description": "Look up the weather", "parameters": _WEATHER_SCHEMA},
}
_CLAUDE_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Look up the weather",
    "input_schema": _WEATHER_SCHEMA,
}
_RESPONSES_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Look up the weather",
    "parameters": _WEATHER_SCHEMA,
}
_GEMINI_TOOL: Final[dict[str, JsonValue]] = {
    "functionDeclarations": [
        {"name": "get_weather", "description": "Look up the weather", "parameters": _WEATHER_SCHEMA}
    ]
}
Kind: TypeAlias = Literal["valid", "rejected", "dropped"]


def _marker(label: str) -> str:
    return f"{label} marker-{uuid.uuid4().hex}"


def _tag(text: str) -> str:
    found: Final = _MARKER.search(text)
    return found.group(1) if found else "none"


def _answer(tag: str) -> str:
    return f"answer {tag}"


def _gemini_body(tag: str) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": _answer(tag)}]}, "index": 0, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2, "totalTokenCount": 11},
            "modelVersion": _GEMINI_BACKEND,
        }
    ).encode()


def _scripted(request: Request) -> Reply:
    text: Final = request.body.decode()
    tag: Final = _tag(text)
    path: Final = request.target.split("?")[0]
    stream: Final = bool(text) and _JSON_OBJECT.validate_json(request.body).get("stream") is True
    if path == "/v1/chat/completions":
        return chat_reply(f"chatcmpl-{tag}", _OPENAI_BACKEND, _answer(tag), stream=stream)
    if path == "/v1/responses":
        return responses_reply(f"resp_{tag}", _OPENAI_BACKEND, _answer(tag), stream=stream)
    if path == "/v1/messages" and _JSON_OBJECT.validate_json(request.body).get("tools") == 5:
        return Reply(status=400, body=error_body(400, _NOT_A_LIST))
    if path == "/v1/messages" and stream:
        return Reply(
            content_type="text/event-stream", chunks=message_stream(f"msg_{tag}", _CLAUDE_BACKEND, _answer(tag))
        )
    if path == "/v1/messages":
        return Reply(body=message_json(f"msg_{tag}", _CLAUDE_BACKEND, _answer(tag)))
    if path.endswith(":streamGenerateContent"):
        return Reply(content_type="text/event-stream", chunks=(b"data: " + _gemini_body(tag) + b"\r\n\r\n",))
    if path.endswith(":generateContent"):
        return Reply(body=_gemini_body(tag))
    return Reply(status=404, body=json.dumps({"error": f"unscripted {request.target}"}).encode())


@dataclass(frozen=True, slots=True)
class _Switchboard:
    down: threading.Event
    holding: threading.Event
    release: threading.Event
    held: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        if self.down.is_set():
            return Reply(drop_connection=True)
        if self.holding.is_set():
            self.held.put(request.target)
            assert self.release.wait(timeout=120), "Held request was never released"
        return _scripted(request)

    def reset(self) -> None:
        self.down.clear()
        self.release.set()
        self.holding.clear()


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire
    switchboard: _Switchboard
    chat: str
    claude: str
    gemini: str


@pytest.fixture(scope="module")
def rig() -> Iterator[_Rig]:
    switchboard: Final = _Switchboard(threading.Event(), threading.Event(), threading.Event(), SimpleQueue())
    with (
        gateway_from_environment() as gateway,
        wire_server(switchboard.respond) as wire,
        gateway.scenario() as scenario,
    ):
        built: Final = _Rig(
            gateway=gateway,
            wire=wire,
            switchboard=switchboard,
            chat=scenario.model(api_base=f"{wire.url}/v1", api_key=_PROVIDER_KEY),
            claude=scenario.model(model=f"anthropic/{_CLAUDE_BACKEND}", api_base=wire.url, api_key=_PROVIDER_KEY),
            gemini=scenario.model(model=f"gemini/{_GEMINI_BACKEND}", api_base=wire.url, api_key=_PROVIDER_KEY),
        )
        for group in (built.chat, built.claude, built.gemini):
            wait_for_worker_readiness(gateway, group, 1, _PROXY_WORKERS)
        try:
            yield built
        finally:
            switchboard.reset()


@pytest.fixture(autouse=True)
def _reset(rig: _Rig) -> Iterator[None]:
    rig.switchboard.reset()
    rig.wire.drain()
    yield
    rig.switchboard.reset()


@dataclass(frozen=True, slots=True)
class _Call:
    path: str
    body: dict[str, JsonValue]
    marker: str
    kind: Kind


def _tools_for(kind: Kind, tool: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if kind == "rejected":
        return {"tools": 5}
    if kind == "dropped":
        return {"tools": [tool, 5], "drop_params": True}
    return {"tools": [tool]}


def _call(rig: _Rig, index: int, kind: Kind, label: str) -> _Call:
    marker: Final = _marker(label)
    stream: Final = (index // 4) % 2 == 1
    family: Final = index % 4
    if family == 0:
        return _Call(
            "/v1/chat/completions",
            {
                "model": rig.chat,
                "messages": [{"role": "user", "content": marker}],
                "stream": stream,
                **_tools_for(kind, _CHAT_TOOL),
            },
            marker,
            kind,
        )
    if family == 1:
        return _Call(
            "/v1/messages",
            {
                "model": rig.claude,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": marker}],
                "stream": stream,
                **_tools_for(kind, _CLAUDE_TOOL),
            },
            marker,
            kind,
        )
    if family == 2:
        return _Call(
            "/v1/responses",
            {"model": rig.chat, "input": marker, "stream": stream, **_tools_for(kind, _RESPONSES_TOOL)},
            marker,
            kind,
        )
    action: Final = "streamGenerateContent?alt=sse" if stream else "generateContent"
    return _Call(
        f"/v1beta/models/{rig.gemini}:{action}",
        {"contents": [{"role": "user", "parts": [{"text": marker}]}], **_tools_for(kind, _GEMINI_TOOL)},
        marker,
        kind,
    )


def _kind(index: int) -> Kind:
    kinds: Final[tuple[Kind, ...]] = ("valid", "rejected", "dropped")
    return kinds[index % 3]


def _clients(stack: ExitStack, base_url: str, count: int) -> tuple[httpx.Client, ...]:
    return tuple(
        stack.enter_context(httpx.Client(base_url=base_url, timeout=120, trust_env=False)) for _ in range(count)
    )


def _post(client: httpx.Client, key: str, call: _Call) -> httpx.Response:
    return client.post(call.path, json=call.body, headers={"Authorization": f"Bearer {key}"})


def _status_or_dropped(client: httpx.Client, key: str, call: _Call) -> int | None:
    try:
        return _post(client, key, call).status_code
    except httpx.TransportError:
        return None


def _deliveries(wire: Wire) -> tuple[str, ...]:
    return tuple(request.body.decode() for request in wire.drain())


def _seen(deliveries: tuple[str, ...], marker: str) -> int:
    return sum(marker in body for body in deliveries)


def _delivered_tool_names(deliveries: tuple[str, ...], marker: str) -> tuple[str, ...]:
    (body,) = tuple(body for body in deliveries if marker in body)
    return tuple(re.findall(r'"name": ?"([a-z_]+)"', body))


def _assert_wave(calls: tuple[_Call, ...], statuses: tuple[int | None, ...], deliveries: tuple[str, ...]) -> None:
    expected_statuses: Final = tuple(400 if call.kind == "rejected" else 200 for call in calls)
    assert statuses == expected_statuses, tuple(zip(statuses, (call.marker for call in calls), strict=True))
    seen: Final = tuple(_seen(deliveries, call.marker) for call in calls)
    assert seen == tuple(0 if call.kind == "rejected" else 1 for call in calls), seen
    names: Final = tuple(_delivered_tool_names(deliveries, call.marker) for call in calls if call.kind != "rejected")
    assert all("get_weather" in found for found in names), names


def test_mixed_burst_rejects_malformed_tools_and_forwards_the_rest_once(rig: _Rig) -> None:
    with ExitStack() as stack:
        calls: Final = tuple(_call(rig, index, _kind(index), "burst") for index in range(40))
        clients: Final = _clients(stack, str(rig.gateway.client.base_url), len(calls))
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(calls)))
        statuses: Final = tuple(
            pool.map(
                lambda pair: _status_or_dropped(pair[0], rig.gateway.key, pair[1]), zip(clients, calls, strict=True)
            )
        )
        _assert_wave(calls, statuses, _deliveries(rig.wire))


def test_upstream_outage_still_rejects_malformed_tools_and_recovers(rig: _Rig) -> None:
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(rig.gateway.client.base_url), 12)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))

        def wave(label: str) -> tuple[tuple[_Call, ...], tuple[int | None, ...]]:
            calls: Final = tuple(_call(rig, index, _kind(index), label) for index in range(len(clients)))
            statuses: Final = tuple(
                pool.map(
                    lambda pair: _status_or_dropped(pair[0], rig.gateway.key, pair[1]), zip(clients, calls, strict=True)
                )
            )
            return calls, statuses

        before_calls, before_statuses = wave("before outage")
        _assert_wave(before_calls, before_statuses, _deliveries(rig.wire))
        rig.switchboard.down.set()
        outage_calls, outage_statuses = wave("during outage")
        for call, status in zip(outage_calls, outage_statuses, strict=True):
            assert (status == 400) if call.kind == "rejected" else (status is not None and status >= 500), (
                call.kind,
                status,
            )
        assert rig.gateway.request("GET", "/health/liveliness").status_code == 200
        rig.wire.drain()
        rig.switchboard.down.clear()
        after_calls, after_statuses = wave("after outage")
        _assert_wave(after_calls, after_statuses, _deliveries(rig.wire))


def test_held_upstream_does_not_delay_malformed_tools_rejections(rig: _Rig) -> None:
    with ExitStack() as stack:
        held_calls: Final = tuple(_call(rig, index * 4, "valid", "held") for index in range(8))
        rejected_calls: Final = tuple(_call(rig, index, "rejected", "rejected") for index in range(8))
        clients: Final = _clients(stack, str(rig.gateway.client.base_url), len(held_calls) + len(rejected_calls))
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        rig.switchboard.release.clear()
        rig.switchboard.holding.set()
        stack.callback(rig.switchboard.release.set)
        pending: Final[tuple[Future[int | None], ...]] = tuple(
            pool.submit(_status_or_dropped, client, rig.gateway.key, call)
            for client, call in zip(clients[: len(held_calls)], held_calls, strict=True)
        )
        eventually(rig.switchboard.held.qsize, lambda size: size == len(held_calls), seconds=60)
        rejected: Final = tuple(
            pool.map(
                lambda pair: _status_or_dropped(pair[0], rig.gateway.key, pair[1]),
                zip(clients[len(held_calls) :], rejected_calls, strict=True),
            )
        )
        assert rejected == (400,) * len(rejected_calls), rejected
        assert not any(future.done() for future in pending), "a held request finished before its release"
        rig.switchboard.release.set()
        assert tuple(future.result(timeout=120) for future in pending) == (200,) * len(held_calls)
        deliveries: Final = _deliveries(rig.wire)
        assert tuple(_seen(deliveries, call.marker) for call in held_calls) == (1,) * len(held_calls)
        assert tuple(_seen(deliveries, call.marker) for call in rejected_calls) == (0,) * len(rejected_calls)


@dataclass(frozen=True, slots=True)
class _Owned:
    proxy: OwnedProxy
    wire: Wire
    held: SimpleQueue[str]
    holding: threading.Event
    release: threading.Event
    blocked_tool: str
    denied_tool: str


def _owned_config(path: Path, upstream: str, denied_tool: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": _OWNED_CHAT,
                        "litellm_params": {
                            "model": f"openai/{_OPENAI_BACKEND}",
                            "api_base": f"{upstream}/v1",
                            "api_key": _PROVIDER_KEY,
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
                "litellm_settings": {"drop_params": True},
                "router_settings": {"disable_cooldowns": True},
                "guardrails": [
                    {
                        "guardrail_name": "malformed-tools-policy",
                        "litellm_params": {"guardrail": "tool_policy", "mode": "pre_call", "default_on": True},
                    },
                    {
                        "guardrail_name": "malformed-tools-permission",
                        "litellm_params": {
                            "guardrail": "tool_permission",
                            "mode": "pre_call",
                            "default_on": False,
                            "default_action": "allow",
                            "on_disallowed_action": "block",
                            "rules": [{"id": "deny_tool", "tool_name": f"^{denied_tool}$", "decision": "deny"}],
                        },
                    },
                ],
                "vector_store_registry": [
                    {
                        "vector_store_name": "malformed-tools-owned-store",
                        "litellm_params": {
                            "vector_store_id": _OWNED_STORE,
                            "custom_llm_provider": "openai",
                            "api_base": upstream,
                            "api_key": _PROVIDER_KEY,
                        },
                    }
                ],
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def owned(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Owned]:
    suffix: Final = uuid.uuid4().hex[:12]
    blocked_tool: Final = f"wipe_disk_{suffix}"
    denied_tool: Final = f"run_shell_{suffix}"
    held: Final[SimpleQueue[str]] = SimpleQueue()
    holding: Final = threading.Event()
    release: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if holding.is_set():
            held.put(request.target)
            assert release.wait(timeout=120), "Held request was never released"
        return _scripted(request)

    directory: Final = tmp_path_factory.mktemp("malformed-tools-owned")
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        policy: Final = gateway.request(
            "POST", "/v1/tool/policy", {"tool_name": blocked_tool, "input_policy": "blocked"}
        )
        assert policy.status_code == 200, policy.text
        stack.callback(write_rows, 'DELETE FROM "LiteLLM_ToolTable" WHERE tool_name = %s', (blocked_tool,))
        wire: Final = stack.enter_context(wire_server(respond))
        stack.callback(release.set)
        proxy: Final = stack.enter_context(
            owned_proxy_process(
                gateway,
                directory,
                {"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": _PROVIDER_KEY},
                config=_owned_config(directory / "malformed-tools.yaml", wire.url, denied_tool),
                workers=2,
            )
        )
        yield _Owned(proxy, wire, held, holding, release, blocked_tool, denied_tool)


def _owned_chat(owned: _Owned, marker: str, **fields: JsonValue) -> httpx.Response:
    return owned.proxy.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": _OWNED_CHAT, "messages": [{"role": "user", "content": marker}], **fields},
    )


def _named_tool(name: str) -> dict[str, JsonValue]:
    return {"type": "function", "function": {"name": name, "parameters": _WEATHER_SCHEMA}}


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_global_drop_params_wins_over_an_explicit_request_false(owned: _Owned) -> None:
    marker: Final = _marker("global drop")
    response: Final = _owned_chat(owned, marker, tools=5, drop_params=False)
    assert response.status_code == 200, response.text
    assert _answer(_tag(marker)) in response.text, response.text
    (sent,) = tuple(request for request in owned.wire.drain() if marker in request.body.decode())
    assert "tools" not in _JSON_OBJECT.validate_json(sent.body), sent.body


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_tool_policy_still_blocks_a_tool_kept_beside_a_dropped_entry(owned: _Owned) -> None:
    marker: Final = _marker("tool policy")
    response: Final = _owned_chat(owned, marker, tools=[_named_tool(owned.blocked_tool), 5])
    assert response.status_code == 400, response.text
    assert "Violated tool policy" in response.text and owned.blocked_tool in response.text, response.text
    assert tuple(request for request in owned.wire.drain() if marker in request.body.decode()) == ()


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_tool_permission_still_denies_a_tool_kept_beside_a_dropped_entry(owned: _Owned) -> None:
    marker: Final = _marker("tool permission")
    response: Final = _owned_chat(
        owned, marker, tools=[_named_tool(owned.denied_tool), "x"], guardrails=["malformed-tools-permission"]
    )
    assert response.status_code == 400, response.text
    assert "Violated guardrail policy" in response.text and owned.denied_tool in response.text, response.text
    assert tuple(request for request in owned.wire.drain() if marker in request.body.decode()) == ()


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_anthropic_pass_through_forwards_malformed_tools_unchanged(owned: _Owned) -> None:
    marker: Final = _marker("pass-through")
    body: Final[dict[str, JsonValue]] = {
        "model": _CLAUDE_BACKEND,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": marker}],
        "tools": 5,
    }
    response: Final = owned.proxy.gateway.request("POST", "/anthropic/v1/messages", body)
    assert response.status_code == 400, response.text
    assert _JSON_OBJECT.validate_json(response.content) == _JSON_OBJECT.validate_json(error_body(400, _NOT_A_LIST))
    (sent,) = tuple(request for request in owned.wire.drain() if marker in request.body.decode())
    assert sent.target == "/v1/messages", sent.target
    assert _JSON_OBJECT.validate_json(sent.body) == body, sent.body


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_anthropic_pass_through_streams_well_formed_tools(owned: _Owned) -> None:
    marker: Final = _marker("pass-through stream")
    body: Final[dict[str, JsonValue]] = {
        "model": _CLAUDE_BACKEND,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": marker}],
        "tools": [_CLAUDE_TOOL],
        "stream": True,
    }
    response: Final = owned.proxy.gateway.request("POST", "/anthropic/v1/messages", body)
    assert response.status_code == 200, response.text
    assert response.content == b"".join(message_stream(f"msg_{_tag(marker)}", _CLAUDE_BACKEND, _answer(_tag(marker))))
    (sent,) = tuple(request for request in owned.wire.drain() if marker in request.body.decode())
    assert _JSON_OBJECT.validate_json(sent.body) == body, sent.body


def _local_port(client: httpx.Client) -> int:
    with client.stream("GET", "/health/liveliness") as response:
        port: Final = int(response.extensions["network_stream"].get_extra_info("client_addr")[1])
        response.read()
    assert response.status_code == 200, response.text
    return port


def _accepted_client_ports(pid: int, proxy_port: int) -> frozenset[int]:
    return frozenset(
        connection.raddr.port
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.raddr and connection.laddr.port == proxy_port
    )


def _owned_call(index: int, label: str) -> _Call:
    marker: Final = _marker(label)
    tools: Final[JsonValue] = [_CHAT_TOOL, 5] if index % 2 else [_CHAT_TOOL]
    return _Call(
        "/v1/chat/completions",
        {
            "model": _OWNED_CHAT,
            "messages": [{"role": "user", "content": marker}],
            "stream": index % 4 == 1,
            "tools": tools,
        },
        marker,
        "dropped" if index % 2 else "valid",
    )


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_worker_sigkill_mid_burst_leaves_the_replacement_dropping_bad_entries(owned: _Owned) -> None:
    proxy_url: Final = owned.proxy.gateway.client.base_url
    workers: Final = eventually(
        lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.proxy.log.read_text())),
        lambda pids: len(pids) >= 2,
        seconds=30,
    )
    started_before: Final = len(workers)
    serving: Final = owned.proxy.worker_pids()
    owned.wire.drain()
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(proxy_url), 12)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        ports: Final = tuple(_local_port(client) for client in clients)
        calls: Final = tuple(_owned_call(index, "worker kill") for index in range(len(clients)))
        owned.release.clear()
        owned.holding.set()
        stack.callback(owned.holding.clear)
        stack.callback(owned.release.set)
        futures: Final = tuple(
            pool.submit(_status_or_dropped, client, owned.proxy.gateway.key, call)
            for client, call in zip(clients, calls, strict=True)
        )
        eventually(owned.held.qsize, lambda size: size == len(clients), seconds=60)
        shares: Final = {pid: _accepted_client_ports(pid, proxy_url.port or 0) & frozenset(ports) for pid in serving}
        assert sum(map(len, shares.values())) == len(clients), shares
        victim: Final = min((pid for pid in serving if shares[pid]), key=lambda pid: len(shares[pid]))
        with owned.proxy.replacing_worker(victim) as kill:
            kill()
            owned.holding.clear()
            owned.release.set()
            results: Final = tuple(future.result(timeout=120) for future in futures)
            for port, result in zip(ports, results, strict=True):
                assert result == (None if port in shares[victim] else 200), (port, result, shares)
        eventually(
            lambda: len(_STARTED_WORKER.findall(owned.proxy.log.read_text())),
            lambda started: started > started_before,
            _RESPAWN_SECONDS,
        )
        second_wave: Final = tuple(_owned_call(index * 2 + 1, "after kill") for index in range(6))
        (fresh,) = _clients(stack, str(proxy_url), 1)
        assert tuple(_post(fresh, owned.proxy.gateway.key, call).status_code for call in second_wave) == (200,) * len(
            second_wave
        )
        deliveries: Final = _deliveries(owned.wire)
        everything: Final = calls + second_wave
        assert tuple(_seen(deliveries, call.marker) for call in everything) == (1,) * len(everything)
        assert tuple(_delivered_tool_names(deliveries, call.marker) for call in everything) == (
            ("get_weather",),
        ) * len(everything)
        assert owned.proxy.process.poll() is None
