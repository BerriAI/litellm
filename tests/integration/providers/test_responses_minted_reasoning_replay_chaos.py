import asyncio
import json
import re
import signal
import socket
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import websockets
import yaml
from integration._support import claude_code as cc
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_GPT: Final = "gpt-5.6"
_CODEX: Final = "gpt-5.3-codex"
_OPENAI_KEY: Final = "synthetic-openai-key"
_CONFIG_MODEL: Final = "responses-minted-reasoning-chaos"
_FOUNDRY_BASE: Final = "http://minted-reasoning-audit.services.ai.azure.com"
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})

Endpoint = Literal["responses", "chat", "messages"]


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
class _Models:
    responses: str
    chat: str
    messages: str

    def of(self, endpoint: Endpoint) -> str:
        match endpoint:
            case "responses":
                return self.responses
            case "chat":
                return self.chat
            case "messages":
                return self.messages


def _register(scenario: Scenario, api_base: str) -> _Models:
    return _Models(
        responses=scenario.model(model=f"openai/{_GPT}", api_base=api_base, api_key=_OPENAI_KEY),
        chat=scenario.model(model=f"openai/{_CODEX}", api_base=api_base, api_key=_OPENAI_KEY),
        messages=scenario.model(model=f"anthropic/{cc.OPUS}", api_base=api_base, api_key=cc.ANTHROPIC_API_KEY),
    )


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "responses":
            return "/v1/responses"
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"


def _body(models: _Models, call: _Call) -> dict[str, JsonValue]:
    common: Final[dict[str, JsonValue]] = {
        "model": models.of(call.endpoint),
        "stream": call.stream,
        "num_retries": 0,
        **_CACHE_BUST,
    }
    match call.endpoint:
        case "responses":
            return {**common, "input": rv.agents_sdk_history(call.marker, rv.minted_item(call.marker))}
        case "chat":
            return {
                **common,
                "messages": [
                    {"role": "user", "content": "Pick a city."},
                    {
                        "role": "assistant",
                        "content": "Prague",
                        "reasoning_items": [
                            {"type": "reasoning", "encrypted_content": f"gAAAAA-stored-{call.marker}", "summary": []}
                        ],
                    },
                    {"role": "user", "content": f"Name a landmark marker-{call.marker}"},
                ],
            }
        case "messages":
            return {
                **common,
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "Pick a city."},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": rv.THOUGHT, "signature": rv.signature(call.marker)},
                            {"type": "text", "text": "Prague"},
                        ],
                    },
                    {"role": "user", "content": f"Name a landmark marker-{call.marker}"},
                ],
            }


def _calls(count: int, endpoints: tuple[Endpoint, ...]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=index % 2 == 1, marker=uuid.uuid4().hex)
        for index in range(count)
    )


async def _send(client: httpx.AsyncClient, key: str, models: _Models, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(models, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode(), response.headers.get("x-litellm-call-id", ""))


async def _burst(
    base_url: str, key: str, models: _Models, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, models, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _frames(text: str) -> list[dict[str, JsonValue]]:
    return [rv.JSON_OBJECT.validate_json(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


def _response_id(served: _Served) -> str:
    if not served.call.stream:
        return str(rv.JSON_OBJECT.validate_json(served.text)["id"])
    frames: Final = _frames(served.text)
    match served.call.endpoint:
        case "responses":
            (completed,) = [frame for frame in frames if frame.get("type") == "response.completed"]
            return str(rv.JSON_OBJECT.validate_python(completed["response"])["id"])
        case "chat":
            return str(frames[0]["id"])
        case "messages":
            (start,) = [frame for frame in frames if frame.get("type") == "message_start"]
            return str(rv.JSON_OBJECT.validate_python(start["message"])["id"])


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(rv.MARKER.findall(served.text)) == {served.call.marker}, served.text


def _assert_forwarded_without_a_minted_item(request: Request, marker: str) -> None:
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    path: Final = urlsplit(request.target).path
    assert "no-cache" not in request.body.decode(), request.body
    if path.endswith("/messages"):
        (assistant,) = [turn for turn in rv.ITEMS.validate_python(body["messages"]) if turn["role"] == "assistant"]
        assert assistant["content"] == [
            {"type": "thinking", "thinking": rv.THOUGHT, "signature": rv.signature(marker)},
            {"type": "text", "text": "Prague"},
        ], assistant
        return
    assert path.endswith("/responses"), request.target
    items: Final = rv.reasoning_items(body)
    if body["model"] == _CODEX:
        assert items == [{"type": "reasoning", "encrypted_content": f"gAAAAA-stored-{marker}", "summary": []}], items
        return
    assert items == [], body["input"]


def _spend_rows(models: _Models, expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group IN (%s, %s, %s)',
            (models.responses, models.chat, models.messages),
        ),
        lambda found: len(found) >= expected,
        seconds=70,
    )


def _assert_each_lands_once(
    rows: list[dict[str, JsonValue]], failed: tuple[_Served, ...], served: tuple[_Served, ...]
) -> None:
    by_status: Final = {str(row["request_id"]): str(row["status"]) for row in rows}
    assert len(by_status) == len(rows) == len(failed) + len(served), rows
    for item in failed:
        assert by_status.get(item.call_id) == "failure", (item.call_id, rows)
    for item in served:
        (match,) = [request_id for request_id in by_status if rv.same_response(request_id, _response_id(item))]
        assert by_status[match] == "success", rows


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _health_counts(gateway: Gateway, model: str) -> tuple[int, int]:
    response: Final = gateway.request("GET", f"/health?model={model}", None)
    assert response.status_code in (200, 503), response.text
    health: Final = rv.JSON_OBJECT.validate_json(response.text)
    return int(str(health["healthy_count"])), int(str(health["unhealthy_count"]))


def _marked(received: tuple[Request, ...]) -> dict[str, Request]:
    marked: Final = {marker: request for request in received if (marker := rv.newest_marker(request.body.decode()))}
    assert len(marked) == sum(1 for request in received if rv.newest_marker(request.body.decode())), received
    return marked


@pytest.mark.timeout(180)
async def test_vendor_outage_fails_each_replay_cleanly_and_the_recovered_vendor_gets_them_without_minted_items(
    gateway: Gateway,
) -> None:
    port: Final = _free_port()
    while_down: Final = _calls(15, ("responses", "chat", "messages"))
    after: Final = _calls(15, ("responses", "chat", "messages"))
    with gateway.scenario() as scenario:
        models: Final = _register(scenario, f"http://127.0.0.1:{port}")
        failed: Final = await _burst(str(gateway.client.base_url), gateway.key, models, while_down)
        assert len(failed) == 15
        for item in failed:
            assert item.status == 500 and "Cannot connect to host" in item.text, (item.status, item.text)
            assert "answer marker" not in item.text, item.text
            assert item.call_id, item
        assert _health_counts(gateway, models.responses) == (0, 1)
        with wire_server(rv.ResponsesVendor().respond, port=port) as wire:
            assert _health_counts(gateway, models.responses) == (1, 0)
            wire.drain()
            served: Final = await _burst(str(gateway.client.base_url), gateway.key, models, after)
            assert len(served) == 15
            for item in served:
                _assert_answered_with_its_own_marker(item)
            forwarded: Final = _marked(wire.drain())
            assert set(forwarded) == {call.marker for call in after}, sorted(forwarded)
            for marker, request in forwarded.items():
                _assert_forwarded_without_a_minted_item(request, marker)
        _assert_each_lands_once(_spend_rows(models, 30), failed, served)


async def test_slow_vendor_streams_are_each_forwarded_once_without_the_minted_item(gateway: Gateway) -> None:
    calls: Final = tuple(_Call("responses", True, uuid.uuid4().hex) for _ in range(10))
    with wire_server(rv.ResponsesVendor(pause_between_chunks=0.3).respond) as wire, gateway.scenario() as scenario:
        models: Final = _register(scenario, wire.url)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, models, calls)
        assert len(served) == 10
        for item in served:
            _assert_answered_with_its_own_marker(item)
            assert "response.completed" in item.text, item.text
        received: Final = wire.drain()
        assert len(received) == 10, [request.target for request in received]
        forwarded: Final = _marked(received)
        assert set(forwarded) == {call.marker for call in calls}
        for marker, request in forwarded.items():
            _assert_forwarded_without_a_minted_item(request, marker)
        _assert_each_lands_once(_spend_rows(models, 10), (), served)


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {"model": f"openai/{_GPT}", "api_base": wire.url, "api_key": _OPENAI_KEY},
        }
    ]
    path: Final = tmp_path / "responses-minted-reasoning-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(240)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_dropping_the_minted_item(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = tuple(_Call("responses", False, uuid.uuid4().hex) for _ in range(20))
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    vendor: Final = rv.ResponsesVendor()

    def held(request: Request) -> Reply:
        if request.method == "GET":
            return vendor.respond(request)
        marker: Final = rv.newest_marker(request.body.decode())
        assert marker is not None, request.body
        held_markers.put(marker)
        assert release.wait(timeout=60), "The burst was never released"
        return vendor.respond(request)

    with wire_server(held) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            models: Final = _Models(_CONFIG_MODEL, _CONFIG_MODEL, _CONFIG_MODEL)
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(str(candidate.client.base_url), candidate.key, models, calls, tolerate_transport_errors=True)
            )
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
                _assert_answered_with_its_own_marker(item)
            follow_up: Final = _Call("responses", False, uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, models, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            forwarded: Final = _marked(tuple(request for request in wire.drain() if request.method == "POST"))
            assert set(forwarded) == {call.marker for call in (*calls, follow_up)}, sorted(forwarded)
            for marker, request in forwarded.items():
                _assert_forwarded_without_a_minted_item(request, marker)


@dataclass(frozen=True, slots=True)
class _Rig:
    wire: Wire
    proxy: OwnedProxy
    cert: Path
    key: Path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("minted-reasoning-rig")
    cert, key = write_self_signed_cert(directory)
    copilot: Final = directory / "copilot"
    chatgpt: Final = directory / "chatgpt"
    copilot.mkdir()
    chatgpt.mkdir()
    with gateway_from_environment() as gateway, wire_server(rv.ResponsesVendor().respond) as wire:
        (copilot / "api-key.json").write_text(
            json.dumps(
                {"token": "synthetic-copilot-token", "expires_at": time.time() + 3600, "endpoints": {"api": wire.url}}
            )
        )
        (chatgpt / "auth.json").write_text(
            json.dumps(
                {
                    "access_token": "synthetic-chatgpt-token",
                    "account_id": "acct-synthetic",
                    "expires_at": time.time() + 3600,
                }
            )
        )
        overrides: Final = {
            "GITHUB_COPILOT_TOKEN_DIR": str(copilot),
            "CHATGPT_TOKEN_DIR": str(chatgpt),
            "CHATGPT_API_BASE": wire.url,
            "SSL_CERT_FILE": str(cert),
            "HTTP_PROXY": wire.url,
            "NO_PROXY": "127.0.0.1,localhost",
        }
        with owned_proxy_process(gateway, directory, overrides, workers=2) as owned:
            yield _Rig(wire, owned, cert, key)


def _replay(gateway: Gateway, model: str, history: list[dict[str, JsonValue]], stream: bool) -> httpx.Response:
    return gateway.request("POST", "/v1/responses", {"model": model, "input": history, "stream": stream, **_CACHE_BUST})


@dataclass(frozen=True, slots=True)
class _LoginDeployment:
    label: str
    model: str
    api_key: str | None


_LOGIN_DEPLOYMENTS: Final = (
    _LoginDeployment("github_copilot", f"github_copilot/{_CODEX}", None),
    _LoginDeployment("chatgpt", f"chatgpt/{_CODEX}", None),
    _LoginDeployment("azure_ai-foundry-host", "azure_ai/deepseek-v3", "synthetic-azure-key"),
)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("deployment", _LOGIN_DEPLOYMENTS, ids=[deployment.label for deployment in _LOGIN_DEPLOYMENTS])
def test_login_backed_and_foundry_deployments_forward_the_minted_item_unchanged(
    rig: _Rig, deployment: _LoginDeployment, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    minted: Final = rv.minted_item(marker, summary=[])
    history: Final = rv.agents_sdk_history(marker, minted)
    api_base: Final = _FOUNDRY_BASE if deployment.label.startswith("azure_ai") else rig.wire.url
    rig.wire.drain()
    with rig.proxy.gateway.scenario() as scenario:
        parameters: Final[dict[str, JsonValue]] = {"model": deployment.model, "api_base": api_base}
        model: Final = scenario.model(
            **parameters, **({} if deployment.api_key is None else {"api_key": deployment.api_key})
        )
        response: Final = _replay(rig.proxy.gateway, model, history, stream)
        received: Final = rig.wire.drain()
        assert len(received) == 1, [(request.method, request.target) for request in received]
        target: Final = urlsplit(received[0].target)
        assert target.path.endswith("/responses"), received[0].target
        if deployment.label.startswith("azure_ai"):
            assert target.scheme == "http" and target.netloc == urlsplit(_FOUNDRY_BASE).netloc, received[0].target
        items: Final = rv.reasoning_items(rv.JSON_OBJECT.validate_json(received[0].body))
        assert items == [minted], items
        assert response.status_code == 404, response.text
        assert f"Item with id '{minted['id']}' not found" in response.text, response.text


@pytest.mark.timeout(240)
async def test_websocket_session_forwards_the_minted_item_as_before(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    minted: Final = rv.minted_item(marker)
    history: Final = rv.agents_sdk_history(marker, minted)
    frames: Final[SimpleQueue[tuple[str, str]]] = SimpleQueue()

    async def vendor(connection: websockets.ServerConnection) -> None:
        first: Final = await connection.recv()
        frames.put((str(connection.request.path), str(first)))
        tag: Final = uuid.uuid4().hex
        response: Final[dict[str, JsonValue]] = {
            "id": f"resp_{tag}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": _GPT,
            "output": [
                {
                    "id": f"msg_{tag}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": rv.answer(marker), "annotations": []}],
                }
            ],
            "usage": rv.USAGE,
        }
        created: Final = {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        }
        await connection.send(json.dumps(created))
        await connection.send(json.dumps({"type": "response.completed", "sequence_number": 1, "response": response}))
        await connection.wait_closed()

    gateway: Final = rig.proxy.gateway
    async with websockets.serve(vendor, "127.0.0.1", 0, ssl=server_context(rig.cert, rig.key)) as server:
        port: Final = server.sockets[0].getsockname()[1]
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model=f"openai/{_GPT}", api_base=f"https://127.0.0.1:{port}", api_key=_OPENAI_KEY
            )
            session_url: Final = (
                f"{str(gateway.client.base_url).rstrip('/').replace('http://', 'ws://')}/v1/responses?model={model}"
            )
            async with websockets.connect(
                session_url, additional_headers={"Authorization": f"Bearer {gateway.key}"}
            ) as session:
                await session.send(json.dumps({"type": "response.create", "model": model, "input": history}))
                received: Final[list[dict[str, JsonValue]]] = []
                while not received or received[-1].get("type") != "response.completed":
                    received.append(rv.JSON_OBJECT.validate_json(str(await session.recv())))
            assert [event["type"] for event in received] == ["response.created", "response.completed"], received
            completed: Final = rv.JSON_OBJECT.validate_python(received[-1]["response"])
            (message,) = rv.ITEMS.validate_python(completed["output"])
            assert rv.ITEMS.validate_python(message["content"])[0]["text"] == rv.answer(marker), message
    assert frames.qsize() == 1
    path, first = frames.get_nowait()
    assert path.startswith("/responses?") and f"model={_GPT}" in path, path
    assert rv.JSON_OBJECT.validate_json(first)["input"] == history, first
