import asyncio
import hashlib
import itertools
import json
import ssl
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final, TypeAlias

import pytest
import websockets
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_stream import RATE_LIMIT_MESSAGE, openai_error_event
from integration._support.tls import server_context, write_self_signed_cert
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection, serve

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

PROVIDER_MODEL: Final = "ws-peer-model"
USAGE: Final = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
TERMINAL: Final = frozenset({"response.completed", "response.failed", "error"})
PROXY_OPENAI_KEY: Final = "sk-proxy-environment-openai-key"
PROXY_AZURE_KEY: Final = "proxy-environment-azure-key"

_Script: TypeAlias = Callable[[str], tuple[dict[str, JsonValue], ...]]


@dataclass(frozen=True, slots=True)
class Credentials:
    authorization: str | None
    api_key: str | None


@dataclass(frozen=True, slots=True)
class Peer:
    url: str
    paths: SimpleQueue[str]
    credentials: SimpleQueue[Credentials]
    frames: SimpleQueue[dict[str, JsonValue]]


def _events(response_id: str, text: str) -> tuple[dict[str, JsonValue], ...]:
    message: Final = {
        "type": "message",
        "id": f"msg_{response_id}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    response: Final = {"id": response_id, "object": "response", "created_at": 1700000000, "model": PROVIDER_MODEL}
    return (
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        {
            "type": "response.output_text.delta",
            "item_id": f"msg_{response_id}",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {
            "type": "response.completed",
            "response": {**response, "status": "completed", "output": [message], "usage": USAGE},
        },
    )


def _seven(response_id: str) -> tuple[dict[str, JsonValue], ...]:
    return _events(response_id, "seven")


def _top_level_rate_limit(response_id: str) -> tuple[dict[str, JsonValue], ...]:
    created: Final = _events(response_id, "")[0]
    return (created, dict(openai_error_event("rate_limit_exceeded", RATE_LIMIT_MESSAGE)))


async def _answer(
    connection: ServerConnection,
    paths: SimpleQueue[str],
    credentials: SimpleQueue[Credentials],
    frames: SimpleQueue[dict[str, JsonValue]],
    script: _Script,
) -> None:
    request: Final = connection.request
    paths.put(request.path if request is not None else "")
    credentials.put(
        Credentials(request.headers.get("Authorization"), request.headers.get("api-key"))
        if request is not None
        else Credentials(None, None)
    )
    turns: Final = itertools.count(1)
    async for raw in connection:
        frame: Final = json.loads(raw)
        frames.put(frame)
        if frame.get("type") != "response.create":
            continue
        for event in script(f"resp_peer_{next(turns)}"):
            await connection.send(json.dumps(event))


async def _serve(
    tls: ssl.SSLContext,
    paths: SimpleQueue[str],
    credentials: SimpleQueue[Credentials],
    frames: SimpleQueue[dict[str, JsonValue]],
    ports: SimpleQueue[int],
    stop: asyncio.Event,
    script: _Script,
) -> None:
    async with serve(
        lambda connection: _answer(connection, paths, credentials, frames, script), "127.0.0.1", 0, ssl=tls
    ) as server:
        ports.put(next(iter(server.sockets)).getsockname()[1])
        await stop.wait()


@contextmanager
def responses_peer(cert: tuple[Path, Path], script: _Script = _seven) -> Iterator[Peer]:
    loop: Final = asyncio.new_event_loop()
    stop: Final = asyncio.Event()
    paths: Final = SimpleQueue[str]()
    credentials: Final = SimpleQueue[Credentials]()
    frames: Final = SimpleQueue[dict[str, JsonValue]]()
    ports: Final = SimpleQueue[int]()
    thread: Final = threading.Thread(
        target=loop.run_until_complete,
        args=(_serve(server_context(*cert), paths, credentials, frames, ports, stop, script),),
        daemon=True,
    )
    thread.start()
    try:
        yield Peer(f"https://127.0.0.1:{ports.get(timeout=10)}/v1", paths, credentials, frames)
    finally:
        loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=10)
        loop.close()


def _create(model: str, text: str, previous_response_id: str | None = None) -> str:
    return json.dumps(
        {
            "type": "response.create",
            "model": model,
            "store": True,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}],
            **({} if previous_response_id is None else {"previous_response_id": previous_response_id}),
        }
    )


async def _turn(connection: websockets.ClientConnection, frame: str) -> tuple[dict[str, JsonValue], ...]:
    await connection.send(frame)
    return await _until_terminal(connection, ())


async def _until_terminal(
    connection: websockets.ClientConnection, received: tuple[dict[str, JsonValue], ...]
) -> tuple[dict[str, JsonValue], ...]:
    event: Final = json.loads(await asyncio.wait_for(connection.recv(), timeout=20))
    collected: Final = (*received, event)
    if event.get("type") in TERMINAL or len(collected) >= 50:
        return collected
    return await _until_terminal(connection, collected)


async def _session(
    proxy_url: str, key: str, model: str, texts: tuple[str, ...]
) -> tuple[tuple[dict[str, JsonValue], ...], ...]:
    proxy: Final = proxy_url.rstrip("/").replace("http://", "ws://")
    async with websockets.connect(
        f"{proxy}/v1/responses?model={model}",
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=10,
    ) as connection:
        first: Final = await _turn(connection, _create(model, texts[0]))
        if len(texts) == 1:
            return (first,)
        previous: Final = str(first[-1]["response"]["id"])
        second: Final = await _turn(connection, _create(model, texts[1], previous))
        return (first, second)


def _completed(events: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    assert events[-1]["type"] == "response.completed", [event.get("type") for event in events]
    return events[-1]["response"]


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("responses-ws-cert"))


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        with owned_proxy(
            base,
            tmp_path_factory.mktemp("responses-ws"),
            {"SSL_CERT_FILE": str(cert[0]), "OPENAI_API_KEY": PROXY_OPENAI_KEY, "AZURE_API_KEY": PROXY_AZURE_KEY},
        ) as proxy:
            yield proxy


def _drain(queue: SimpleQueue[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(queue.get_nowait() for _ in range(queue.qsize()))


def test_a_response_create_frame_streams_from_the_provider_socket_back_to_the_client(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
        key: Final = scenario.key(models=[model])
        text: Final = f"say seven {uuid.uuid4().hex}"
        (events,) = asyncio.run(_session(str(candidate.client.base_url), key, model, (text,)))
        assert [event["type"] for event in events] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ]
        completed: Final = _completed(events)
        assert completed["status"] == "completed"
        assert completed["usage"] == USAGE
        assert peer.paths.get_nowait() == f"/v1/responses?model={PROVIDER_MODEL}"
        (forwarded,) = _drain(peer.frames)
        assert forwarded["type"] == "response.create"
        assert forwarded["model"] == PROVIDER_MODEL
        assert forwarded["input"][0]["content"][0]["text"] == text


def test_previous_response_id_from_the_first_turn_reaches_the_provider_as_its_own_id(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
        key: Final = scenario.key(models=[model])
        texts: Final = (f"remember seven {uuid.uuid4().hex}", f"which number {uuid.uuid4().hex}")
        first, second = asyncio.run(_session(str(candidate.client.base_url), key, model, texts))
        assert _completed(first)["status"] == "completed"
        assert str(_completed(first)["id"]).startswith("resp_") and _completed(first)["id"] != "resp_peer_1"
        assert _completed(second)["status"] == "completed"
        assert peer.paths.qsize() == 1
        forwarded: Final = _drain(peer.frames)
        assert [frame["input"][0]["content"][0]["text"] for frame in forwarded] == list(texts)
        assert "previous_response_id" not in forwarded[0]
        assert forwarded[1]["previous_response_id"] == "resp_peer_1"


def test_a_top_level_error_frame_reaches_the_client_and_logs_the_provider_code_and_message(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert, _top_level_rate_limit) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
        key: Final = scenario.key(models=[model])
        text: Final = f"say seven {uuid.uuid4().hex}"
        (events,) = asyncio.run(_session(str(candidate.client.base_url), key, model, (text,)))
        assert [event["type"] for event in events] == ["response.created", "error"], events
        assert (events[-1]["code"], events[-1]["message"]) == ("rate_limit_exceeded", RATE_LIMIT_MESSAGE), events
        (forwarded,) = _drain(peer.frames)
        assert forwarded["input"][0]["content"][0]["text"] == text
        (row,) = eventually(
            lambda: read_rows(
                'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (hashlib.sha256(key.encode()).hexdigest(),),
            ),
            lambda rows: len(rows) >= 1,
            seconds=70,
        )
        assert row["status"] == "failure", row
        error: Final = object_value(object_value(row["metadata"])["error_information"])
        assert str(error["error_code"]) == "429", error
        assert RATE_LIMIT_MESSAGE in string_value(error["error_message"]), error


def _one_turn(candidate: Gateway, key: str, model: str) -> tuple[dict[str, JsonValue], ...]:
    (events,) = asyncio.run(_session(str(candidate.client.base_url), key, model, (f"say seven {uuid.uuid4().hex}",)))
    return events


def test_a_bare_model_declared_as_openai_sends_its_deployment_key_to_the_provider_socket(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    deployment_key: Final = f"sk-deployment-{uuid.uuid4().hex}"
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(
            model=PROVIDER_MODEL, custom_llm_provider="openai", api_base=peer.url, api_key=deployment_key
        )
        events: Final = _one_turn(candidate, scenario.key(models=[model]), model)
        assert _completed(events)["status"] == "completed"
        assert peer.credentials.get_nowait().authorization == f"Bearer {deployment_key}"


def test_an_openai_deployment_without_its_own_key_uses_the_proxy_openai_key(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url, api_key=None)
        events: Final = _one_turn(candidate, scenario.key(models=[model]), model)
        assert _completed(events)["status"] == "completed"
        assert peer.credentials.get_nowait().authorization == f"Bearer {PROXY_OPENAI_KEY}"


def test_an_azure_deployment_without_its_own_key_gets_the_azure_key_and_never_the_openai_key(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{PROVIDER_MODEL}", api_base=peer.url.removesuffix("/v1"), api_key=None
        )
        events: Final = _one_turn(candidate, scenario.key(models=[model]), model)
        assert _completed(events)["status"] == "completed"
        credentials: Final = peer.credentials.get_nowait()
        assert credentials.api_key == PROXY_AZURE_KEY
        assert PROXY_OPENAI_KEY not in (credentials.authorization or "")


def test_a_declared_openai_provider_wins_over_another_providers_model_prefix(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    gateway_model: Final = f"anthropic/{PROVIDER_MODEL}"
    with responses_peer(cert) as peer, candidate.scenario() as scenario:
        model: Final = scenario.model(model=gateway_model, custom_llm_provider="openai", api_base=peer.url)
        key: Final = scenario.key(models=[model])
        events: Final = _one_turn(candidate, key, model)
        assert _completed(events)["status"] == "completed"
        (forwarded,) = _drain(peer.frames)
        assert forwarded["model"] == gateway_model
        (row,) = eventually(
            lambda: read_rows(
                'SELECT custom_llm_provider FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (hashlib.sha256(key.encode()).hexdigest(),),
            ),
            lambda rows: len(rows) >= 1,
            seconds=70,
        )
        assert row["custom_llm_provider"] == "openai", row
