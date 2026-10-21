"""A scripted provider-side Responses WebSocket peer.

Answers every ``response.create`` frame with the events a script returns for that turn over
``wss://`` (the proxy only dials TLS provider sockets). Clients drive it through ``session``.
"""

import asyncio
import itertools
import json
import ssl
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final, TypeAlias

import websockets
from integration._support.tls import server_context
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection, serve

PROVIDER_MODEL: Final = "ws-peer-model"
USAGE: Final = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
TERMINAL: Final = frozenset({"response.completed", "response.failed", "error"})

Script: TypeAlias = Callable[[str], tuple[dict[str, JsonValue], ...]]


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


def events(response_id: str, text: str) -> tuple[dict[str, JsonValue], ...]:
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


def seven(response_id: str) -> tuple[dict[str, JsonValue], ...]:
    return events(response_id, "seven")


async def _answer(
    connection: ServerConnection,
    paths: SimpleQueue[str],
    credentials: SimpleQueue[Credentials],
    frames: SimpleQueue[dict[str, JsonValue]],
    script: Script,
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
    script: Script,
) -> None:
    async with serve(
        lambda connection: _answer(connection, paths, credentials, frames, script), "127.0.0.1", 0, ssl=tls
    ) as server:
        ports.put(next(iter(server.sockets)).getsockname()[1])
        await stop.wait()


@contextmanager
def responses_peer(cert: tuple[Path, Path], script: Script = seven) -> Iterator[Peer]:
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


def create(model: str, text: str, previous_response_id: str | None = None) -> str:
    return json.dumps(
        {
            "type": "response.create",
            "model": model,
            "store": True,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}],
            **({} if previous_response_id is None else {"previous_response_id": previous_response_id}),
        }
    )


async def turn(connection: websockets.ClientConnection, frame: str) -> tuple[dict[str, JsonValue], ...]:
    await connection.send(frame)
    return await until_terminal(connection, ())


async def until_terminal(
    connection: websockets.ClientConnection, received: tuple[dict[str, JsonValue], ...]
) -> tuple[dict[str, JsonValue], ...]:
    event: Final = json.loads(await asyncio.wait_for(connection.recv(), timeout=20))
    collected: Final = (*received, event)
    if event.get("type") in TERMINAL or len(collected) >= 50:
        return collected
    return await until_terminal(connection, collected)


async def session(
    proxy_url: str, key: str, model: str, texts: tuple[str, ...]
) -> tuple[tuple[dict[str, JsonValue], ...], ...]:
    proxy: Final = proxy_url.rstrip("/").replace("http://", "ws://")
    async with websockets.connect(
        f"{proxy}/v1/responses?model={model}",
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=10,
    ) as connection:
        first: Final = await turn(connection, create(model, texts[0]))
        if len(texts) == 1:
            return (first,)
        previous: Final = str(first[-1]["response"]["id"])
        second: Final = await turn(connection, create(model, texts[1], previous))
        return (first, second)


def completed(received: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    assert received[-1]["type"] == "response.completed", [event.get("type") for event in received]
    return received[-1]["response"]


def drain(queue: SimpleQueue[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(queue.get_nowait() for _ in range(queue.qsize()))
