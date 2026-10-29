import asyncio
from collections.abc import Mapping
from types import TracebackType
from typing import Final

import pytest
from websockets.exceptions import ConnectionClosedOK

from litellm.llms.elevenlabs.realtime.backend import ElevenLabsRealtimeBackend
from litellm.types.llms.elevenlabs import ElevenLabsRealtimeConfigure

URL: Final = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
HEADERS: Final = {"xi-api-key": "secret"}
CONFIGURE: Final = ElevenLabsRealtimeConfigure(query="model_id=scribe_v2_realtime&audio_format=pcm_24000").model_dump_json()


class FakeUpstream:
    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self.incoming: asyncio.Queue[str | bytes] = asyncio.Queue()
        self.closed = False

    async def __aenter__(self) -> "FakeUpstream":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self, decode: bool | None = None) -> str | bytes:
        return await self.incoming.get()

    async def close(self) -> None:
        self.closed = True


class RecordingConnect:
    def __init__(self, upstream: FakeUpstream, gate: asyncio.Event | None = None) -> None:
        self.upstream = upstream
        self.gate = gate
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    async def __call__(self, url: str, headers: Mapping[str, str]) -> FakeUpstream:
        self.calls.append((url, headers))
        if self.gate is not None:
            await self.gate.wait()
        return self.upstream


async def test_configure_opens_one_upstream_connection_with_the_query_and_auth_header():
    upstream = FakeUpstream()
    connect = RecordingConnect(upstream)
    backend = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)

    assert connect.calls == []
    await backend.send(CONFIGURE)
    await backend.send('{"message_type":"input_audio_chunk"}')

    assert connect.calls == [(f"{URL}?model_id=scribe_v2_realtime&audio_format=pcm_24000", HEADERS)]
    assert upstream.sent == ['{"message_type":"input_audio_chunk"}']


async def test_audio_before_configure_is_rejected_without_connecting():
    connect = RecordingConnect(FakeUpstream())
    backend = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)

    with pytest.raises(RuntimeError, match="before the ElevenLabs realtime session was configured"):
        await backend.send('{"message_type":"input_audio_chunk"}')
    assert connect.calls == []


async def test_recv_waits_for_the_upstream_connection():
    upstream = FakeUpstream()
    backend = ElevenLabsRealtimeBackend(URL, HEADERS, connect=RecordingConnect(upstream))
    pending = asyncio.create_task(backend.recv())
    await asyncio.sleep(0)

    assert not pending.done()
    await backend.send(CONFIGURE)
    upstream.incoming.put_nowait('{"message_type":"session_started"}')
    assert await asyncio.wait_for(pending, timeout=1) == '{"message_type":"session_started"}'


async def test_close_before_configure_ends_recv_and_prevents_connecting():
    connect = RecordingConnect(FakeUpstream())
    backend = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)
    pending = asyncio.create_task(backend.recv())
    await asyncio.sleep(0)

    await backend.close()

    with pytest.raises(ConnectionClosedOK):
        await asyncio.wait_for(pending, timeout=1)
    with pytest.raises(ConnectionClosedOK):
        await backend.send(CONFIGURE)
    assert connect.calls == []


async def test_close_after_configure_closes_the_upstream_connection():
    upstream = FakeUpstream()
    async with ElevenLabsRealtimeBackend(URL, HEADERS, connect=RecordingConnect(upstream)) as backend:
        await backend.send(CONFIGURE)

    assert upstream.closed


async def test_close_during_connect_closes_the_late_upstream_connection():
    upstream = FakeUpstream()
    gate = asyncio.Event()
    backend = ElevenLabsRealtimeBackend(URL, HEADERS, connect=RecordingConnect(upstream, gate))
    configuring = asyncio.create_task(backend.send(CONFIGURE))
    await asyncio.sleep(0)

    await backend.close()
    gate.set()

    with pytest.raises(ConnectionClosedOK):
        await asyncio.wait_for(configuring, timeout=1)
    assert upstream.closed
