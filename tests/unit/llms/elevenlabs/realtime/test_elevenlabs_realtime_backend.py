import asyncio
from collections.abc import Mapping
from types import TracebackType
from typing import Final

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidHandshake, InvalidStatus
from websockets.http11 import Response

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
    def __init__(
        self,
        upstream: FakeUpstream,
        gate: asyncio.Event | None = None,
        failure: OSError | TimeoutError | InvalidHandshake | None = None,
    ) -> None:
        self.upstream = upstream
        self.gate = gate
        self.failure = failure
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    async def __call__(self, url: str, headers: Mapping[str, str]) -> FakeUpstream:
        self.calls.append((url, headers))
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        return self.upstream


async def test_configure_opens_one_upstream_connection_with_the_query_and_auth_header():
    upstream: Final = FakeUpstream()
    connect: Final = RecordingConnect(upstream)
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)

    assert connect.calls == []
    await backend.send(CONFIGURE)
    await backend.send('{"message_type":"input_audio_chunk"}')

    assert connect.calls == [(f"{URL}?model_id=scribe_v2_realtime&audio_format=pcm_24000", HEADERS)]
    assert upstream.sent == ['{"message_type":"input_audio_chunk"}']


async def test_audio_before_configure_is_rejected_without_connecting():
    connect: Final = RecordingConnect(FakeUpstream())
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)

    with pytest.raises(RuntimeError, match="before the ElevenLabs realtime session was configured"):
        await backend.send('{"message_type":"input_audio_chunk"}')
    assert connect.calls == []


async def test_recv_waits_for_the_upstream_connection():
    upstream: Final = FakeUpstream()
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=RecordingConnect(upstream))
    pending: Final = asyncio.create_task(backend.recv())
    await asyncio.sleep(0)

    assert not pending.done()
    await backend.send(CONFIGURE)
    upstream.incoming.put_nowait('{"message_type":"session_started"}')
    assert await asyncio.wait_for(pending, timeout=1) == '{"message_type":"session_started"}'


async def test_close_before_configure_ends_recv_and_prevents_connecting():
    connect: Final = RecordingConnect(FakeUpstream())
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)
    pending: Final = asyncio.create_task(backend.recv())
    await asyncio.sleep(0)

    await backend.close()

    with pytest.raises(ConnectionClosedOK):
        await asyncio.wait_for(pending, timeout=1)
    with pytest.raises(ConnectionClosedOK):
        await backend.send(CONFIGURE)
    assert connect.calls == []


async def test_close_after_configure_closes_the_upstream_connection():
    upstream: Final = FakeUpstream()
    async with ElevenLabsRealtimeBackend(URL, HEADERS, connect=RecordingConnect(upstream)) as backend:
        await backend.send(CONFIGURE)

    assert upstream.closed


async def test_close_during_connect_closes_the_late_upstream_connection():
    upstream: Final = FakeUpstream()
    gate: Final = asyncio.Event()
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=RecordingConnect(upstream, gate))
    configuring: Final = asyncio.create_task(backend.send(CONFIGURE))
    await asyncio.sleep(0)

    await backend.close()
    gate.set()

    with pytest.raises(ConnectionClosedOK):
        await asyncio.wait_for(configuring, timeout=1)
    assert upstream.closed


@pytest.mark.parametrize("message", [CONFIGURE, '{"message_type":"input_audio_chunk"}'])
async def test_send_during_connect_does_not_open_a_second_connection(message: str):
    gate: Final = asyncio.Event()
    connect: Final = RecordingConnect(FakeUpstream(), gate)
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)
    configuring: Final = asyncio.create_task(backend.send(CONFIGURE))
    await asyncio.sleep(0)

    try:
        with pytest.raises(RuntimeError, match="the ElevenLabs realtime session is already being configured"):
            await asyncio.wait_for(backend.send(message), timeout=1)
    finally:
        gate.set()
        await asyncio.wait_for(configuring, timeout=1)

    assert len(connect.calls) == 1


async def test_cancelled_connect_allows_a_new_configure_attempt():
    upstream: Final = FakeUpstream()
    gate: Final = asyncio.Event()
    connect: Final = RecordingConnect(upstream, gate)
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)
    configuring: Final = asyncio.create_task(backend.send(CONFIGURE))
    await asyncio.sleep(0)

    configuring.cancel()
    with pytest.raises(asyncio.CancelledError):
        await configuring

    new_gate: Final = asyncio.Event()
    connect.gate = new_gate
    retry: Final = asyncio.create_task(backend.send(CONFIGURE))
    await asyncio.sleep(0)
    try:
        assert len(connect.calls) == 2
    finally:
        new_gate.set()
        await asyncio.wait_for(retry, timeout=1)

    await backend.send('{"message_type":"input_audio_chunk"}')
    assert upstream.sent == ['{"message_type":"input_audio_chunk"}']


@pytest.mark.parametrize(
    "failure", [OSError("network failure"), TimeoutError("connect timeout"), asyncio.TimeoutError("handshake timeout")]
)
async def test_connection_failure_wakes_recv_and_rejects_later_sends(
    failure: OSError | TimeoutError | asyncio.TimeoutError,
):
    connect: Final = RecordingConnect(FakeUpstream(), failure=failure)
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=connect)
    pending: Final = asyncio.create_task(backend.recv())
    await asyncio.sleep(0)

    try:
        await backend.send(CONFIGURE)
        with pytest.raises(ConnectionClosedError) as recv_error:
            await asyncio.wait_for(pending, timeout=1)
        assert recv_error.value.rcvd is not None
        assert recv_error.value.rcvd.code == 1011
        assert "ElevenLabs realtime connection failed" in recv_error.value.rcvd.reason
        with pytest.raises(ConnectionClosedError) as send_error:
            await backend.send(CONFIGURE)
        assert send_error.value.rcvd == recv_error.value.rcvd
        assert len(connect.calls) == 1
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_handshake_failure_includes_http_status_in_close_reason():
    failure: Final = InvalidStatus(Response(401, "Unauthorized", Headers()))
    backend: Final = ElevenLabsRealtimeBackend(URL, HEADERS, connect=RecordingConnect(FakeUpstream(), failure=failure))

    await backend.send(CONFIGURE)

    with pytest.raises(ConnectionClosedError) as caught:
        await asyncio.wait_for(backend.recv(), timeout=1)
    assert caught.value.rcvd is not None
    assert caught.value.rcvd.code == 1011
    assert caught.value.rcvd.reason == "ElevenLabs realtime connection failed (HTTP 401)"


async def test_close_after_connection_failure_does_not_raise():
    backend: Final = ElevenLabsRealtimeBackend(
        URL, HEADERS, connect=RecordingConnect(FakeUpstream(), failure=OSError("network failure"))
    )

    await backend.send(CONFIGURE)
    await backend.close()

    with pytest.raises(ConnectionClosedError) as closed:
        await backend.recv()
    assert closed.value.rcvd is not None
    assert closed.value.rcvd.code == 1011
