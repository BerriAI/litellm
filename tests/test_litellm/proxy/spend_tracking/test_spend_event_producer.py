import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Final

import pytest
import uvloop

from litellm.proxy.spend_tracking.spend_event_producer import (
    AddressError,
    CollectorAddress,
    CollectorSettings,
    SpendEventProducer,
    TcpAddress,
    UnixAddress,
    build_spend_event_producer,
    open_collector_connection,
    parse_collector_address,
)


class _Sidecar:
    """A unix-socket server that records every line it receives, standing in for the collector."""

    def __init__(self, path: Path, reads: bool = True, limit: int = 2**16) -> None:
        self.path = path
        self.reads = reads
        self.limit = limit
        self.lines: list[bytes] = []  # mutable-ok: test double records what the producer sent
        self._server: asyncio.Server | None = None
        self._stopped = asyncio.Event()
        self._connections: list[asyncio.StreamWriter] = []  # mutable-ok: test double tracks peers to hang up on

    async def __aenter__(self) -> "_Sidecar":
        self._server = await asyncio.start_unix_server(self._on_connection, path=str(self.path), limit=self.limit)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stopped.set()
        await self.hang_up()

    async def hang_up(self) -> None:
        """Exit the way a stopped sidecar does: stop listening and close every producer connection."""
        assert self._server is not None
        self._server.close()
        for connection in self._connections:
            connection.close()
            await connection.wait_closed()
        await self._server.wait_closed()

    async def _on_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._connections.append(writer)
        if not self.reads:
            await self._stopped.wait()
            return
        while line := await reader.readline():
            self.lines.append(line)
        writer.close()


class _CrashingSidecar(_Sidecar):
    """Bills a few lines, then dies mid-stream with the producer's backlog still queued behind them."""

    def __init__(self, path: Path, lines_before_crash: int) -> None:
        super().__init__(path, limit=2**20)
        self._lines_before_crash = lines_before_crash

    async def _on_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._connections.append(writer)
        for _ in range(self._lines_before_crash):
            self.lines.append(await reader.readline())
        writer.transport.abort()


class _Fallback:
    def __init__(self) -> None:
        self.lines: list[bytes] = []  # mutable-ok: test double records what fell back to in-process

    async def __call__(self, line: bytes) -> None:
        self.lines.append(line)


class _GatedFallback(_Fallback):
    """A fallback that blocks, like a slow database write, until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, line: bytes) -> None:
        self.started.set()
        await self.release.wait()
        await super().__call__(line)


class _StalledDrainWriter(asyncio.StreamWriter):
    """Hands bytes to the real transport but never wakes ``drain()``: the loop iteration between a flush
    completing and the writer task resuming, frozen in place."""

    def __init__(self, real: asyncio.StreamWriter, reader: asyncio.StreamReader) -> None:
        super().__init__(real.transport, real.transport.get_protocol(), reader, asyncio.get_running_loop())
        self._real_writer_whose_finalizer_would_close_the_transport = real

    async def drain(self) -> None:
        await asyncio.Event().wait()


async def _open_with_stalled_drain(
    address: CollectorAddress, timeout: float
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await open_collector_connection(address, timeout)
    return reader, _StalledDrainWriter(writer, reader)


def _producer(
    path: Path,
    fallback: _Fallback,
    on_unavailable="fallback",
    buffer_size: int = 100,
    open_connection: Callable[
        [CollectorAddress, float], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
    ] = open_collector_connection,
) -> SpendEventProducer:
    return SpendEventProducer(
        address=UnixAddress(path=str(path)),
        on_unavailable=on_unavailable,
        buffer_size=buffer_size,
        connect_timeout=1.0,
        fallback=fallback,
        open_connection=open_connection,
    )


def test_parse_collector_address():
    assert parse_collector_address("unix:///var/run/litellm/collector.sock") == UnixAddress(
        path="/var/run/litellm/collector.sock"
    )
    assert parse_collector_address("tcp://127.0.0.1:4100") == TcpAddress(host="127.0.0.1", port=4100)
    assert parse_collector_address("tcp://localhost:4100") == TcpAddress(host="localhost", port=4100)
    assert parse_collector_address("tcp://[::1]:4100") == TcpAddress(host="::1", port=4100)
    assert isinstance(parse_collector_address("redis://localhost:6379"), AddressError)
    assert isinstance(parse_collector_address("tcp://127.0.0.1"), AddressError)


@pytest.mark.parametrize("address", ["tcp://0.0.0.0:4100", "tcp://10.0.0.5:4100", "tcp://collector.svc:4100"])
def test_tcp_address_outside_loopback_is_refused(address: str):
    """The socket has no authentication, so anything reachable from outside the pod would accept forged spend."""
    error: Final = parse_collector_address(address)
    assert isinstance(error, AddressError)
    assert "loopback" in error.reason
    assert build_spend_event_producer(CollectorSettings(enabled=True, address=address), _Fallback()) is None


def test_gateway_produces_only_when_enabled_and_not_the_sidecar_itself():
    fallback: Final = _Fallback()
    assert build_spend_event_producer(CollectorSettings(enabled=False), fallback) is None
    assert build_spend_event_producer(CollectorSettings(enabled=True, job_role="collector"), fallback) is None
    assert build_spend_event_producer(CollectorSettings(enabled=True, address="redis://x"), fallback) is None
    assert isinstance(build_spend_event_producer(CollectorSettings(enabled=True), fallback), SpendEventProducer)


def test_settings_read_the_documented_env(monkeypatch):
    monkeypatch.setenv("LITELLM_COLLECTOR_ENABLED", "true")
    monkeypatch.setenv("LITELLM_COLLECTOR_ADDRESS", "tcp://127.0.0.1:4100")
    monkeypatch.setenv("LITELLM_COLLECTOR_BUFFER_SIZE", "50")
    monkeypatch.setenv("LITELLM_COLLECTOR_ON_UNAVAILABLE", "drop")
    monkeypatch.setenv("LITELLM_JOB_ROLE", "collector")
    settings: Final = CollectorSettings()
    assert (settings.enabled, settings.address, settings.buffer_size, settings.on_unavailable) == (
        True,
        "tcp://127.0.0.1:4100",
        50,
        "drop",
    )
    assert settings.produces is False


@pytest.mark.asyncio
async def test_events_reach_the_sidecar_once_and_in_order(tmp_path: Path):
    fallback: Final = _Fallback()
    async with _Sidecar(tmp_path / "spend.sock") as sidecar:
        producer: Final = _producer(sidecar.path, fallback)
        outcomes: Final = [await producer.publish(f"event-{i}\n".encode()) for i in range(20)]
        await producer.close(drain_timeout=5.0)
        await asyncio.sleep(0.05)

    assert outcomes == ["queued"] * 20
    assert sidecar.lines == [f"event-{i}\n".encode() for i in range(20)]
    assert fallback.lines == []
    stats: Final = producer.stats()
    assert (stats.queued, stats.sent, stats.fallback, stats.dropped) == (20, 20, 0, 0)


@pytest.mark.asyncio
async def test_unreachable_sidecar_falls_back_in_process_and_backs_off(tmp_path: Path):
    fallback: Final = _Fallback()
    producer: Final = _producer(tmp_path / "missing.sock", fallback)
    first: Final = await producer.publish(b"event-1\n")
    await asyncio.sleep(0.05)
    second: Final = await producer.publish(b"event-2\n")
    await producer.close(drain_timeout=5.0)

    assert first == "queued"
    assert second == "fallback"
    assert fallback.lines == [b"event-1\n", b"event-2\n"]
    stats: Final = producer.stats()
    assert (stats.sent, stats.fallback, stats.dropped, stats.connected) == (0, 2, 0, False)


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop], ids=["asyncio", "uvloop"])
def test_sidecar_hang_up_falls_back_instead_of_losing_events(
    tmp_path: Path, loop_factory: Callable[[], asyncio.AbstractEventLoop]
):
    async def scenario() -> tuple[list[bytes], list[bytes], tuple[int, int, int]]:
        fallback: Final = _Fallback()
        sidecar: Final = _Sidecar(tmp_path / "spend.sock")
        async with sidecar:
            producer: Final = _producer(sidecar.path, fallback)
            await producer.publish(b"event-1\n")
            await asyncio.sleep(0.05)
            await sidecar.hang_up()
            await asyncio.sleep(0.05)
            await producer.publish(b"event-2\n")
            await producer.close(drain_timeout=5.0)
        stats: Final = producer.stats()
        return sidecar.lines, fallback.lines, (stats.sent, stats.fallback, stats.dropped)

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        sidecar_lines, fallback_lines, counts = runner.run(scenario())

    assert sidecar_lines == [b"event-1\n"]
    assert fallback_lines == [b"event-2\n"]
    assert counts == (1, 1, 0)


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop], ids=["asyncio", "uvloop"])
def test_mid_stream_crash_never_bills_an_event_on_both_sides(
    tmp_path: Path, loop_factory: Callable[[], asyncio.AbstractEventLoop]
):
    """Events large enough to straddle the kernel buffer, a sidecar that reads some and then drops the socket: a
    failed write may only fall back when the sidecar cannot have read the whole line."""
    events: Final = tuple(f"event-{i:03d}-".encode() + b"x" * 65536 + b"\n" for i in range(64))

    async def scenario() -> tuple[list[bytes], list[bytes], tuple[int, int, int]]:
        fallback: Final = _Fallback()
        sidecar: Final = _CrashingSidecar(tmp_path / "spend.sock", lines_before_crash=3)
        async with sidecar:
            producer: Final = _producer(sidecar.path, fallback)
            for event in events:
                assert await producer.publish(event) == "queued"
            await asyncio.sleep(0.2)
            await producer.close(drain_timeout=5.0)
        stats: Final = producer.stats()
        return sidecar.lines, fallback.lines, (stats.sent, stats.fallback, stats.dropped)

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        sidecar_lines, fallback_lines, counts = runner.run(scenario())

    assert sidecar_lines == list(events[:3])
    assert set(sidecar_lines).isdisjoint(fallback_lines)
    assert len(fallback_lines) == len(set(fallback_lines))
    assert fallback_lines[-1] == events[-1]
    assert counts[0] + counts[1] == len(events) and counts[2] == 0
    assert counts[0] >= len(sidecar_lines)


@pytest.mark.asyncio
async def test_drain_timeout_hands_the_in_flight_event_to_fallback(tmp_path: Path):
    """A sidecar that stops reading leaves one event half-written; cancelling the writer must not lose it."""
    fallback: Final = _Fallback()
    stuck: Final = b"x" * (4 * 1024 * 1024) + b"\n"
    async with _Sidecar(tmp_path / "spend.sock", reads=False) as sidecar:
        producer: Final = _producer(sidecar.path, fallback)
        assert await producer.publish(stuck) == "queued"
        await asyncio.sleep(0.1)
        await producer.close(drain_timeout=0.2)

    assert fallback.lines == [stuck]
    stats: Final = producer.stats()
    assert (stats.sent, stats.fallback, stats.connected) == (0, 1, False)


@pytest.mark.asyncio
async def test_shutdown_lets_the_writer_finish_a_fallback_already_in_progress(tmp_path: Path):
    """Cancelling the writer while it runs the pipeline in-process must neither lose nor repeat that event."""
    fallback: Final = _GatedFallback()
    producer: Final = _producer(tmp_path / "missing.sock", fallback)
    assert await producer.publish(b"event-1\n") == "queued"
    await asyncio.wait_for(fallback.started.wait(), 5.0)
    closing: Final = asyncio.ensure_future(producer.close(drain_timeout=0.05))
    await asyncio.sleep(0.2)
    assert fallback.lines == []
    fallback.release.set()
    await asyncio.wait_for(closing, 5.0)

    assert fallback.lines == [b"event-1\n"]
    assert producer.stats().fallback == 1


@pytest.mark.asyncio
async def test_shutdown_does_not_replay_an_event_the_kernel_already_took(tmp_path: Path):
    """Cancelling a drain whose bytes already left the process must not run the event a second time in-process."""
    fallback: Final = _Fallback()
    async with _Sidecar(tmp_path / "spend.sock") as sidecar:
        producer: Final = _producer(sidecar.path, fallback, open_connection=_open_with_stalled_drain)
        assert await producer.publish(b"event-1\n") == "queued"
        await asyncio.sleep(0.1)
        await producer.close(drain_timeout=0.2)
        await asyncio.sleep(0.05)

    assert sidecar.lines == [b"event-1\n"]
    assert fallback.lines == []
    stats: Final = producer.stats()
    assert (stats.fallback, stats.dropped, stats.connected) == (0, 0, False)


@pytest.mark.asyncio
async def test_drop_policy_counts_instead_of_running_in_process(tmp_path: Path):
    fallback: Final = _Fallback()
    producer: Final = _producer(tmp_path / "missing.sock", fallback, on_unavailable="drop")
    await producer.publish(b"event-1\n")
    await producer.close(drain_timeout=5.0)
    assert await producer.publish(b"event-2\n") == "dropped"

    assert fallback.lines == []
    assert producer.stats().dropped == 2


@pytest.mark.asyncio
async def test_full_buffer_applies_the_unavailable_policy_immediately(tmp_path: Path):
    fallback: Final = _Fallback()
    async with _Sidecar(tmp_path / "spend.sock") as sidecar:
        producer: Final = _producer(sidecar.path, fallback, buffer_size=2)
        outcomes: Final = [await producer.publish(f"event-{i}\n".encode()) for i in range(3)]
        await producer.close(drain_timeout=5.0)
        await asyncio.sleep(0.05)

    assert outcomes == ["queued", "queued", "fallback"]
    assert fallback.lines == [b"event-2\n"]
    assert sidecar.lines == [b"event-0\n", b"event-1\n"]


@pytest.mark.asyncio
async def test_close_flushes_buffered_events_then_refuses_new_ones(tmp_path: Path):
    fallback: Final = _Fallback()
    async with _Sidecar(tmp_path / "spend.sock") as sidecar:
        producer: Final = _producer(sidecar.path, fallback)
        for i in range(50):
            await producer.publish(f"event-{i}\n".encode())
        assert sidecar.lines == []
        await producer.close(drain_timeout=5.0)
        await asyncio.sleep(0.05)
        after_close: Final = await producer.publish(b"late\n")

    assert len(sidecar.lines) == 50
    assert after_close == "fallback"
    assert fallback.lines == [b"late\n"]
    assert producer.stats().connected is False
