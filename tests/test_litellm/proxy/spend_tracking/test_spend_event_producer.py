import asyncio
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.spend_tracking.spend_event_producer import (
    AddressError,
    SpendEventProducer,
    SpendWorkerSettings,
    TcpAddress,
    UnixAddress,
    build_spend_event_producer,
    parse_spend_worker_address,
)


class _Sidecar:
    """A unix-socket server that records every line it receives, standing in for the spend worker."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[bytes] = []  # mutable-ok: test double records what the producer sent
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> "_Sidecar":
        self._server = await asyncio.start_unix_server(self._on_connection, path=str(self.path))
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _on_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while line := await reader.readline():
            self.lines.append(line)
        writer.close()


class _Fallback:
    def __init__(self) -> None:
        self.lines: list[bytes] = []  # mutable-ok: test double records what fell back to in-process

    async def __call__(self, line: bytes) -> None:
        self.lines.append(line)


def _producer(path: Path, fallback: _Fallback, on_unavailable="fallback", buffer_size: int = 100) -> SpendEventProducer:
    return SpendEventProducer(
        address=UnixAddress(path=str(path)),
        on_unavailable=on_unavailable,
        buffer_size=buffer_size,
        connect_timeout=1.0,
        fallback=fallback,
    )


def test_parse_spend_worker_address():
    assert parse_spend_worker_address("unix:///var/run/litellm/spend-worker.sock") == UnixAddress(
        path="/var/run/litellm/spend-worker.sock"
    )
    assert parse_spend_worker_address("tcp://127.0.0.1:4100") == TcpAddress(host="127.0.0.1", port=4100)
    assert isinstance(parse_spend_worker_address("redis://localhost:6379"), AddressError)
    assert isinstance(parse_spend_worker_address("tcp://127.0.0.1"), AddressError)


def test_gateway_produces_only_when_enabled_and_not_the_sidecar_itself():
    fallback: Final = _Fallback()
    assert build_spend_event_producer(SpendWorkerSettings(enabled=False), fallback) is None
    assert build_spend_event_producer(SpendWorkerSettings(enabled=True, job_role="spend_worker"), fallback) is None
    assert build_spend_event_producer(SpendWorkerSettings(enabled=True, address="redis://x"), fallback) is None
    assert isinstance(build_spend_event_producer(SpendWorkerSettings(enabled=True), fallback), SpendEventProducer)


def test_settings_read_the_documented_env(monkeypatch):
    monkeypatch.setenv("LITELLM_SPEND_WORKER_ENABLED", "true")
    monkeypatch.setenv("LITELLM_SPEND_WORKER_ADDRESS", "tcp://127.0.0.1:4100")
    monkeypatch.setenv("LITELLM_SPEND_WORKER_BUFFER_SIZE", "50")
    monkeypatch.setenv("LITELLM_SPEND_WORKER_ON_UNAVAILABLE", "drop")
    monkeypatch.setenv("LITELLM_JOB_ROLE", "spend_worker")
    settings: Final = SpendWorkerSettings()
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
