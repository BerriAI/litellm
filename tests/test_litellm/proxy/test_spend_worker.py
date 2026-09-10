import asyncio
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.spend_tracking.spend_event_producer import (
    AddressError,
    SpendEventProducer,
    TcpAddress,
    UnixAddress,
    open_spend_worker_connection,
)
from litellm.proxy.spend_worker import SpendEventConsumer, _address_argument


class _Handler:
    def __init__(self, fail_on: bytes | None = None) -> None:
        self.lines: list[bytes] = []  # mutable-ok: test double records the events the consumer handed over
        self._fail_on = fail_on

    async def __call__(self, line: bytes) -> None:
        if line == self._fail_on:
            raise RuntimeError("pipeline failed")
        self.lines.append(line)


async def _no_fallback(line: bytes) -> None:
    raise AssertionError(f"unexpected fallback for {line!r}")


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "tcp"])
async def test_consumer_handles_each_producer_line_once_in_order(tmp_path: Path, transport: str):
    handler: Final = _Handler(fail_on=b"event-3\n")
    consumer: Final = SpendEventConsumer(handler)
    server: Final = await consumer.serve(
        UnixAddress(path=str(tmp_path / "spend.sock")) if transport == "unix" else TcpAddress("127.0.0.1", 0)
    )
    address: Final = (
        UnixAddress(path=str(tmp_path / "spend.sock"))
        if transport == "unix"
        else TcpAddress("127.0.0.1", server.sockets[0].getsockname()[1])
    )
    producer: Final = SpendEventProducer(
        address=address, on_unavailable="fallback", buffer_size=100, connect_timeout=1.0, fallback=_no_fallback
    )
    for i in range(6):
        await producer.publish(f"event-{i}\n".encode())
    await producer.close(drain_timeout=5.0)

    server.close()
    assert await consumer.drain(timeout=5.0) == 0
    assert handler.lines == [f"event-{i}\n".encode() for i in range(6) if i != 3]
    assert (consumer.received, consumer.handled, consumer.failed) == (6, 5, 1)


@pytest.mark.asyncio
async def test_consumer_discards_a_truncated_trailing_event(tmp_path: Path):
    handler: Final = _Handler()
    consumer: Final = SpendEventConsumer(handler)
    address: Final = UnixAddress(path=str(tmp_path / "spend.sock"))
    server: Final = await consumer.serve(address)
    _, writer = await open_spend_worker_connection(address, timeout=1.0)
    writer.write(b"whole\npartial-without-newline")
    await writer.drain()
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)

    server.close()
    assert await consumer.drain(timeout=5.0) == 0
    assert handler.lines == [b"whole\n"]
    assert consumer.received == 1


@pytest.mark.asyncio
async def test_drain_reports_producers_still_connected_after_the_timeout(tmp_path: Path):
    consumer: Final = SpendEventConsumer(_Handler())
    address: Final = UnixAddress(path=str(tmp_path / "spend.sock"))
    server: Final = await consumer.serve(address)
    _, writer = await open_spend_worker_connection(address, timeout=1.0)
    await asyncio.sleep(0.05)

    server.close()
    assert await consumer.drain(timeout=0.1) == 1
    writer.close()
    await writer.wait_closed()
    assert await consumer.drain(timeout=5.0) == 0


def test_address_argument():
    assert _address_argument((), default="unix:///tmp/x.sock") == "unix:///tmp/x.sock"
    assert _address_argument(("--address", "tcp://127.0.0.1:4100"), default="unix:///tmp/x.sock") == (
        "tcp://127.0.0.1:4100"
    )
    assert isinstance(_address_argument(("--listen", "x"), default="unix:///tmp/x.sock"), AddressError)
