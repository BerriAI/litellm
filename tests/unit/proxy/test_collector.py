import asyncio
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import pytest
import uvloop

from litellm._logging import verbose_logger, verbose_proxy_logger, verbose_router_logger
from litellm.proxy.collector import (
    SpendEventConsumer,
    address_argument,
    apply_log_level,
    pod_pgbouncer_database_url,
)
from litellm.proxy.db.pgbouncer import PgBouncerError, PgBouncerSettings
from litellm.proxy.spend_tracking.spend_event_producer import (
    AddressError,
    SpendEventProducer,
    TcpAddress,
    UnixAddress,
    open_collector_connection,
)


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


class _Fallback:
    def __init__(self) -> None:
        self.lines: list[bytes] = []  # mutable-ok: test double records the events run in-process

    async def __call__(self, line: bytes) -> None:
        self.lines.append(line)


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
    _, writer = await open_collector_connection(address, timeout=1.0)
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
    _, writer = await open_collector_connection(address, timeout=1.0)
    await asyncio.sleep(0.05)

    server.close()
    assert await consumer.drain(timeout=0.1) == 1
    writer.close()
    await writer.wait_closed()
    assert await consumer.drain(timeout=5.0) == 0


@pytest.mark.asyncio
async def test_graceful_stop_hands_the_producer_over_to_its_fallback_without_losing_events(tmp_path: Path):
    handler: Final = _Handler()
    fallback: Final = _Fallback()
    consumer: Final = SpendEventConsumer(handler)
    address: Final = UnixAddress(path=str(tmp_path / "spend.sock"))
    server: Final = await consumer.serve(address)
    producer: Final = SpendEventProducer(
        address=address, on_unavailable="fallback", buffer_size=100, connect_timeout=1.0, fallback=fallback
    )
    await producer.publish(b"event-1\n")
    await asyncio.sleep(0.05)

    server.close()
    draining: Final = asyncio.ensure_future(consumer.drain(timeout=5.0))
    await asyncio.sleep(0.05)
    await producer.publish(b"event-2\n")
    await producer.close(drain_timeout=5.0)

    assert await draining == 0
    assert handler.lines == [b"event-1\n"]
    assert fallback.lines == [b"event-2\n"]
    assert (producer.stats().sent, producer.stats().fallback) == (1, 1)


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop], ids=["asyncio", "uvloop"])
def test_drain_still_hands_over_live_producers_when_another_connection_already_died(
    tmp_path: Path, loop_factory: Callable[[], asyncio.AbstractEventLoop]
):
    """A transport the loop force-closed under a busy handler must not abort the half-close of the others."""

    async def scenario() -> tuple[int, list[bytes]]:
        release: Final = asyncio.Event()

        async def slow_handler(line: bytes) -> None:
            await release.wait()

        consumer: Final = SpendEventConsumer(slow_handler)
        address: Final = UnixAddress(path=str(tmp_path / "spend.sock"))
        server: Final = await consumer.serve(address)
        _, dead = await open_collector_connection(address, timeout=1.0)
        dead.write(b"stuck\n")
        await dead.drain()
        await asyncio.sleep(0.05)
        for connection in consumer._open_connections:  # pyright: ignore[reportPrivateUsage]  # force-close like uvloop does on a socket error
            connection.transport.close()
        dead.close()
        fallback: Final = _Fallback()
        producer: Final = SpendEventProducer(
            address=address, on_unavailable="fallback", buffer_size=100, connect_timeout=1.0, fallback=fallback
        )
        await producer.publish(b"event-1\n")
        await asyncio.sleep(0.05)

        server.close()
        draining: Final = asyncio.ensure_future(consumer.drain(timeout=0.5))
        await asyncio.sleep(0.05)
        await producer.publish(b"event-2\n")
        await producer.close(drain_timeout=5.0)
        still_open: Final = await draining
        release.set()
        await asyncio.sleep(0.05)
        return still_open, fallback.lines

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        still_open, fallback_lines = runner.run(scenario())

    assert still_open == 2
    assert fallback_lines == [b"event-2\n"]


def test_address_argument():
    assert address_argument((), default="unix:///tmp/x.sock") == "unix:///tmp/x.sock"
    assert address_argument(("--address", "tcp://127.0.0.1:4100"), default="unix:///tmp/x.sock") == (
        "tcp://127.0.0.1:4100"
    )
    assert isinstance(address_argument(("--listen", "x"), default="unix:///tmp/x.sock"), AddressError)


def test_pod_pgbouncer_database_url_points_at_the_proxy_containers_pooler():
    """With pgbouncer on, the sidecar must not open its own upstream connections but share the pod's pooler."""
    upstream: Final = "postgresql://u:p@db.internal:5432/litellm?schema=public"
    environ: Final = {"DATABASE_URL": upstream}
    assert pod_pgbouncer_database_url(PgBouncerSettings(enabled=False), environ, token_auth=False) is None
    assert (
        pod_pgbouncer_database_url(PgBouncerSettings(enabled=True, port=6543), environ, token_auth=False)
        == "postgresql://u:p@127.0.0.1:6543/litellm?schema=public&pgbouncer=true"
    )
    assert isinstance(pod_pgbouncer_database_url(PgBouncerSettings(enabled=True), {}, token_auth=False), PgBouncerError)


def test_pod_pgbouncer_database_url_goes_direct_under_token_auth():
    """The proxy's pgbouncer only knows the token that container minted, so the sidecar must mint its own upstream."""
    iam_upstream: Final = "postgresql://u@db.internal:5432/litellm?schema=public"
    assert (
        pod_pgbouncer_database_url(PgBouncerSettings(enabled=True), {"DATABASE_URL": iam_upstream}, token_auth=True)
        is None
    )
    assert pod_pgbouncer_database_url(PgBouncerSettings(enabled=True), {}, token_auth=True) is None


@pytest.fixture
def restore_log_levels() -> Iterator[None]:
    loggers: Final = (verbose_logger, verbose_router_logger, verbose_proxy_logger)
    levels: Final = tuple(logger.level for logger in loggers)
    yield
    for logger, level in zip(loggers, levels, strict=True):
        logger.setLevel(level)


@pytest.mark.usefixtures("restore_log_levels")
@pytest.mark.parametrize(
    ("litellm_log", "expected"),
    [("DEBUG", logging.DEBUG), ("info", logging.INFO), (None, logging.WARNING), ("loud", logging.WARNING)],
)
def test_apply_log_level_mirrors_the_proxy_env_contract(litellm_log: str | None, expected: int):
    verbose_proxy_logger.setLevel(logging.WARNING)
    apply_log_level(litellm_log)
    assert verbose_proxy_logger.isEnabledFor(expected)
    assert not verbose_proxy_logger.isEnabledFor(expected - 10)
