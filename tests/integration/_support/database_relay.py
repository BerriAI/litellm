import asyncio
import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from pydantic import TypeAdapter

PORT: Final = TypeAdapter(int)
OUTAGE_SECONDS: Final = 10.0


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return PORT.validate_python(reserve.getsockname()[1])


class DatabaseRelay:
    def __init__(self, upstream_host: str, upstream_port: int, trigger: bytes) -> None:
        self.port: Final = _free_port()
        self._upstream_host: Final = upstream_host
        self._upstream_port: Final = upstream_port
        self._trigger: Final = trigger
        self._loop: Final = asyncio.new_event_loop()
        self._armed: Final = threading.Event()
        self.tripped: Final = threading.Event()
        self.refused = 0
        self.reconnected: Final = threading.Event()
        self._tripped_at = 0.0
        self._writers: tuple[asyncio.StreamWriter, ...] = ()
        self._ready: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, daemon=True)

    def arm(self) -> None:
        self._armed.set()

    def start(self) -> None:
        self._thread.start()
        assert self._ready.wait(10), "Database relay did not start"

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(asyncio.start_server(self._serve, "127.0.0.1", self.port))
        self._ready.set()
        self._loop.run_forever()

    def _drop_all(self) -> None:
        for writer in self._writers:
            writer.close()
        self._writers = ()

    async def _serve(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        if self.tripped.is_set() and time.monotonic() - self._tripped_at < OUTAGE_SECONDS:
            self.refused += 1
            client_writer.close()
            return
        if self.tripped.is_set():
            self.reconnected.set()
        server_reader, server_writer = await asyncio.open_connection(self._upstream_host, self._upstream_port)
        self._writers = (*self._writers, client_writer, server_writer)

        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, inspect: bool) -> None:
            try:
                while chunk := await reader.read(65536):
                    if inspect and self._armed.is_set() and not self.tripped.is_set() and self._trigger in chunk:
                        self._tripped_at = time.monotonic()
                        self.tripped.set()
                        self._drop_all()
                        return
                    writer.write(chunk)
                    await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                return
            finally:
                writer.close()

        await asyncio.gather(
            forward(client_reader, server_writer, True),
            forward(server_reader, client_writer, False),
        )


class HeldStatementRelay:
    def __init__(self, upstream_host: str, upstream_port: int, trigger: bytes) -> None:
        self.port: Final = _free_port()
        self._upstream_host: Final = upstream_host
        self._upstream_port: Final = upstream_port
        self._trigger: Final = trigger
        self._loop: Final = asyncio.new_event_loop()
        self._released: Final = asyncio.Event()
        self.held: Final = threading.Event()
        self._ready: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, daemon=True)

    def release(self) -> None:
        self._loop.call_soon_threadsafe(self._released.set)

    def start(self) -> None:
        self._thread.start()
        assert self._ready.wait(10), "Database relay did not start"

    def stop(self) -> None:
        self.release()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(asyncio.start_server(self._serve, "127.0.0.1", self.port))
        self._ready.set()
        self._loop.run_forever()

    def _holds(self, window: bytes) -> bool:
        return not self.held.is_set() and self._trigger in window

    async def _serve(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        server_reader, server_writer = await asyncio.open_connection(self._upstream_host, self._upstream_port)

        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, inspect: bool) -> None:
            tail = b""  # rebind-ok: carries the previous read's end so a trigger split across reads still matches
            try:
                while chunk := await reader.read(65536):
                    window: Final = tail + chunk
                    if inspect and self._holds(window):
                        self.held.set()
                        await self._released.wait()
                    tail = window[-(len(self._trigger) - 1) :]
                    writer.write(chunk)
                    await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                return
            finally:
                writer.close()

        await asyncio.gather(
            forward(client_reader, server_writer, True),
            forward(server_reader, client_writer, False),
        )


class TriggerScanner:
    def __init__(self, trigger: bytes) -> None:
        self._trigger: Final = trigger
        self._tail: bytes = b""

    def feed(self, chunk: bytes) -> bool:
        window: Final = self._tail + chunk
        self._tail = window[-(len(self._trigger) - 1) :]
        return self._trigger in window


class DroppedConnectionRelay:
    def __init__(self, upstream_host: str, upstream_port: int, trigger: bytes) -> None:
        self.port: Final = _free_port()
        self._upstream_host: Final = upstream_host
        self._upstream_port: Final = upstream_port
        self._trigger: Final = trigger
        self._loop: Final = asyncio.new_event_loop()
        self._armed: Final = threading.Event()
        self.dropped: Final = threading.Event()
        self._ready: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, daemon=True)

    def arm(self) -> None:
        self._armed.set()

    def disarm(self) -> None:
        self._armed.clear()

    def start(self) -> None:
        self._thread.start()
        assert self._ready.wait(10), "Database relay did not start"

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(asyncio.start_server(self._serve, "127.0.0.1", self.port))
        self._ready.set()
        self._loop.run_forever()

    async def _serve(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        server_reader, server_writer = await asyncio.open_connection(self._upstream_host, self._upstream_port)

        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, inspect: bool) -> None:
            scanner: Final = TriggerScanner(self._trigger)
            try:
                while chunk := await reader.read(65536):
                    matched: Final = scanner.feed(chunk)
                    if inspect and self._armed.is_set() and matched:
                        self.dropped.set()
                        client_writer.close()
                        return
                    writer.write(chunk)
                    await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                return
            finally:
                writer.close()

        await asyncio.gather(
            forward(client_reader, server_writer, True),
            forward(server_reader, client_writer, False),
        )


def _relayed_url(database_url: str, port: int) -> str:
    parts: Final = urlsplit(database_url)
    credentials: Final = f"{parts.username}:{parts.password}@" if parts.username else ""
    return urlunsplit(parts._replace(netloc=f"{credentials}127.0.0.1:{port}"))


@contextmanager
def database_relay(database_url: str, trigger: bytes) -> Generator[tuple[DatabaseRelay, str]]:
    parts: Final = urlsplit(database_url)
    assert parts.hostname is not None and parts.port is not None, database_url
    relay: Final = DatabaseRelay(parts.hostname, parts.port, trigger)
    relay.start()
    try:
        yield relay, _relayed_url(database_url, relay.port)
    finally:
        relay.stop()


@contextmanager
def held_statement_relay(database_url: str, trigger: bytes) -> Generator[tuple[HeldStatementRelay, str]]:
    parts: Final = urlsplit(database_url)
    assert parts.hostname is not None and parts.port is not None, database_url
    relay: Final = HeldStatementRelay(parts.hostname, parts.port, trigger)
    relay.start()
    try:
        yield relay, _relayed_url(database_url, relay.port)
    finally:
        relay.stop()


@contextmanager
def dropped_connection_relay(database_url: str, trigger: bytes) -> Generator[tuple[DroppedConnectionRelay, str]]:
    parts: Final = urlsplit(database_url)
    assert parts.hostname is not None and parts.port is not None, database_url
    relay: Final = DroppedConnectionRelay(parts.hostname, parts.port, trigger)
    relay.start()
    try:
        yield relay, _relayed_url(database_url, relay.port)
    finally:
        relay.stop()
