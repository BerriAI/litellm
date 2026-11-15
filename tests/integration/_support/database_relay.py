import asyncio
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tests.integration._support.async_server import LoopbackServer, close_writer
from tests.integration._support.ports import allocate_port

OUTAGE_SECONDS: Final = 10.0


class DatabaseRelay:
    def __init__(self, upstream_host: str, upstream_port: int, trigger: bytes) -> None:
        self.port: Final = allocate_port()
        self._upstream_host: Final = upstream_host
        self._upstream_port: Final = upstream_port
        self._trigger: Final = trigger
        self._server: Final = LoopbackServer(self.port, self._serve)
        self._loop: Final = self._server.loop
        self._armed: Final = threading.Event()
        self.tripped: Final = threading.Event()
        self.refused = 0
        self.reconnected: Final = threading.Event()
        self._tripped_at = 0.0
        self._writers: tuple[asyncio.StreamWriter, ...] = ()

    def arm(self) -> None:
        self._armed.set()

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.stop()

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
                await close_writer(writer)

        await asyncio.gather(
            forward(client_reader, server_writer, True),
            forward(server_reader, client_writer, False),
        )


class HeldStatementRelay:
    """Holds the first client statement carrying ``trigger`` until ``release``. With ``after``, only a
    trigger statement that follows a statement carrying ``after`` is held, so a test can pin the hold to
    one point of a longer sequence. Starts armed unless ``armed`` is false, then ``arm`` starts it."""

    def __init__(
        self,
        upstream_host: str,
        upstream_port: int,
        trigger: bytes,
        *,
        after: bytes | None = None,
        armed: bool = True,
    ) -> None:
        self.port: Final = allocate_port()
        self._upstream_host: Final = upstream_host
        self._upstream_port: Final = upstream_port
        self._trigger: Final = trigger
        self._after: Final = after
        self._server: Final = LoopbackServer(self.port, self._serve)
        self._loop: Final = self._server.loop
        self._released: Final = asyncio.Event()
        self._armed: Final = threading.Event()
        if armed:
            self._armed.set()
        self._after_seen: Final = threading.Event()
        self.held: Final = threading.Event()

    def arm(self) -> None:
        self._armed.set()

    def release(self) -> None:
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._released.set)

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self.release()
        self._server.stop()

    def _holds(self, markers: frozenset[bytes]) -> bool:
        if not self._armed.is_set() or self.held.is_set():
            return False
        if self._after is not None and not self._after_seen.is_set():
            if self._after in markers:
                self._after_seen.set()
            return False
        return self._trigger in markers

    async def _serve(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        server_reader, server_writer = await asyncio.open_connection(self._upstream_host, self._upstream_port)

        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, inspect: bool) -> None:
            scanner: Final = StatementMarkerScanner((self._trigger,) + ((self._after,) if self._after else ()))
            try:
                while chunk := await reader.read(65536):
                    if inspect and any(self._holds(markers) for markers in scanner.feed(chunk)):
                        self.held.set()
                        await self._released.wait()
                    writer.write(chunk)
                    await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                return
            finally:
                await close_writer(writer)

        await asyncio.gather(
            forward(client_reader, server_writer, True),
            forward(server_reader, client_writer, False),
        )


class StatementMarkerScanner:
    def __init__(self, markers: tuple[bytes, ...]) -> None:
        self._markers: Final = markers
        self._statements: Final[dict[bytes, frozenset[bytes]]] = {}
        self._buffer = b""
        self._startup = True

    def feed(self, chunk: bytes) -> Generator[frozenset[bytes]]:
        self._buffer += chunk
        while len(self._buffer) >= (4 if self._startup else 5):
            offset: Final = 0 if self._startup else 1
            length: Final = int.from_bytes(self._buffer[offset : offset + 4], "big")
            if length < 4:
                raise ValueError("Invalid PostgreSQL message length")
            if len(self._buffer) < length + offset:
                return
            message: Final = self._buffer[: length + offset]
            self._buffer = self._buffer[length + offset :]
            if self._startup:
                self._startup = message[4:8] == b"\x04\xd2\x16/"
                continue
            payload: Final = message[5:]
            if message[:1] == b"P":
                name, _, query = payload.partition(b"\x00")
                matches: Final = self._matches(query.split(b"\x00", 1)[0])
                if name:
                    self._statements[name] = matches
                yield matches
            elif message[:1] == b"B":
                fields: Final = payload.split(b"\x00", 2)
                prepared: Final = self._statements.get(fields[1], frozenset()) if len(fields) > 1 else frozenset()
                yield prepared | self._matches(payload)
            elif message[:1] == b"Q":
                yield self._matches(payload)
            elif message[:1] == b"C" and payload[:1] == b"S":
                self._statements.pop(payload[1:].split(b"\x00", 1)[0], None)

    def _matches(self, query: bytes) -> frozenset[bytes]:
        return frozenset(marker for marker in self._markers if marker in query)


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
        self.port: Final = allocate_port()
        self._upstream_host: Final = upstream_host
        self._upstream_port: Final = upstream_port
        self._trigger: Final = trigger
        self._server: Final = LoopbackServer(self.port, self._serve)
        self._loop: Final = self._server.loop
        self._armed: Final = threading.Event()
        self.dropped: Final = threading.Event()

    def arm(self) -> None:
        self._armed.set()

    def disarm(self) -> None:
        self._armed.clear()

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.stop()

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
                await close_writer(writer)

        await asyncio.gather(
            forward(client_reader, server_writer, True),
            forward(server_reader, client_writer, False),
        )


SSL_REQUEST_CODE: Final = 80877103
GSSENC_REQUEST_CODE: Final = 80877104


class FrontendStatementScanner:
    """Counts the statements one Postgres client sends: a Sync closes every extended-protocol statement and a
    Query is a simple-protocol statement, so the count moves with the number of statements, never their size."""

    def __init__(self) -> None:
        self._buffer = b""
        self._started = False

    def feed(self, chunk: bytes) -> int:
        self._buffer += chunk  # rebind-ok: a message can span two reads
        statements = 0  # rebind-ok: summed over the messages this chunk completes
        while (frame := self._next_frame()) is not None:
            length, is_statement = frame
            statements += int(is_statement)
            self._buffer = self._buffer[length:]  # rebind-ok: drops the parsed message
        return statements

    def _next_frame(self) -> tuple[int, bool] | None:
        if not self._started:
            if len(self._buffer) < 8:
                return None
            startup_length: Final = int.from_bytes(self._buffer[:4], "big")
            if len(self._buffer) < startup_length:
                return None
            request_code: Final = int.from_bytes(self._buffer[4:8], "big")
            self._started = request_code not in (SSL_REQUEST_CODE, GSSENC_REQUEST_CODE)
            return startup_length, False
        if len(self._buffer) < 5:
            return None
        frame_length: Final = 1 + int.from_bytes(self._buffer[1:5], "big")
        if len(self._buffer) < frame_length:
            return None
        return frame_length, self._buffer[:1] in (b"S", b"Q")


class StatementCountingRelay:
    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        self.port: Final = allocate_port()
        self._upstream_host: Final = upstream_host
        self._upstream_port: Final = upstream_port
        self._loop: Final = asyncio.new_event_loop()
        self.statements = 0
        self._ready: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, daemon=True)

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

        async def forward(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter, scanner: FrontendStatementScanner | None
        ) -> None:
            try:
                while chunk := await reader.read(65536):
                    if scanner is not None:
                        self.statements += scanner.feed(chunk)
                    writer.write(chunk)
                    await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                return
            finally:
                writer.close()

        await asyncio.gather(
            forward(client_reader, server_writer, FrontendStatementScanner()),
            forward(server_reader, client_writer, None),
        )


def plaintext_database_url(database_url: str) -> str:
    parts: Final = urlsplit(database_url)
    parameters: Final = tuple(
        (name, value) for name, value in parse_qsl(parts.query, keep_blank_values=True) if name != "sslmode"
    ) + (("sslmode", "disable"),)
    return urlunsplit(parts._replace(query=urlencode(parameters)))


def _relayed_url(database_url: str, port: int) -> str:
    parts: Final = urlsplit(database_url)
    credentials: Final = f"{parts.username}:{parts.password}@" if parts.username else ""
    return urlunsplit(parts._replace(netloc=f"{credentials}127.0.0.1:{port}"))


@contextmanager
def database_relay(database_url: str, trigger: bytes) -> Generator[tuple[DatabaseRelay, str]]:
    parts: Final = urlsplit(database_url)
    assert parts.hostname is not None and parts.port is not None, database_url
    relay: Final = DatabaseRelay(parts.hostname, parts.port, trigger)
    try:
        relay.start()
        yield relay, _relayed_url(database_url, relay.port)
    finally:
        relay.stop()


@contextmanager
def held_statement_relay(
    database_url: str, trigger: bytes, *, after: bytes | None = None, armed: bool = True
) -> Generator[tuple[HeldStatementRelay, str]]:
    parts: Final = urlsplit(database_url)
    assert parts.hostname is not None and parts.port is not None, database_url
    relay: Final = HeldStatementRelay(parts.hostname, parts.port, trigger, after=after, armed=armed)
    try:
        relay.start()
        yield relay, _relayed_url(database_url, relay.port)
    finally:
        relay.stop()


@contextmanager
def dropped_connection_relay(database_url: str, trigger: bytes) -> Generator[tuple[DroppedConnectionRelay, str]]:
    parts: Final = urlsplit(database_url)
    assert parts.hostname is not None and parts.port is not None, database_url
    relay: Final = DroppedConnectionRelay(parts.hostname, parts.port, trigger)
    try:
        relay.start()
        yield relay, _relayed_url(database_url, relay.port)
    finally:
        relay.stop()


@contextmanager
def statement_counting_relay(database_url: str) -> Generator[tuple[StatementCountingRelay, str]]:
    parts: Final = urlsplit(database_url)
    assert parts.hostname is not None and parts.port is not None, database_url
    relay: Final = StatementCountingRelay(parts.hostname, parts.port)
    relay.start()
    try:
        yield relay, plaintext_database_url(_relayed_url(database_url, relay.port))
    finally:
        relay.stop()
