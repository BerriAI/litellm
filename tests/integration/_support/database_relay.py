import asyncio
import socket
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from pydantic import TypeAdapter

PORT: Final = TypeAdapter(int)


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
        if self.tripped.is_set() and self.refused < 5:
            self.refused += 1
            client_writer.close()
            return
        server_reader, server_writer = await asyncio.open_connection(self._upstream_host, self._upstream_port)
        self._writers = (*self._writers, client_writer, server_writer)

        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, inspect: bool) -> None:
            try:
                while chunk := await reader.read(65536):
                    if inspect and self._armed.is_set() and not self.tripped.is_set() and self._trigger in chunk:
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


@contextmanager
def database_relay(database_url: str, trigger: bytes) -> Generator[tuple[DatabaseRelay, str]]:
    parts: Final = urlsplit(database_url)
    assert parts.hostname is not None and parts.port is not None, database_url
    relay: Final = DatabaseRelay(parts.hostname, parts.port, trigger)
    relay.start()
    credentials: Final = f"{parts.username}:{parts.password}@" if parts.username else ""
    relayed: Final = urlunsplit(parts._replace(netloc=f"{credentials}127.0.0.1:{relay.port}"))
    try:
        yield relay, relayed
    finally:
        relay.stop()
