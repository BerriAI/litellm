from __future__ import annotations

import asyncio
import shlex
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Final

from pydantic import TypeAdapter
from redis import Redis

from tests.integration._support.async_server import LoopbackServer, close_writer
from tests.integration._support.redis_process import free_loopback_port

MONITOR_RECORD: Final = TypeAdapter(dict[str, object])
CRLF: Final = b"\r\n"


@dataclass(frozen=True, slots=True)
class Command:
    arguments: tuple[str, ...]
    raw: bytes


def _line_end(buffer: bytes, start: int) -> int | None:
    end: Final = buffer.find(CRLF, start)
    return None if end < 0 else end


def _bulk(buffer: bytes, start: int) -> tuple[str, int] | None:
    end: Final = _line_end(buffer, start)
    if end is None or buffer[start : start + 1] != b"$":
        return None
    data_start: Final = end + 2
    data_end: Final = data_start + int(buffer[start + 1 : end])
    if len(buffer) < data_end + 2:
        return None
    return buffer[data_start:data_end].decode(errors="replace"), data_end + 2


def _bulks(buffer: bytes, start: int, count: int) -> tuple[tuple[str, ...], int] | None:
    if count == 0:
        return (), start
    head: Final = _bulk(buffer, start)
    if head is None:
        return None
    rest: Final = _bulks(buffer, head[1], count - 1)
    if rest is None:
        return None
    return (head[0], *rest[0]), rest[1]


def _command(buffer: bytes, start: int) -> tuple[Command, int] | None:
    end: Final = _line_end(buffer, start)
    if end is None:
        return None
    if buffer[start : start + 1] != b"*":
        return Command((), buffer[start : end + 2]), end + 2
    parsed: Final = _bulks(buffer, end + 2, int(buffer[start + 1 : end]))
    if parsed is None:
        return None
    return Command(parsed[0], buffer[start : parsed[1]]), parsed[1]


def split_commands(buffer: bytes) -> tuple[tuple[Command, ...], bytes]:
    parsed: Final = _command(buffer, 0)
    if parsed is None:
        return (), buffer
    command, consumed = parsed
    rest, remainder = split_commands(buffer[consumed:])
    return (command, *rest), remainder


@dataclass(frozen=True, slots=True)
class Interception:
    key: str
    commands: frozenset[str]
    action: Callable[[], None]
    label: str


class RedisRelay:
    def __init__(self, host: str, port: int) -> None:
        self.port: Final = free_loopback_port()
        self._host: Final = host
        self._upstream_port: Final = port
        self._server: Final = LoopbackServer(self.port, self._serve)
        self._lock: Final = threading.Lock()
        self._armed: tuple[Interception, ...] = ()
        self.fired: Final = SimpleQueue[str]()

    def intercept(self, interception: Interception) -> None:
        with self._lock:
            self._armed = (*self._armed, interception)

    def pending(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(interception.label for interception in self._armed)

    def clear(self) -> None:
        with self._lock:
            self._armed = ()

    def drain_fired(self) -> tuple[str, ...]:
        return tuple(self.fired.get_nowait() for _ in range(self.fired.qsize()))

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._server.stop()

    def _matched(self, command: Command) -> Interception | None:
        with self._lock:
            if not self._armed or len(command.arguments) < 2:
                return None
            head: Final = self._armed[0]
            if command.arguments[0].upper() not in head.commands or command.arguments[1] != head.key:
                return None
            self._armed = self._armed[1:]
            return head

    async def _serve(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        server_reader, server_writer = await asyncio.open_connection(self._host, self._upstream_port)

        async def forward_commands() -> None:
            pending = b""  # rebind-ok: carries the bytes of a command split across reads
            try:
                while chunk := await client_reader.read(65536):
                    commands, pending = split_commands(pending + chunk)  # rebind-ok: the unparsed tail rolls over
                    for command in commands:
                        matched: Final = self._matched(command)
                        if matched is not None:
                            await asyncio.get_running_loop().run_in_executor(None, matched.action)
                            self.fired.put(f"{matched.label}:{command.arguments[0].upper()}")
                        server_writer.write(command.raw)
                        await server_writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                return
            finally:
                await close_writer(server_writer)

        async def forward_replies() -> None:
            try:
                while chunk := await server_reader.read(65536):
                    client_writer.write(chunk)
                    await client_writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                return
            finally:
                await close_writer(client_writer)

        await asyncio.gather(forward_commands(), forward_replies())


@contextmanager
def redis_relay(host: str, port: int) -> Iterator[RedisRelay]:
    relay: Final = RedisRelay(host, port)
    try:
        relay.start()
        yield relay
    finally:
        relay.stop()


@contextmanager
def monitored_commands(host: str, port: int) -> Iterator[SimpleQueue[str]]:
    commands: Final = SimpleQueue[str]()
    started: Final = threading.Event()
    armed: Final = threading.Event()
    stopped: Final = threading.Event()
    ready_marker: Final = f"monitor-ready-{uuid.uuid4().hex}"
    stop_marker: Final = f"monitor-stop-{uuid.uuid4().hex}"

    def capture() -> None:
        with Redis(host=host, port=port, decode_responses=True) as client:
            with client.monitor() as monitor:
                started.set()
                stream: Final = iter(monitor.listen())
                while not stopped.is_set():
                    record: Final = MONITOR_RECORD.validate_python(next(stream))
                    command: Final = record.get("command")
                    if not isinstance(command, str):
                        continue
                    commands.put(command)
                    if ready_marker in command:
                        armed.set()

    thread: Final = threading.Thread(target=capture, daemon=True)
    thread.start()
    try:
        assert started.wait(timeout=5), "Redis MONITOR did not start"
        with Redis(host=host, port=port, socket_timeout=1, decode_responses=True) as client:
            client.set(ready_marker, "ready", ex=1)
        assert armed.wait(timeout=5), "Redis MONITOR did not capture its readiness command"
        yield commands
    finally:
        stopped.set()
        with Redis(host=host, port=port, socket_timeout=1, decode_responses=True) as client:
            client.set(stop_marker, "stop", ex=1)
        thread.join(timeout=5)
        assert not thread.is_alive(), "Redis MONITOR thread survived cleanup"


def commands_mentioning(commands: SimpleQueue[str], needle: str) -> tuple[tuple[str, ...], ...]:
    captured: Final = tuple(commands.get_nowait() for _ in range(commands.qsize()))
    return tuple(tuple(shlex.split(line)) for line in captured if needle in line)
