"""The extension host: a process that runs v1 callbacks for a gateway that does not.

    python -m litellm.callbacks_v1.host --fd 3 --load my_pkg.audit --load my_pkg.cb:callback

The gateway never imports a callback. It starts (or attaches to) this worker, which imports
the callbacks exactly as an SDK process would, reports what they subscribed to, and then
only answers: the gateway sends envelopes and interception requests over one socket, the
worker runs the handlers and sends back patches. The worker cannot call into the gateway;
there is no frame for it. A callback file is the same file in the SDK and here.

This module isolates nothing by itself. What the worker can reach (environment, files,
network) is whatever the process it runs in was given; confinement is the spawner's job.

Frames are a 4-byte big-endian length and that many bytes of JSON, one object with a
`type`. `litellm-rust/crates/callbacks-v1-exthost/golden` pins every frame for both sides.

    worker -> gateway   hello      {schema, subscriptions: [{name, events, observes, intercepts}]}
    gateway -> worker   event      {subscriber, envelope}                       no reply
    gateway -> worker   intercept  {id, subscriber, request}
    worker -> gateway   patch      {id, patch}     or     error {id, error_class, message}
    worker -> gateway   report     {subscriber, event, message}     an observer raised
    gateway -> worker   flush      {id}
    worker -> gateway   flushed    {id}            every event sent before the flush was handled

`subscriber` is an index into the hello's `subscriptions`. Envelopes reach one subscriber
in the order they were sent; an interception never waits behind events.

A handler is scheduled by the kind its author declared, which is the author's decision and
not this module's guess. An `async` handler is a task on the worker's loop, so one that only
waits costs no thread and concurrency is the loop's rather than a pool size; a sync handler
gets a thread, because running it blocks one, and the loop keeps turning while it does. Every
frame leaves through one writer, so no handler and no loop callback blocks on the socket.
"""

import argparse
import asyncio
import importlib
import json
import queue
import socket
import sys
import threading
from collections.abc import Coroutine, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias

from typing_extensions import ReadOnly, TypedDict

from litellm.callbacks_v1 import SCHEMA, JSONValue, SchemaVersion, Subscriber, register, snapshot

MAX_FRAME_BYTES: Final = 64 * 1024 * 1024
INTERCEPT_THREADS: Final = 8
OBSERVER_QUEUE: Final = 1024


class Handler(Protocol):
    """A handler the worker calls on a thread, because running it blocks one."""

    def __call__(self, argument: JSONValue, /) -> object: ...


class AsyncHandler(Protocol):
    """A handler the worker awaits on its loop, because waiting is all it does."""

    def __call__(self, argument: JSONValue, /) -> Coroutine[object, object, object]: ...


class SubscriptionData(TypedDict):
    name: ReadOnly[str]
    events: ReadOnly[Sequence[str]]
    observes: ReadOnly[bool]
    intercepts: ReadOnly[bool]


class HelloFrame(TypedDict):
    type: ReadOnly[Literal["hello"]]
    schema: ReadOnly[SchemaVersion]
    subscriptions: ReadOnly[Sequence[SubscriptionData]]


class PatchFrame(TypedDict):
    type: ReadOnly[Literal["patch"]]
    id: ReadOnly[JSONValue]
    patch: ReadOnly[JSONValue]


class ErrorFrame(TypedDict):
    type: ReadOnly[Literal["error"]]
    id: ReadOnly[JSONValue]
    error_class: ReadOnly[str]
    message: ReadOnly[str]


class ReportFrame(TypedDict):
    type: ReadOnly[Literal["report"]]
    subscriber: ReadOnly[int]
    event: ReadOnly[JSONValue]
    message: ReadOnly[str]


class FlushedFrame(TypedDict):
    type: ReadOnly[Literal["flushed"]]
    id: ReadOnly[JSONValue]


Sent: TypeAlias = HelloFrame | PatchFrame | ErrorFrame | ReportFrame | FlushedFrame
Received: TypeAlias = Mapping[str, JSONValue]


class ProtocolError(Exception):
    """The peer sent something that is not a frame of this protocol; the worker stops."""


def _read_exactly(sock: socket.socket, size: int) -> bytes | None:
    """`size` bytes, or `None` when the peer closed before the first of them."""
    chunks: Final[list[bytes]] = []  # mutable-ok: a socket yields a frame in pieces
    remaining = size  # rebind-ok: counts down as the pieces arrive
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))  # rebind-ok: one piece per pass
        if not chunk:
            if chunks:
                raise ProtocolError("connection closed inside a frame")
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> Received | None:
    """The next frame, or `None` once the peer has closed the connection."""
    header: Final = _read_exactly(sock, 4)
    if header is None:
        return None
    size: Final = int.from_bytes(header, "big")
    if size > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame of {size} bytes exceeds {MAX_FRAME_BYTES}")
    body: Final = _read_exactly(sock, size)
    if body is None:
        raise ProtocolError("connection closed inside a frame")
    frame: Final[JSONValue] = json.loads(body)  # pyright: ignore[reportAny]  # untyped until the checks below
    if not isinstance(frame, Mapping) or not isinstance(frame.get("type"), str):
        raise ProtocolError("a frame is a JSON object with a string `type`")
    return frame


def encode_frame(frame: Sent | Received) -> bytes:
    body: Final = json.dumps(frame, separators=(",", ":")).encode()
    return len(body).to_bytes(4, "big") + body


def load(reference: str) -> None:
    """Imports `module`, whose decorators register as they run, or registers `module:attr`."""
    module_name, _, attribute = reference.partition(":")
    module: Final = importlib.import_module(module_name)
    if attribute:
        register(getattr(module, attribute))  # pyright: ignore[reportAny]  # the attribute is whatever the operator named; `register` validates it


def _subscription(subscriber: Subscriber) -> SubscriptionData:
    data: Final[SubscriptionData] = {
        "name": subscriber.name,
        "events": tuple(sorted(subscriber.events)),
        "observes": subscriber.on_event is not None or subscriber.async_on_event is not None,
        "intercepts": subscriber.before_send is not None or subscriber.async_before_send is not None,
    }
    return data


def hello(subscribers: Sequence[Subscriber]) -> HelloFrame:
    frame: Final[HelloFrame] = {
        "type": "hello",
        "schema": SCHEMA,
        "subscriptions": tuple(_subscription(subscriber) for subscriber in subscribers),
    }
    return frame


def _plain(value: object) -> JSONValue:
    """A handler's return value as JSON can carry it; callbacks may build patches from read-only views."""
    if isinstance(value, Mapping):
        # A handler's return value is untyped until it is encoded, and json.dumps wants a dict.
        return {str(k): _plain(v) for k, v in value.items()}  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # mutable-ok: json.dumps encodes dict, not Mapping
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_plain(item) for item in value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # a handler's return value is untyped until it is encoded
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"a callback returned {type(value).__name__}, which JSON cannot carry")


def _sync(handler: object | None) -> Handler | None:
    """A handler the worker calls on a thread; `register` has already checked its kind."""
    return handler if callable(handler) else None


def _asynchronous(handler: object | None) -> AsyncHandler | None:
    """A handler the worker awaits on its loop; `register` has already checked its kind."""
    return handler if callable(handler) else None  # pyright: ignore[reportReturnType]  # `register` validated the kind and `callable` narrows no further


@dataclass(frozen=True, slots=True)
class _Marker:
    """A flush's place in one lane's queue, resolved once everything queued before it is handled."""

    reached: Future[None]


@dataclass(frozen=True, slots=True)
class _Stop:
    """The end of one lane's queue; the consumer returns once everything before it is handled."""


_Queued: TypeAlias = JSONValue | _Marker | _Stop


@dataclass(frozen=True, slots=True)
class _Lane:
    """One subscriber's handlers, each kept in the kind its author declared.

    The kind is the author's scheduling decision and the worker routes on it: a handler that
    waits on the network is `async` and costs no thread, one that occupies the CPU is sync
    and gets a thread so that the loop keeps turning while it runs. Collapsing the two would
    discard that decision, and it cannot be recovered by calling a handler to see what comes
    back: calling a sync one has already run it, on whatever thread asked. `events` is the
    thread a sync observer's envelopes are handled on, in order, and exists only for one.
    """

    observe: Handler | None
    async_observe: AsyncHandler | None
    intercept: Handler | None
    async_intercept: AsyncHandler | None
    events: ThreadPoolExecutor | None


class Worker:
    """Serves one gateway connection over the subscribers registered when it was built.

    Nothing that waits holds a thread it does not need. An async handler is a task on the
    worker's loop, so concurrency is the loop's and not a pool size; a sync handler has a
    thread because it blocks one; and every frame leaves through one writer, so no handler
    and no loop callback ever blocks on the socket.
    """

    def __init__(self, sock: socket.socket, subscribers: Sequence[Subscriber]) -> None:
        self._sock: Final = sock
        self._subscribers: Final = tuple(subscribers)
        self._loop: Final = asyncio.new_event_loop()
        self._outgoing: Final[queue.Queue[bytes | None]] = queue.Queue()
        self._writer: Final = threading.Thread(target=self._write, name="callback-writer", daemon=True)
        self._intercepts: Final = ThreadPoolExecutor(INTERCEPT_THREADS, thread_name_prefix="callback-intercept")
        self._queues: Final[
            dict[int, asyncio.Queue[_Queued]]
        ] = {}  # mutable-ok: filled once on the loop before serving
        self._consumers: Final[list[asyncio.Task[None]]] = []  # mutable-ok: filled once on the loop before serving
        self._pending: Final[set[asyncio.Task[None]]] = (
            set()
        )  # mutable-ok: the loop's strong references to its live tasks
        self._lanes: Final = tuple(
            _Lane(
                _sync(subscriber.on_event),
                _asynchronous(subscriber.async_on_event),
                _sync(subscriber.before_send),
                _asynchronous(subscriber.async_before_send),
                ThreadPoolExecutor(1, thread_name_prefix=f"callback-events-{index}")
                if subscriber.on_event is not None and subscriber.async_on_event is None
                else None,
            )
            for index, subscriber in enumerate(self._subscribers)
        )

    def _write(self) -> None:
        """The only thread that writes to the socket, so that no caller ever blocks on it."""
        for data in iter(self._outgoing.get, None):
            try:
                self._sock.sendall(data)
            except OSError:
                return  # the gateway is gone; the read loop ends on its own

    def _send(self, frame: Sent) -> None:
        self._outgoing.put(encode_frame(frame))

    def _report(self, index: int, envelope: JSONValue, error: BaseException) -> ReportFrame:
        event: Final = envelope.get("event") if isinstance(envelope, Mapping) else None
        kind: Final = event.get("type") if isinstance(event, Mapping) else None
        frame: Final[ReportFrame] = {"type": "report", "subscriber": index, "event": kind, "message": str(error)}
        return frame

    def _patched(self, frame_id: JSONValue, result: object) -> PatchFrame:
        frame: Final[PatchFrame] = {"type": "patch", "id": frame_id, "patch": _plain(result)}
        return frame

    def _failed(self, frame_id: JSONValue, error: BaseException) -> ErrorFrame:
        name: Final = f"{type(error).__module__}.{type(error).__qualname__}"
        frame: Final[ErrorFrame] = {"type": "error", "id": frame_id, "error_class": name, "message": str(error)}
        return frame

    def _observe(self, index: int, handler: Handler, envelope: JSONValue) -> None:
        """One sync observer on its lane's thread, in the order its envelopes arrived."""
        try:
            handler(envelope)
        except Exception as error:  # noqa: BLE001  # an observer's failure is reported and swallowed, by contract
            self._send(self._report(index, envelope, error))

    async def _consume(self, index: int, handler: AsyncHandler, envelopes: "asyncio.Queue[_Queued]") -> None:
        """One async observer's whole lane: one item at a time, so order holds without a thread."""
        while True:
            item: _Queued = await envelopes.get()
            if isinstance(item, _Stop):
                return
            if isinstance(item, _Marker):
                item.reached.set_result(None)
                continue
            try:
                await handler(item)
            except Exception as error:  # noqa: BLE001  # an observer's failure is reported and swallowed, by contract
                self._send(self._report(index, item, error))

    def _intercept(self, frame_id: JSONValue, handler: Handler, request: JSONValue) -> None:
        """One sync interceptor on a pool thread, which it holds for as long as it runs."""
        try:
            self._send(self._patched(frame_id, handler(request)))
        except Exception as error:  # noqa: BLE001  # an interceptor's failure is the gateway's to act on
            self._send(self._failed(frame_id, error))

    async def _await_intercept(self, frame_id: JSONValue, handler: AsyncHandler, request: JSONValue) -> None:
        """One async interceptor as a task, so that a thousand of them wait on one thread."""
        try:
            self._send(self._patched(frame_id, await handler(request)))
        except Exception as error:  # noqa: BLE001  # an interceptor's failure is the gateway's to act on
            self._send(self._failed(frame_id, error))

    def _keep(self, task: "asyncio.Task[None]") -> None:
        """asyncio keeps only a weak reference to a task; this is the strong one."""
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _spawn(self, frame_id: JSONValue, handler: AsyncHandler, request: JSONValue) -> None:
        """On the loop: starts the interception and returns, so the read loop stays free."""
        self._keep(self._loop.create_task(self._await_intercept(frame_id, handler, request)))

    def _enqueue(self, index: int, envelope: JSONValue) -> None:
        """On the loop: a bounded lane drops rather than grow without limit, and says so."""
        try:
            self._queues[index].put_nowait(envelope)
        except asyncio.QueueFull:
            full: Final = RuntimeError(f"observer queue is full at {OBSERVER_QUEUE}; this envelope was dropped")
            self._send(self._report(index, envelope, full))

    def _mark(self, index: int, reached: "Future[None]") -> None:
        """On the loop: a marker waits for room rather than drop, so it keeps its place."""
        self._keep(self._loop.create_task(self._queues[index].put(_Marker(reached))))

    async def _start(self) -> None:
        """On the loop: one queue and one consumer per async observer, before the first frame."""
        for index, lane in enumerate(self._lanes):
            if lane.async_observe is not None:
                envelopes: asyncio.Queue[_Queued] = asyncio.Queue(OBSERVER_QUEUE)
                self._queues[index] = envelopes
                self._consumers.append(self._loop.create_task(self._consume(index, lane.async_observe, envelopes)))

    async def _drain(self) -> None:
        """On the loop: every lane sees a stop behind what was queued, then every task ends."""
        for envelopes in self._queues.values():
            await envelopes.put(_Stop())
        outstanding: Final = (*self._consumers, *self._pending)
        if outstanding:
            _ = await asyncio.gather(*outstanding, return_exceptions=True)

    def _lane(self, frame: Received) -> tuple[int, _Lane]:
        index: Final = frame.get("subscriber")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self._lanes):
            raise ProtocolError(f"unknown subscriber: {index!r}")
        return index, self._lanes[index]

    def _flush(self, frame_id: JSONValue) -> None:
        """A marker at the back of each lane passes only after everything queued before it."""
        markers: Final[list[Future[None]]] = []  # mutable-ok: one marker per lane, collected and then waited on
        for index, lane in enumerate(self._lanes):
            if index in self._queues:
                reached: Future[None] = Future()
                self._loop.call_soon_threadsafe(self._mark, index, reached)
                markers.append(reached)
            elif lane.events is not None:
                markers.append(lane.events.submit(lambda: None))
        for marker in markers:
            marker.result()
        flushed: Final[FlushedFrame] = {"type": "flushed", "id": frame_id}
        self._send(flushed)

    def _dispatch(self, frame: Received) -> None:
        """Routes one frame and returns; nothing here runs a handler or touches the socket."""
        kind: Final = frame["type"]
        if kind == "event":
            index, observed = self._lane(frame)
            envelope: Final = frame.get("envelope")
            if observed.async_observe is not None:
                self._loop.call_soon_threadsafe(self._enqueue, index, envelope)
            elif observed.observe is not None and observed.events is not None:
                _ = observed.events.submit(self._observe, index, observed.observe, envelope)
        elif kind == "intercept":
            _, asked = self._lane(frame)
            if asked.async_intercept is not None:
                self._loop.call_soon_threadsafe(
                    self._spawn, frame.get("id"), asked.async_intercept, frame.get("request")
                )
            elif asked.intercept is not None:
                _ = self._intercepts.submit(self._intercept, frame.get("id"), asked.intercept, frame.get("request"))
            else:
                unpatched: Final[PatchFrame] = {"type": "patch", "id": frame.get("id"), "patch": None}
                self._send(unpatched)
        elif kind == "flush":
            _ = self._intercepts.submit(self._flush, frame.get("id"))
        else:
            raise ProtocolError(f"unknown frame type: {kind!r}")

    def serve(self) -> None:
        """Says hello, then answers until the gateway closes the connection."""
        threading.Thread(target=self._loop.run_forever, name="callback-loop", daemon=True).start()
        self._writer.start()
        try:
            _ = asyncio.run_coroutine_threadsafe(self._start(), self._loop).result()
            self._send(hello(self._subscribers))
            while (frame := read_frame(self._sock)) is not None:
                self._dispatch(frame)
        finally:
            # In this order: no new flush can start, so every marker still resolves behind a
            # live consumer; then the lanes end; then the writer, once nothing can send again.
            self._intercepts.shutdown(wait=True)
            _ = asyncio.run_coroutine_threadsafe(self._drain(), self._loop).result()
            for lane in self._lanes:
                if lane.events is not None:
                    lane.events.shutdown(wait=True)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._outgoing.put(None)
            self._writer.join()


@dataclass(frozen=True, slots=True)
class Arguments:
    fd: int | None
    connect: str | None
    load: tuple[str, ...]


def _connect(arguments: Arguments) -> socket.socket:
    if arguments.fd is not None:
        return socket.socket(fileno=arguments.fd)
    if arguments.connect is None:
        raise ValueError("one of --fd and --connect is required")
    connected: Final = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connected.connect(arguments.connect)
    return connected


def _option(parsed: Mapping[str, object], name: str) -> object:
    return parsed.get(name)


def _arguments(argv: Sequence[str] | None) -> Arguments:
    parser: Final = argparse.ArgumentParser(prog="python -m litellm.callbacks_v1.host")
    where: Final = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--fd", type=int, help="an inherited, connected socket (the gateway spawned this worker)")
    where.add_argument("--connect", metavar="PATH", help="a unix socket the gateway listens on (a sidecar)")
    parser.add_argument("--load", action="append", metavar="MODULE[:ATTR]", help="a callback to import")
    parsed: Final = vars(parser.parse_args(argv))
    fd: Final = _option(parsed, "fd")
    connect: Final = _option(parsed, "connect")
    load_: Final = _option(parsed, "load")
    return Arguments(
        fd if isinstance(fd, int) else None,
        connect if isinstance(connect, str) else None,
        tuple(str(item) for item in load_) if isinstance(load_, (list, tuple)) else (),  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # argparse collects `--load` untyped
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments: Final = _arguments(argv)
    for reference in arguments.load:
        load(reference)
    with _connect(arguments) as sock:
        Worker(sock, snapshot()).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
