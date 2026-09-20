"""The in-process host of a port: everything a port is not allowed to do itself.

Joining a call's envelopes, reading the clock, batching off the call path and talking to
the vendor all live here, once, so a port stays pure methods over values (`port.py`).
`Sink` and `Patcher` are the two subscribers: each names a port of its kind and wraps it,
and they are the only things in this package that implement the v1 contract, so a port
never writes `name`, `events` or a handler. The name is the caller's because it is the
registration's, not the port's: one port configured twice is two subscribers. This is not a sandbox and isolates nothing: it is plain
in-process Python. It is a stand-in for what a host should own (join, queue, batch,
deliver). When the contract grows off-path delivery and a joined call record, this module
is deleted and the ports do not change. Custom callbacks are isolated by the gateway's
extension host, in Rust.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Generic, Protocol, TypeVar

from litellm.callbacks_v1 import (
    CallStartedV1,
    EnvelopeV1,
    EventName,
    Interceptor,
    Observer,
    RequestFactsV1,
    RequestSendingV1,
    WirePatchV1,
)
from litellm.callbacks_v1.builtin.outbox import Outbox
from litellm.callbacks_v1.builtin.port import CallRecord, Delivery, InterceptorPort, SinkPort

R = TypeVar("R")  # rebind-ok: a TypeVar must bind to a bare name to be recognised as one


class Transport(Protocol):
    def send(self, delivery: Delivery) -> None: ...


class HttpxTransport:
    """Sends through the same client the legacy loggers use, so proxy TLS settings apply."""

    def __init__(self) -> None:
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        self._client: Final = HTTPHandler()

    def send(self, delivery: Delivery) -> None:
        headers: Final = dict(delivery.headers)  # mutable-ok: the httpx client takes its headers as a dict
        self._client.post(url=delivery.url, headers=headers, content=delivery.body)  # pyright: ignore[reportUnknownMemberType]  # the legacy HTTP handler is only partly typed


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class _OpenCall:
    started: CallStartedV1 | None = None
    request: RequestSendingV1 | None = None


JOINED_EVENTS: Final[frozenset[EventName]] = frozenset(
    {"call.started", "request.sending", "call.succeeded", "call.failed"}
)


class CallJoin:
    """Joins a call's envelopes into one `CallRecord` at its terminal envelope.

    The terminal envelope carries neither the model nor the metadata, so a sink has to
    remember the earlier ones. A cancelled call never gets a terminal envelope, which is
    why the open calls are bounded: the oldest is forgotten rather than leaked.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self._capacity: Final = capacity
        self._lock: Final = threading.Lock()
        self._open: Final[dict[str, _OpenCall]] = {}  # mutable-ok: per-call join state is an evolving keyed store

    def accept(self, envelope: EnvelopeV1) -> CallRecord | None:
        call_id: Final = envelope["call_id"]
        event: Final = envelope["event"]
        with self._lock:
            if event["type"] == "call.succeeded" or event["type"] == "call.failed":
                seen: Final = self._open.pop(call_id, _OpenCall())
                return CallRecord(call_id, envelope["call_type"], seen.started, seen.request, event)
            current: Final = self._open.get(call_id, _OpenCall())
            if event["type"] == "call.started":
                self._open[call_id] = _OpenCall(event, current.request)
            elif event["type"] == "request.sending":
                self._open[call_id] = _OpenCall(current.started, event)
            while len(self._open) > self._capacity:
                del self._open[next(iter(self._open))]
        return None

    def open_calls(self) -> int:
        with self._lock:
            return len(self._open)


class Sink(Observer, Generic[R]):
    """The subscriber of every `SinkPort`: the port supplies pure methods, this runs them.

    The handler only joins and enqueues, so the sync one serves async calls too and never
    blocks; the outbox thread calls `deliveries` and sends.
    """

    def __init__(
        self,
        name: str,
        port: SinkPort[R],
        transport: Transport,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.name = name
        self.events = JOINED_EVENTS
        self._port: Final = port
        self._transport: Final = transport
        self._clock: Final = clock
        self._calls: Final = CallJoin()
        self.outbox: Final[Outbox[R]] = Outbox(
            name, self._deliver, batch_size=port.batching.size, flush_interval=port.batching.interval
        )

    def _deliver(self, batch: tuple[R, ...]) -> None:
        for delivery in self._port.deliveries(batch):
            self._transport.send(delivery)

    def on_event(self, event: EnvelopeV1) -> None:
        call: Final = self._calls.accept(event)
        if call is None:
            return
        record: Final = self._port.payload(call, self._clock())
        if record is not None:
            self.outbox.put(record)

    def open_calls(self) -> int:
        return self._calls.open_calls()


class Patcher(Interceptor):
    """The subscriber of every `InterceptorPort`: the contract's `before_send` over `payload`.

    An interceptor runs on the call path by definition, so there is nothing to join, batch
    or defer here; this exists so that an interceptor port declares as little as a sink does.
    """

    def __init__(self, name: str, port: InterceptorPort) -> None:
        self.name = name
        self.events: frozenset[EventName] = frozenset()
        self._port: Final = port

    def before_send(self, request: RequestFactsV1) -> WirePatchV1 | None:
        return self._port.payload(request)
