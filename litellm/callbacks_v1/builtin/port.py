"""The shape every port has, and nothing that can act.

A port is a frozen class that holds its `Config` and names one of the protocols below as
its base: pure methods from contract values to what the port produces. Declaring the base
is what holds it to the shape, at the class rather than at some call site far away; a
method that drifts fails there, and a missing one fails wherever the port is built. It opens no socket,
starts no thread and reads neither the environment nor the clock; the host hands it all of
those as values. That is the shape a sandboxed custom callback is confined to, and
built-ins take it first so both share one interface.

A port has no identity of its own. It is not named, does not know which events reach it
and implements nothing of the v1 contract: `runtime.py` names it and wraps it into a
subscriber, so the same port configured twice is two registrations rather than a clash.
Nothing in this file is migration scaffolding; `manifest.py` holds all of that.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeVar

from litellm.callbacks_v1 import (
    CallFailedV1,
    CallStartedV1,
    CallSucceededV1,
    RequestFactsV1,
    RequestSendingV1,
    WirePatchV1,
)

R = TypeVar("R")  # rebind-ok: a TypeVar must bind to a bare name to be recognised as one


@dataclass(frozen=True, slots=True)
class Delivery:
    """One request to a vendor, fully decided; sending it is the only thing left."""

    url: str
    headers: tuple[tuple[str, str], ...]
    body: bytes


@dataclass(frozen=True, slots=True)
class CallRecord:
    """Every envelope of one finished call a sink needs, joined by `call_id`.

    `request` is absent when the call failed before anything was sent.
    """

    call_id: str
    call_type: str
    started: CallStartedV1 | None
    request: RequestSendingV1 | None
    terminal: CallSucceededV1 | CallFailedV1


@dataclass(frozen=True, slots=True)
class Batching:
    """How many records the host carries to the vendor at once, and how long it waits."""

    size: int = 1
    interval: float = 5.0


class SinkPort(Protocol[R]):
    """A port that observes finished calls and tells a vendor.

    `payload` is the port: one finished call and the current time to the vendor's record,
    or `None` for nothing to send, which is also how a port filters. `deliveries` frames a
    batch of those records as requests, headers and framing included, and `batching` says
    how large a batch the host should collect. The two are separate because the handler
    runs inline in the call and the delivery does not.
    """

    def payload(self, call: CallRecord, now: datetime, /) -> R | None: ...

    def deliveries(self, batch: tuple[R, ...], /) -> tuple[Delivery, ...]: ...

    @property
    def batching(self) -> Batching: ...


class InterceptorPort(Protocol):
    """A port that patches the wire request before it is sent; its failure fails the call.

    `payload` is the patch the request needs, or `None` to leave it alone. It decides
    whether the request is one this port acts on at all: the host applies what it returns
    and filters nothing.
    """

    def payload(self, request: RequestFactsV1, /) -> WirePatchV1 | None: ...
