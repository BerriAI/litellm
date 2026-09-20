"""The shapes a port can have, and what the host gives every one of them.

A port is a frozen class that holds its `Config` and names one of the protocols below as
its base. Declaring the base is what holds it to the shape, at the class rather than at some
call site far away; a method that drifts fails there, and a missing one fails wherever the
port is built.

What every port is held to is its input: contract facts and its `Config`, values all the way
down, never a live object of the call. That is the whole of what the legacy callbacks got
wrong -- they are handed the caller's own objects and mutate them, and the host carries an
obligation to keep that aliasing intact -- and it is what makes a port testable, relocatable
and safe to run anywhere.

What a port does with them is its own business, and the protocols differ only there. A
vendor reached by one fire-and-forget request fits `SinkPort`, which hands back `Delivery`
values for the host to send: the strictest shape, worth taking whenever it is free, because
a pure function over JSON can be governed at its egress, moved out of the process or
rewritten in Rust. A vendor that needs its own client, a token refresh, a retry decided on
what came back, or an SDK does not fit that and is not bent into it -- a `Delivery` never
sees a response, so bending loses the behaviour, not the I/O. It declares `ExporterPort` and
talks to its vendor itself. Doing I/O was never the legacy problem.

Both observing shapes get the same two things from the host, which are the two the contract
does not give an observer yet (`off_path_delivery` and `terminal_model` in `manifest.py`): a
`CallRecord` joined from one call's envelopes, and a thread that is not the call's.

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


class ExporterPort(Protocol):
    """A port that observes finished calls and reaches its vendor itself.

    `export` gets a batch of finished calls on the host's outbox thread, never on the call's,
    and may do whatever reaching the vendor takes: its own client, a session, a token
    refresh, a decision made on what came back. It may raise, and the host logs and drops the
    batch, as the legacy batch loggers do. Nothing filters before it, so a port that wants
    only some calls filters inside `export`.

    `SinkPort` is the same job under a stricter shape. Take that one where the vendor fits
    it, and this one rather than bend a vendor that does not.
    """

    def export(self, batch: tuple[CallRecord, ...], /) -> None: ...

    @property
    def batching(self) -> Batching: ...


class InterceptorPort(Protocol):
    """A port that patches the wire request before it is sent; its failure fails the call.

    `payload` is the patch the request needs, or `None` to leave it alone. It decides
    whether the request is one this port acts on at all: the host applies what it returns
    and filters nothing.
    """

    def payload(self, request: RequestFactsV1, /) -> WirePatchV1 | None: ...
