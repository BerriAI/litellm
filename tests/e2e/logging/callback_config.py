"""Turn a logging integration on for the duration of a test, then put the proxy
back the way it was found.

A litellm logging integration is a process-wide callback, not a per-request
option, so a delivery test can only assert against a proxy that has the
integration registered for the event it cares about. Rather than depend on how
the proxy under test happened to be launched, a test declares what it needs:
``callback_enabled`` reads the registered callbacks back from
/get/config/callbacks, registers the missing one through /config/update, waits
until the proxy reports it, and unregisters exactly what it registered on the
way out. A proxy that already ships the integration is left untouched, so this
is a no-op wherever the destination is already wired into the deployment.

Success and failure are separate registrations in litellm and are removed by
separate routes: /config/update unions into success_callback (callbacks are
additive there, so a shorter list cannot remove one) and /config/callback/delete
is the only way back out, while failure_callback is written wholesale. Every
write here is therefore a read-modify-write of the list as it stands at that
moment, adding or dropping just this caller's entry. Restoring a snapshot taken
at setup would unregister whatever a concurrently running test had registered in
the meantime, which on a shared proxy is a real way to break someone else's run.

A read-modify-write is still not atomic, and nothing available here can make it
so: litellm exposes a whole-list write and a server-side read-remove-write, with
no per-entry update and no conditional write, so there is no compare-and-set to
build ownership on, and a reference count would be one more read-then-write over
the same shared state. What is available is detection. Every write re-reads the
list afterwards and fails, naming the entries, if anything that was registered
before it and belongs to somebody else has gone, which turns a silent change to
a shared proxy's logging configuration into a diagnosable failure. Entries that
appear only after a write are somebody else's later registration, not damage,
and are left alone.

All of this assumes one sequential pytest session per proxy, which is what the
rest of the harness assumes anyway; the session-scoped proxy fixture and the
destructive spend-log truncate in the root conftest are both single-session by
construction. Run against a shared proxy under xdist the detection still fires,
but it reports the collision rather than preventing it.

/get/config/callbacks also returns each integration's resolved credentials. The
model here keeps only the name and the event type, so no secret is ever parsed
into a test.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Literal, assert_never

import pytest
from pydantic import BaseModel, ConfigDict

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT
from e2e_http import NoBody, unwrap
from logging_client import LoggingClient

type CallbackEvent = Literal["success", "failure"]


def _matching_types(event: CallbackEvent) -> frozenset[str]:
    """The /get/config/callbacks ``type`` values that register for ``event``:
    litellm_settings.callbacks registers for both and is reported as
    success_and_failure."""
    match event:
        case "success":
            return frozenset({"success", "success_and_failure"})
        case "failure":
            return frozenset({"failure", "success_and_failure"})
        case _:
            assert_never(event)


class _ConfiguredCallback(BaseModel):
    """One /get/config/callbacks entry. The route also returns the
    integration's resolved credentials under ``variables``; leaving them
    unmodelled keeps them out of the test process."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: str


class _ConfiguredCallbacks(BaseModel):
    model_config = ConfigDict(extra="ignore")

    callbacks: list[_ConfiguredCallback] = []


class _CallbackLists(BaseModel):
    """The litellm_settings slice /config/update needs. Only the list for the
    event being changed is sent; the other stays None and is dropped."""

    success_callback: list[str] | None = None
    failure_callback: list[str] | None = None


class _ConfigUpdateBody(BaseModel):
    litellm_settings: _CallbackLists


class _ConfigUpdateAck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str


class _CallbackDeleteBody(BaseModel):
    callback_name: str


class _CallbackDeleteAck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    removed_callback: str


def registered_callbacks(client: LoggingClient, event: CallbackEvent) -> tuple[str, ...]:
    """The integrations the proxy reports as registered for ``event``."""
    reported = unwrap(
        client.proxy.transport.get(
            "/get/config/callbacks",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=_ConfiguredCallbacks,
        )
    )
    wanted = _matching_types(event)
    return tuple(callback.name for callback in reported.callbacks if callback.type in wanted)


def _write_callbacks(client: LoggingClient, event: CallbackEvent, names: tuple[str, ...]) -> None:
    settings = (
        _CallbackLists(success_callback=list(names))
        if event == "success"
        else _CallbackLists(failure_callback=list(names))
    )
    ack = unwrap(
        client.proxy.transport.post(
            "/config/update",
            headers=client.proxy.transport.master,
            json=_ConfigUpdateBody(litellm_settings=settings),
            response_type=_ConfigUpdateAck,
        )
    )
    assert "success" in ack.message.lower(), f"POST /config/update must acknowledge the write; got {ack.message!r}"


#: Why a vanished entry is worth stopping for, appended to both loss reports.
_LOST_ENTRIES = (
    "the callback list has no atomic per-entry update, so a registration made between this "
    "read-modify-write's read and its write is overwritten rather than merged; those entries are "
    "gone from the proxy's configuration and anything relying on them is now logged differently"
)


def _register(client: LoggingClient, name: str, event: CallbackEvent) -> None:
    before = registered_callbacks(client, event)
    _write_callbacks(client, event, before + (name,))
    deadline = time.monotonic() + POLL_TIMEOUT
    while True:
        settled = frozenset(registered_callbacks(client, event))
        if name in settled:
            lost = frozenset(before) - settled
            if lost:
                pytest.fail(
                    f"registering {name!r} for {event} calls dropped {sorted(lost)}: {_LOST_ENTRIES}"
                )
            return
        if time.monotonic() >= deadline:
            pytest.fail(f"the proxy never reported the {name!r} callback registered for {event} calls")
        time.sleep(POLL_INTERVAL)


def _unregister(client: LoggingClient, name: str, event: CallbackEvent) -> frozenset[str]:
    """Drop only this test's own entry, and report anything else that vanished.

    The list is re-read here and the name filtered out of what is registered
    *now*, never replaced with the snapshot taken at setup: the proxy is shared,
    so writing back a stale whole list would silently unregister a callback
    something else added in between. The re-read afterwards catches the case that
    remains, where something registered inside the window between this read and
    this write and the write overwrote it. Returns those lost names rather than
    failing, so the caller can finish unregistering every event before it
    reports; leaving half the events registered would be a worse outcome than
    the loss being reported one moment later."""
    before = registered_callbacks(client, event)
    if event == "failure":
        _write_callbacks(client, event, tuple(n for n in before if n != name))
    else:
        _ = client.proxy.transport.post(
            "/config/callback/delete",
            headers=client.proxy.transport.master,
            json=_CallbackDeleteBody(callback_name=name),
            response_type=_CallbackDeleteAck,
        )
    return (frozenset(before) - {name}) - frozenset(registered_callbacks(client, event))


def callback_enabled(client: LoggingClient, name: str, *, events: tuple[CallbackEvent, ...]) -> Iterator[None]:
    """Guarantee ``name`` is registered for every event in ``events`` while the
    generator is suspended, and no longer registered for the ones this call
    added once it resumes. Every write is a read-modify-write of the live list
    followed by a re-read, so concurrent registrations by other tests survive,
    and one lost to the gap between the read and the write is reported instead
    of disappearing. Registration happens inside the try, so a failure part way
    through a multi-event registration still unregisters what it managed to add.
    Drive it from a fixture with ``yield from``."""
    added: tuple[CallbackEvent, ...] = tuple(
        event for event in events if name not in registered_callbacks(client, event)
    )
    try:
        for event in added:
            _register(client, name, event)
        yield
    finally:
        lost = tuple(
            f"{event}/{missing}"
            for event in added
            for missing in sorted(_unregister(client, name, event))
        )
        if lost:
            pytest.fail(f"unregistering {name!r} dropped {list(lost)}: {_LOST_ENTRIES}")
