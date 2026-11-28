"""Process-local registry of requests the proxy is serving right now.

Each worker tracks its own in-flight requests so an operator can list them and
cancel a stuck one without dropping the worker. The registry is deliberately
in-memory and per-process: a cancel reaches the asyncio task owning the
upstream connection, and closing that connection is what makes the provider
stop generating. Cross-worker fan-out (Redis pub/sub) is a follow-up; on
multi-worker deployments call the worker serving the request.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

InFlightState: TypeAlias = Literal["running", "cancelling"]


@dataclass(slots=True, frozen=True)
class InFlightEntry:
    handle: str
    request_id: str
    started_at: float
    model: str
    user_id: str | None
    key_alias: str | None
    is_streaming: bool
    task: asyncio.Task[object]
    state: InFlightState = "running"


_LOCK: Final = threading.Lock()
_ITEMS: Final[dict[str, InFlightEntry]] = {}  # mutable-ok: live request registry, keyed by handle, guarded by _LOCK


def register(
    request_id: str,
    task: asyncio.Task[object] | None,
    *,
    model: str,
    user_id: str | None = None,
    key_alias: str | None = None,
    is_streaming: bool = False,
) -> str | None:
    """Track the current request and return its cancellation handle, or None
    when nothing was registered. A missing request_id or task registers
    nothing: there would be no way to attribute or cancel the entry.

    The handle is server-generated and unique, unlike ``request_id`` (the
    client-supplied ``litellm_call_id``): two concurrent requests may share a
    correlation id, but never a handle, so an entry can always be addressed and
    is never overwritten by a later registration."""
    if not request_id or task is None:
        return None
    handle: Final = uuid.uuid4().hex
    entry: Final = InFlightEntry(
        handle=handle,
        request_id=request_id,
        started_at=time.time(),
        model=model,
        user_id=user_id,
        key_alias=key_alias,
        is_streaming=is_streaming,
        task=task,
    )
    with _LOCK:
        _ITEMS[handle] = entry
    return handle


def unregister_if_owned(handle: str, task: asyncio.Task[object]) -> None:
    """Remove the entry for ``handle`` only while it still belongs to ``task``.
    A completion callback must never drop an entry another request registered."""
    with _LOCK:
        entry: Final = _ITEMS.get(handle)
        if entry is not None and entry.task is task:
            del _ITEMS[handle]


def snapshot() -> tuple[MappingProxyType[str, object], ...]:
    """Running and cancelling requests, oldest first, without task handles."""
    with _LOCK:
        entries: Final = sorted(_ITEMS.values(), key=lambda entry: entry.started_at)
        return tuple(
            MappingProxyType(
                {
                    "handle": entry.handle,
                    "request_id": entry.request_id,
                    "started_at": entry.started_at,
                    "model": entry.model,
                    "user_id": entry.user_id,
                    "key_alias": entry.key_alias,
                    "is_streaming": entry.is_streaming,
                    "state": entry.state,
                }
            )
            for entry in entries
        )


def cancel(handles: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Cancel running requests by handle, or every running one when handles is None.

    Marks the entry cancelling under the lock, then cancels the tasks outside
    it. Returns the handles actually cancelled; unknown, already-cancelling and
    finished requests are skipped."""
    wanted: Final = set(handles) if handles is not None else None
    targets: Final[list[InFlightEntry]] = []  # mutable-ok: filled while holding the lock, cancelled after release
    with _LOCK:
        for handle, entry in _ITEMS.items():
            if wanted is not None and handle not in wanted:
                continue
            if entry.state != "running" or entry.task.done():
                continue
            _ITEMS[handle] = replace(entry, state="cancelling")
            targets.append(_ITEMS[handle])
    for entry in targets:
        entry.task.cancel()
    return tuple(entry.handle for entry in targets)
