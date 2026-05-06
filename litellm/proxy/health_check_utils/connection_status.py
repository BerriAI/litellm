"""
In-memory tri-state connection status for proxy dependencies (DB, Redis).

Background: see Linear LIT-2607. A long-running pod can stay alive while
its DB or Redis dependency silently degrades, leaving the UI empty even
though /health/liveliness returns 200. We piggy-back on the existing
ServiceLogging success/failure hooks (which fire on every DB and Redis
op) to record the latest observed state, with no new background pollers
or per-event history — both of which would reintroduce the same
memory-leak class as the original incident.

Status semantics:
- ``up``      — last observation succeeded
- ``down``    — last observation failed
- ``unknown`` — no observation yet (just-started pod *or* dependency not
                configured for this deployment). Treated as healthy by
                liveness — we must not flap pods that haven't exercised
                the dependency.

Only ``down`` should fail liveness. ``up`` / ``unknown`` keep the pod
alive.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

ConnectionState = Literal["up", "down", "unknown"]
Component = Literal["db", "redis"]


@dataclass
class _ComponentStatus:
    state: ConnectionState = "unknown"
    last_updated: Optional[datetime] = None
    last_error: Optional[str] = None


@dataclass
class _StatusSnapshot:
    db: _ComponentStatus = field(default_factory=_ComponentStatus)
    redis: _ComponentStatus = field(default_factory=_ComponentStatus)


class ConnectionStatusTracker:
    """
    Process-local tri-state tracker for DB and Redis connection health.

    All state is a fixed-size dict; no history is retained. Reads and
    writes are protected by a synchronous ``threading.Lock`` so the
    tracker is safe to call from sync code, async code, and from inside
    callbacks without depending on an event loop.
    """

    _MAX_ERROR_LEN = 500

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = _StatusSnapshot()

    def _set(
        self,
        component: Component,
        state: ConnectionState,
        error: Optional[str] = None,
    ) -> None:
        truncated_error: Optional[str] = None
        if error is not None:
            truncated_error = error[: self._MAX_ERROR_LEN]
        new_status = _ComponentStatus(
            state=state,
            last_updated=datetime.now(timezone.utc),
            last_error=truncated_error,
        )
        with self._lock:
            setattr(self._snapshot, component, new_status)

    def mark_up(self, component: Component) -> None:
        self._set(component, "up", None)

    def mark_down(self, component: Component, error: Optional[str] = None) -> None:
        self._set(component, "down", error)

    def get(self, component: Component) -> _ComponentStatus:
        with self._lock:
            current = getattr(self._snapshot, component)
            return _ComponentStatus(
                state=current.state,
                last_updated=current.last_updated,
                last_error=current.last_error,
            )

    def snapshot(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Return a JSON-friendly view of the current state."""
        out: Dict[str, Dict[str, Optional[str]]] = {}
        with self._lock:
            for component in ("db", "redis"):
                status = getattr(self._snapshot, component)
                out[component] = {
                    "status": status.state,
                    "last_updated": (
                        status.last_updated.isoformat()
                        if status.last_updated is not None
                        else None
                    ),
                    "last_error": status.last_error,
                }
        return out

    def is_any_down(self) -> bool:
        with self._lock:
            return (
                self._snapshot.db.state == "down"
                or self._snapshot.redis.state == "down"
            )

    def reset(self) -> None:
        """Reset to initial state. Used by tests."""
        with self._lock:
            self._snapshot = _StatusSnapshot()


connection_status_tracker = ConnectionStatusTracker()
