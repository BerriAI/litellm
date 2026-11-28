from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock


@dataclass(slots=True)
class _ArchiveRetryState:
    last_failure_at: datetime | None = None
    next_retry_at: datetime | None = None
    in_flight: bool = False


class PluginArchiveRetryCache:
    def __init__(self, *, failure_ttl: timedelta, retry_interval: timedelta) -> None:
        self._failure_ttl = failure_ttl
        self._retry_interval = retry_interval
        self._lock = Lock()
        self._states: dict[str, _ArchiveRetryState] = {}

    def _state(self, plugin_name: str) -> _ArchiveRetryState:
        state: _ArchiveRetryState | None = self._states.get(plugin_name)
        if state is None:
            state = _ArchiveRetryState()
            self._states[plugin_name] = state
        return state

    def should_skip(self, plugin_name: str, *, now: datetime | None = None) -> bool:
        timestamp = now or datetime.now(timezone.utc)
        with self._lock:
            state = self._state(plugin_name)
            if state.in_flight:
                return True
            if state.last_failure_at is not None and timestamp < state.last_failure_at + self._failure_ttl:
                return True
            if state.next_retry_at is not None and timestamp < state.next_retry_at:
                return True
            return False

    def begin(self, plugin_name: str, *, now: datetime | None = None) -> bool:
        timestamp = now or datetime.now(timezone.utc)
        with self._lock:
            state = self._state(plugin_name)
            if state.in_flight:
                return False
            if state.last_failure_at is not None and timestamp < state.last_failure_at + self._failure_ttl:
                return False
            if state.next_retry_at is not None and timestamp < state.next_retry_at:
                return False
            state.in_flight = True
            return True

    def failure(self, plugin_name: str, *, now: datetime | None = None) -> None:
        timestamp = now or datetime.now(timezone.utc)
        with self._lock:
            state = self._state(plugin_name)
            state.in_flight = False
            state.last_failure_at = timestamp
            state.next_retry_at = timestamp + self._retry_interval

    def success(self, plugin_name: str, *, now: datetime | None = None) -> None:
        timestamp = now or datetime.now(timezone.utc)
        with self._lock:
            state = self._state(plugin_name)
            state.in_flight = False
            state.last_failure_at = None
            state.next_retry_at = timestamp + self._retry_interval

