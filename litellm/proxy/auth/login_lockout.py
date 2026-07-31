"""In-memory login lockout policy for the Admin UI.

The state is local to one process and is lost when that process restarts. It is
not shared between workers, so deployments requiring distributed enforcement
must use an external rate-limiting mechanism. The tracked identity count is
capped; during a flood of distinct usernames, an entry can be evicted early, so
the cap should remain well above the number of realistic users.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


_MAX_TRACKED_IDENTITIES = 4096


@dataclass(frozen=True, slots=True)
class LoginBlock:
    """A temporary block and the number of seconds until it expires."""

    kind: Literal["cooldown", "lockout"]
    remaining_seconds: int


class LoginLockout:
    """Track failed login attempts and determine whether an identity is blocked."""

    __slots__ = (
        "_failures",
        "_time_fn",
        "_window_seconds",
        "_cooldown_threshold",
        "_cooldown_seconds",
        "_lockout_threshold",
        "_lockout_seconds",
    )

    def __init__(
        self,
        *,
        time_fn: Callable[[], float] = time.monotonic,
        window_seconds: float = 900,
        cooldown_threshold: int = 3,
        cooldown_seconds: float = 60,
        lockout_threshold: int = 5,
        lockout_seconds: float = 900,
    ) -> None:
        self._failures: dict[str, tuple[float, ...]] = {}
        self._time_fn = time_fn
        self._window_seconds = window_seconds
        self._cooldown_threshold = cooldown_threshold
        self._cooldown_seconds = cooldown_seconds
        self._lockout_threshold = lockout_threshold
        self._lockout_seconds = lockout_seconds

    @staticmethod
    def normalize_username(username: str) -> str:
        return username.strip().lower()

    def check(self, username: str) -> LoginBlock | None:
        key = self.normalize_username(username)
        now = self._time_fn()
        failures = self._prune(key, now)
        if len(failures) >= self._lockout_threshold:
            return self._block("lockout", failures[-1], now, self._lockout_seconds)
        if len(failures) >= self._cooldown_threshold:
            return self._block("cooldown", failures[-1], now, self._cooldown_seconds)
        return None

    def record_failure(self, username: str) -> None:
        key = self.normalize_username(username)
        now = self._time_fn()
        failures = self._prune(key, now)
        self._failures[key] = (*failures, now)
        if len(self._failures) > _MAX_TRACKED_IDENTITIES:
            self._failures.pop(next(iter(self._failures)))

    def clear(self, username: str) -> None:
        self._failures.pop(self.normalize_username(username), None)

    def _prune(self, key: str, now: float) -> tuple[float, ...]:
        cutoff = now - self._window_seconds
        failures = tuple(timestamp for timestamp in self._failures.get(key, ()) if timestamp > cutoff)
        if failures:
            self._failures[key] = failures
        else:
            self._failures.pop(key, None)
        return failures

    @staticmethod
    def _block(
        kind: Literal["cooldown", "lockout"],
        latest_failure: float,
        now: float,
        duration: float,
    ) -> LoginBlock | None:
        remaining_seconds = max(0, math.ceil(duration - (now - latest_failure)))
        if remaining_seconds == 0:
            return None
        return LoginBlock(kind=kind, remaining_seconds=remaining_seconds)
