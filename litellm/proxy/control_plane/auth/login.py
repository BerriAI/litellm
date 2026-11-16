"""Username and password login for the admin configured through UI_USERNAME / UI_PASSWORD."""

import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import ceil
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.rbac import ADMIN_ROLE, RolePolicy, build_principal
from litellm.types.proxy.control_plane.auth import SessionTokenResponse

LOCKOUT_FAILURES: Final = 5
LOCKOUT_SECONDS: Final = 300.0
MAX_TRACKED_LOGIN_KEYS: Final = 10_000


@dataclass(frozen=True, slots=True)
class InvalidCredentials:
    pass


@dataclass(frozen=True, slots=True)
class Throttled:
    retry_after_seconds: int


LoginResult: TypeAlias = SessionTokenResponse | InvalidCredentials | Throttled


class LoginThrottle:
    """Locks a source-and-account pair after repeated misses."""

    def __init__(
        self, now: Callable[[], float], failures: int = LOCKOUT_FAILURES, lockout_seconds: float = LOCKOUT_SECONDS
    ) -> None:
        self._now = now
        self._failures = failures
        self._lockout_seconds = lockout_seconds
        self._misses: Mapping[tuple[str, str], tuple[float, ...]] = MappingProxyType({})

    def _recent(self, key: tuple[str, str]) -> tuple[float, ...]:
        cutoff: Final = self._now() - self._lockout_seconds
        return tuple(at for at in self._misses.get(key, ()) if at > cutoff)

    def locked_for(self, key: tuple[str, str]) -> float:
        recent: Final = self._recent(key)
        if len(recent) < self._failures:
            return 0.0
        return max(0.0, recent[-1] + self._lockout_seconds - self._now())

    def record_failure(self, key: tuple[str, str]) -> None:
        now: Final = self._now()
        cutoff: Final = now - self._lockout_seconds
        active: Final = {
            tracked_key: tuple(at for at in misses if at > cutoff)
            for tracked_key, misses in self._misses.items()
            if any(at > cutoff for at in misses)
        }
        updated: Final = {**active, key: (*active.get(key, ()), now)}
        self._misses = self._bounded(updated)

    @staticmethod
    def _bounded(
        misses: Mapping[tuple[str, str], tuple[float, ...]],
    ) -> Mapping[tuple[str, str], tuple[float, ...]]:
        if len(misses) <= MAX_TRACKED_LOGIN_KEYS:
            return MappingProxyType(dict(misses))
        oldest: Final = min(misses, key=lambda tracked_key: misses[tracked_key][-1])
        return MappingProxyType(
            {tracked_key: values for tracked_key, values in misses.items() if tracked_key != oldest}
        )

    def tracked(self) -> int:
        return len(self._misses)

    def reset(self, key: tuple[str, str]) -> None:
        self._misses = MappingProxyType(
            {tracked_key: misses for tracked_key, misses in self._misses.items() if tracked_key != key}
        )


class LoginService:
    def __init__(
        self,
        sessions: SessionTokenCodec,
        policy: RolePolicy,
        ttl_seconds: int,
        admin_username: str | None,
        admin_password: str | None,
        throttle: LoginThrottle,
    ) -> None:
        self._sessions = sessions
        self._policy = policy
        self._ttl_seconds = ttl_seconds
        self._admin_username = admin_username
        self._admin_password = admin_password
        self._throttle = throttle

    def login(self, username: str, password: str, client_source: str = "") -> LoginResult:
        if self._admin_username is None or self._admin_password is None:
            return InvalidCredentials()
        throttle_key: Final = (client_source, self._admin_username)
        locked: Final = self._throttle.locked_for(throttle_key)
        if locked > 0:
            return Throttled(retry_after_seconds=max(1, ceil(locked)))
        if not self._credentials_match(username, password):
            self._throttle.record_failure(throttle_key)
            return InvalidCredentials()
        self._throttle.reset(throttle_key)
        principal: Final = build_principal(
            policy=self._policy,
            subject=username,
            email=None,
            roles=frozenset({ADMIN_ROLE}),
            claimed_scopes=None,
            dataplane_ids=frozenset(),
            token_kind="session",
        )
        return SessionTokenResponse(access_token=self._sessions.mint(principal), expires_in=self._ttl_seconds)

    def _credentials_match(self, username: str, password: str) -> bool:
        if self._admin_username is None or self._admin_password is None:
            return False
        username_ok: Final = secrets.compare_digest(username.encode("utf-8"), self._admin_username.encode("utf-8"))
        password_ok: Final = secrets.compare_digest(password.encode("utf-8"), self._admin_password.encode("utf-8"))
        return username_ok and password_ok
