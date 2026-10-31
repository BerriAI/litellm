from datetime import timedelta
from typing import Final

import pytest

from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.login import (
    LOCKOUT_FAILURES,
    LOCKOUT_SECONDS,
    InvalidCredentials,
    LoginService,
    LoginThrottle,
    MAX_TRACKED_LOGIN_KEYS,
    Throttled,
)
from litellm.proxy.control_plane.auth.rbac import default_policy
from litellm.types.proxy.control_plane.auth import SessionTokenResponse


class Clock:  # mutable-ok: the test advances it
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _service(clock: Clock, throttle: LoginThrottle | None = None) -> LoginService:
    policy: Final = default_policy()
    sessions: Final = SessionTokenCodec(
        master_key="sk-master", issuer="cp", audience="cp-api", ttl=timedelta(minutes=5), policy=policy
    )
    return LoginService(
        sessions=sessions,
        policy=policy,
        ttl_seconds=300,
        admin_username="admin",
        admin_password="hunter2",
        throttle=throttle if throttle is not None else LoginThrottle(clock),
    )


def test_account_locks_after_repeated_misses_and_unlocks_after_the_window() -> None:
    clock: Final = Clock()
    service: Final = _service(clock)
    misses: Final = [service.login("admin", "wrong") for i in range(LOCKOUT_FAILURES)]
    assert all(isinstance(miss, InvalidCredentials) for miss in misses)
    locked: Final = service.login("admin", "hunter2")
    assert locked == Throttled(retry_after_seconds=int(LOCKOUT_SECONDS))
    clock.now += LOCKOUT_SECONDS / 2
    assert service.login("admin", "hunter2") == Throttled(retry_after_seconds=int(LOCKOUT_SECONDS / 2))
    clock.now += LOCKOUT_SECONDS / 2 + 1
    assert isinstance(service.login("admin", "hunter2"), SessionTokenResponse)


def test_login_throttle_is_scoped_to_the_client_source() -> None:
    clock: Final = Clock()
    service: Final = _service(clock)
    misses: Final = tuple(service.login("admin", "wrong", client_source="client-a") for _ in range(LOCKOUT_FAILURES))
    assert all(isinstance(miss, InvalidCredentials) for miss in misses)
    assert isinstance(service.login("admin", "hunter2", client_source="client-a"), Throttled)
    assert isinstance(service.login("admin", "hunter2", client_source="client-b"), SessionTokenResponse)


def test_unknown_usernames_do_not_grow_the_throttle_or_lock_out_the_admin() -> None:
    clock: Final = Clock()
    throttle: Final = LoginThrottle(clock)
    service: Final = _service(clock, throttle)
    for index in range(100):
        assert isinstance(service.login(f"unknown-{index}", "wrong"), InvalidCredentials)
    assert throttle.tracked() == 0
    assert isinstance(service.login("admin", "hunter2"), SessionTokenResponse)


def test_failure_recording_prunes_keys_whose_misses_expired() -> None:
    clock: Final = Clock()
    throttle: Final = LoginThrottle(clock)
    throttle.record_failure(("old-source", "admin"))
    clock.now += LOCKOUT_SECONDS + 1

    throttle.record_failure(("new-source", "admin"))

    assert throttle.tracked() == 1


def test_rotating_sources_stay_bounded_without_evicting_a_hot_lockout() -> None:
    clock: Final = Clock()
    throttle: Final = LoginThrottle(clock)
    hot_key: Final = ("hot-source", "admin")
    for _ in range(LOCKOUT_FAILURES - 1):
        throttle.record_failure(hot_key)
    for index in range(MAX_TRACKED_LOGIN_KEYS - 1):
        clock.now += 0.001
        throttle.record_failure((f"source-{index}", "admin"))

    clock.now += 0.001
    throttle.record_failure(hot_key)
    clock.now += 0.001
    throttle.record_failure(("source-over-cap", "admin"))

    assert throttle.tracked() == MAX_TRACKED_LOGIN_KEYS
    assert throttle.locked_for(hot_key) > 0


def test_a_success_clears_the_accounts_misses() -> None:
    clock: Final = Clock()
    service: Final = _service(clock)
    for _ in range(LOCKOUT_FAILURES - 1):
        service.login("admin", "wrong")
    assert isinstance(service.login("admin", "hunter2"), SessionTokenResponse)
    for _ in range(LOCKOUT_FAILURES - 1):
        assert isinstance(service.login("admin", "wrong"), InvalidCredentials)
    assert isinstance(service.login("admin", "hunter2"), SessionTokenResponse)


@pytest.mark.parametrize("username", ["admin", "nobody"])
def test_no_configured_admin_rejects_everyone(username: str) -> None:
    throttle: Final = LoginThrottle(Clock())
    sessions: Final = SessionTokenCodec(
        master_key="sk-master", issuer="cp", audience="cp-api", ttl=timedelta(minutes=5), policy=default_policy()
    )
    service: Final = LoginService(sessions, default_policy(), 300, None, None, throttle)
    assert isinstance(service.login(username, "hunter2"), InvalidCredentials)
    assert throttle.tracked() == 0


@pytest.mark.parametrize("username", ["", "u" * 257])
def test_login_request_bounds_the_username(username: str) -> None:
    from pydantic import ValidationError

    from litellm.types.proxy.control_plane.auth import LoginRequest

    with pytest.raises(ValidationError):
        LoginRequest(username=username, password="x")
