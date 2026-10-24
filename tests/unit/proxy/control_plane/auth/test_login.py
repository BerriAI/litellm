from datetime import timedelta

import pytest

from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.login import (
    LOCKOUT_FAILURES,
    LOCKOUT_SECONDS,
    InvalidCredentials,
    LoginService,
    LoginThrottle,
    Throttled,
)
from litellm.proxy.control_plane.auth.rbac import default_policy
from litellm.types.proxy.control_plane.auth import SessionTokenResponse


class Clock:  # mutable-ok: the test advances it
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _service(clock: Clock) -> LoginService:
    policy = default_policy()
    sessions = SessionTokenCodec(
        master_key="sk-master", issuer="cp", audience="cp-api", ttl=timedelta(minutes=5), policy=policy
    )
    return LoginService(
        sessions=sessions,
        policy=policy,
        ttl_seconds=300,
        admin_username="admin",
        admin_password="hunter2",
        throttle=LoginThrottle(clock),
    )


def test_account_locks_after_repeated_misses_and_unlocks_after_the_window() -> None:
    clock = Clock()
    service = _service(clock)
    misses = [service.login("admin", "wrong") for i in range(LOCKOUT_FAILURES)]
    assert all(isinstance(miss, InvalidCredentials) for miss in misses)
    locked = service.login("admin", "hunter2")
    assert locked == Throttled(retry_after_seconds=int(LOCKOUT_SECONDS))
    clock.now += LOCKOUT_SECONDS / 2
    assert service.login("admin", "hunter2") == Throttled(retry_after_seconds=int(LOCKOUT_SECONDS / 2))
    clock.now += LOCKOUT_SECONDS / 2 + 1
    assert isinstance(service.login("admin", "hunter2"), SessionTokenResponse)


def test_expired_misses_are_pruned_so_distinct_usernames_cannot_grow_the_store_forever() -> None:
    clock = Clock()
    throttle = LoginThrottle(clock)
    for i in range(100):
        throttle.record_failure(f"user{i}")
        clock.now += LOCKOUT_SECONDS / 10
    assert throttle.tracked() <= 10
    clock.now += LOCKOUT_SECONDS + 1
    throttle.record_failure("last")
    assert throttle.tracked() == 1


def test_a_success_clears_the_accounts_misses() -> None:
    clock = Clock()
    service = _service(clock)
    for _ in range(LOCKOUT_FAILURES - 1):
        service.login("admin", "wrong")
    assert isinstance(service.login("admin", "hunter2"), SessionTokenResponse)
    for _ in range(LOCKOUT_FAILURES - 1):
        assert isinstance(service.login("admin", "wrong"), InvalidCredentials)
    assert isinstance(service.login("admin", "hunter2"), SessionTokenResponse)


@pytest.mark.parametrize("username", ["admin", "nobody"])
def test_no_configured_admin_rejects_everyone(username: str) -> None:
    sessions = SessionTokenCodec(
        master_key="sk-master", issuer="cp", audience="cp-api", ttl=timedelta(minutes=5), policy=default_policy()
    )
    service = LoginService(sessions, default_policy(), 300, None, None, LoginThrottle(Clock()))
    assert isinstance(service.login(username, "hunter2"), InvalidCredentials)


@pytest.mark.parametrize("username", ["", "u" * 257])
def test_login_request_bounds_the_username(username: str) -> None:
    from pydantic import ValidationError

    from litellm.types.proxy.control_plane.auth import LoginRequest

    with pytest.raises(ValidationError):
        LoginRequest(username=username, password="x")
