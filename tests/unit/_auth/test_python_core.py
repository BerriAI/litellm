from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm._auth.decision import (
    AdminKey,
    Allow,
    CachedAdminKey,
    Deny,
    ExpiredKey,
    MalformedKey,
    MasterKey,
    MasterKeyOnlyRoute,
    MissingKey,
    NoDatabase,
    NoMasterKey,
    NonAdminKey,
    NotAVirtualKey,
    Outcome,
    PresentedKey,
    RequestFacts,
    Settings,
    UserAuthDisabled,
    UserAuthRoute,
    VirtualKey,
)
from litellm._auth.driver import drive
from litellm._auth.key import hash_token
from litellm._auth.python_core import PythonCore

MASTER: Final = "sk-master"
NOW: Final = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
SETTINGS: Final = Settings(
    master_key=MASTER,
    allow_user_auth=False,
    has_db=True,
    master_key_only_routes=frozenset({"/global/spend/reset"}),
)


@dataclass
class RecordingHandler:
    presented: PresentedKey | None = None
    calls: tuple[tuple[str, object], ...] = ()

    async def resolve_end_user(self, prefetch_key: bool) -> None:
        self.calls = (*self.calls, ("resolve_end_user", prefetch_key))

    async def resolve_presented_key(self, hashed_token: str) -> PresentedKey | None:
        self.calls = (*self.calls, ("resolve_presented_key", hashed_token))
        return self.presented

    async def evict_presented_key(self, hashed_token: str) -> None:
        self.calls = (*self.calls, ("evict_presented_key", hashed_token))

    async def load_key(self, hashed_token: str) -> None:
        self.calls = (*self.calls, ("load_key", hashed_token))


async def _decide(
    token: str | None,
    *,
    route: str = "/chat/completions",
    settings: Settings = SETTINGS,
    presented: PresentedKey | None = None,
) -> tuple[Outcome, tuple[tuple[str, object], ...]]:
    handler: Final = RecordingHandler(presented=presented)
    outcome: Final = await drive(PythonCore().start(RequestFacts(route=route, token=token), settings, NOW), handler)
    return outcome, handler.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "token", "settings", "expected"),
    [
        ("/chat/completions", None, Settings(None, False, True, frozenset()), Allow(NoMasterKey(token=None))),
        ("/chat/completions", "sk-x", Settings(None, False, False, frozenset()), Allow(NoMasterKey(token="sk-x"))),
        ("/chat/completions", None, SETTINGS, Deny(MissingKey())),
        ("/chat/completions", "", SETTINGS, Deny(MalformedKey())),
        ("/user/auth", "", Settings(MASTER, True, True, frozenset()), Deny(MalformedKey())),
        ("/user/auth", "sk-x", Settings(MASTER, True, True, frozenset()), Allow(UserAuthRoute())),
        ("/user/auth", "sk-x", SETTINGS, Deny(UserAuthDisabled())),
        ("/user/auth", MASTER, SETTINGS, Deny(UserAuthDisabled())),
    ],
)
async def test_credential_decisions_do_no_io(route: str, token: str | None, settings: Settings, expected: Outcome):
    assert await _decide(token, route=route, settings=settings) == (expected, ())


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/chat/completions", "/global/spend/reset"])
async def test_master_key_resolves_end_user_without_key_prefetch(route: str):
    assert await _decide(MASTER, route=route) == (Allow(MasterKey()), (("resolve_end_user", False),))


@pytest.mark.asyncio
async def test_unknown_virtual_key_is_loaded_from_the_db_after_a_cache_miss():
    hashed: Final = hash_token("sk-x")
    assert await _decide("sk-x") == (
        Allow(VirtualKey(loaded_from_db=True)),
        (("resolve_end_user", True), ("resolve_presented_key", hashed), ("load_key", hashed)),
    )


@pytest.mark.asyncio
async def test_cached_virtual_key_is_not_loaded_again():
    assert await _decide("sk-x", presented=NonAdminKey()) == (
        Allow(VirtualKey(loaded_from_db=False)),
        (("resolve_end_user", True), ("resolve_presented_key", hash_token("sk-x"))),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("expires", [None, NOW + timedelta(seconds=1), NOW])
async def test_unexpired_cached_admin_key_skips_route_and_db_checks(expires: datetime | None):
    no_db: Final = Settings(MASTER, False, False, frozenset({"/global/spend/reset"}))
    outcome, _ = await _decide(
        "sk-admin", route="/global/spend/reset", settings=no_db, presented=AdminKey(expires=expires)
    )
    assert outcome == Allow(CachedAdminKey())


@pytest.mark.asyncio
async def test_expired_cached_admin_key_is_evicted_then_denied():
    expires: Final = NOW - timedelta(seconds=1)
    hashed: Final = hash_token("sk-admin")
    assert await _decide("sk-admin", presented=AdminKey(expires=expires)) == (
        Deny(ExpiredKey(expires=expires, now=NOW)),
        (("resolve_end_user", True), ("resolve_presented_key", hashed), ("evict_presented_key", hashed)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("presented", [None, NonAdminKey()])
async def test_non_admin_key_cannot_use_master_key_only_routes(presented: PresentedKey | None):
    outcome, calls = await _decide("sk-x", route="/global/spend/reset", presented=presented)
    assert outcome == Deny(MasterKeyOnlyRoute())
    assert "load_key" not in {name for name, _ in calls}


@pytest.mark.asyncio
@pytest.mark.parametrize("presented", [None, NonAdminKey()])
async def test_virtual_key_needs_a_database(presented: PresentedKey | None):
    no_db: Final = Settings(MASTER, False, False, frozenset())
    outcome, _ = await _decide("sk-x", settings=no_db, presented=presented)
    assert outcome == Deny(NoDatabase())


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["eyJhbGciOiJIUzI1NiJ9.e30.sig", "a" * 64, "sk"])
async def test_uncached_token_without_virtual_key_prefix_is_rejected_before_the_db(token: str):
    outcome, calls = await _decide(token)
    assert outcome == Deny(NotAVirtualKey())
    assert [name for name, _ in calls] == ["resolve_end_user", "resolve_presented_key"]


@pytest.mark.asyncio
async def test_cached_token_without_virtual_key_prefix_is_accepted():
    outcome, _ = await _decide("ui-login-blob", presented=NonAdminKey())
    assert outcome == Allow(VirtualKey(loaded_from_db=False))
