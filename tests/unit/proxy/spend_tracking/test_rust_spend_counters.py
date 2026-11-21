"""The ledger's post-call counter step, which ``increment_spend_counters`` hands to Rust, run on the Rust spend
counter store: in memory, and over a Redis server."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from typing import Final
from unittest.mock import AsyncMock

import pytest
import redis
from fakeredis import TcpFakeServer
from pydantic import TypeAdapter

import litellm.proxy.proxy_server as ps
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import (
    SpendCounterStore,
    user_window_counter,
)
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth
from tests.unit.proxy.spend_tracking.native_budgets_replica import native_reservation

pytestmark = pytest.mark.usefixtures("native_budgets_on")

_ENTRIES: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_EVERY_BUCKET: Final = (
    "spend:key:hashed",
    "spend:team:team-1",
    "spend:team_member:user-1:team-1",
    "spend:user:user-1",
    "spend:end_user:eu-1",
    "spend:tag:tag-a",
    "spend:tag:tag-b",
    "spend:org:org-1",
)


@pytest.fixture
def key_cache(monkeypatch: pytest.MonkeyPatch) -> UserApiKeyCache:
    cache: Final = UserApiKeyCache()
    monkeypatch.setattr(ps, "user_api_key_cache", cache)
    monkeypatch.setattr(ps, "prisma_client", None)
    return cache


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, key_cache: UserApiKeyCache) -> SpendCounterStore:
    installed: Final = SpendCounterStore(connect(None).spend_counters())
    monkeypatch.setattr(ps, "spend_counter_store", installed)
    return installed


def _stored_spend(monkeypatch: pytest.MonkeyPatch, spend: Mapping[str, float]) -> AsyncMock:
    async def from_db(prisma_client: object, counter_key: str) -> float | None:
        if counter_key == "spend:team:fails":
            raise RuntimeError("reseed failed")
        return spend.get(counter_key)

    read: Final = AsyncMock(side_effect=from_db)
    monkeypatch.setattr(ps.SpendCounterReseed, "from_db", read)
    return read


async def _charge_every_bucket(cost: float, **options: object) -> None:
    await ps.increment_spend_counters(
        token="hashed",
        team_id="team-1",
        user_id="user-1",
        response_cost=cost,
        org_id="org-1",
        end_user_id="eu-1",
        tags=["tag-a", "tag-b"],
        windows_from_ledger=True,
        **options,  # pyright: ignore[reportArgumentType]  # each test passes the step's typed options
    )


@pytest.mark.asyncio
async def test_cold_counters_seed_from_stored_spend_and_take_the_charge(
    monkeypatch: pytest.MonkeyPatch, store: SpendCounterStore
) -> None:
    read: Final = _stored_spend(monkeypatch, {"spend:key:hashed": 3.0, "spend:team:team-1": 4.0})

    await _charge_every_bucket(0.5)

    assert store.snapshot() == (
        {
            "spend:key:hashed": 3.5,
            "spend:team:team-1": 4.5,
            "spend:team_member:user-1:team-1": 0.5,
            "spend:user:user-1": 0.5,
            "spend:end_user:eu-1": 0.5,
            "spend:tag:tag-a": 0.5,
            "spend:tag:tag-b": 0.5,
            "spend:org:org-1": 0.5,
        }
    )
    assert {call.kwargs["counter_key"] for call in read.await_args_list} >= {"spend:key:hashed", "spend:team:team-1"}


@pytest.mark.asyncio
async def test_warm_counters_take_the_charge_without_reading_stored_spend(
    monkeypatch: pytest.MonkeyPatch, store: SpendCounterStore
) -> None:
    for counter in _EVERY_BUCKET:
        store.set_local(counter, 1.0)
    read: Final = _stored_spend(monkeypatch, {})

    await _charge_every_bucket(0.5)

    assert store.snapshot() == dict.fromkeys(_EVERY_BUCKET, 1.5)
    read.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_reserved_counter_is_reconciled_and_the_reservation_stamped_and_finalized(
    monkeypatch: pytest.MonkeyPatch, store: SpendCounterStore
) -> None:
    store.set_local("spend:key:hashed", 10.0)
    store.set_local("spend:team:team-1", 1.0)
    _stored_spend(monkeypatch, {})
    reservation: Final = await native_reservation(
        store,
        UserAPIKeyAuth(token="hashed"),
        [{"counter_key": "spend:key:hashed", "entity_type": "Key", "entity_id": "hashed"}],
        2.0,
    )
    assert store.get_local("spend:key:hashed") == 12.0

    await ps.increment_spend_counters(
        token="hashed",
        team_id="team-1",
        user_id=None,
        response_cost=0.5,
        budget_reservation=reservation,
        windows_from_ledger=True,
    )

    assert store.get_local("spend:key:hashed") == 10.5
    assert store.get_local("spend:team:team-1") == 1.5
    assert reservation["finalized"] is True
    assert [entry["applied_adjustment"] for entry in _ENTRIES.validate_python(reservation["entries"])] == [-1.5]


_USER_WINDOWS: Final = [
    {"budget_duration": "1h", "max_budget": 5.0, "reset_at": "2026-08-01T01:00:00Z"},
    {"budget_duration": "30d", "max_budget": 100.0, "reset_at": "2026-08-31T00:00:00Z"},
]


@pytest.mark.asyncio
async def test_user_windows_missing_from_the_auth_cache_come_from_the_stored_row(
    monkeypatch: pytest.MonkeyPatch, store: SpendCounterStore
) -> None:
    for counter in ("spend:user:u", user_window_counter("u", "1h"), user_window_counter("u", "30d")):
        store.set_local(counter, 1.0)
    stored: Final = AsyncMock(return_value=_USER_WINDOWS)
    monkeypatch.setattr(ps, "_stored_budget_limits", stored)

    await ps.increment_spend_counters(
        token=None, team_id=None, user_id="u", response_cost=2.0, windows_from_ledger=True
    )

    assert store.snapshot() == (
        {"spend:user:u": 3.0, user_window_counter("u", "1h"): 3.0, user_window_counter("u", "30d"): 3.0}
    )
    assert [call.args[1] for call in stored.await_args_list] == ["u"]


@pytest.mark.asyncio
async def test_a_failing_stored_spend_read_propagates_after_every_other_counter_is_charged(
    monkeypatch: pytest.MonkeyPatch, store: SpendCounterStore
) -> None:
    warm: Final = ("spend:key:hashed", "spend:user:user-1", "spend:end_user:eu-1", "spend:tag:tag-a", "spend:org:org-1")
    for counter in (*warm, "spend:team_member:user-1:fails"):
        store.set_local(counter, 1.0)
    _stored_spend(monkeypatch, {})
    reservation: Final[dict[str, object]] = {"finalized": False}

    with pytest.raises(RuntimeError, match="reseed failed"):
        await ps.increment_spend_counters(
            token="hashed",
            team_id="fails",
            user_id="user-1",
            response_cost=5.0,
            org_id="org-1",
            end_user_id="eu-1",
            tags=["tag-a"],
            budget_reservation=reservation,
            windows_from_ledger=True,
        )

    assert reservation["finalized"] is False
    assert store.snapshot() == dict.fromkeys((*warm, "spend:team_member:user-1:fails"), 6.0)


@pytest.mark.asyncio
async def test_the_customer_hierarchy_settles_before_the_charge(store: SpendCounterStore) -> None:
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="c",
        customer_team_id="ct",
        scopes=(
            CustomerScopeAuth(key="customer_org", entity_id="c", spend=2.0),
            CustomerScopeAuth(key="customer_team", entity_id="ct", spend=1.0),
        ),
    )

    await ps.increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.5,
        customer_hierarchy=hierarchy,
        windows_from_ledger=True,
    )

    assert store.snapshot() == ({"spend:customer_org:c:lifetime": 2.5, "spend:customer_team:ct:lifetime": 1.5})


@pytest.fixture
def redis_port() -> Iterator[int]:
    server: Final = TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


@pytest.mark.asyncio
async def test_on_redis_every_bucket_is_charged_in_the_shared_counters(
    monkeypatch: pytest.MonkeyPatch, key_cache: UserApiKeyCache, redis_port: int
) -> None:
    shared: Final = redis.Redis(host="127.0.0.1", port=redis_port, decode_responses=True)
    for counter in _EVERY_BUCKET:
        shared.set(counter, 1.0)
    installed: Final = SpendCounterStore(connect(RedisCache(host="127.0.0.1", port=redis_port)).spend_counters())
    monkeypatch.setattr(ps, "spend_counter_store", installed)

    await _charge_every_bucket(0.5)

    assert {counter: float(shared.get(counter) or "nan") for counter in _EVERY_BUCKET} == (
        dict.fromkeys(_EVERY_BUCKET, 1.5)
    )
    assert installed.snapshot() == dict.fromkeys(_EVERY_BUCKET, 1.5)
