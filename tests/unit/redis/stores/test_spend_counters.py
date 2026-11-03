import json
from typing import Final

import pytest

from litellm._internal_context import current_service_target
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.spend_counters import (
    SPEND_COUNTERS_TARGET,
    SpendCounterStore,
    end_user_counter,
    is_unavailable,
    key_counter,
    key_window_counter,
    model_access_group_counter,
    org_counter,
    project_counter,
    shadow_eval_counter,
    tag_counter,
    team_counter,
    team_member_counter,
    team_window_counter,
    user_counter,
)
from litellm.rust_bridge._native import NativeRedisClient, NativeSpendCounterError, NativeSpendCounters
from tests.unit.caching.test_redis_batch import FakeNatives

from .spend_counters_double import FakeSpendCounters

_KEYS: Final = (
    (
        key_counter("88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"),
        "spend:key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
    ),
    (key_window_counter("88dc28d0", "30d"), "spend:key:88dc28d0:window:30d"),
    (team_counter("team-a"), "spend:team:team-a"),
    (team_window_counter("team-a", "1h"), "spend:team:team-a:window:1h"),
    (team_member_counter("user:with:colons", "team-a"), "spend:team_member:user:with:colons:team-a"),
    (user_counter("user@example.com"), "spend:user:user@example.com"),
    (end_user_counter("customer é 1"), "spend:end_user:customer é 1"),
    (org_counter("org-1"), "spend:org:org-1"),
    (tag_counter("prod:eu"), "spend:tag:prod:eu"),
    (project_counter("proj-1"), "spend:project:proj-1"),
    (model_access_group_counter("gpt-group"), "spend:model_access_group:gpt-group"),
    (shadow_eval_counter("job-7"), "spend:shadow_eval:job-7"),
)


@pytest.mark.parametrize(("built", "key"), _KEYS, ids=[key for _, key in _KEYS])
def test_python_builds_every_key_the_native_store_formats(built: str, key: str) -> None:
    store: Final = NativeSpendCounters(None)

    store.set_local(built, 1.5)

    assert built == key
    assert store.snapshot() == {key: 1.5}


@pytest.mark.parametrize(
    ("spend", "wire"), [(0.0, "0.0"), (12, "12"), (1e-05, "1e-05"), (0.30000000000000004, "0.30000000000000004")]
)
def test_a_reset_writes_what_redis_cache_set_cache_writes(spend: float, wire: str) -> None:
    assert json.dumps(spend) == wire


@pytest.mark.parametrize(("spend", "wire"), [(2.5, "2.5"), (100, "100"), (1e-05, "1e-05")])
def test_a_repair_writes_what_redis_cache_set_max_writes(spend: float, wire: str) -> None:
    assert str(spend) == wire


@pytest.mark.parametrize(("wire", "spend"), [("7.25", 7.25), ("3", 3), ("0.0", 0.0), ("1e-05", 1e-05)])
def test_a_read_decodes_what_redis_cache_decodes(wire: str, spend: float) -> None:
    decoded: Final = RedisCache._get_cache_logic(  # pyright: ignore[reportPrivateUsage]  # main's codec is the contract
        RedisCache.__new__(RedisCache), wire.encode()
    )

    assert decoded == spend
    assert type(decoded) is type(spend)


@pytest.mark.parametrize("key", ["spend_db_floor:spend:key:k", "virtual_key_spend:k:gpt:1d", "spend:unknown:k"])
def test_the_native_store_refuses_keys_outside_the_spend_counter_keyspace(key: str) -> None:
    with pytest.raises(ValueError, match="is not a spend counter"):
        NativeSpendCounters(None).get_local(key)


@pytest.mark.asyncio
async def test_without_redis_the_native_store_counts_and_answers_from_memory() -> None:
    store: Final = SpendCounterStore(None)

    first: Final = await store.increment("spend:key:k", 2)
    second: Final = await store.increment_many((("spend:key:k", 0.5), ("spend:team:t", 1.0)))
    seeded: Final = await store.seed("spend:org:o", 4.0, increment_if_lost=False)
    repaired: Final = await store.set_max("spend:key:k", 9.0)

    assert (first, second, seeded, repaired) == (2, (2.5, 1.0), 4.0, None)
    assert await store.read("spend:key:k") == (9.0, False)
    assert await store.read_many(("spend:key:k",)) is None
    assert not store.has_redis
    assert not store.increment_many_post_call((("spend:key:k", 1.0),))


def test_a_broadcast_writes_and_drops_only_spend_counter_copies() -> None:
    store: Final = SpendCounterStore(None)
    store.set_local("spend:key:k", 5.0)
    store.set_local("spend:team:t", 5.0)

    store.apply_broadcast("spend:key:k", 0.0, 60)
    store.apply_broadcast("spend:team:t", None, None)
    store.apply_broadcast("spend_db_floor:spend:key:k", 1.0, 5)
    store.apply_broadcast("spend:unknown:x", 1.0, None)

    assert store.snapshot() == {"spend:key:k": 0.0}


@pytest.mark.parametrize(
    "kind, dropped",
    [("timeout", True), ("breaker_open", True), ("transport", False), ("rejected", False), ("memory", False)],
)
def test_only_a_timeout_or_an_open_breaker_lets_a_post_call_increment_drop(kind: str, dropped: bool) -> None:
    assert is_unavailable(NativeSpendCounterError(kind, "message")) is dropped
    assert not is_unavailable(TimeoutError(kind))


def test_redis_attaches_once_and_its_ttl_is_the_lease_the_store_reports() -> None:
    built: Final[list[tuple[NativeRedisClient | None, int]]] = []

    def handle(client: NativeRedisClient | None, redis_ttl: int) -> FakeSpendCounters:
        built.append((client, redis_ttl))
        return FakeSpendCounters(redis=None if client is None else {})

    store: Final = SpendCounterStore(None, handle=handle)
    first: Final = NativeRedisClient.connect({"host": "127.0.0.1", "port": 1})
    second: Final = NativeRedisClient.connect({"host": "127.0.0.2", "port": 1})

    store.attach_redis(None, redis_ttl=10)
    store.attach_redis(first, redis_ttl=30)
    store.attach_redis(second, redis_ttl=90)

    assert built == [(None, 60), (first, 30)]
    assert store.has_redis
    assert store.redis_ttl == 30


@pytest.mark.asyncio
async def test_post_call_increments_go_on_the_requests_native_post_call_batch_under_the_spend_counters_target() -> None:
    natives: Final = FakeNatives()
    targets: Final[list[str | None]] = []

    class Recording(FakeSpendCounters):
        def increment_many_post_call(self, request: object, increments: object) -> bool:
            targets.append(current_service_target())
            return super().increment_many_post_call(request, increments)  # pyright: ignore[reportArgumentType]  # recorded as given

    spend: Final = Recording()
    store: Final = spend.store()

    outside: Final = store.increment_many_post_call((("spend:key:k", 1.0),))
    with request_redis_batch_scope(native_factory=natives):  # pyright: ignore[reportArgumentType]  # test double
        inside: Final = store.increment_many_post_call((("spend:key:k", 1.0),))
        empty: Final = store.increment_many_post_call(())

    assert (outside, inside, empty) == (False, True, False)
    assert spend.post_call == [(natives.built[0], (("spend:key:k", 1.0),))]
    assert targets == [SPEND_COUNTERS_TARGET]


@pytest.mark.asyncio
async def test_an_unreachable_redis_fails_writes_as_transport_and_serves_reads_from_memory() -> None:
    store: Final = NativeSpendCounters(
        NativeRedisClient.connect({"host": "127.0.0.1", "port": 1, "socket_timeout": 0.2})
    )
    store.set_local("spend:key:k", 4.0)

    with pytest.raises(NativeSpendCounterError) as failure:
        await store.increment("spend:key:k", 1.0)

    assert failure.value.args[0] == "transport"
    assert not is_unavailable(failure.value)
    assert await store.read("spend:key:k") == ("memory", None)
