import binascii
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.rate_limits import (
    AtomicCounter,
    CounterIncrement,
    RateLimitScope,
    RateLimitStore,
)
from tests.unit.caching.test_redis_batch import FakeNatives


@dataclass(frozen=True, slots=True)
class _WireScope:
    descriptor_key: str
    descriptor_value: str
    window: str
    requests: str
    tokens: str
    max_parallel_requests: str


_WIRE_SCOPES: Final = (
    _WireScope(
        descriptor_key="api_key",
        descriptor_value="88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
        window="{api_key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b}:window",
        requests="{api_key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b}:requests",
        tokens="{api_key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b}:tokens",
        max_parallel_requests=(
            "{api_key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b}:max_parallel_requests"
        ),
    ),
    _WireScope(
        descriptor_key="team_member",
        descriptor_value="team-1:user@example.com",
        window="{team_member:team-1:user@example.com}:window",
        requests="{team_member:team-1:user@example.com}:requests",
        tokens="{team_member:team-1:user@example.com}:tokens",
        max_parallel_requests="{team_member:team-1:user@example.com}:max_parallel_requests",
    ),
    _WireScope(
        descriptor_key="model_per_key",
        descriptor_value="sk-hash:anthropic/claude-sonnet-5-5",
        window="{model_per_key:sk-hash:anthropic/claude-sonnet-5-5}:window",
        requests="{model_per_key:sk-hash:anthropic/claude-sonnet-5-5}:requests",
        tokens="{model_per_key:sk-hash:anthropic/claude-sonnet-5-5}:tokens",
        max_parallel_requests="{model_per_key:sk-hash:anthropic/claude-sonnet-5-5}:max_parallel_requests",
    ),
    _WireScope(
        descriptor_key="tag",
        descriptor_value="tier}é",
        window="{tag:tier}é}:window",
        requests="{tag:tier}é}:requests",
        tokens="{tag:tier}é}:tokens",
        max_parallel_requests="{tag:tier}é}:max_parallel_requests",
    ),
)


def _mains_key(key: str, value: str, rate_limit_type: str) -> str:
    """``create_rate_limit_keys`` and the window key as ``main``'s limiter formatted them."""
    return f"{{{key}:{value}}}:{rate_limit_type}"


def _mains_keyslot(key: str) -> int:
    """``keyslot_for_redis_cluster`` from ``main``'s limiter."""
    start: Final = key.find("{")
    if start != -1:
        end: Final = key.find("}", start + 1)
        if end != -1 and end != start + 1:
            key = key[start + 1 : end]
    return binascii.crc_hqx(key.encode("utf-8"), 0) % 16384


@pytest.mark.parametrize("scope", _WIRE_SCOPES, ids=("api_key", "team_member", "model_per_key", "tag"))
def test_the_wire_keys_are_the_keys_mains_limiter_formats(scope: _WireScope) -> None:
    assert (scope.window, scope.requests, scope.tokens, scope.max_parallel_requests) == tuple(
        _mains_key(scope.descriptor_key, scope.descriptor_value, suffix)
        for suffix in ("window", "requests", "tokens", "max_parallel_requests")
    )


@pytest.mark.parametrize(("raw", "decoded"), (("1730000000", 1730000000), ("7", 7), ("7.5", 7.5), ("-3", -3)))
def test_the_wire_values_decode_as_mains_redis_cache_reads_them(raw: str, decoded: float) -> None:
    redis_cache: Final = RedisCache.__new__(RedisCache)
    read: Final = redis_cache._get_cache_logic(raw.encode())  # pyright: ignore[reportPrivateUsage]  # main's read codec is the contract
    assert read == decoded
    assert type(read) is type(decoded)


@pytest.mark.parametrize("force", [True, False])
def test_slot_groups_follow_mains_hash_tag_slots_only_when_grouping_is_forced(force: bool):
    store: Final = RateLimitStore(None)
    scopes: Final = (
        *(RateLimitScope(s.descriptor_key, s.descriptor_value) for s in _WIRE_SCOPES),
        RateLimitScope("api_key", "a"),
        RateLimitScope("team", "t"),
        RateLimitScope("api_key", "a"),
        RateLimitScope("tag", ""),
    )
    slots: Final = [_mains_keyslot(_mains_key(s.descriptor_key, s.descriptor_value, "window")) for s in scopes]
    expected: Final = tuple(
        tuple(index for index, slot in enumerate(slots) if slot == group_slot) for group_slot in dict.fromkeys(slots)
    )

    grouped: Final = store.slot_groups(scopes, force)

    assert grouped == (expected if force else (tuple(range(len(scopes))),))


def test_the_memory_tier_keeps_each_value_type_it_was_given():
    store: Final = RateLimitStore(None)
    scope: Final = RateLimitScope("api_key", "k1")
    store.set_local(scope.window, "1730000000", 60)
    store.set_local(scope.counter("requests"), 3, 60)
    store.set_local(scope.counter("tokens"), 2.5, 60)
    store.set_local(scope.gauge, {"slot-1": 1730000000.5}, 60)

    values: Final = store.local((scope.window, scope.counter("requests"), scope.counter("tokens"), scope.gauge))

    assert values == ("1730000000", 3, 2.5, {"slot-1": 1730000000.5})
    assert [type(value) for value in values] == [str, int, float, dict]
    assert store.local((RateLimitScope("api_key", "k2").counter("requests"),)) == (None,)


@pytest.mark.asyncio
async def test_without_redis_increments_add_in_memory_as_python_numbers_do():
    store: Final = RateLimitStore(None)
    tokens: Final = RateLimitScope("team", "t1").counter("tokens")

    assert await store.increment(CounterIncrement(tokens, 5, 60)) == 5
    await store.increment_many((CounterIncrement(tokens, 2.5, 60), CounterIncrement(tokens, -1, 60)))
    await store.increment_many_post_call((CounterIncrement(tokens, 1, 60),))

    assert store.local((tokens,)) == (7.5,)
    assert not store.has_redis
    assert not store.has_post_call_batch


@pytest.mark.asyncio
async def test_without_redis_nothing_is_declared_on_the_request_and_post_call_writes_are_refused():
    store: Final = RateLimitStore(None)
    counter: Final = RateLimitScope("api_key", "k1").counter("requests")
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives):  # pyright: ignore[reportArgumentType]  # test double
        assert store.declare_sliding_window((counter,), 0, 60) is None
        assert store.declare_check_and_increment((AtomicCounter(counter, 10, 1, 60, 60),)) is None
        assert not await store.token_increment_post_call((CounterIncrement(counter, 1, 60),))
        assert not await store.parallel_release_post_call((counter.scope,), "slot")

    assert natives.built == []
