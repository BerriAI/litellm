"""
LIT-5273: enqueued-token accounting for batch submissions.

Covers the ``BatchEnqueuedTokenStore`` (reserve / refund / reservation
records), the metadata-driven scope resolution, and the batch-id and
response-shape helpers the v3 limiter's post-call hooks rely on.
"""

import base64
import logging
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Final

import fakeredis
import pytest
from pydantic import TypeAdapter
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import NoScriptError
from redis.exceptions import TimeoutError as RedisTimeoutError

import litellm
from litellm.caching.caching import DualCache
from litellm.constants import BATCH_ENQUEUED_TOKEN_TTL_SECONDS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BatchEnqueuedTokenOverLimit,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenScope,
    BatchEnqueuedTokenStore,
    batch_response_view,
    canonical_provider_batch_id,
    resolve_batch_enqueued_token_scopes,
)
from litellm.proxy.utils import InternalUsageCache
from tests._support.fake_redis import fake_redis_cache

_RESERVATION_ADAPTER: Final = TypeAdapter(BatchEnqueuedTokenReservation)


def _in_memory_store() -> BatchEnqueuedTokenStore:
    return BatchEnqueuedTokenStore(internal_usage_cache=InternalUsageCache(DualCache(default_in_memory_ttl=60)))


def _scope(limit: int, key: str = "api_key") -> BatchEnqueuedTokenScope:
    return BatchEnqueuedTokenScope(key=key, value=f"{key}-{uuid.uuid4().hex}", limit=limit)


def test_scope_resolution_reads_key_and_team_metadata():
    user = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={"batch_enqueued_token_limit": 100},
        team_id="team-1",
        team_metadata={"batch_enqueued_token_limit": "150"},
    )
    scopes = resolve_batch_enqueued_token_scopes(user)
    assert scopes == (
        BatchEnqueuedTokenScope(key="api_key", value="hashed-key", limit=100),
        BatchEnqueuedTokenScope(key="team", value="team-1", limit=150),
    )


def test_scope_resolution_returns_empty_without_opt_in():
    assert resolve_batch_enqueued_token_scopes(UserAPIKeyAuth(api_key="k")) == ()
    assert resolve_batch_enqueued_token_scopes(UserAPIKeyAuth(api_key="k", metadata={}, team_metadata=None)) == ()


@pytest.mark.parametrize("bad_value", ["not-a-number", 0, -5, None, [1000]])
def test_scope_resolution_ignores_invalid_limits(bad_value):
    user = UserAPIKeyAuth(api_key="k", metadata={"batch_enqueued_token_limit": bad_value})
    assert resolve_batch_enqueued_token_scopes(user) == ()


def test_scope_resolution_skips_team_scope_without_team_id():
    user = UserAPIKeyAuth(api_key="k", team_metadata={"batch_enqueued_token_limit": 100})
    assert resolve_batch_enqueued_token_scopes(user) == ()


@pytest.mark.asyncio
async def test_reserve_rejects_once_allowance_is_exhausted():
    store = _in_memory_store()
    scope = _scope(limit=100)
    first = await store.reserve(tokens=80, scopes=(scope,))
    assert isinstance(first, BatchEnqueuedTokenReservation)
    second = await store.reserve(tokens=30, scopes=(scope,))
    assert second == BatchEnqueuedTokenOverLimit(scope=scope, enqueued=80)
    third = await store.reserve(tokens=20, scopes=(scope,))
    assert isinstance(third, BatchEnqueuedTokenReservation)


@pytest.mark.asyncio
async def test_reserve_is_all_or_nothing_across_scopes():
    store = _in_memory_store()
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")
    over = await store.reserve(tokens=60, scopes=(key_scope, team_scope))
    assert over == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)
    exact_fit = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(exact_fit, BatchEnqueuedTokenReservation)


@pytest.mark.asyncio
async def test_refund_restores_allowance_and_never_goes_negative():
    store = _in_memory_store()
    scope = _scope(limit=100)
    reservation = await store.reserve(tokens=30, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.refund(reservation)
    await store.refund(reservation)
    refill = await store.reserve(tokens=100, scopes=(scope,))
    assert isinstance(refill, BatchEnqueuedTokenReservation)
    assert isinstance(await store.reserve(tokens=1, scopes=(scope,)), BatchEnqueuedTokenOverLimit)


@pytest.mark.asyncio
async def test_reservation_record_roundtrip_pops_exactly_once():
    store = _in_memory_store()
    scope = _scope(limit=100)
    reservation = await store.reserve(tokens=40, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.save_reservation("batch_abc", reservation)
    assert await store.pop_reservation("batch_abc") == reservation
    assert await store.pop_reservation("batch_abc") is None
    assert await store.pop_reservation("batch_never_saved") is None


@pytest.mark.asyncio
async def test_zero_token_reserve_charges_nothing():
    store = _in_memory_store()
    scope = _scope(limit=100)
    empty = await store.reserve(tokens=0, scopes=(scope,))
    assert empty == BatchEnqueuedTokenReservation(tokens=0, scopes=(scope,))
    full = await store.reserve(tokens=100, scopes=(scope,))
    assert isinstance(full, BatchEnqueuedTokenReservation)


class _FaultyClient:
    def __init__(
        self,
        inner: fakeredis.FakeAsyncRedis,
        fail_from_call: Mapping[str, int],
        raise_after_landing_keys: frozenset[str],
    ) -> None:
        self.inner = inner
        self.fail_from_call = fail_from_call
        self.raise_after_landing_keys = raise_after_landing_keys
        self.calls: Mapping[str, int] = MappingProxyType({})

    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: str | bytes | float) -> object:
        if not (await self.inner.script_exists(sha))[0]:
            raise NoScriptError("No matching script")
        return await self._run(self.inner.evalsha, sha, numkeys, keys_and_args)

    async def eval(self, script: str, numkeys: int, *keys_and_args: str | bytes | float) -> object:
        return await self._run(self.inner.eval, script, numkeys, keys_and_args)

    async def _run(self, run, script: str, numkeys: int, keys_and_args: tuple[str | bytes | int | float, ...]):
        key: Final = str(keys_and_args[0])
        self.calls = MappingProxyType({**self.calls, key: self.calls.get(key, 0) + 1})
        if self.calls[key] >= self.fail_from_call.get(key, sys.maxsize):
            raise RedisConnectionError(f"simulated redis failure for {key}")
        reply: Final = await run(script, numkeys, *keys_and_args)
        if key in self.raise_after_landing_keys:
            raise RedisTimeoutError(f"simulated redis timeout after landing for {key}")
        return reply


def _redis_store(
    server: fakeredis.FakeServer,
    fail_from_call: Mapping[str, int] = MappingProxyType({}),
    raise_after_landing_keys: frozenset[str] = frozenset(),
    monotonic: Callable[[], float] = time.monotonic,
    namespace: str | None = None,
) -> tuple[BatchEnqueuedTokenStore, fakeredis.FakeAsyncRedis]:
    cache = fake_redis_cache(server, namespace=namespace)
    raw = fakeredis.FakeAsyncRedis(server=server)
    litellm.in_memory_llm_clients_cache.set_cache(
        key=cache._get_async_client_cache_key(),
        value=_FaultyClient(raw, fail_from_call=fail_from_call, raise_after_landing_keys=raise_after_landing_keys),
    )
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=cache, default_in_memory_ttl=60)),
        monotonic=monotonic,
    )
    return store, raw


def _counter(scope: BatchEnqueuedTokenScope) -> str:
    return f"batch_enqueued_tokens:{scope.key}:{scope.value}"


def _record(batch_id: str) -> str:
    return f"batch_enqueued_token_reservation:{batch_id}"


def _saved_reservation() -> BatchEnqueuedTokenReservation:
    return BatchEnqueuedTokenReservation(
        tokens=5,
        scopes=(
            BatchEnqueuedTokenScope(key="api_key", value='clé "one"', limit=100),
            BatchEnqueuedTokenScope(key="team", value="組/二", limit=200),
        ),
        backend="memory",
        owner="worker\n雪",
        reserved_at_monotonic=1000.25,
    )


_SAVED_DOCUMENT: Final = (
    '{"tokens":5,"scopes":[{"key":"api_key","value":"clé \\"one\\"","limit":100},'
    '{"key":"team","value":"組/二","limit":200}],"backend":"memory",'
    '"owner":"worker\\n雪","reserved_at_monotonic":1000.25}'
)


@pytest.mark.asyncio
async def test_redis_reserve_rolls_back_on_over_limit_and_refund_clears_the_counters():
    store, raw = _redis_store(fakeredis.FakeServer())
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")

    over = await store.reserve(tokens=60, scopes=(key_scope, team_scope))
    assert over == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)
    assert await raw.keys("*") == []

    fits = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(fits, BatchEnqueuedTokenReservation)
    assert fits.backend == "redis"
    assert await raw.get(_counter(key_scope)) == b"50"
    assert 0 < await raw.pttl(_counter(team_scope)) <= BATCH_ENQUEUED_TOKEN_TTL_SECONDS * 1000
    await store.refund(fits)
    assert await raw.keys("*") == []


@pytest.mark.asyncio
async def test_a_counter_an_old_pod_wrote_counts_against_the_allowance():
    store, raw = _redis_store(fakeredis.FakeServer())
    scope = _scope(limit=100)
    await raw.set(_counter(scope), 70, ex=BATCH_ENQUEUED_TOKEN_TTL_SECONDS)

    assert await store.reserve(tokens=40, scopes=(scope,)) == BatchEnqueuedTokenOverLimit(scope=scope, enqueued=70)
    assert isinstance(await store.reserve(tokens=30, scopes=(scope,)), BatchEnqueuedTokenReservation)
    assert await raw.get(_counter(scope)) == b"100"


@pytest.mark.asyncio
async def test_a_reservation_record_round_trips_with_old_pods():
    store, raw = _redis_store(fakeredis.FakeServer())
    scope = _scope(limit=100)
    reservation = BatchEnqueuedTokenReservation(tokens=5, scopes=(scope,), backend="redis")
    old_writer_document = _RESERVATION_ADAPTER.dump_json(reservation)
    await raw.set(_record("batch_old"), old_writer_document, ex=60)

    assert await store.pop_reservation("batch_old") == reservation
    assert await raw.get(_record("batch_old")) == b""

    await store.save_reservation("batch_new", reservation)
    assert await raw.get(_record("batch_new")) == _RESERVATION_ADAPTER.dump_json(reservation)


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", (None, "test-namespace"))
async def test_redis_reservation_bytes_include_unicode_and_pop_leaves_tombstone(
    namespace: str | None,
) -> None:
    store, raw = _redis_store(fakeredis.FakeServer(), monotonic=lambda: 1000.25, namespace=namespace)
    reservation: Final = _saved_reservation()
    batch_id: Final = "batch_wire"
    record_key: Final = f"{namespace}:{_record(batch_id)}" if namespace else _record(batch_id)

    await store.save_reservation(batch_id, reservation)

    assert await raw.get(record_key) == _SAVED_DOCUMENT.encode("utf-8")
    if namespace is not None:
        assert await raw.get(_record(batch_id)) is None
    popped: Final = await store.pop_reservation(batch_id)
    assert popped == reservation
    assert await raw.get(record_key) == b""
    assert 0 < await raw.pttl(record_key) <= BATCH_ENQUEUED_TOKEN_TTL_SECONDS * 1000


@pytest.mark.asyncio
async def test_malformed_local_reservation_is_consumed_without_a_cache_ghost() -> None:
    store, _ = _redis_store(fakeredis.FakeServer())
    record_key: Final = _record("batch_malformed_local")
    await store.internal_usage_cache.async_set_cache(
        key=record_key,
        value="{",
        litellm_parent_otel_span=None,
        local_only=True,
    )

    assert await store.pop_reservation("batch_malformed_local") is None
    assert (
        await store.internal_usage_cache.async_get_cache(key=record_key, litellm_parent_otel_span=None, local_only=True)
        is None
    )


@pytest.mark.asyncio
async def test_redis_outage_fallback_preserves_timestamp_and_consumes_once() -> None:
    server: Final = fakeredis.FakeServer()
    store, _ = _redis_store(server, monotonic=lambda: 1000.25)
    reservation: Final = _saved_reservation()
    server.connected = False

    await store.save_reservation("batch_outage", reservation)

    popped: Final = await store.pop_reservation("batch_outage")
    assert isinstance(popped, BatchEnqueuedTokenReservation)
    assert popped == reservation
    assert popped.reserved_at_monotonic == reservation.reserved_at_monotonic
    assert await store.pop_reservation("batch_outage") is None
    assert (
        await store.internal_usage_cache.async_get_cache(
            key=_record("batch_outage"), litellm_parent_otel_span=None, local_only=True
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_reservation_record_that_does_not_decode_is_discarded(caplog):
    store, raw = _redis_store(fakeredis.FakeServer())
    await raw.set(_record("batch_bad"), "not a reservation", ex=60)

    with caplog.at_level(logging.WARNING):
        assert await store.pop_reservation("batch_bad") is None

    assert any("Discarding malformed" in record.getMessage() for record in caplog.records)
    assert await raw.get(_record("batch_bad")) == b""


@pytest.mark.asyncio
async def test_partial_redis_reserve_failure_rolls_back_and_grants_in_memory():
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")
    store, raw = _redis_store(fakeredis.FakeServer(), fail_from_call=MappingProxyType({_counter(team_scope): 1}))

    outcome = await store.reserve(tokens=10, scopes=(key_scope, team_scope))
    assert isinstance(outcome, BatchEnqueuedTokenReservation)
    assert outcome.backend == "memory"
    assert await raw.exists(_counter(key_scope)) == 0

    await store.refund(outcome)
    refilled = await store.reserve(tokens=50, scopes=(team_scope,))
    assert isinstance(refilled, BatchEnqueuedTokenReservation)
    assert refilled.backend == "memory"


@pytest.mark.asyncio
async def test_over_limit_verdict_survives_a_failing_rollback():
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=5, key="team")
    store, raw = _redis_store(fakeredis.FakeServer(), fail_from_call=MappingProxyType({_counter(key_scope): 2}))

    outcome = await store.reserve(tokens=10, scopes=(key_scope, team_scope))
    assert outcome == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)
    assert await raw.get(_counter(key_scope)) == b"10"


@pytest.mark.asyncio
async def test_pop_falls_back_to_local_record_when_the_redis_save_failed():
    scope = _scope(limit=100)
    store, raw = _redis_store(
        fakeredis.FakeServer(), fail_from_call=MappingProxyType({_record("batch_local_record"): 1})
    )

    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.save_reservation("batch_local_record", reservation)
    assert await raw.exists(_record("batch_local_record")) == 0

    popped = await store.pop_reservation("batch_local_record")
    assert popped == reservation
    await store.refund(popped)
    assert await raw.exists(_counter(scope)) == 0
    assert await store.pop_reservation("batch_local_record") is None


@pytest.mark.asyncio
async def test_local_ghost_left_by_landed_save_never_refunds_twice():
    scope = _scope(limit=100)
    store, raw = _redis_store(fakeredis.FakeServer(), raise_after_landing_keys=frozenset({_record("batch_ghost")}))

    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    await store.save_reservation("batch_ghost", reservation)
    assert await raw.get(_record("batch_ghost"))

    first = await store.pop_reservation("batch_ghost")
    assert first == reservation
    await store.refund(first)
    assert await raw.exists(_counter(scope)) == 0
    assert await raw.get(_record("batch_ghost")) == b""

    assert await store.pop_reservation("batch_ghost") is None
    assert (
        await store.internal_usage_cache.async_get_cache(
            key=_record("batch_ghost"), litellm_parent_otel_span=None, local_only=True
        )
        is None
    )
    assert await store.pop_reservation("batch_ghost") is None
    assert isinstance(await store.reserve(tokens=100, scopes=(scope,)), BatchEnqueuedTokenReservation)
    assert await raw.get(_counter(scope)) == b"100"


@pytest.mark.asyncio
async def test_record_ttl_shrinks_by_elapsed_time_so_stale_records_never_outlive_their_counters():
    scope = _scope(limit=100)
    ticks = iter((1_000.0, 1_030.5))
    store, raw = _redis_store(fakeredis.FakeServer(), monotonic=lambda: next(ticks))
    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert reservation.reserved_at_monotonic == 1_000.0
    await store.save_reservation("batch_ttl_clamp", reservation)
    assert await raw.ttl(_record("batch_ttl_clamp")) == BATCH_ENQUEUED_TOKEN_TTL_SECONDS - 31
    assert await store.pop_reservation("batch_ttl_clamp") == reservation


@pytest.mark.asyncio
async def test_memory_refund_skips_reservations_granted_by_another_worker():
    store = _in_memory_store()
    scope = _scope(limit=100)
    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert reservation.backend == "memory"
    assert reservation.owner

    foreign: Final = BatchEnqueuedTokenReservation(
        tokens=60, scopes=reservation.scopes, backend="memory", owner="another-worker"
    )
    await store.refund(foreign)
    assert await store.reserve(tokens=50, scopes=(scope,)) == BatchEnqueuedTokenOverLimit(scope=scope, enqueued=60)

    await store.refund(reservation)
    assert isinstance(await store.reserve(tokens=100, scopes=(scope,)), BatchEnqueuedTokenReservation)


@pytest.mark.asyncio
async def test_failed_redis_refund_leaves_local_counters_untouched():
    scope = _scope(limit=100)
    store, raw = _redis_store(fakeredis.FakeServer(), fail_from_call=MappingProxyType({_counter(scope): 2}))

    reservation = await store.reserve(tokens=60, scopes=(scope,))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert reservation.backend == "redis"
    store.internal_usage_cache.dual_cache.in_memory_cache.set_cache(key=_counter(scope), value=45)

    await store.refund(reservation)
    assert store.internal_usage_cache.dual_cache.in_memory_cache.get_cache(key=_counter(scope)) == 45


@pytest.mark.asyncio
async def test_pop_reservation_defaults_legacy_records_to_redis_backend():
    store, raw = _redis_store(fakeredis.FakeServer())
    legacy = '{"tokens": 5, "scopes": [{"key": "api_key", "value": "k", "limit": 10}]}'
    await raw.set(_record("batch_legacy"), legacy, ex=60)
    popped = await store.pop_reservation("batch_legacy")
    assert popped == BatchEnqueuedTokenReservation(
        tokens=5, scopes=(BatchEnqueuedTokenScope(key="api_key", value="k", limit=10),), backend="redis"
    )


def test_canonical_provider_batch_id_passes_raw_ids_through():
    assert canonical_provider_batch_id("batch_abc123") == "batch_abc123"


def test_canonical_provider_batch_id_decodes_unified_batch_ids():
    unified = "litellm_proxy;model_id:m-1;llm_batch_id:batch_prov_9;llm_output_file_id:file-9"
    encoded = base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")
    assert canonical_provider_batch_id(encoded) == "batch_prov_9"


def test_canonical_provider_batch_id_decodes_model_embedded_ids():
    from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

    encoded = encode_file_id_with_model(file_id="batch_prov_7", model="my-alias", id_type="batch")
    assert canonical_provider_batch_id(encoded) == "batch_prov_7"


def test_batch_response_view_accepts_batch_objects_only():
    batch = SimpleNamespace(id="batch_1", status="completed", object="batch")
    view = batch_response_view(batch)
    assert view is not None and view.id == "batch_1" and view.status == "completed"
    assert batch_response_view({"id": "chatcmpl-1", "object": "chat.completion"}) is None
    assert batch_response_view(None) is None
    assert batch_response_view("batch_1") is None


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_keeps_reservations_in_memory_without_a_warning(caplog):
    scope = _scope(limit=100)
    cache = fake_redis_cache(fakeredis.FakeServer())
    for _ in range(cache._circuit_breaker.failure_threshold):
        cache._circuit_breaker.record_failure()
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=cache, default_in_memory_ttl=60))
    )
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        reservation = await store.reserve(tokens=60, scopes=(scope,))
        assert isinstance(reservation, BatchEnqueuedTokenReservation)
        assert reservation.backend == "memory"
        await store.save_reservation("batch_quiet", reservation)
        assert await store.pop_reservation("batch_quiet") == reservation

    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert sum("circuit breaker is open" in record.getMessage() for record in caplog.records) == 3
