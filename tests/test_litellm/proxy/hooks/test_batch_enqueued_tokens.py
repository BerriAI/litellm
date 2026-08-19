"""
LIT-5273: enqueued-token accounting for batch submissions.

Covers the ``BatchEnqueuedTokenStore`` (reserve / refund / reservation
records), the metadata-driven scope resolution, and the batch-id and
response-shape helpers the v3 limiter's post-call hooks rely on.
"""

import base64
import socket
import uuid
from types import SimpleNamespace

import pytest

from litellm.caching.caching import DualCache
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


def _local_redis_port() -> int | None:
    for port in (6379,):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return port
    return None


@pytest.mark.asyncio
@pytest.mark.skipif(_local_redis_port() is None, reason="requires a local Redis on 6379 for the Lua script path")
async def test_redis_lua_path_full_lifecycle():
    from litellm.caching.redis_cache import RedisCache

    port = _local_redis_port()
    redis_cache = RedisCache(host="127.0.0.1", port=port)
    store = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=redis_cache, default_in_memory_ttl=60))
    )
    key_scope = _scope(limit=100, key="api_key")
    team_scope = _scope(limit=50, key="team")

    over = await store.reserve(tokens=60, scopes=(key_scope, team_scope))
    assert over == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)

    reservation = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(reservation, BatchEnqueuedTokenReservation)
    assert isinstance(await store.reserve(tokens=1, scopes=(key_scope, team_scope)), BatchEnqueuedTokenOverLimit)

    batch_id = f"batch_{uuid.uuid4().hex}"
    await store.save_reservation(batch_id, reservation)
    popped = await store.pop_reservation(batch_id)
    assert popped == reservation
    assert await store.pop_reservation(batch_id) is None

    await store.refund(popped)
    refill = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
    assert isinstance(refill, BatchEnqueuedTokenReservation)
    await store.refund(refill)
