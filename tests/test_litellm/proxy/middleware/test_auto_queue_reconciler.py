import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.insert(0, REPO_ROOT)
_loaded_litellm = sys.modules.get("litellm")
_loaded_litellm_path = getattr(_loaded_litellm, "__file__", None)
_needs_reload = _loaded_litellm_path is not None and not os.path.abspath(
    _loaded_litellm_path
).startswith(REPO_ROOT)
if _needs_reload:
    for _name in list(sys.modules):
        if _name == "litellm" or _name.startswith("litellm."):
            sys.modules.pop(_name, None)

import fakeredis.aioredis
import pytest
import pytest_asyncio
from redis.exceptions import RedisError

from litellm.proxy.middleware.auto_queue_reconciler import AutoQueueReconciler
from litellm.proxy.middleware.auto_queue_lease import ActiveLeaseHeartbeat
from litellm.proxy.middleware.auto_queue_scripts import AutoQueueRedis
from litellm.proxy.middleware.auto_queue_state import (
    active_key,
    active_lease_key,
    claim_key,
    queue_key,
    request_key,
    request_state_from_hash,
)


@pytest_asyncio.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def aqr_factory(redis):
    def _factory(
        *,
        default_max_concurrent: int = 1,
        ceiling: int = 10,
        scale_up_threshold: int = 3,
        scale_down_step: int = 1,
        max_queue_depth: int = 100,
    ):
        return AutoQueueRedis(
            redis=redis,
            default_max_concurrent=default_max_concurrent,
            ceiling=ceiling,
            scale_up_threshold=scale_up_threshold,
            scale_down_step=scale_down_step,
            max_queue_depth=max_queue_depth,
        )

    return _factory


@pytest.mark.asyncio
async def test_active_lease_heartbeat_promotes_claimed_request_to_active(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    model = "gpt-4o"

    await aqr.admit_or_enqueue(
        model=model,
        request_id="req-active",
        priority=10,
        deadline_at_ms=9_999_999_999_999,
        worker_id="worker-a",
    )
    await aqr.admit_or_enqueue(
        model=model,
        request_id="req-claimed",
        priority=10,
        deadline_at_ms=9_999_999_999_999,
        worker_id="worker-b",
    )
    transfer = await aqr.release_and_claim_next(model, "req-active")
    assert transfer.claimed_request_id == "req-claimed"

    heartbeat = ActiveLeaseHeartbeat(
        redis=redis,
        request_id="req-claimed",
        worker_id="worker-b",
        claim_token=transfer.claim_token,
    )

    refreshed = await heartbeat.refresh_once()

    assert refreshed is True
    request_state = request_state_from_hash(await redis.hgetall(request_key("req-claimed")))
    assert request_state.state == "active"
    assert request_state.claim_token == transfer.claim_token
    assert request_state.started_at_ms is not None

    lease = await redis.hgetall(active_lease_key("req-claimed"))
    assert lease[b"worker_id"] == b"worker-b"
    assert lease[b"claim_token"] == transfer.claim_token.encode()
    assert b"heartbeat_at_ms" in lease


@pytest.mark.asyncio
async def test_reconciler_releases_stale_active_lease_and_claims_next_waiter(aqr_factory, redis):
    aqr = aqr_factory(default_max_concurrent=1)
    model = "gpt-4o"

    active = await aqr.admit_or_enqueue(
        model=model,
        request_id="req-stale",
        priority=10,
        deadline_at_ms=9_999_999_999_999,
        worker_id="worker-a",
    )
    queued = await aqr.admit_or_enqueue(
        model=model,
        request_id="req-next",
        priority=10,
        deadline_at_ms=9_999_999_999_999,
        worker_id="worker-b",
    )
    assert active.decision == "admit_now"
    assert queued.decision == "queued"

    await redis.delete(active_lease_key("req-stale"))

    reconciler = AutoQueueReconciler(aqr=aqr, interval_seconds=60)
    reconciled = await reconciler.reconcile_once()

    assert reconciled == 1
    info = await aqr.get_model_info(model)
    assert info["active"] == 1
    assert info["queued"] == 0
    assert await redis.get(active_key(model)) == b"1"
    assert await redis.zrange(queue_key(model), 0, -1) == []

    stale_state = request_state_from_hash(await redis.hgetall(request_key("req-stale")))
    assert stale_state.state == "cancelled"
    assert stale_state.claim_token is None
    assert await redis.get(claim_key("req-stale")) is None

    next_state = request_state_from_hash(await redis.hgetall(request_key("req-next")))
    assert next_state.state == "claimed"
    assert next_state.claim_token is not None

    next_lease = await redis.hgetall(active_lease_key("req-next"))
    assert next_lease[b"worker_id"] == b"worker-b"
    assert next_lease[b"claim_token"] == next_state.claim_token.encode()


@pytest.mark.asyncio
async def test_reconciler_skips_expired_waiter_when_recovering_stale_active_lease(
    aqr_factory, redis
):
    aqr = aqr_factory(default_max_concurrent=1)
    model = "gpt-4o"

    active = await aqr.admit_or_enqueue(
        model=model,
        request_id="req-stale",
        priority=10,
        deadline_at_ms=9_999_999_999_999,
        worker_id="worker-a",
    )
    expired = await aqr.admit_or_enqueue(
        model=model,
        request_id="req-expired",
        priority=10,
        deadline_at_ms=1,
        worker_id="worker-b",
    )
    live = await aqr.admit_or_enqueue(
        model=model,
        request_id="req-live",
        priority=10,
        deadline_at_ms=9_999_999_999_999,
        worker_id="worker-c",
    )

    assert active.decision == "admit_now"
    assert expired.decision == "queued"
    assert live.decision == "queued"

    await redis.delete(active_lease_key("req-stale"))

    reconciler = AutoQueueReconciler(aqr=aqr, interval_seconds=60)
    reconciled = await reconciler.reconcile_once()

    assert reconciled == 1

    info = await aqr.get_model_info(model)
    assert info["active"] == 1
    assert info["queued"] == 0
    assert await redis.zrange(queue_key(model), 0, -1) == []

    stale_state = request_state_from_hash(await redis.hgetall(request_key("req-stale")))
    assert stale_state.state == "cancelled"
    assert stale_state.claim_token is None
    assert await redis.get(claim_key("req-stale")) is None

    expired_state = request_state_from_hash(
        await redis.hgetall(request_key("req-expired"))
    )
    assert expired_state.state == "timed_out"
    assert expired_state.claim_token is None
    assert await redis.get(claim_key("req-expired")) is None

    live_state = request_state_from_hash(await redis.hgetall(request_key("req-live")))
    assert live_state.state == "claimed"
    assert live_state.claim_token is not None

    live_lease = await redis.hgetall(active_lease_key("req-live"))
    assert live_lease[b"worker_id"] == b"worker-c"
    assert live_lease[b"claim_token"] == live_state.claim_token.encode()


@pytest.mark.asyncio
async def test_active_lease_heartbeat_retries_after_transient_refresh_error(monkeypatch, redis):
    heartbeat = ActiveLeaseHeartbeat(
        redis=redis,
        request_id="req-claimed",
        worker_id="worker-a",
        claim_token="claim-1",
        interval_seconds=0.01,
    )
    calls = []

    async def fake_sleep(_seconds):
        return None

    async def fake_refresh_once():
        calls.append("refresh")
        if len(calls) == 1:
            raise RedisError("temporary redis failure")
        return False

    monkeypatch.setattr("litellm.proxy.middleware.auto_queue_lease.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(heartbeat, "refresh_once", fake_refresh_once)

    await heartbeat._run()

    assert calls == ["refresh", "refresh"]
