"""Live e2e: concurrent cold-counter reseeds keep the spend counter equal to DB spend (#26829).

Regression for the cross-pod spend-counter multiplication. Real requests build a key's
DB spend through the spend writer; the Redis spend counter then expires (the e2e proxy
sets a short default_redis_ttl) and goes cold. The proxy runs several workers sharing one
Redis, so a concurrent burst makes more than one worker reseed the same cold counter at
once. The fix seeds with SET NX - one worker initializes the counter at the DB spend and
the rest read it back - so the counter still equals the DB spend (plus the burst's own
small cost). The pre-#26829 additive reseed stacked the DB spend once per worker, leaving
the counter at ~N x the real spend.

The test reads the shared counter straight from Redis and asserts it equals the DB spend,
not a multiple. It also asserts the counter actually went cold before the burst, so a proxy
that never expires the counter (no short TTL) fails loudly instead of passing vacuously.
Skipped when the e2e Redis is not reachable.
"""

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter, ValidationError

from budget_client import BudgetClient
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager

if TYPE_CHECKING:
    import redis
    from redis.cluster import RedisCluster

pytestmark = pytest.mark.e2e

MODEL = "claude-haiku-4-5"
ACCUMULATE_CALLS = 24
BURST = 6
# proxy_batch_write_at (60s) flushes the spend to the DB and default_redis_ttl (20s)
# expires the counter; this waits out both.
COLD_WAIT_SECONDS = 80

_JSON_FLOAT: TypeAdapter[float] = TypeAdapter(float)


def _redis() -> "redis.Redis[str] | RedisCluster[str]":
    """The proxy's Redis. The deployed runner sets REDIS_HOST to the serverless
    ElastiCache, which is always TLS + cluster-mode; without it, fall back to a
    local standalone redis for docker-compose runs."""
    import redis

    host = os.getenv("REDIS_HOST")
    if not host:
        return redis.Redis(host="localhost", port=6380, decode_responses=True, socket_connect_timeout=2)

    from redis.cluster import RedisCluster

    return RedisCluster(
        host=host,
        port=int(os.getenv("REDIS_PORT", "6379")),
        ssl=True,
        decode_responses=True,
        socket_connect_timeout=2,
    )


def _parse_counter(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return _JSON_FLOAT.validate_json(text)
    except ValidationError:
        return None


def _spend_counter(rds: "redis.Redis[str] | RedisCluster[str]", key: str) -> float | None:
    """The shared spend counter for `key`, or None if it is cold.

    The gateway keys counters as ``spend:key:{sha256(raw_sk)}``, optionally under a
    redis namespace prefix. Cluster mode cannot SCAN all shards, so try the bare key
    and a few common namespaces; standalone redis uses a suffix SCAN.
    """
    from redis.cluster import RedisCluster

    digest = hashlib.sha256(key.encode()).hexdigest()
    suffix = f"spend:key:{digest}"
    if isinstance(rds, RedisCluster):
        for candidate in (suffix, f"litellm:{suffix}", f"litellm.caching:{suffix}"):
            parsed = _parse_counter(rds.get(candidate))
            if parsed is not None:
                return parsed
        return None

    matches = list(rds.scan_iter(match=f"*{suffix}"))
    if not matches:
        return None
    return _parse_counter(rds.get(matches[0]))


def _chat(client: BudgetClient, key: str) -> StreamingResponse:
    return client.chat(key, MODEL, f"reseed {unique_marker()}", max_tokens=16)


def _accumulate(client: BudgetClient, key: str, count: int) -> None:
    def one(_: int) -> StreamingResponse:
        return _chat(client, key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, range(count)))


@pytest.mark.covers("quota_management.budget.spend_counter.reseed_matches_db")
def test_cold_counter_reseed_keeps_counter_equal_to_db_spend(
    client: BudgetClient, resources: ResourceManager
) -> None:
    try:
        rds = _redis()
        rds.ping()
    except Exception as exc:  # noqa: BLE001 - any connect failure means skip
        pytest.skip(f"e2e redis not reachable (set REDIS_HOST/REDIS_PORT): {exc}")

    key = client.generate_key(max_budget=1.0, models=[MODEL])
    resources.defer(lambda: client.delete_key(key))

    _accumulate(client, key, ACCUMULATE_CALLS)
    time.sleep(COLD_WAIT_SECONDS)

    assert _spend_counter(rds, key) is None, (
        "the spend counter never went cold; default_redis_ttl must be short enough for it "
        "to expire, otherwise the burst reads a warm counter and the reseed is never exercised"
    )
    db_spend = client.gateway.key_info(key).spend or 0.0
    assert db_spend > 0, f"no DB spend accumulated from real calls: {db_spend}"

    burst_results = []
    barrier = Barrier(BURST)

    def one(_: int) -> StreamingResponse:
        barrier.wait()
        return _chat(client, key)

    with ThreadPoolExecutor(max_workers=BURST) as pool:
        burst_results = list(pool.map(one, range(BURST)))
    assert all(r.ok for r in burst_results), (
        "some burst calls failed; cannot exercise concurrent reseed. "
        f"statuses={[r.status_code for r in burst_results]}"
    )

    counter: float | None = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        counter = _spend_counter(rds, key)
        if counter is not None:
            break
        time.sleep(0.5)

    assert counter is not None, "the burst did not reseed the cold counter"
    assert db_spend * 0.95 <= counter < db_spend * 1.7, (
        f"redis spend counter {counter} does not equal DB spend {db_spend} (expected ~equal "
        f"plus the burst's small cost); a near-multiple means the cold-counter reseed stacked "
        f"the DB spend once per worker instead of seeding it once (#26829)"
    )
