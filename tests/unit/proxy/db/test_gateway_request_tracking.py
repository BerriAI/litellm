"""
Tests for the gateway request (SGR) fold and its commit to
LiteLLM_DailyGatewayRequests.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from itertools import chain
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import (
    GATEWAY_REQUESTS_MAX_ROWS_PER_UPSERT,
    MAX_REDIS_BUFFER_DEQUEUE_COUNT,
    REDIS_GATEWAY_REQUESTS_BUFFER_KEY,
    REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2,
)
from litellm.proxy.db.gateway_request_tracking import (
    GATEWAY_REQUESTS_JOB_NAME,
    GatewayRequestAccumulator,
    GatewayRequestRedisBuffer,
    commit_gateway_requests_to_db,
    flush_gateway_requests,
)
from litellm.proxy.db.log_db_metrics import record_db_io
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory
from litellm.types.proxy.gateway_requests import GatewayRequestCounts, GatewayRequestKey

_NOW: Final = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)
_DAY: Final = _NOW.strftime("%Y-%m-%d")


def _accumulator() -> GatewayRequestAccumulator:
    return GatewayRequestAccumulator(clock=lambda: _NOW)


def _record(accumulator: GatewayRequestAccumulator, status_code: int, **overrides) -> None:
    accumulator.record(
        category=overrides.get("category", BillableCategory.LLM),
        route=overrides.get("route", "/chat/completions"),
        status_code=status_code,
    )


# ── fold ──────────────────────────────────────────────────────────────────────


def test_folds_repeated_requests_into_one_key():
    acc = _accumulator()
    for _ in range(3):
        _record(acc, 200)
    _record(acc, 500)

    snapshot = acc.drain()
    assert snapshot == {
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=200): (
            GatewayRequestCounts(successful_requests=3, failed_requests=0)
        ),
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=500): (
            GatewayRequestCounts(successful_requests=0, failed_requests=1)
        ),
    }


def test_folds_failed_status_codes_into_distinct_keys_for_the_same_route():
    acc: Final = _accumulator()
    _record(acc, 429)
    _record(acc, 500)

    assert acc.drain() == {
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=429): (
            GatewayRequestCounts(successful_requests=0, failed_requests=1)
        ),
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=500): (
            GatewayRequestCounts(successful_requests=0, failed_requests=1)
        ),
    }


def test_records_each_request_under_the_date_of_its_clock_read():
    accumulator: Final = GatewayRequestAccumulator(
        clock=iter(
            (
                datetime(2026, 1, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
                datetime(2026, 2, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
            )
        ).__next__
    )
    _record(accumulator, 200)
    _record(accumulator, 200)

    assert accumulator.drain() == {
        GatewayRequestKey(
            date="2026-01-31", category="llm", route="/chat/completions", status_code=200
        ): GatewayRequestCounts(successful_requests=1, failed_requests=0),
        GatewayRequestKey(
            date="2026-02-01", category="llm", route="/chat/completions", status_code=200
        ): GatewayRequestCounts(successful_requests=1, failed_requests=0),
    }


@pytest.mark.parametrize(
    "status_code, expected_successful, expected_failed",
    [(200, 1, 0), (201, 1, 0), (204, 1, 0), (299, 1, 0), (300, 0, 1), (400, 0, 1), (500, 0, 1)],
)
def test_success_boundary_is_2xx(status_code: int, expected_successful: int, expected_failed: int):
    acc = _accumulator()
    _record(acc, status_code)
    counts = next(iter(acc.drain().values()))
    assert (counts.successful_requests, counts.failed_requests) == (expected_successful, expected_failed)


def test_distinct_dimensions_do_not_merge():
    acc = _accumulator()
    _record(acc, 200, route="/chat/completions")
    _record(acc, 200, route="/embeddings")
    _record(acc, 200, category=BillableCategory.MCP, route="/mcp")
    assert len(acc.drain()) == 3


def test_drain_empties_the_fold():
    acc = _accumulator()
    _record(acc, 200)
    assert len(acc.drain()) == 1
    assert acc.drain() == {}


def test_drain_snapshot_is_not_mutated_by_later_records():
    acc = _accumulator()
    _record(acc, 200)
    snapshot = acc.drain()
    _record(acc, 200)
    assert next(iter(snapshot.values())).successful_requests == 1


# ── commit ────────────────────────────────────────────────────────────────────


class FakeDB:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    async def execute_raw(self, query: str, *args: object) -> int:
        record_db_io()
        self.statements.append((query, args))
        return len(args) // 5


class FakePrismaClient:
    def __init__(self) -> None:
        self.db = FakeDB()


def _rows_written(client: FakePrismaClient) -> list[tuple[object, ...]]:
    """Every (date, category, route, successful, failed) tuple the database received, in statement order."""
    return list(chain.from_iterable(_main_rows_for_statement(query, params) for query, params in client.db.statements))


def _main_param_offset(query: str) -> int:
    main_query: Final[str] = query[query.index('INSERT INTO "LiteLLM_DailyGatewayRequests"') :]
    first_placeholder: Final[int] = main_query.index("$")
    placeholder_type: Final[int] = main_query.index("::", first_placeholder)
    offset: Final[int] = int(main_query[first_placeholder + 1 : placeholder_type]) - 1
    return offset


def _main_rows_for_statement(query: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
    offset: Final[int] = _main_param_offset(query)
    return [params[offset + i : offset + i + 5] for i in range(0, len(params) - offset, 5)]


def _failure_rows_written(client: FakePrismaClient) -> list[tuple[object, ...]]:
    return list(
        chain.from_iterable(_failure_rows_for_statement(query, params) for query, params in client.db.statements)
    )


def _failure_rows_for_statement(query: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
    if 'INSERT INTO "LiteLLM_DailyGatewayFailedRequests"' not in query:
        return []
    return [params[i : i + 5] for i in range(0, _main_param_offset(query), 5)]


def test_commit_increments_with_a_single_statement_for_the_whole_snapshot():
    """One statement per flush is the whole point: the previous per-key upsert cost
    the primary (workers x routes) statements per interval."""
    client = FakePrismaClient()
    snapshot = {
        GatewayRequestKey(date="2026-08-01", category="llm", route=route): (
            GatewayRequestCounts(successful_requests=7, failed_requests=2)
        )
        for route in ("/chat/completions", "/embeddings", "/responses", "/v1/messages", "/mcp")
    }

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    assert len(client.db.statements) == 1
    sql, params = client.db.statements[0]
    assert sql.count("ON CONFLICT") == 1
    assert sql.count("(NOW() AT TIME ZONE 'UTC'))") == 5
    assert len(params) == 25


def test_commit_sql_adds_to_the_existing_row_instead_of_replacing_it():
    """A worker only knows its own share; the SQL must add EXCLUDED onto the stored count."""
    client = FakePrismaClient()
    snapshot = {
        GatewayRequestKey(date="2026-08-01", category="llm", route="/chat/completions"): (
            GatewayRequestCounts(successful_requests=7, failed_requests=2)
        )
    }

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    sql, params = client.db.statements[0]
    assert 'INSERT INTO "LiteLLM_DailyGatewayRequests"' in sql
    assert 'ON CONFLICT ("date", "category", "route") DO UPDATE SET' in sql
    assert (
        '"successful_requests" = "LiteLLM_DailyGatewayRequests"."successful_requests" + EXCLUDED."successful_requests"'
        in sql
    )
    assert '"failed_requests" = "LiteLLM_DailyGatewayRequests"."failed_requests" + EXCLUDED."failed_requests"' in sql
    assert params == ("2026-08-01", "llm", "/chat/completions", 7, 2)


def test_commit_placeholders_line_up_with_params():
    """$n positions are generated per row; a drift here silently swaps a route for a count."""
    client = FakePrismaClient()
    snapshot = {
        GatewayRequestKey(date="2026-08-01", category="llm", route="/chat/completions"): (
            GatewayRequestCounts(successful_requests=1, failed_requests=0)
        ),
        GatewayRequestKey(date="2026-08-01", category="mcp", route="/mcp"): (
            GatewayRequestCounts(successful_requests=0, failed_requests=3)
        ),
    }

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    sql, params = client.db.statements[0]
    assert "($1::text, $2::text, $3::text, $4::bigint, $5::bigint," in sql
    assert "($6::text, $7::text, $8::text, $9::bigint, $10::bigint," in sql
    assert "$11" not in sql
    assert params == ("2026-08-01", "llm", "/chat/completions", 1, 0, "2026-08-01", "mcp", "/mcp", 0, 3)


def test_commit_upserts_per_status_and_sums_main_table_counts_in_one_statement():
    client: Final = FakePrismaClient()
    accumulator: Final = _accumulator()
    _record(accumulator, 200)
    _record(accumulator, 201)
    _record(accumulator, 429)
    _record(accumulator, 500)

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=accumulator.drain()))

    assert len(client.db.statements) == 1
    sql, params = client.db.statements[0]
    assert sql.startswith('WITH failed AS (INSERT INTO "LiteLLM_DailyGatewayFailedRequests"')
    assert sql.index('INSERT INTO "LiteLLM_DailyGatewayFailedRequests"') < sql.index(
        'INSERT INTO "LiteLLM_DailyGatewayRequests"'
    )
    assert "$1::text, $2::text, $3::text, $4::integer, $5::bigint" in sql
    assert "$6::text, $7::text, $8::text, $9::integer, $10::bigint" in sql
    assert "$11::text, $12::text, $13::text, $14::bigint, $15::bigint" in sql
    assert "$16" not in sql
    assert params == (
        _DAY,
        "llm",
        "/chat/completions",
        429,
        1,
        _DAY,
        "llm",
        "/chat/completions",
        500,
        1,
        _DAY,
        "llm",
        "/chat/completions",
        2,
        2,
    )
    assert _failure_rows_written(client) == [
        (_DAY, "llm", "/chat/completions", 429, 1),
        (_DAY, "llm", "/chat/completions", 500, 1),
    ]
    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", 2, 2)]


def test_success_only_snapshot_uses_only_the_existing_table_upsert():
    client: Final = FakePrismaClient()
    accumulator: Final = _accumulator()
    _record(accumulator, 200)
    snapshot: Final = accumulator.drain()

    assert snapshot == {
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=200): (
            GatewayRequestCounts(successful_requests=1, failed_requests=0)
        )
    }

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    sql, _ = client.db.statements[0]
    assert sql.startswith('INSERT INTO "LiteLLM_DailyGatewayRequests"')
    assert "LiteLLM_DailyGatewayFailedRequests" not in sql
    assert _failure_rows_written(client) == []


def test_commit_is_deterministically_ordered():
    """Concurrent writers must touch rows in the same order or they deadlock."""
    client = FakePrismaClient()
    keys = [
        GatewayRequestKey(date="2026-08-02", category="llm", route="/embeddings"),
        GatewayRequestKey(date="2026-08-01", category="mcp", route="/mcp"),
        GatewayRequestKey(date="2026-08-01", category="llm", route="/chat/completions"),
    ]
    snapshot = {key: GatewayRequestCounts(successful_requests=1, failed_requests=0) for key in keys}

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    written_order = [(row[0], row[1]) for row in _rows_written(client)]
    assert written_order == [("2026-08-01", "llm"), ("2026-08-01", "mcp"), ("2026-08-02", "llm")]


def _backlog(size: int) -> dict[GatewayRequestKey, GatewayRequestCounts]:
    return {
        GatewayRequestKey(date=_DAY, category="llm", route=f"/route-{index:05d}", status_code=200): (
            GatewayRequestCounts(successful_requests=1, failed_requests=0)
        )
        for index in range(size)
    }


def test_commit_splits_a_backlog_into_bounded_statements():
    client: Final = MagicMock()
    client.db.execute_raw = AsyncMock()

    uncommitted: Final = asyncio.run(
        commit_gateway_requests_to_db(prisma_client=client, snapshot=_backlog(GATEWAY_REQUESTS_MAX_ROWS_PER_UPSERT + 1))
    )

    assert uncommitted == {}
    assert client.db.execute_raw.await_count == 2
    assert len(client.db.execute_raw.await_args_list[0].args) - 1 == GATEWAY_REQUESTS_MAX_ROWS_PER_UPSERT * 5


def test_commit_keeps_a_route_total_and_its_status_rows_in_one_statement():
    client: Final = MagicMock()
    client.db.execute_raw = AsyncMock()
    snapshot: Final = {
        **_backlog(GATEWAY_REQUESTS_MAX_ROWS_PER_UPSERT - 1),
        GatewayRequestKey(date=_DAY, category="llm", route="/last", status_code=429): (
            GatewayRequestCounts(successful_requests=0, failed_requests=1)
        ),
        GatewayRequestKey(date=_DAY, category="llm", route="/last", status_code=500): (
            GatewayRequestCounts(successful_requests=0, failed_requests=1)
        ),
    }

    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    assert client.db.execute_raw.await_count == 2
    with_last: Final = [
        awaited.args for awaited in client.db.execute_raw.await_args_list if "/last" in awaited.args[1:]
    ]
    assert len(with_last) == 1
    assert with_last[0][1:].count("/last") == 3


def test_commit_hands_back_only_the_chunks_that_did_not_land():
    client: Final = MagicMock()
    client.db.execute_raw = AsyncMock(side_effect=(None, RuntimeError("db down")))
    snapshot: Final = _backlog(GATEWAY_REQUESTS_MAX_ROWS_PER_UPSERT + 1)

    uncommitted: Final = asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot=snapshot))

    last_key: Final = GatewayRequestKey(
        date=_DAY, category="llm", route=f"/route-{GATEWAY_REQUESTS_MAX_ROWS_PER_UPSERT:05d}", status_code=200
    )
    assert uncommitted == {last_key: snapshot[last_key]}
    assert client.db.execute_raw.await_count == 2


def test_commit_skips_the_database_entirely_when_nothing_accumulated():
    client = FakePrismaClient()
    asyncio.run(commit_gateway_requests_to_db(prisma_client=client, snapshot={}))
    assert client.db.statements == []


# ── flush ─────────────────────────────────────────────────────────────────────


def test_flush_drains_and_commits():
    client = FakePrismaClient()
    acc = _accumulator()
    _record(acc, 200)

    asyncio.run(flush_gateway_requests(client, acc))

    assert len(client.db.statements) == 1
    assert acc.drain() == {}


class ExplodingDB:
    async def execute_raw(self, query: str, *args: object) -> int:
        raise RuntimeError("db gone")


class ExplodingClient:
    db = ExplodingDB()


def test_flush_swallows_commit_failure_so_the_scheduler_survives():
    acc = _accumulator()
    _record(acc, 200)

    asyncio.run(flush_gateway_requests(ExplodingClient(), acc))


def test_failed_flush_keeps_counts_for_the_next_attempt():
    """A dropped flush would silently undercount the SGR source of truth."""
    acc = _accumulator()
    _record(acc, 200)
    _record(acc, 500)

    asyncio.run(flush_gateway_requests(ExplodingClient(), acc))

    client = FakePrismaClient()
    asyncio.run(flush_gateway_requests(client, acc))

    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", 1, 1)]


def test_restored_counts_merge_with_requests_recorded_meanwhile():
    acc = _accumulator()
    _record(acc, 200)
    asyncio.run(flush_gateway_requests(ExplodingClient(), acc))

    _record(acc, 200)
    client = FakePrismaClient()
    asyncio.run(flush_gateway_requests(client, acc))

    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", 2, 0)]


class ExplodingDBWithInFlightRequest:
    """Fails the write after a request has been recorded while it was in flight."""

    def __init__(self, accumulator: GatewayRequestAccumulator) -> None:
        self.accumulator = accumulator

    async def execute_raw(self, query: str, *args: object) -> int:
        _record(self.accumulator, 500)
        raise RuntimeError("db gone")


class ExplodingClientWithInFlightRequest:
    def __init__(self, accumulator: GatewayRequestAccumulator) -> None:
        self.db = ExplodingDBWithInFlightRequest(accumulator)


def test_restore_keeps_requests_recorded_while_the_failed_write_was_in_flight():
    acc = _accumulator()
    _record(acc, 200)
    asyncio.run(flush_gateway_requests(ExplodingClientWithInFlightRequest(acc), acc))

    client = FakePrismaClient()
    asyncio.run(flush_gateway_requests(client, acc))

    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", 1, 1)]


# ── redis buffer ──────────────────────────────────────────────────────────────


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}

    async def async_rpush(self, key: str, values: list[str]) -> int:
        self.lists.setdefault(key, []).extend(values)
        return len(self.lists[key])

    async def async_lpop(self, key: str, count: int) -> list[str] | None:
        queue = self.lists.get(key, [])
        if not queue:
            return None
        popped, self.lists[key] = queue[:count], queue[count:]
        return popped


class FakePodLock:
    def __init__(self, *, leader: bool) -> None:
        self.leader = leader
        self.held: list[str] = []
        self.released: list[str] = []

    async def acquire_lock(self, cronjob_id: str) -> bool:
        self.held.append(cronjob_id)
        return self.leader

    async def release_lock(self, cronjob_id: str) -> None:
        self.released.append(cronjob_id)


class FakeLease:
    """Redis-side view of the job lock: SET NX by pod id, re-entrant for the holder, freed only by release or TTL."""

    def __init__(self) -> None:
        self.holder: str | None = None


class FakeLeasePodLock:
    def __init__(self, lease: FakeLease, pod_id: str) -> None:
        self.lease = lease
        self.pod_id = pod_id

    async def acquire_lock(self, cronjob_id: str) -> bool:
        if self.lease.holder is None:
            self.lease.holder = self.pod_id
        return self.lease.holder == self.pod_id

    async def release_lock(self, cronjob_id: str) -> None:
        if self.lease.holder == self.pod_id:
            self.lease.holder = None


def _buffer(redis: FakeRedis, *, leader: bool) -> tuple[GatewayRequestRedisBuffer, FakePodLock]:
    lock = FakePodLock(leader=leader)
    return GatewayRequestRedisBuffer(redis_cache=redis, pod_lock_manager=lock), lock  # pyright: ignore[reportArgumentType]  # duck-typed fakes


def test_non_leader_workers_push_to_redis_and_never_touch_the_database():
    redis = FakeRedis()
    client = FakePrismaClient()
    for _ in range(3):
        acc = _accumulator()
        _record(acc, 200)
        buffer, _ = _buffer(redis, leader=False)
        asyncio.run(flush_gateway_requests(client, acc, buffer))

    assert client.db.statements == []
    assert len(redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2]) == 3
    assert redis.lists.get(REDIS_GATEWAY_REQUESTS_BUFFER_KEY, []) == []


def test_push_writes_six_field_rows_to_the_v2_key():
    redis: Final = FakeRedis()
    accumulator: Final = _accumulator()
    _record(accumulator, 429)
    buffer, _ = _buffer(redis, leader=False)

    asyncio.run(buffer.push(accumulator.drain()))

    assert json.loads(redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2][0]) == [
        [_DAY, "llm", "/chat/completions", 429, 0, 1]
    ]
    assert redis.lists.get(REDIS_GATEWAY_REQUESTS_BUFFER_KEY, []) == []


def test_leader_combines_legacy_and_v2_buffer_rows():
    redis: Final = FakeRedis()
    redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY] = [json.dumps(((_DAY, "llm", "/chat/completions", 2, 1),))]
    redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2] = [json.dumps(((_DAY, "llm", "/chat/completions", 429, 0, 2),))]
    client: Final = FakePrismaClient()
    leader, _ = _buffer(redis, leader=True)

    asyncio.run(flush_gateway_requests(client, _accumulator(), leader))

    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", 2, 3)]
    assert _failure_rows_written(client) == [(_DAY, "llm", "/chat/completions", 429, 2)]
    assert redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY] == []
    assert redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2] == []


def test_leader_folds_every_workers_snapshot_into_one_statement():
    """Fifty workers each flushing the same routes must cost the primary one statement, not fifty."""
    redis = FakeRedis()
    client = FakePrismaClient()
    for _ in range(50):
        acc = _accumulator()
        _record(acc, 200)
        _record(acc, 500, route="/responses")
        buffer, _ = _buffer(redis, leader=False)
        asyncio.run(flush_gateway_requests(client, acc, buffer))

    leader_acc = _accumulator()
    _record(leader_acc, 200)
    leader, lock = _buffer(redis, leader=True)
    asyncio.run(flush_gateway_requests(client, leader_acc, leader))

    assert len(client.db.statements) == 1
    assert _rows_written(client) == [
        (_DAY, "llm", "/chat/completions", 51, 0),
        (_DAY, "llm", "/responses", 0, 50),
    ]
    assert redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2] == []
    assert lock.held == [GATEWAY_REQUESTS_JOB_NAME]
    assert lock.released == []


def test_leader_keeps_the_lease_so_staggered_pods_cost_one_statement_per_interval():
    """Pods flush on their own clocks; without the lease each one would win the lock in turn and commit alone."""
    redis = FakeRedis()
    client = FakePrismaClient()
    lease = FakeLease()
    pods = tuple(
        GatewayRequestRedisBuffer(redis_cache=redis, pod_lock_manager=FakeLeasePodLock(lease, f"pod-{i}"))  # pyright: ignore[reportArgumentType]  # duck-typed fakes
        for i in range(4)
    )

    for _interval in range(3):
        for pod in pods:
            acc = _accumulator()
            _record(acc, 200)
            asyncio.run(flush_gateway_requests(client, acc, pod))

    assert lease.holder == "pod-0"
    assert len(client.db.statements) == 3
    assert [row[3] for row in _rows_written(client)] == [1, 4, 4]
    assert len(redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2]) == 3


def test_leader_drains_a_backlog_deeper_than_one_capped_pop():
    """More workers than MAX_REDIS_BUFFER_DEQUEUE_COUNT must not leave a growing tail queued behind the cap."""
    redis = FakeRedis()
    client = FakePrismaClient()
    workers = MAX_REDIS_BUFFER_DEQUEUE_COUNT * 2 + 1
    for _ in range(workers):
        acc = _accumulator()
        _record(acc, 200)
        buffer, _ = _buffer(redis, leader=False)
        asyncio.run(flush_gateway_requests(client, acc, buffer))

    leader, _ = _buffer(redis, leader=True)
    asyncio.run(flush_gateway_requests(client, _accumulator(), leader))

    assert len(client.db.statements) == 1
    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", workers, 0)]
    assert redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2] == []


def test_leader_with_nothing_buffered_writes_nothing():
    redis = FakeRedis()
    client = FakePrismaClient()
    leader, lock = _buffer(redis, leader=True)

    asyncio.run(flush_gateway_requests(client, _accumulator(), leader))

    assert client.db.statements == []
    assert lock.released == []


def test_leader_requeues_to_redis_when_the_database_commit_fails():
    """Counts popped from Redis are gone from every worker; a failed commit must put them back."""
    redis = FakeRedis()
    acc = _accumulator()
    _record(acc, 200)
    _record(acc, 200)
    leader, lock = _buffer(redis, leader=True)

    asyncio.run(flush_gateway_requests(ExplodingClient(), acc, leader))

    assert len(redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2]) == 1
    assert lock.released == []
    assert acc.drain() == {}

    client = FakePrismaClient()
    retry, _ = _buffer(redis, leader=True)
    asyncio.run(flush_gateway_requests(client, _accumulator(), retry))
    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", 2, 0)]


class ExplodingRedis(FakeRedis):
    async def async_rpush(self, key: str, values: list[str]) -> int:
        raise RuntimeError("redis gone")


class UnreadableRedis(FakeRedis):
    async def async_lpop(self, key: str, count: int) -> list[str] | None:
        raise RuntimeError("redis gone mid-flush")


class LegacyKeyReadFailureRedis(FakeRedis):
    async def async_lpop(self, key: str, count: int) -> list[str] | None:
        if key == REDIS_GATEWAY_REQUESTS_BUFFER_KEY:
            raise RuntimeError("legacy redis key unreadable")
        return await super().async_lpop(key, count)


class FailureAfterFullV2BatchRedis(FakeRedis):
    async def async_lpop(self, key: str, count: int) -> list[str] | None:
        if key == REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2 and not self.lists.get(key):
            raise RuntimeError("redis read failed after full batch")
        return await super().async_lpop(key, count)


class UnwritableRedis(FakeRedis):
    """Pops succeed, pushes fail: a Redis that went read-only between the leader's pop and its re-queue."""

    async def async_rpush(self, key: str, values: list[str]) -> int:
        raise RuntimeError("redis read-only")


def test_leader_keeps_popped_counts_in_memory_when_both_the_database_and_the_requeue_fail():
    """The pop removed the only copy; if Redis will not take it back the leader itself must carry it."""
    redis = FakeRedis()
    worker_acc = _accumulator()
    _record(worker_acc, 200)
    _record(worker_acc, 200)
    worker, _ = _buffer(redis, leader=False)
    asyncio.run(flush_gateway_requests(FakePrismaClient(), worker_acc, worker))

    degraded = UnwritableRedis()
    degraded.lists = redis.lists
    leader_acc = _accumulator()
    leader, _ = _buffer(degraded, leader=True)
    asyncio.run(flush_gateway_requests(ExplodingClient(), leader_acc, leader))
    assert degraded.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2] == []

    client = FakePrismaClient()
    retry, _ = _buffer(redis, leader=True)
    asyncio.run(flush_gateway_requests(client, leader_acc, retry))
    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", 2, 0)]


def test_leader_whose_redis_read_fails_leaves_the_pushed_rows_for_the_next_flush():
    """The scheduler job must not raise, and nothing is popped so nothing needs restoring anywhere."""
    redis = UnreadableRedis()
    acc = _accumulator()
    _record(acc, 200)
    client = FakePrismaClient()
    leader, _ = _buffer(redis, leader=True)

    asyncio.run(flush_gateway_requests(client, acc, leader))

    assert client.db.statements == []
    assert acc.drain() == {}
    assert len(redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2]) == 1


def test_pop_keeps_v2_counts_when_legacy_key_read_fails():
    redis: Final = LegacyKeyReadFailureRedis()
    redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2] = [
        json.dumps(((_DAY, "llm", "/chat/completions", 500, 0, 2),)),
        json.dumps(((_DAY, "llm", "/embeddings", 429, 0, 3),)),
    ]
    buffer: Final = _buffer(redis, leader=True)[0]

    snapshot: Final = asyncio.run(buffer.pop())

    assert snapshot == {
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=500): (
            GatewayRequestCounts(successful_requests=0, failed_requests=2)
        ),
        GatewayRequestKey(date=_DAY, category="llm", route="/embeddings", status_code=429): (
            GatewayRequestCounts(successful_requests=0, failed_requests=3)
        ),
    }


def test_commit_if_leader_writes_v2_counts_when_legacy_key_read_fails():
    redis: Final = LegacyKeyReadFailureRedis()
    redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2] = [json.dumps(((_DAY, "llm", "/chat/completions", 500, 0, 2),))]
    buffer: Final = _buffer(redis, leader=True)[0]
    client: Final = FakePrismaClient()

    asyncio.run(buffer.commit_if_leader(client))

    assert _rows_written(client) == [(_DAY, "llm", "/chat/completions", 0, 2)]
    assert _failure_rows_written(client) == [(_DAY, "llm", "/chat/completions", 500, 2)]


def test_pop_keeps_v2_first_batch_when_a_later_batch_read_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.proxy.db.gateway_request_tracking.MAX_REDIS_BUFFER_DEQUEUE_COUNT", 2)
    redis: Final = FailureAfterFullV2BatchRedis()
    redis.lists[REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2] = [
        json.dumps(((_DAY, "llm", "/chat/completions", 500, 0, 2),)),
        json.dumps(((_DAY, "llm", "/chat/completions", 429, 0, 3),)),
    ]
    buffer: Final = _buffer(redis, leader=True)[0]

    snapshot: Final = asyncio.run(buffer.pop())

    assert snapshot == {
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=500): (
            GatewayRequestCounts(successful_requests=0, failed_requests=2)
        ),
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=429): (
            GatewayRequestCounts(successful_requests=0, failed_requests=3)
        ),
    }


def test_failed_redis_push_keeps_counts_locally_for_the_next_flush():
    acc = _accumulator()
    _record(acc, 200)
    _record(acc, 500)
    buffer, lock = _buffer(ExplodingRedis(), leader=True)

    asyncio.run(flush_gateway_requests(FakePrismaClient(), acc, buffer))

    assert lock.held == []
    assert acc.drain() == {
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=200): (
            GatewayRequestCounts(successful_requests=1, failed_requests=0)
        ),
        GatewayRequestKey(date=_DAY, category="llm", route="/chat/completions", status_code=500): (
            GatewayRequestCounts(successful_requests=0, failed_requests=1)
        ),
    }


@pytest.mark.asyncio
async def test_a_gateway_request_flush_renders_a_postgres_upsert_span(
    postgres_span_names: Callable[[], Awaitable[tuple[str, ...]]],
) -> None:
    prisma = FakePrismaClient()
    snapshot = {
        GatewayRequestKey(date="2026-08-01", category="llm", route="/chat/completions"): (
            GatewayRequestCounts(successful_requests=7, failed_requests=2)
        )
    }

    await commit_gateway_requests_to_db(prisma_client=prisma, snapshot=snapshot)

    assert await postgres_span_names() == ("postgres.upsert LiteLLM_DailyGatewayRequests",)
