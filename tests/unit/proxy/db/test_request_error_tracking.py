import asyncio
import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import MAX_REDIS_BUFFER_DEQUEUE_COUNT, REQUEST_ERRORS_MAX_ROWS_PER_UPSERT
from litellm.proxy.db.request_error_tracking import (
    RequestErrorAccumulator,
    RequestErrorRedisBuffer,
    build_request_errors_upsert,
    commit_request_errors_to_db,
    flush_request_errors,
    fold_counts,
    status_code_from_metadata,
)
from litellm.types.proxy.request_errors import RequestErrorKey


def _key(status_code: int = 429, api_key: str = "hash-1", date: str = "2026-10-08") -> RequestErrorKey:
    return RequestErrorKey(
        date=date, api_key=api_key, team_id="team-a", user_id="user-a", model_group="gpt-4o", status_code=status_code
    )


def _payload(status: str = "failure", error_code: str | None = "429", **overrides: object) -> dict[str, object]:
    error_information: Final = (
        {} if error_code is None else {"error_information": {"error_code": error_code, "error_class": "RateLimitError"}}
    )
    metadata: Final[dict[str, object]] = {"status": status, **error_information}
    return {
        "api_key": "hash-1",
        "team_id": "team-a",
        "user": "user-a",
        "model_group": "gpt-4o",
        "model": "gpt-4o-2024-08-06",
        "metadata": json.dumps(metadata),
        **overrides,
    }


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"error_information": {"error_code": "429"}}, 429),
        ({"error_information": {"error_code": 503}}, 503),
        ({"error_information": {"error_code": ""}}, 0),
        ({"error_information": {"error_code": "RateLimitError"}}, 0),
        ({"error_information": {"error_code": "9999"}}, 0),
        ({"error_information": None}, 0),
        ({}, 0),
    ],
)
def test_status_code_from_metadata(metadata: dict[str, object], expected: int) -> None:
    assert status_code_from_metadata(metadata) == expected


def test_record_counts_failures_by_caller_and_status() -> None:
    accumulator: Final = RequestErrorAccumulator()
    for _ in range(2):
        accumulator.record(payload=_payload(), request_status="failure", date="2026-10-08", is_internal_call=False)
    accumulator.record(
        payload=_payload(error_code="500"), request_status="failure", date="2026-10-08", is_internal_call=False
    )
    assert accumulator.drain() == {_key(429): 2, _key(500): 1}
    assert accumulator.drain() == {}


def test_record_skips_successes_and_internal_sub_calls() -> None:
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(
        payload=_payload(status="success"), request_status="success", date="2026-10-08", is_internal_call=False
    )
    accumulator.record(payload=_payload(), request_status="failure", date="2026-10-08", is_internal_call=True)
    assert accumulator.drain() == {}


def test_record_normalizes_missing_dimensions_to_empty_text() -> None:
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(
        payload=_payload(error_code=None, team_id=None, user=None, model_group=None, model=None),
        request_status="failure",
        date="2026-10-08",
        is_internal_call=False,
    )
    assert accumulator.drain() == {
        RequestErrorKey(date="2026-10-08", api_key="hash-1", team_id="", user_id="", model_group="", status_code=0): 1
    }


def test_record_falls_back_to_model_when_model_group_is_missing() -> None:
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(
        payload=_payload(model_group=None), request_status="failure", date="2026-10-08", is_internal_call=False
    )
    (key,) = accumulator.drain()
    assert key.model_group == "gpt-4o-2024-08-06"


def test_record_tolerates_unparseable_metadata() -> None:
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(
        payload=_payload(metadata="{not json"), request_status="failure", date="2026-10-08", is_internal_call=False
    )
    assert accumulator.drain() == {_key(0): 1}


def test_restore_folds_into_pending_counts() -> None:
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(payload=_payload(), request_status="failure", date="2026-10-08", is_internal_call=False)
    accumulator.restore({_key(429): 3, _key(500): 1})
    assert accumulator.drain() == {_key(429): 4, _key(500): 1}


def test_fold_counts_sums_duplicate_keys() -> None:
    assert fold_counts([(_key(), 1), (_key(), 2), (_key(500), 5)]) == {_key(): 3, _key(500): 5}


def test_upsert_orders_rows_by_conflict_key_and_increments_on_conflict() -> None:
    sql, params = build_request_errors_upsert({_key(500): 1, _key(429, api_key="hash-0"): 2, _key(429): 0})
    assert params == (
        "2026-10-08", "hash-0", "team-a", "user-a", "gpt-4o", 429, 2,
        "2026-10-08", "hash-1", "team-a", "user-a", "gpt-4o", 500, 1,
    )  # fmt: skip
    assert sql.count("($") == 2
    assert 'ON CONFLICT ("date", "api_key", "team_id", "user_id", "model_group", "status_code")' in sql
    assert '"failed_requests" = "LiteLLM_DailyRequestErrors"."failed_requests" + EXCLUDED."failed_requests"' in sql


@pytest.mark.asyncio
async def test_flush_commits_one_statement_and_drains() -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock()
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(payload=_payload(), request_status="failure", date="2026-10-08", is_internal_call=False)

    await flush_request_errors(prisma_client, accumulator)

    prisma_client.db.execute_raw.assert_awaited_once()
    assert prisma_client.db.execute_raw.await_args.args[1:] == (
        "2026-10-08", "hash-1", "team-a", "user-a", "gpt-4o", 429, 1,
    )  # fmt: skip
    assert accumulator.drain() == {}


@pytest.mark.asyncio
async def test_flush_restores_counts_when_commit_fails() -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(side_effect=RuntimeError("db down"))
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(payload=_payload(), request_status="failure", date="2026-10-08", is_internal_call=False)

    await flush_request_errors(prisma_client, accumulator)

    assert accumulator.drain() == {_key(429): 1}


@pytest.mark.asyncio
async def test_commit_splits_a_large_backlog_into_bounded_statements() -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock()
    snapshot: Final = {_key(api_key=f"hash-{index:05d}"): 1 for index in range(REQUEST_ERRORS_MAX_ROWS_PER_UPSERT + 1)}

    uncommitted: Final = await commit_request_errors_to_db(prisma_client=prisma_client, snapshot=snapshot)

    assert uncommitted == {}
    assert prisma_client.db.execute_raw.await_count == 2
    first, second = prisma_client.db.execute_raw.await_args_list
    assert len(first.args) - 1 == REQUEST_ERRORS_MAX_ROWS_PER_UPSERT * 7
    assert len(second.args) - 1 == 7


@pytest.mark.asyncio
async def test_commit_hands_back_only_the_chunks_that_did_not_land() -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(side_effect=(None, RuntimeError("db down")))
    snapshot: Final = {_key(api_key=f"hash-{index:05d}"): 1 for index in range(REQUEST_ERRORS_MAX_ROWS_PER_UPSERT + 1)}

    uncommitted: Final = await commit_request_errors_to_db(prisma_client=prisma_client, snapshot=snapshot)

    assert uncommitted == {_key(api_key=f"hash-{REQUEST_ERRORS_MAX_ROWS_PER_UPSERT:05d}"): 1}
    assert prisma_client.db.execute_raw.await_count == 2


@pytest.mark.asyncio
async def test_commit_stops_at_the_first_failed_chunk_and_hands_back_the_rest() -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(side_effect=RuntimeError("db down"))
    snapshot: Final = {_key(api_key=f"hash-{index:05d}"): 1 for index in range(REQUEST_ERRORS_MAX_ROWS_PER_UPSERT + 1)}

    uncommitted: Final = await commit_request_errors_to_db(prisma_client=prisma_client, snapshot=snapshot)

    assert uncommitted == snapshot
    prisma_client.db.execute_raw.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_commit_restores_only_the_unwritten_chunks_and_propagates() -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(side_effect=(None, asyncio.CancelledError()))
    accumulator: Final = RequestErrorAccumulator()
    snapshot: Final = {_key(api_key=f"hash-{index:05d}"): 1 for index in range(REQUEST_ERRORS_MAX_ROWS_PER_UPSERT + 1)}
    accumulator.restore(snapshot)

    with pytest.raises(asyncio.CancelledError):
        await flush_request_errors(prisma_client, accumulator)

    assert accumulator.drain() == {_key(api_key=f"hash-{REQUEST_ERRORS_MAX_ROWS_PER_UPSERT:05d}"): 1}
    assert prisma_client.db.execute_raw.await_count == 2


@pytest.mark.asyncio
async def test_cancelled_leader_commit_requeues_the_unwritten_chunks_to_redis() -> None:
    redis_cache: Final = MagicMock()
    stored: Final[list[str]] = []  # mutable-ok: stands in for the Redis list

    async def rpush(key: str, values: tuple[str, ...]) -> int:
        stored.extend(values)
        return len(stored)

    async def lpop(key: str, count: int) -> list[str]:
        popped: Final = stored[:count]
        del stored[:count]
        return popped

    redis_cache.async_rpush = AsyncMock(side_effect=rpush)
    redis_cache.async_lpop = AsyncMock(side_effect=lpop)
    pod_lock_manager: Final = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(side_effect=(None, asyncio.CancelledError()))
    buffer: Final = RequestErrorRedisBuffer(redis_cache=redis_cache, pod_lock_manager=pod_lock_manager)
    accumulator: Final = RequestErrorAccumulator()
    accumulator.restore(
        {_key(api_key=f"hash-{index:05d}"): 1 for index in range(REQUEST_ERRORS_MAX_ROWS_PER_UPSERT + 1)}
    )

    with pytest.raises(asyncio.CancelledError):
        await flush_request_errors(prisma_client, accumulator, buffer)

    assert accumulator.drain() == {}
    assert await buffer.pop() == {_key(api_key=f"hash-{REQUEST_ERRORS_MAX_ROWS_PER_UPSERT:05d}"): 1}
    assert prisma_client.db.execute_raw.await_count == 2


@pytest.mark.asyncio
async def test_cancelled_leader_commit_keeps_rows_in_memory_when_redis_refuses_them() -> None:
    redis_cache: Final = MagicMock()
    stored: Final[list[str]] = []  # mutable-ok: stands in for the Redis list

    async def rpush_once(key: str, values: tuple[str, ...]) -> int:
        if stored:
            raise RuntimeError("redis down")
        stored.extend(values)
        return len(stored)

    async def lpop(key: str, count: int) -> list[str]:
        return list(stored)

    redis_cache.async_rpush = AsyncMock(side_effect=rpush_once)
    redis_cache.async_lpop = AsyncMock(side_effect=lpop)
    pod_lock_manager: Final = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock(side_effect=asyncio.CancelledError())
    buffer: Final = RequestErrorRedisBuffer(redis_cache=redis_cache, pod_lock_manager=pod_lock_manager)
    accumulator: Final = RequestErrorAccumulator()
    accumulator.restore({_key(429): 2})

    with pytest.raises(asyncio.CancelledError):
        await flush_request_errors(prisma_client, accumulator, buffer)

    assert accumulator.drain() == {_key(429): 2}


@pytest.mark.asyncio
async def test_cancelled_drain_puts_the_popped_entries_back_in_redis() -> None:
    redis_cache: Final = MagicMock()
    entry: Final = json.dumps([["2026-10-08", "hash-1", "team-a", "user-a", "gpt-4o", 429, 1]])
    stored: Final[list[str]] = [entry] * MAX_REDIS_BUFFER_DEQUEUE_COUNT  # mutable-ok: stands in for the Redis list

    async def lpop(key: str, count: int) -> list[str]:
        if not stored:
            raise asyncio.CancelledError()
        popped: Final = stored[:count]
        del stored[:count]
        return popped

    async def rpush(key: str, values: tuple[str, ...]) -> int:
        stored.extend(values)
        return len(stored)

    redis_cache.async_lpop = AsyncMock(side_effect=lpop)
    redis_cache.async_rpush = AsyncMock(side_effect=rpush)
    pod_lock_manager: Final = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock()
    buffer: Final = RequestErrorRedisBuffer(redis_cache=redis_cache, pod_lock_manager=pod_lock_manager)
    accumulator: Final = RequestErrorAccumulator()

    with pytest.raises(asyncio.CancelledError):
        await flush_request_errors(prisma_client, accumulator, buffer)

    assert len(stored) == MAX_REDIS_BUFFER_DEQUEUE_COUNT
    prisma_client.db.execute_raw.assert_not_awaited()
    assert accumulator.drain() == {}


@pytest.mark.asyncio
async def test_flush_skips_the_database_when_nothing_failed() -> None:
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock()
    await flush_request_errors(prisma_client, RequestErrorAccumulator())
    prisma_client.db.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_redis_buffer_round_trips_and_leader_commits() -> None:
    redis_cache: Final = MagicMock()
    stored: Final[list[str]] = []  # mutable-ok: stands in for the Redis list

    async def rpush(key: str, values: tuple[str, ...]) -> int:
        stored.extend(values)
        return len(stored)

    async def lpop(key: str, count: int) -> list[str]:
        popped: Final = stored[:count]
        del stored[:count]
        return popped

    redis_cache.async_rpush = AsyncMock(side_effect=rpush)
    redis_cache.async_lpop = AsyncMock(side_effect=lpop)
    pod_lock_manager: Final = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock()
    buffer: Final = RequestErrorRedisBuffer(redis_cache=redis_cache, pod_lock_manager=pod_lock_manager)
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(payload=_payload(), request_status="failure", date="2026-10-08", is_internal_call=False)

    await buffer.push({_key(429): 2})
    await flush_request_errors(prisma_client, accumulator, buffer)

    prisma_client.db.execute_raw.assert_awaited_once()
    assert prisma_client.db.execute_raw.await_args.args[1:] == (
        "2026-10-08", "hash-1", "team-a", "user-a", "gpt-4o", 429, 3,
    )  # fmt: skip
    assert stored == []
    assert accumulator.drain() == {}


@pytest.mark.asyncio
async def test_redis_buffer_follower_leaves_rows_in_redis() -> None:
    redis_cache: Final = MagicMock()
    redis_cache.async_rpush = AsyncMock()
    redis_cache.async_lpop = AsyncMock()
    pod_lock_manager: Final = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=False)
    prisma_client: Final = MagicMock()
    prisma_client.db.execute_raw = AsyncMock()
    buffer: Final = RequestErrorRedisBuffer(redis_cache=redis_cache, pod_lock_manager=pod_lock_manager)
    accumulator: Final = RequestErrorAccumulator()
    accumulator.record(payload=_payload(), request_status="failure", date="2026-10-08", is_internal_call=False)

    await flush_request_errors(prisma_client, accumulator, buffer)

    redis_cache.async_rpush.assert_awaited_once()
    redis_cache.async_lpop.assert_not_awaited()
    prisma_client.db.execute_raw.assert_not_awaited()
    assert accumulator.drain() == {}
