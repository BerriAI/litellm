from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from litellm.proxy._types import ScheduledJobStaggerSettings
from litellm.proxy.common_utils.scheduled_job_stagger import apply_scheduled_job_stagger

from litellm.proxy.db.proxy_worker_heartbeat import (
    BEAT_SQL,
    COUNT_SQL,
    DEREGISTER_SQL,
    PROXY_WORKER_LIVENESS_WINDOW_SECONDS,
    PRUNE_SQL,
    STALE_ROW_RETENTION_SECONDS,
    ProxyWorkerHeartbeat,
    count_live_proxy_workers,
)
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from tests.unit.proxy.db.fake_prisma_engine import engine_call


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", (0, 14, 15, 120))
async def test_startup_heartbeat_survives_an_explicit_stagger_offset(
    monkeypatch: pytest.MonkeyPatch, offset: int
) -> None:
    monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", "15")
    scheduler: Final = AsyncIOScheduler()
    await ProxyWorkerHeartbeat(_prisma(), worker_id="staggered").start(scheduler)
    job: Final = scheduler.get_job("proxy_worker_heartbeat_job")
    assert job is not None
    started_at: Final = job.trigger.start_date - timedelta(seconds=15)
    apply_scheduled_job_stagger(
        scheduler=scheduler,
        settings=ScheduledJobStaggerSettings(offsets={"proxy_worker_heartbeat_job": offset}),
        identity="test-worker",
    )
    next_fire: Final = job.trigger.get_next_fire_time(None, started_at)
    assert next_fire is not None
    assert next_fire - started_at < timedelta(seconds=30)


def _prisma():
    prisma = MagicMock()
    prisma.db.execute_raw = AsyncMock()
    prisma.db.query_raw = AsyncMock()
    return prisma


@pytest.mark.asyncio
async def test_disabled_heartbeat_does_not_access_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", "0")
    prisma: Final = _prisma()
    heartbeat: Final = ProxyWorkerHeartbeat(prisma_client=prisma, worker_id="disabled-worker")

    await heartbeat.beat()
    await heartbeat.deregister()

    assert await count_live_proxy_workers(prisma) is None
    prisma.db.execute_raw.assert_not_awaited()
    prisma.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_interval", "expected_interval"),
    ((None, 60), ("15", 15), ("1800", 1800), ("-1", 60), ("invalid", 60), ("", 60), ("1.5", 60)),
)
async def test_heartbeat_uses_configured_liveness_and_retention(
    monkeypatch: pytest.MonkeyPatch, configured_interval: str | None, expected_interval: int
) -> None:
    if configured_interval is None:
        monkeypatch.delenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    else:
        monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", configured_interval)
    prisma: Final = _prisma()
    prisma.db.query_raw.return_value = [{"live_workers": 2}]
    heartbeat: Final = ProxyWorkerHeartbeat(prisma_client=prisma, worker_id="configured-worker")

    await heartbeat.beat()

    assert prisma.db.execute_raw.call_args_list[0].args == (
        BEAT_SQL,
        "configured-worker",
        heartbeat.hostname,
        3 * expected_interval,
    )
    assert prisma.db.execute_raw.call_args_list[1].args == (
        PRUNE_SQL,
        STALE_ROW_RETENTION_SECONDS,
        PROXY_WORKER_LIVENESS_WINDOW_SECONDS,
    )
    assert await count_live_proxy_workers(prisma) == 2
    prisma.db.query_raw.assert_awaited_once_with(COUNT_SQL, PROXY_WORKER_LIVENESS_WINDOW_SECONDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("interval_seconds", (0, 15, 60, 1800))
async def test_start_registers_only_enabled_heartbeats(monkeypatch: pytest.MonkeyPatch, interval_seconds: int) -> None:
    monkeypatch.setenv("PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS", str(interval_seconds))
    prisma: Final = _prisma()
    heartbeat: Final = ProxyWorkerHeartbeat(prisma_client=prisma, worker_id="scheduled-worker")
    scheduler: Final = AsyncIOScheduler()

    await heartbeat.start(scheduler)

    job: Final = scheduler.get_job("proxy_worker_heartbeat_job")
    if interval_seconds == 0:
        assert job is None
        prisma.db.execute_raw.assert_not_awaited()
        return

    assert job is not None
    assert isinstance(job.trigger, IntervalTrigger)
    assert job.trigger.interval.total_seconds() == interval_seconds
    assert prisma.db.execute_raw.await_count == 2
    await job.func()
    assert prisma.db.execute_raw.await_count == 4
    assert prisma.db.execute_raw.call_args_list[2].args == (
        BEAT_SQL,
        "scheduled-worker",
        heartbeat.hostname,
        3 * interval_seconds,
    )


@pytest.mark.asyncio
async def test_beat_upserts_own_row_then_prunes_stale_rows():
    prisma = _prisma()
    heartbeat = ProxyWorkerHeartbeat(prisma_client=prisma, worker_id="worker-1")
    await heartbeat.beat()
    calls = prisma.db.execute_raw.call_args_list
    assert calls[0].args == (BEAT_SQL, "worker-1", heartbeat.hostname, PROXY_WORKER_LIVENESS_WINDOW_SECONDS)
    assert calls[1].args == (PRUNE_SQL, STALE_ROW_RETENTION_SECONDS, PROXY_WORKER_LIVENESS_WINDOW_SECONDS)


@pytest.mark.asyncio
async def test_beat_survives_a_database_error():
    prisma = _prisma()
    prisma.db.execute_raw = AsyncMock(side_effect=RuntimeError("db down"))
    await ProxyWorkerHeartbeat(prisma_client=prisma).beat()


def test_each_worker_process_gets_its_own_id():
    prisma = _prisma()
    first = ProxyWorkerHeartbeat(prisma_client=prisma)
    second = ProxyWorkerHeartbeat(prisma_client=prisma)
    assert first.worker_id != second.worker_id


@pytest.mark.asyncio
async def test_deregister_deletes_only_its_own_row():
    prisma = _prisma()
    await ProxyWorkerHeartbeat(prisma_client=prisma, worker_id="worker-1").deregister()
    assert prisma.db.execute_raw.call_args.args == (DEREGISTER_SQL, "worker-1")


@pytest.mark.asyncio
async def test_deregister_survives_a_database_error():
    prisma = _prisma()
    prisma.db.execute_raw = AsyncMock(side_effect=RuntimeError("db down"))
    await ProxyWorkerHeartbeat(prisma_client=prisma, worker_id="worker-1").deregister()


@pytest.mark.asyncio
async def test_count_reads_workers_within_the_liveness_window():
    prisma = _prisma()
    prisma.db.query_raw.return_value = [{"live_workers": 3}]
    assert await count_live_proxy_workers(prisma) == 3
    assert prisma.db.query_raw.call_args.args == (COUNT_SQL, PROXY_WORKER_LIVENESS_WINDOW_SECONDS)


@pytest.mark.asyncio
async def test_count_reads_from_the_primary_when_reads_route_to_a_replica():
    writer = MagicMock()
    writer.query_raw = AsyncMock(return_value=[{"live_workers": 2}])
    reader = MagicMock()
    reader.query_raw = AsyncMock(return_value=[{"live_workers": 1}])
    prisma = MagicMock()
    prisma.db = RoutingPrismaWrapper(writer=writer, reader=reader)
    assert await count_live_proxy_workers(prisma) == 2
    reader.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_count_returns_unknown_when_the_query_fails():
    prisma = _prisma()
    prisma.db.query_raw.side_effect = RuntimeError("db down")
    assert await count_live_proxy_workers(prisma) is None


@pytest.mark.asyncio
async def test_count_returns_unknown_for_a_malformed_row():
    prisma = _prisma()
    prisma.db.query_raw.return_value = [{"unexpected": "shape"}]
    assert await count_live_proxy_workers(prisma) is None


@pytest.mark.asyncio
async def test_a_heartbeat_tick_renders_one_postgres_span_per_round_trip(
    postgres_span_names: Callable[[], Awaitable[tuple[str, ...]]],
) -> None:
    prisma = _prisma()
    prisma.db.execute_raw = engine_call()
    prisma.db.query_raw = engine_call([{"live_workers": 2}])

    await ProxyWorkerHeartbeat(prisma_client=prisma, worker_id="worker-1").beat()
    assert await count_live_proxy_workers(prisma) == 2

    assert await postgres_span_names() == (
        "postgres.upsert LiteLLM_ProxyWorkerHeartbeat",
        "postgres.delete LiteLLM_ProxyWorkerHeartbeat",
        "postgres.select LiteLLM_ProxyWorkerHeartbeat",
    )
