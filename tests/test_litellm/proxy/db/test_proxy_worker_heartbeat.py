from unittest.mock import AsyncMock, MagicMock

import pytest

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


def _prisma():
    prisma = MagicMock()
    prisma.db.execute_raw = AsyncMock()
    prisma.db.query_raw = AsyncMock()
    return prisma


@pytest.mark.asyncio
async def test_beat_upserts_own_row_then_prunes_stale_rows():
    prisma = _prisma()
    heartbeat = ProxyWorkerHeartbeat(prisma_client=prisma, worker_id="worker-1")
    await heartbeat.beat()
    calls = prisma.db.execute_raw.call_args_list
    assert calls[0].args == (BEAT_SQL, "worker-1", heartbeat.hostname)
    assert calls[1].args == (PRUNE_SQL, STALE_ROW_RETENTION_SECONDS)


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
