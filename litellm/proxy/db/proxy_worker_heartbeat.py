"""
Live proxy worker census, one row per worker process.

Every uvicorn worker upserts its own row on a fixed heartbeat, so counting
rows with a recent heartbeat answers "how many workers share this database?"
without any coordination. The Admin UI's "no Redis" banner uses that count to
hide itself for deployments that are provably a single worker, where per-worker
rate limits, budgets, and router state are already global. All timestamps are
written and compared with the database's own clock, so pods with skewed clocks
still agree.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS: Final = 60
PROXY_WORKER_LIVENESS_WINDOW_SECONDS: Final = 3 * PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS
STALE_ROW_RETENTION_SECONDS: Final = 3600

BEAT_SQL: Final = """
INSERT INTO "LiteLLM_ProxyWorkerHeartbeat" (worker_id, hostname, last_heartbeat_at)
VALUES ($1, $2, NOW())
ON CONFLICT (worker_id) DO UPDATE SET last_heartbeat_at = NOW()
"""

PRUNE_SQL: Final = """
DELETE FROM "LiteLLM_ProxyWorkerHeartbeat"
WHERE last_heartbeat_at < NOW() - make_interval(secs => $1)
"""

COUNT_SQL: Final = """
SELECT COUNT(*)::int AS live_workers FROM "LiteLLM_ProxyWorkerHeartbeat"
WHERE last_heartbeat_at > NOW() - make_interval(secs => $1)
"""

DEREGISTER_SQL: Final = """
DELETE FROM "LiteLLM_ProxyWorkerHeartbeat" WHERE worker_id = $1
"""


class _LiveWorkerCountRow(TypedDict):
    live_workers: ReadOnly[int]


_COUNT_ROWS_ADAPTER: Final = TypeAdapter(tuple[_LiveWorkerCountRow, ...])


class ProxyWorkerHeartbeat:
    def __init__(self, prisma_client: PrismaClient, worker_id: str | None = None) -> None:
        self.prisma_client: Final = prisma_client
        self.worker_id: Final[str] = worker_id or str(uuid.uuid4())
        self.hostname: Final = socket.gethostname()

    async def beat(self) -> None:
        try:
            await self.prisma_client.db.execute_raw(BEAT_SQL, self.worker_id, self.hostname)
            await self.prisma_client.db.execute_raw(PRUNE_SQL, STALE_ROW_RETENTION_SECONDS)
        except Exception as beat_err:  # noqa: BLE001  # a missed heartbeat must never take down the worker
            verbose_proxy_logger.debug("Proxy worker heartbeat write failed: %s", beat_err)

    async def deregister(self) -> None:
        try:
            await self.prisma_client.db.execute_raw(DEREGISTER_SQL, self.worker_id)
        except Exception as deregister_err:  # noqa: BLE001  # best-effort cleanup; the liveness window ages the row out anyway
            verbose_proxy_logger.debug("Proxy worker heartbeat deregister failed: %s", deregister_err)


async def count_live_proxy_workers(prisma_client: PrismaClient) -> int | None:
    """
    The number of workers with a recent heartbeat, or None when the database
    cannot answer. Callers must treat None as "unknown", not as zero. Always
    counts on the primary: a lagging read replica must never undercount.
    """
    try:
        db: Final = prisma_client.db
        primary_db: Final = db.writer if isinstance(db, RoutingPrismaWrapper) else db
        rows: Final = await primary_db.query_raw(COUNT_SQL, PROXY_WORKER_LIVENESS_WINDOW_SECONDS)
        return _COUNT_ROWS_ADAPTER.validate_python(rows)[0]["live_workers"]
    except Exception as count_err:  # noqa: BLE001  # an unknown count must degrade to "warn", never to a 503
        verbose_proxy_logger.debug("Live proxy worker count unavailable: %s", count_err)
        return None
