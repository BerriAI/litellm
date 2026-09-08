"""
Accumulates gateway request counts (SGR) recorded at the ASGI edge and commits
them to ``LiteLLM_DailyGatewayRequests``.

Unlike the spend queues this keeps no per-request item. A count is a pure
aggregate, so requests fold into an in-memory map as they finish. Every
dimension of the key is server-chosen and drawn from a fixed set: the date, the
category, and a route that the classifier maps to one of a closed list of
strings rather than passing the raw path through. Nothing a caller sends can
add a key, so the fold and the table it commits to are bounded by (days x
routes) however much traffic arrives, and the response path carries no
unbounded queue that would block once full.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.middleware.billable_request_metrics_middleware import BillableCategory
from litellm.types.proxy.gateway_requests import (
    GatewayRequestCounts,
    GatewayRequestKey,
    GatewayRequestSnapshot,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_EMPTY: Final = GatewayRequestCounts(successful_requests=0, failed_requests=0)


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class GatewayRequestAccumulator:
    """Sink for the request-metrics middleware. ``record`` is sync and never awaits."""

    def __init__(self) -> None:
        self._counts: dict[GatewayRequestKey, GatewayRequestCounts] = {}  # mutable-ok: bounded fold, drained per flush

    def record(self, *, category: BillableCategory, route: str, status_code: int) -> None:
        key: Final = GatewayRequestKey(date=_utc_date(), category=category.value, route=route)
        self._counts[key] = self._counts.get(key, _EMPTY).plus(succeeded=200 <= status_code < 300)

    def drain(self) -> GatewayRequestSnapshot:
        drained: Final = self._counts
        self._counts = {}  # mutable-ok: the fold restarts empty; the drained map is handed off whole
        return drained

    def restore(self, snapshot: GatewayRequestSnapshot) -> None:
        """
        Merge un-committed counts back so the next flush retries them.

        A dropped flush would silently undercount the metric the dashboard now
        treats as the source of truth. Merging cannot grow without bound: keys
        collapse on collision, so the fold stays bounded by (date x category x
        route) however long the database is unreachable.

        This buys at-least-once, not exactly-once, and the cost is worth stating.
        The batch commits inside its context manager's ``__aexit__``, so a failure
        raised after the transaction committed (a connection dropped while reading
        the acknowledgement) restores counts that are already persisted, and the
        next flush increments them a second time. Exactly-once would need a dedup
        key the upserts could ignore on replay. For a traffic-volume metric a rare
        overcount on a dropped acknowledgement beats losing a whole interval to
        every database blip, so the trade is deliberate.
        """
        for key, counts in snapshot.items():
            existing = self._counts.get(key, _EMPTY)
            self._counts[key] = GatewayRequestCounts(
                successful_requests=existing.successful_requests + counts.successful_requests,
                failed_requests=existing.failed_requests + counts.failed_requests,
            )


async def commit_gateway_requests_to_db(
    *,
    prisma_client: "PrismaClient",
    snapshot: GatewayRequestSnapshot,
) -> None:
    """Upsert one incrementing row per (date, category, route)."""
    if not snapshot:
        return

    ordered: Final = sorted(snapshot.items(), key=lambda item: (item[0].date, item[0].category, item[0].route))

    # pyright: ignore[reportAny] on both lines -- prisma's generated client is untyped,
    # so .db and every table action off it resolve to Any at this boundary. The dict
    # literals below are the shape prisma's generated inputs require.
    async with prisma_client.db.batch_() as batcher:  # pyright: ignore[reportAny]  # untyped prisma client
        for key, counts in ordered:
            columns = asdict(key)
            batcher.litellm_dailygatewayrequests.upsert(  # pyright: ignore[reportAny]  # untyped prisma client
                where={"date_category_route": columns},  # mutable-ok: prisma input is dict-shaped
                data={  # mutable-ok: prisma input is dict-shaped
                    "create": {  # mutable-ok: prisma input is dict-shaped
                        **columns,
                        "successful_requests": counts.successful_requests,
                        "failed_requests": counts.failed_requests,
                    },
                    "update": {  # mutable-ok: prisma input is dict-shaped
                        "successful_requests": {"increment": counts.successful_requests},  # mutable-ok: as above
                        "failed_requests": {"increment": counts.failed_requests},  # mutable-ok: as above
                    },
                },
            )

    verbose_proxy_logger.debug("Gateway request tracking - committed %d aggregated rows", len(ordered))


async def flush_gateway_requests(
    prisma_client: "PrismaClient",
    accumulator: GatewayRequestAccumulator,
) -> None:
    """
    Scheduler entrypoint. Never raises: a metering failure must not kill the job.

    ``CancelledError`` is deliberately not caught, so a flush cancelled during
    shutdown drops its snapshot rather than restoring counts onto an accumulator
    the process is about to discard.
    """
    snapshot: Final = accumulator.drain()
    try:
        await commit_gateway_requests_to_db(prisma_client=prisma_client, snapshot=snapshot)
    except Exception:  # noqa: BLE001 -- a failed flush must not stop the scheduler
        accumulator.restore(snapshot)
        verbose_proxy_logger.warning(
            "Gateway request tracking - failed to commit %d rows, retrying on the next flush",
            len(snapshot),
            exc_info=True,
        )
