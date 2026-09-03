"""Pod-local queue of shadow-eval funnel increments, drained by the spend-update job.

The shadow-eval success hook counts the sampled-traffic outcomes that never produce an
attempt row (a lost sampling dice roll, an unjudgeable request shape, a concurrency
shed), so a job's results can state what share of its eligible traffic the judged rows
represent. Counters are advisory coverage stats: a pod dying loses at most one flush
interval, and a failed flush drops its batch because a repeated increment is worse
than an undercount (same call as the auto-router session rollup flush).
"""

from typing import TYPE_CHECKING, Final, Literal

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

ShadowEvalFunnelStage = Literal["not_sampled", "unjudgeable", "shed", "withheld"]

FUNNEL_STAGES: Final[tuple[ShadowEvalFunnelStage, ...]] = ("not_sampled", "unjudgeable", "shed", "withheld")

_pending: dict[str, dict[ShadowEvalFunnelStage, int]] = {}  # mutable-ok: module-level queue, single event loop

_FUNNEL_PLACEHOLDERS: Final = ", ".join(f"${n + 2}" for n in range(len(FUNNEL_STAGES)))

_UPSERT_FUNNEL_SQL: Final = f"""
INSERT INTO "LiteLLM_ShadowEvalFunnel" (job_id, {", ".join(FUNNEL_STAGES)})
VALUES ($1, {_FUNNEL_PLACEHOLDERS})
ON CONFLICT (job_id) DO UPDATE SET
    {", ".join(f'{stage} = "LiteLLM_ShadowEvalFunnel".{stage} + EXCLUDED.{stage}' for stage in FUNNEL_STAGES)}
"""


def pending_shadow_eval_funnel_events() -> int:
    """Queue census for the drain triggers: entries not yet flushed, so a funnel-only
    batch still wakes the spend job that would otherwise skip an empty-queue run."""
    return sum(sum(counters.values()) for counters in _pending.values())


def record_shadow_eval_funnel_event(job_id: str, stage: ShadowEvalFunnelStage) -> None:
    """Count one skipped request for one job leg; synchronous so the hook's read-modify-
    write cannot interleave with the flush's snapshot on the shared event loop."""
    counters: Final = _pending.setdefault(job_id, dict.fromkeys(FUNNEL_STAGES, 0))  # mutable-ok: queue entry
    counters[stage] += 1


async def flush_shadow_eval_funnel(prisma_client: "PrismaClient") -> None:
    if not _pending:
        return
    batch: Final = dict(_pending)  # mutable-ok: snapshot drained from the queue
    _pending.clear()
    for job_id, counters in batch.items():
        try:
            await prisma_client.db.execute_raw(
                _UPSERT_FUNNEL_SQL,
                job_id,
                *(counters[stage] for stage in FUNNEL_STAGES),
            )
        except Exception as flush_err:  # noqa: BLE001  # drop this leg's batch: a repeated increment is worse than an undercount
            verbose_proxy_logger.error(
                "Spend tracking - shadow eval funnel flush failed for job %s, %s dropped: %s",
                job_id,
                counters,
                flush_err,
            )
