"""
Durable, cross-replica claim for "this budget alert has already been sent".

The in-process claim used by the email alerting path lives in ``DualCache``, which
is memory-only when Redis is not configured. Every replica therefore wins its own
claim and sends its own copy of the same alert, and the claim is lost on restart.
This module records the claim in the database instead, so it is shared by all
replicas and survives restarts.

The claim is scoped to the entity's budget window: the budget period it belongs to
plus the budget it was measured against. The alert therefore re-arms when the budget
period rolls over or the budget itself is changed, rather than on a fixed TTL.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler


def get_budget_window(budget_reset_at: datetime | None, max_budget: float | None) -> str:
    """
    Stable identity for the budget an alert was raised against.

    ``budget_reset_at`` alone is not enough: a key with no ``budget_duration`` never
    rolls over, so a claim keyed on it alone would silence that threshold for the
    lifetime of the key. Including ``max_budget`` means raising or lowering the budget
    moves every threshold and re-arms the alerts, which is what an operator expects.
    """
    period = budget_reset_at.isoformat() if budget_reset_at is not None else ""
    return f"{period}|{max_budget}"


def _claim_identity(
    entity_type: str,
    entity_id: str,
    alert_type: str,
    threshold_pct: int,
) -> Mapping[str, str | int]:
    """The columns the unique constraint is built on, as a prisma query payload."""
    return {  # mutable-ok: prisma query payloads must be plain dicts
        "entity_type": entity_type,
        "entity_id": entity_id,
        "alert_type": alert_type,
        "threshold_pct": threshold_pct,
    }


async def claim_budget_alert_slot(
    entity_type: str,
    entity_id: str,
    alert_type: str,
    threshold_pct: int,
    budget_window: str,
) -> bool:
    """
    Try to become the single sender of one budget alert.

    Returns True iff this caller won the claim and must send the alert, and False
    iff another replica, or an earlier request in this budget window, already sent
    it.

    Inserting the claim row is the atomic part: the unique constraint lets exactly
    one replica succeed. A row that already exists is taken over only when it
    belongs to an older budget window, which is the conditional update below.

    Fails open. If there is no database, or the database rejects the claim for any
    reason other than the alert already being claimed, this returns True, because a
    duplicate alert is a better failure than a missed budget alert.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return True

    identity = _claim_identity(entity_type, entity_id, alert_type, threshold_pct)

    try:
        await prisma_client.db.litellm_budgetalertsent.create(
            data={**identity, "budget_window": budget_window}  # mutable-ok: prisma query payload
        )
        return True
    except Exception as e:  # noqa: BLE001  # best-effort dedup, any failure must fall through to sending
        if not PrismaDBExceptionHandler.is_unique_constraint_violation(e):
            verbose_proxy_logger.warning(
                "Could not record budget alert claim for %s %s at %d%%, sending anyway: %s",
                entity_type,
                entity_id,
                threshold_pct,
                e,
            )
            return True

    try:
        rows_updated = await prisma_client.db.litellm_budgetalertsent.update_many(
            where={**identity, "budget_window": {"not": budget_window}},  # mutable-ok: prisma query payload
            data={  # mutable-ok: prisma query payload
                "budget_window": budget_window,
                "sent_at": datetime.now(timezone.utc),
            },
        )
        return int(rows_updated) > 0
    except Exception as e:  # noqa: BLE001  # best-effort dedup, any failure must fall through to sending
        verbose_proxy_logger.warning(
            "Could not roll over budget alert claim for %s %s at %d%%, sending anyway: %s",
            entity_type,
            entity_id,
            threshold_pct,
            e,
        )
        return True


async def release_budget_alert_slot(
    entity_type: str,
    entity_id: str,
    alert_type: str,
    threshold_pct: int,
    budget_window: str,
) -> None:
    """
    Give a won claim back after the alert failed to send, so the next request can
    retry it. Without this, one failed send would silence the whole fleet for the
    rest of the budget window.

    Scoped to ``budget_window`` so a slow failing send cannot delete a claim that
    another replica has already taken over for a later window.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return

    identity = _claim_identity(entity_type, entity_id, alert_type, threshold_pct)

    try:
        await prisma_client.db.litellm_budgetalertsent.delete_many(
            where={**identity, "budget_window": budget_window}  # mutable-ok: prisma query payload
        )
    except Exception as e:  # noqa: BLE001  # releasing the claim is best-effort, it also expires with the window
        verbose_proxy_logger.debug(
            "Failed to release budget alert claim for %s %s at %d%%: %s",
            entity_type,
            entity_id,
            threshold_pct,
            e,
        )


async def delete_budget_alert_claims(entity_type: str, entity_ids: Sequence[str]) -> None:
    """
    Drop the claim rows for entities that no longer exist, so deleting a key does
    not leave its alert claims behind.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None or not entity_ids:
        return

    try:
        await prisma_client.db.litellm_budgetalertsent.delete_many(
            where={  # mutable-ok: prisma query payload
                "entity_type": entity_type,
                "entity_id": {"in": tuple(entity_ids)},  # mutable-ok: prisma query payload
            }
        )
    except Exception as e:  # noqa: BLE001  # cleanup is best-effort, it must never fail a deletion
        verbose_proxy_logger.warning("Failed to delete budget alert claims for %s %s: %s", entity_type, entity_ids, e)
