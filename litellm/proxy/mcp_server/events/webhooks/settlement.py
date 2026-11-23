from datetime import datetime
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from litellm.types.mcp_server.events import CallbackFailure, GapEnvelope

if TYPE_CHECKING:
    from litellm.proxy.mcp_server.events.webhooks.store import QueuedDelivery
    from litellm.proxy.utils import PrismaClient

SettlementOutcome: TypeAlias = Literal["delivered", "terminated", "discarded"] | CallbackFailure
SUSPEND_AFTER_FAILURES: Final = 20


async def settle_delivery(
    client: "PrismaClient",
    delivery: "QueuedDelivery",
    generation: str,
    at: datetime,
    outcome: SettlementOutcome,
    *,
    gap: bool = False,
) -> bool:
    if delivery.claim_id is None:
        return False
    async with client.tx() as tx:
        locked: Final = await tx.query_raw(
            """SELECT cursor FROM "LiteLLM_MCPEventSubscription"
            WHERE id = $1 AND generation = $2 AND ($3::boolean OR status IN ('active', 'suspended')) FOR UPDATE""",
            delivery.subscription_id,
            generation,
            outcome == "discarded" and not gap,
            model=GapEnvelope,
        )
        if not locked:
            return False
        finished: Final = await tx.litellm_mcpeventdelivery.update_many(
            where={
                "subscription_id": delivery.subscription_id,
                "delivery_id": delivery.delivery_id,
                "claim_id": delivery.claim_id,
                "delivered_at": None,
            },
            data={"delivered_at": at, "body": "", "claim_id": None, "claimed_until": None},
        )
        if not finished:
            return False
        if outcome == "delivered":
            await tx.litellm_mcpeventsubscription.update_many(
                where={"id": delivery.subscription_id},
                data={"last_delivery_at": at, "consecutive_failures": 0, "last_error": None, "failed_since": None},
            )
        elif outcome == "terminated":
            await tx.litellm_mcpeventsubscription.update_many(
                where={"id": delivery.subscription_id}, data={"status": "terminated"}
            )
            await tx.litellm_mcpeventdelivery.delete_many(
                where={"subscription_id": delivery.subscription_id, "delivered_at": None}
            )
        elif outcome != "discarded":
            await tx.execute_raw(
                """UPDATE "LiteLLM_MCPEventSubscription" SET consecutive_failures = consecutive_failures + 1,
                last_error = $2, failed_since = COALESCE(failed_since, $3::timestamp), updated_at = now(),
                status = CASE WHEN consecutive_failures + 1 >= $4 THEN 'suspended' ELSE status END WHERE id = $1""",
                delivery.subscription_id,
                outcome,
                at.isoformat(),
                SUSPEND_AFTER_FAILURES,
            )
        if gap and outcome not in ("delivered", "terminated"):
            gap_id: Final = f"msg_gap_{delivery.delivery_id}"
            await tx.execute_raw(
                """INSERT INTO "LiteLLM_MCPEventDelivery"
                (delivery_id, subscription_id, webhook_id, body, next_attempt_at, prepared)
                VALUES ($1, $2, $1, $3, $4::timestamp, true)""",
                gap_id,
                delivery.subscription_id,
                locked[0].model_dump_json(),
                at.isoformat(),
            )
    return True
