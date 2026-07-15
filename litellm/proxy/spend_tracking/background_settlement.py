"""
Prisma-backed durable settlement for background interaction billing.

The creating pod's poll task persists a settlement context row here so that a
delete routed to any pod, or a sweep after a pod restart, can rebuild a
billable logging object and settle the interaction exactly once. The row's
``status`` column is the cross-pod claim: whoever flips ``pending`` to
``settled`` via a conditional update owns billing or release, and the
``outcome`` column records what the winner did (``billed``, ``released``,
``abandoned``, or ``error``) for audit of anything that could not be billed.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Protocol

from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    BACKGROUND_SETTLEMENT_SWEEP_MIN_AGE_SECONDS,
    MAX_BACKGROUND_SETTLEMENTS_PER_SWEEP,
)
from litellm.interactions.background_cost_polling import (
    _TERMINAL_STATUSES,
    BackgroundSettlementContext,
    SettlementOutcome,
    _release_open_budget_reservation,
)
from litellm.types.interactions import InteractionsAPIResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy.utils import PrismaClient


class PendingSettlementRow(BaseModel):
    interaction_id: str
    context: BackgroundSettlementContext
    timeout_at: datetime


class SettlementRowStore(Protocol):
    async def claim(self, interaction_id: str) -> bool: ...

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None: ...

    async def list_due(self, older_than: datetime, limit: int) -> tuple[PendingSettlementRow, ...]: ...


SettlementFetch = Callable[[BackgroundSettlementContext], Awaitable[InteractionsAPIResponse]]


class DeploymentResolver(Protocol):
    async def async_get_available_deployment(self, model: str, request_kwargs: dict) -> dict: ...


def _get_proxy_router() -> Optional[DeploymentResolver]:
    from litellm.proxy.proxy_server import llm_router

    return llm_router


@dataclass(frozen=True, slots=True)
class _SettlementCredentials:
    api_key: Optional[str]
    api_base: Optional[str]


async def _resolve_settlement_credentials(
    context: BackgroundSettlementContext,
    router: Optional[DeploymentResolver] = None,
) -> _SettlementCredentials:
    if context.model_group is None:
        return _SettlementCredentials(api_key=None, api_base=None)
    active_router = router if router is not None else _get_proxy_router()
    if active_router is None:
        return _SettlementCredentials(api_key=None, api_base=None)
    try:
        deployment = await active_router.async_get_available_deployment(
            model=context.model_group,
            request_kwargs={},
        )
    except Exception:  # noqa: BLE001  # a failed deployment lookup falls back to provider env credentials
        verbose_proxy_logger.warning(
            "Could not resolve a deployment for %s to settle background interaction %s; "
            "falling back to provider environment credentials",
            context.model_group,
            context.interaction_id,
            exc_info=True,
        )
        return _SettlementCredentials(api_key=None, api_base=None)
    litellm_params = deployment.get("litellm_params") or {}
    return _SettlementCredentials(
        api_key=litellm_params.get("api_key"),
        api_base=litellm_params.get("api_base"),
    )


async def fetch_interaction_for_settlement(
    context: BackgroundSettlementContext,
    router: Optional[DeploymentResolver] = None,
) -> InteractionsAPIResponse:
    from litellm.interactions import aget

    credentials = await _resolve_settlement_credentials(context, router=router)
    return await aget(
        interaction_id=context.interaction_id,
        custom_llm_provider=context.custom_llm_provider,
        **{"api_key": credentials.api_key, "api_base": credentials.api_base, "no-log": True},
    )


def rebuild_logging_for_settlement(context: BackgroundSettlementContext) -> "LiteLLMLoggingObj":
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = Logging(
        model=context.model,
        messages=[{"role": "user", "content": f"<background_interaction_settlement/{context.interaction_id}>"}],
        stream=False,
        call_type=context.call_type,
        start_time=datetime.now(),
        litellm_call_id=context.litellm_call_id,
        function_id=str(uuid.uuid4()),
        litellm_trace_id=context.litellm_trace_id,
    )
    attribution = {key: value for key, value in context.attribution.model_dump().items() if value is not None}
    reservation = context.budget_reservation.model_dump() if context.budget_reservation is not None else None
    metadata = attribution if reservation is None else {**attribution, "user_api_key_budget_reservation": reservation}
    logging_obj.update_environment_variables(
        litellm_params={
            "litellm_call_id": context.litellm_call_id,
            "proxy_server_request": {"headers": {"user-agent": "litellm-background-settlement"}},
            "metadata": metadata,
        },
        optional_params={},
        model=context.model,
        custom_llm_provider=context.custom_llm_provider,
    )
    return logging_obj


async def _release_persisted_reservation(context: BackgroundSettlementContext) -> None:
    if context.budget_reservation is None:
        return
    from litellm.proxy.spend_tracking.budget_reservation import release_budget_reservation

    try:
        await release_budget_reservation(budget_reservation=context.budget_reservation.model_dump())
    except Exception:  # noqa: BLE001  # a failed release must not fail settlement; counters expire via TTL
        verbose_proxy_logger.exception(
            "Failed to release persisted budget reservation for background interaction %s",
            context.interaction_id,
        )


async def _claim_row_best_effort(store: SettlementRowStore, interaction_id: str) -> bool:
    try:
        return await store.claim(interaction_id)
    except Exception:  # noqa: BLE001  # an unreachable store defers to the sweep instead of risking a double bill
        verbose_proxy_logger.warning(
            "Failed to claim settlement for background interaction %s; deferring to the settlement sweep",
            interaction_id,
            exc_info=True,
        )
        return False


async def _record_row_outcome_best_effort(
    store: SettlementRowStore,
    interaction_id: str,
    outcome: SettlementOutcome,
) -> None:
    try:
        await store.record_outcome(interaction_id, outcome)
    except Exception:  # noqa: BLE001  # the outcome column is observability, never worth failing settlement over
        verbose_proxy_logger.warning(
            "Failed to record settlement outcome %s for background interaction %s",
            outcome,
            interaction_id,
            exc_info=True,
        )


async def settle_claimed_row(
    row: PendingSettlementRow,
    response: InteractionsAPIResponse,
    store: SettlementRowStore,
) -> None:
    if response.status in _TERMINAL_STATUSES and response.usage is not None:
        try:
            logging_obj = rebuild_logging_for_settlement(row.context)
        except Exception:  # noqa: BLE001  # an unbuildable logging context settles by releasing the reservation
            verbose_proxy_logger.exception(
                "Could not rebuild a billable logging context for background interaction %s; "
                "its spend may be under-tracked",
                row.interaction_id,
            )
            await _release_persisted_reservation(row.context)
            await _record_row_outcome_best_effort(store, row.interaction_id, "error")
            return
        try:
            await logging_obj.async_log_background_interaction_completion(result=response)
        except Exception:  # noqa: BLE001  # a billing failure after winning the claim must be surfaced, not retried
            verbose_proxy_logger.exception(
                "Billing failed after claiming settlement for background interaction %s; "
                "its spend may be under-tracked",
                row.interaction_id,
            )
            await _release_open_budget_reservation(logging_obj=logging_obj)
            await _record_row_outcome_best_effort(store, row.interaction_id, "error")
            return
        await _record_row_outcome_best_effort(store, row.interaction_id, "billed")
        return
    await _release_persisted_reservation(row.context)
    await _record_row_outcome_best_effort(store, row.interaction_id, "released")


async def settle_row_before_delete(
    row: PendingSettlementRow,
    store: SettlementRowStore,
    fetch: SettlementFetch,
) -> None:
    try:
        response = await fetch(row.context)
    except Exception as e:  # noqa: BLE001  # unfetchable pre-delete state settles by releasing the reservation
        verbose_proxy_logger.debug(
            "Could not fetch background interaction %s before delete, releasing its reservation: %s",
            row.interaction_id,
            e,
        )
        if not await _claim_row_best_effort(store, row.interaction_id):
            return
        await _release_persisted_reservation(row.context)
        await _record_row_outcome_best_effort(store, row.interaction_id, "released")
        return
    if not await _claim_row_best_effort(store, row.interaction_id):
        return
    await settle_claimed_row(row=row, response=response, store=store)


async def sweep_pending_settlements(
    store: SettlementRowStore,
    fetch: SettlementFetch = fetch_interaction_for_settlement,
    min_age_seconds: float = BACKGROUND_SETTLEMENT_SWEEP_MIN_AGE_SECONDS,
    limit: int = MAX_BACKGROUND_SETTLEMENTS_PER_SWEEP,
) -> None:
    now = datetime.now(timezone.utc)
    rows = await store.list_due(older_than=now - timedelta(seconds=min_age_seconds), limit=limit)
    for row in rows:
        try:
            await _sweep_row(row=row, store=store, fetch=fetch, now=now)
        except Exception:  # noqa: BLE001  # one bad row must not stop the sweep from settling the rest
            verbose_proxy_logger.exception(
                "Settlement sweep failed for background interaction %s; leaving it for the next cycle",
                row.interaction_id,
            )


async def _sweep_row(
    row: PendingSettlementRow,
    store: SettlementRowStore,
    fetch: SettlementFetch,
    now: datetime,
) -> None:
    if row.timeout_at <= now:
        if not await _claim_row_best_effort(store, row.interaction_id):
            return
        verbose_proxy_logger.warning(
            "Abandoning settlement for background interaction %s past its %s timeout; its usage will not be tracked",
            row.interaction_id,
            row.timeout_at,
        )
        await _release_persisted_reservation(row.context)
        await _record_row_outcome_best_effort(store, row.interaction_id, "abandoned")
        return
    try:
        response = await fetch(row.context)
    except Exception as e:  # noqa: BLE001  # a sweep fetch error must leave the row pending for a later cycle
        verbose_proxy_logger.debug(
            "Settlement sweep could not fetch background interaction %s, leaving it pending: %s",
            row.interaction_id,
            e,
        )
        return
    if response.status not in _TERMINAL_STATUSES:
        return
    if not await _claim_row_best_effort(store, row.interaction_id):
        return
    await settle_claimed_row(row=row, response=response, store=store)


def _parse_row(interaction_id: str, context_value: object, timeout_at: datetime) -> Optional[PendingSettlementRow]:
    try:
        raw = json.loads(context_value) if isinstance(context_value, str) else context_value
        return PendingSettlementRow(
            interaction_id=interaction_id,
            context=BackgroundSettlementContext.model_validate(raw),
            timeout_at=timeout_at,
        )
    except Exception:  # noqa: BLE001  # an unparseable row is surfaced and skipped rather than crashing the sweep
        verbose_proxy_logger.exception(
            "Could not parse persisted settlement context for background interaction %s",
            interaction_id,
        )
        return None


class PrismaBackgroundSettlementStore:
    def __init__(
        self,
        prisma_client: "PrismaClient",
        fetch: SettlementFetch = fetch_interaction_for_settlement,
        claimed_by: Optional[str] = None,
    ) -> None:
        self.prisma_client = prisma_client
        self.fetch = fetch
        self.claimed_by = claimed_by or str(uuid.uuid4())

    @property
    def _table(self):
        return self.prisma_client.db.litellm_backgroundinteractionsettlementtable

    async def persist_pending(self, context: BackgroundSettlementContext, timeout_at: datetime) -> None:
        context_json = context.model_dump_json()
        await self._table.upsert(
            where={"interaction_id": context.interaction_id},
            data={
                "create": {
                    "interaction_id": context.interaction_id,
                    "context": context_json,
                    "timeout_at": timeout_at,
                },
                "update": {"context": context_json, "timeout_at": timeout_at},
            },
        )

    async def claim(self, interaction_id: str) -> bool:
        updated = await self._table.update_many(
            where={"interaction_id": interaction_id, "status": "pending"},
            data={"status": "settled", "claimed_by": self.claimed_by},
        )
        if updated > 0:
            return True
        row = await self._table.find_unique(where={"interaction_id": interaction_id})
        return row is None

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None:
        await self._table.update_many(
            where={"interaction_id": interaction_id},
            data={"outcome": outcome},
        )

    async def is_pending(self, interaction_id: str) -> bool:
        row = await self._table.find_unique(where={"interaction_id": interaction_id})
        return row is not None and row.status == "pending"

    async def get_pending(self, interaction_id: str) -> Optional[PendingSettlementRow]:
        row = await self._table.find_unique(where={"interaction_id": interaction_id})
        if row is None or row.status != "pending":
            return None
        return _parse_row(interaction_id=row.interaction_id, context_value=row.context, timeout_at=row.timeout_at)

    async def list_due(self, older_than: datetime, limit: int) -> tuple[PendingSettlementRow, ...]:
        rows = await self._table.find_many(
            where={"status": "pending", "created_at": {"lt": older_than}},
            take=limit,
            order={"created_at": "asc"},
        )
        parsed = (
            _parse_row(interaction_id=row.interaction_id, context_value=row.context, timeout_at=row.timeout_at)
            for row in rows
        )
        return tuple(row for row in parsed if row is not None)

    async def settle_pending_before_delete(self, interaction_id: str) -> None:
        try:
            row = await self.get_pending(interaction_id)
        except Exception:  # noqa: BLE001  # a store failure must not fail the caller's delete; the sweep settles later
            verbose_proxy_logger.warning(
                "Could not read the pending settlement for background interaction %s before delete; "
                "deferring to the settlement sweep",
                interaction_id,
                exc_info=True,
            )
            return
        if row is None:
            return
        await settle_row_before_delete(row=row, store=self, fetch=self.fetch)
