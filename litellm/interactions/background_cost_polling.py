"""
Cost tracking for background interactions.

A create request with ``background=true`` returns ``in_progress`` with no
usage block, and GET polls are deliberately never billed (billing them would
double-charge every poll; the GET response also does not echo ``background``,
so a poll cannot be told apart from a re-fetch of an already-billed
interaction). The create call is therefore the only place that can own
billing: it schedules a poll task that fetches the interaction until it
reaches a terminal status and logs the final usage as a single success event
attributed to the original request.

Deleting an interaction makes every subsequent poll fail, which would let a
caller retrieve the completed output themselves and then delete it before the
poll task settles, leaving the work unbilled and the budget reservation
refunded at the poll timeout. ``adelete`` therefore settles any pending poll
for the interaction before dispatching the delete: it fetches the current
state with the create's credentials, bills it if it is terminal with usage,
and releases the reservation otherwise. A settlement gate on the create's
logging object makes the poll task and the delete path mutually exclusive, so
the interaction is billed exactly once no matter who settles first.

The poll registry and the create's logging object only live in the pod that
served the create, so a delete routed to another pod or a proxy restart would
leave the completed work unbilled. When a durable settlement store is
registered (the proxy registers a Prisma-backed one at startup), the poll
task persists enough create context for any pod to settle, and every outcome
must win the store's cross-pod compare-and-swap claim before touching spend
counters; the in-memory gate remains the same-pod fast path and the only gate
when no store is registered (SDK usage).
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterator, Literal, Optional, Protocol

from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.constants import (
    BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS,
    BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS,
    BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS,
    BACKGROUND_INTERACTION_COST_POLLING_ENABLED,
)
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.types.interactions import InteractionsAPIResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "incomplete", "budget_exceeded"})

SettlementOutcome = Literal["billed", "released", "abandoned", "error"]


class SettlementReservationEntry(BaseModel):
    counter_key: str
    entity_type: str
    entity_id: str
    reserved_cost: float
    applied_adjustment: float = 0.0


class SettlementReservation(BaseModel):
    reserved_cost: float
    entries: list[SettlementReservationEntry]
    finalized: bool
    input_cost: float


class SettlementKeyAttribution(BaseModel):
    user_api_key: Optional[str] = None
    user_api_key_user_id: Optional[str] = None
    user_api_key_team_id: Optional[str] = None
    user_api_key_org_id: Optional[str] = None
    user_api_key_alias: Optional[str] = None
    user_api_key_team_alias: Optional[str] = None
    user_api_key_user_email: Optional[str] = None
    user_api_key_end_user_id: Optional[str] = None


class BackgroundSettlementContext(BaseModel):
    interaction_id: str
    custom_llm_provider: str
    model: str
    model_group: Optional[str] = None
    litellm_call_id: str
    litellm_trace_id: Optional[str] = None
    call_type: str
    attribution: SettlementKeyAttribution
    budget_reservation: Optional[SettlementReservation] = None


class BackgroundSettlementStore(Protocol):
    async def persist_pending(self, context: BackgroundSettlementContext, timeout_at: datetime) -> None: ...

    async def claim(self, interaction_id: str) -> bool: ...

    async def record_outcome(self, interaction_id: str, outcome: SettlementOutcome) -> None: ...

    async def is_pending(self, interaction_id: str) -> bool: ...

    async def settle_pending_before_delete(self, interaction_id: str) -> None: ...


_SETTLEMENT_STORE: Optional[BackgroundSettlementStore] = None


def get_settlement_store() -> Optional[BackgroundSettlementStore]:
    return _SETTLEMENT_STORE


def set_settlement_store(store: Optional[BackgroundSettlementStore]) -> None:
    global _SETTLEMENT_STORE
    _SETTLEMENT_STORE = store


@dataclass(frozen=True, slots=True)
class BackgroundInteractionPollContext:
    interaction_id: str
    custom_llm_provider: str
    logging_obj: "LiteLLMLoggingObj"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    initial_interval_seconds: float = BACKGROUND_INTERACTION_COST_POLL_INITIAL_INTERVAL_SECONDS
    max_interval_seconds: float = BACKGROUND_INTERACTION_COST_POLL_MAX_INTERVAL_SECONDS
    timeout_seconds: float = BACKGROUND_INTERACTION_COST_POLL_TIMEOUT_SECONDS


FetchInteraction = Callable[[BackgroundInteractionPollContext], Awaitable[InteractionsAPIResponse]]


async def _fetch_interaction(context: BackgroundInteractionPollContext) -> InteractionsAPIResponse:
    from litellm.interactions import aget

    return await aget(
        interaction_id=context.interaction_id,
        custom_llm_provider=context.custom_llm_provider,
        **{"api_key": context.api_key, "api_base": context.api_base, "no-log": True},
    )


def _poll_intervals(initial: float, maximum: float, timeout: float) -> Iterator[float]:
    elapsed = 0.0
    interval = initial
    while elapsed + interval <= timeout:
        yield interval
        elapsed += interval
        interval = min(interval * 2, maximum)


_SETTLED_KEY = "background_interaction_settled"
_SETTLED_OUTCOME_KEY = "background_interaction_settled_outcome"
_SETTLEMENT_OUTCOMES: tuple[SettlementOutcome, ...] = ("billed", "released", "abandoned", "error")


def _is_settled(logging_obj: "LiteLLMLoggingObj") -> bool:
    return logging_obj.model_call_details.get(_SETTLED_KEY) is True


def _stash_settlement_outcome(logging_obj: "LiteLLMLoggingObj", outcome: SettlementOutcome) -> None:
    logging_obj.model_call_details[_SETTLED_OUTCOME_KEY] = outcome


def _stashed_settlement_outcome(logging_obj: "LiteLLMLoggingObj") -> SettlementOutcome:
    stashed = logging_obj.model_call_details.get(_SETTLED_OUTCOME_KEY)
    for outcome in _SETTLEMENT_OUTCOMES:
        if stashed == outcome:
            return outcome
    return "released"


def _claim_settlement(logging_obj: "LiteLLMLoggingObj") -> bool:
    """
    Exactly-once gate between the poll task and the delete-time settlement:
    both run on the same event loop and neither awaits between reading and
    setting the flag, so whichever claims first owns billing or release.
    """
    if _is_settled(logging_obj):
        return False
    logging_obj.model_call_details[_SETTLED_KEY] = True
    return True


def _get_reservation_dict(logging_obj: "LiteLLMLoggingObj") -> Optional[dict]:
    metadata = get_litellm_metadata_from_kwargs(kwargs=logging_obj.model_call_details)
    budget_reservation = metadata.get("user_api_key_budget_reservation")
    return budget_reservation if isinstance(budget_reservation, dict) else None


def _finalize_reservation_locally(logging_obj: "LiteLLMLoggingObj") -> None:
    """
    Losing the durable store's claim means another pod owns (or already
    performed) the billing or release for this interaction. The local copy of
    the reservation dict must be marked finalized without touching the shared
    counters, or a later reconcile of this copy would re-apply its full
    adjustment on top of the winner's.
    """
    budget_reservation = _get_reservation_dict(logging_obj)
    if budget_reservation is not None:
        budget_reservation["finalized"] = True


def build_settlement_context(
    context: BackgroundInteractionPollContext,
) -> BackgroundSettlementContext:
    metadata = get_litellm_metadata_from_kwargs(kwargs=context.logging_obj.model_call_details)
    reservation_dict = metadata.get("user_api_key_budget_reservation")
    model_group = metadata.get("deployment_model_name")
    return BackgroundSettlementContext(
        interaction_id=context.interaction_id,
        custom_llm_provider=context.custom_llm_provider,
        model=str(context.logging_obj.model or context.logging_obj.model_call_details.get("model")),
        model_group=model_group if isinstance(model_group, str) else None,
        litellm_call_id=str(context.logging_obj.litellm_call_id),
        litellm_trace_id=context.logging_obj.model_call_details.get("litellm_trace_id"),
        call_type=str(context.logging_obj.call_type),
        attribution=SettlementKeyAttribution.model_validate(metadata),
        budget_reservation=(
            SettlementReservation.model_validate(reservation_dict) if isinstance(reservation_dict, dict) else None
        ),
    )


async def _persist_pending_settlement(
    context: BackgroundInteractionPollContext,
    store: BackgroundSettlementStore,
) -> bool:
    try:
        settlement_context = build_settlement_context(context)
        timeout_at = datetime.now(timezone.utc) + timedelta(seconds=context.timeout_seconds)
        await store.persist_pending(context=settlement_context, timeout_at=timeout_at)
        return True
    except Exception:  # noqa: BLE001  # a failed persist degrades to in-memory-only settlement, never kills the poll
        verbose_logger.warning(
            "Failed to persist pending settlement for background interaction %s; settlement will be in-memory only",
            context.interaction_id,
            exc_info=True,
        )
        return False


async def _claim_in_store(store: BackgroundSettlementStore, interaction_id: str) -> bool:
    try:
        return await store.claim(interaction_id)
    except Exception:  # noqa: BLE001  # an unreachable store defers to the sweep instead of risking a double bill
        verbose_logger.exception(
            "Failed to claim settlement for background interaction %s; deferring to the settlement sweep",
            interaction_id,
        )
        return False


async def _record_outcome_best_effort(
    store: BackgroundSettlementStore,
    interaction_id: str,
    outcome: SettlementOutcome,
) -> None:
    try:
        await store.record_outcome(interaction_id, outcome)
    except Exception:  # noqa: BLE001  # the outcome column is observability, never worth failing settlement over
        verbose_logger.warning(
            "Failed to record settlement outcome %s for background interaction %s",
            outcome,
            interaction_id,
            exc_info=True,
        )


async def _claim_across_gates(
    context: BackgroundInteractionPollContext,
    store: Optional[BackgroundSettlementStore],
    intended_outcome: SettlementOutcome,
) -> bool:
    if not _claim_settlement(context.logging_obj):
        return False
    _stash_settlement_outcome(context.logging_obj, intended_outcome)
    if store is not None and not await _claim_in_store(store, context.interaction_id):
        _finalize_reservation_locally(context.logging_obj)
        return False
    return True


async def _settle_claimed(
    context: BackgroundInteractionPollContext,
    response: InteractionsAPIResponse,
    store: Optional[BackgroundSettlementStore],
) -> None:
    outcome: SettlementOutcome
    try:
        if response.usage is not None:
            await context.logging_obj.async_log_background_interaction_completion(result=response)
            outcome = "billed"
        else:
            await _release_open_budget_reservation(logging_obj=context.logging_obj)
            outcome = "released"
    except Exception:  # noqa: BLE001  # a billing failure after winning the claim must be surfaced, not retried
        verbose_logger.exception(
            "Billing failed after claiming settlement for background interaction %s; its spend may be under-tracked",
            context.interaction_id,
        )
        await _release_open_budget_reservation(logging_obj=context.logging_obj)
        outcome = "error"
    _stash_settlement_outcome(context.logging_obj, outcome)
    if store is not None:
        await _record_outcome_best_effort(store, context.interaction_id, outcome)


async def _no_longer_pending_in_store(
    store: BackgroundSettlementStore,
    interaction_id: str,
) -> bool:
    try:
        return not await store.is_pending(interaction_id)
    except Exception:  # noqa: BLE001  # an unreachable store must not change the poll loop's behavior
        return False


async def poll_and_log_background_interaction_cost(
    context: BackgroundInteractionPollContext,
    fetch_interaction: FetchInteraction = _fetch_interaction,
    store: Optional[BackgroundSettlementStore] = None,
) -> None:
    configured_store = store if store is not None else get_settlement_store()
    active_store = (
        configured_store
        if configured_store is not None and await _persist_pending_settlement(context, configured_store)
        else None
    )
    if active_store is not None and _is_settled(context.logging_obj):
        if await _claim_in_store(active_store, context.interaction_id):
            await _record_outcome_best_effort(
                active_store,
                context.interaction_id,
                _stashed_settlement_outcome(context.logging_obj),
            )
        return
    for interval in _poll_intervals(
        initial=context.initial_interval_seconds,
        maximum=context.max_interval_seconds,
        timeout=context.timeout_seconds,
    ):
        await asyncio.sleep(interval)
        if _is_settled(context.logging_obj):
            return
        try:
            response = await fetch_interaction(context)
        except Exception as e:  # noqa: BLE001  # any fetch error must not kill the billing poll loop
            if active_store is not None and await _no_longer_pending_in_store(active_store, context.interaction_id):
                if _claim_settlement(context.logging_obj):
                    _finalize_reservation_locally(context.logging_obj)
                return
            verbose_logger.debug(
                "Background interaction cost poll for %s failed, will retry: %s",
                context.interaction_id,
                e,
            )
            continue
        if response.status not in _TERMINAL_STATUSES:
            continue
        intended_outcome: SettlementOutcome = "billed" if response.usage is not None else "released"
        if not await _claim_across_gates(context, active_store, intended_outcome):
            return
        await _settle_claimed(context=context, response=response, store=active_store)
        return
    if not await _claim_across_gates(context, active_store, "abandoned"):
        return
    verbose_logger.warning(
        "Gave up cost polling for background interaction %s after %ss; its usage will not be tracked",
        context.interaction_id,
        context.timeout_seconds,
    )
    await _release_open_budget_reservation(logging_obj=context.logging_obj)
    if active_store is not None:
        await _record_outcome_best_effort(active_store, context.interaction_id, "abandoned")


async def _release_open_budget_reservation(logging_obj: "LiteLLMLoggingObj") -> None:
    """
    The proxy keeps the pre-call budget reservation open for an in-progress
    background interaction so concurrent creates cannot stack past the budget.
    The completion success event reconciles it to the actual cost; when the
    interaction terminates without billable usage (or polling gives up, or it
    is deleted before settling), no such event fires, so whoever claims the
    settlement must release the reservation here or the spend counters stay
    pinned at the estimated cost.
    """
    budget_reservation = _get_reservation_dict(logging_obj)
    if budget_reservation is None:
        return

    from litellm.proxy.spend_tracking.budget_reservation import release_budget_reservation

    try:
        await release_budget_reservation(budget_reservation=budget_reservation)
    except Exception:  # noqa: BLE001  # a failed release must not crash the poll task; counters expire via TTL
        verbose_logger.exception("Failed to release budget reservation for an unbilled background interaction")


@dataclass(frozen=True, slots=True)
class _ActiveBackgroundPoll:
    task: "asyncio.Task[None]"
    context: BackgroundInteractionPollContext


_ACTIVE_POLLS: dict[
    str, _ActiveBackgroundPoll
] = {}  # mutable-ok: asyncio requires strong refs to running tasks, and delete settlement looks polls up by interaction id


def _discard_poll(interaction_id: str, task: "asyncio.Task[None]") -> None:
    entry = _ACTIVE_POLLS.get(interaction_id)
    if entry is not None and entry.task is task:
        del _ACTIVE_POLLS[interaction_id]


def maybe_schedule_background_interaction_cost_polling(
    response: Any,
    create_kwargs: dict[str, Any],
    custom_llm_provider: str,
) -> Optional["asyncio.Task[None]"]:
    from litellm.litellm_core_utils.litellm_logging import Logging

    if not BACKGROUND_INTERACTION_COST_POLLING_ENABLED:
        return None
    if not isinstance(response, InteractionsAPIResponse):
        return None
    if response.status != "in_progress" or not response.id:
        return None
    logging_obj = create_kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return None
    context = BackgroundInteractionPollContext(
        interaction_id=response.id,
        custom_llm_provider=custom_llm_provider,
        logging_obj=logging_obj,
        api_key=create_kwargs.get("api_key"),
        api_base=create_kwargs.get("api_base"),
    )
    task = asyncio.create_task(poll_and_log_background_interaction_cost(context))
    _ACTIVE_POLLS[context.interaction_id] = _ActiveBackgroundPoll(task=task, context=context)
    task.add_done_callback(
        lambda finished, interaction_id=context.interaction_id: _discard_poll(interaction_id, finished)
    )
    return task


async def maybe_settle_background_interaction_before_delete(
    interaction_id: str,
    fetch_interaction: FetchInteraction = _fetch_interaction,
    store: Optional[BackgroundSettlementStore] = None,
) -> None:
    active_store = store if store is not None else get_settlement_store()
    entry = _ACTIVE_POLLS.get(interaction_id)
    if entry is None:
        if active_store is not None:
            await active_store.settle_pending_before_delete(interaction_id)
        return
    context = entry.context
    try:
        response = await fetch_interaction(context)
    except Exception as e:  # noqa: BLE001  # unfetchable pre-delete state settles by releasing the reservation
        verbose_logger.debug(
            "Could not fetch background interaction %s before delete, releasing its reservation: %s",
            interaction_id,
            e,
        )
        if not await _claim_across_gates(context, active_store, "released"):
            return
        await _release_open_budget_reservation(logging_obj=context.logging_obj)
        if active_store is not None:
            await _record_outcome_best_effort(active_store, interaction_id, "released")
        return
    terminal_with_usage = response.status in _TERMINAL_STATUSES and response.usage is not None
    if not await _claim_across_gates(context, active_store, "billed" if terminal_with_usage else "released"):
        return
    if terminal_with_usage:
        await _settle_claimed(context=context, response=response, store=active_store)
        return
    await _release_open_budget_reservation(logging_obj=context.logging_obj)
    if active_store is not None:
        await _record_outcome_best_effort(active_store, interaction_id, "released")
