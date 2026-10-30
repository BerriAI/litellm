import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from typing import Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_cost_calc.finalization import FinalizedCost, PreparedCost, finalize_cost
from litellm.proxy.hooks.search_free_tier import SharedCounter, allocate_search_free_tier
from litellm.types.utils import CostBreakdown, StandardLoggingPayload


def prepare_response_cost(logging_obj: Logging, response: object, now: datetime) -> PreparedCost | None:
    try:
        return logging_obj.prepare_response_cost(response, now)
    except Exception:  # noqa: BLE001  # preparation must not prevent serving an already completed response
        verbose_proxy_logger.exception("Unable to prepare completed response cost for %s", logging_obj.litellm_call_id)
        return None


async def finalize_response_cost(
    logging_obj: Logging,
    response: object,
    counter: SharedCounter | None,
    now: datetime,
    guardrail_cost: float | None = None,
) -> FinalizedCost | None:
    """The finalized cost, or ``None`` when it cannot be known yet: the request is still served and the legacy
    callback path bills it, so a finalization failure never costs the client a response or the proxy a spend row."""
    try:
        return await _finalize_response_cost(logging_obj, response, counter, now, guardrail_cost)
    except Exception:  # noqa: BLE001  # the response was already paid for; billing falls back to the callback path
        verbose_proxy_logger.exception(
            "Cost finalization failed for %s, billing through the callback path", logging_obj.litellm_call_id
        )
        return None


async def _finalize_response_cost(
    logging_obj: Logging,
    response: object,
    counter: SharedCounter | None,
    now: datetime,
    guardrail_cost: float | None,
) -> FinalizedCost | None:
    async with logging_obj.cost_finalization_lock:
        if logging_obj.finalized_cost is not None:
            return logging_obj.finalized_cost
        prepared: Final = prepare_response_cost(logging_obj, response, now)
        if prepared is None:
            return None
        payload: Final = prepared.payload
        amount: Final = payload["response_cost"] if guardrail_cost is None else prepared.amount + guardrail_cost
        breakdown: Final[CostBreakdown | None] = (
            payload["cost_breakdown"]
            if guardrail_cost is None or guardrail_cost == 0
            else {
                **(payload["cost_breakdown"] or {}),
                "guardrail_cost": guardrail_cost,
                "total_cost": amount,
            }
        )
        task: Final = asyncio.create_task(
            _finish_finalization(logging_obj, {"usage": prepared.usage}, payload, amount, breakdown, counter, now)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(task)
            task.result()
            raise


async def _finish_finalization(
    logging_obj: Logging,
    response: object,
    payload: StandardLoggingPayload,
    amount: float,
    breakdown: CostBreakdown | None,
    counter: SharedCounter | None,
    now: datetime,
) -> FinalizedCost:
    if logging_obj.cost_allowance_discount is None:
        allocation: Final = (
            await allocate_search_free_tier(payload, amount, response, counter, now)
            if litellm.apply_search_free_tier
            else None
        )
        logging_obj.cost_allowance_allocation = allocation  # rebind-ok: retain the allocation for durable capture
        logging_obj.cost_allowance_discount = (  # rebind-ok: retain allocation across snapshot retries
            allocation.discount if allocation is not None else 0.0
        )
    finalized: Final = finalize_cost(amount, breakdown, logging_obj.cost_allowance_discount)
    logging_obj.record_finalized_cost(finalized)
    return finalized


async def finalize_proxy_response_cost(
    logging_obj: Logging, response: object, guardrail_cost: float | None = None
) -> FinalizedCost | None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    return await finalize_response_cost(
        logging_obj,
        response,
        proxy_logging_obj.internal_usage_cache.dual_cache.redis_cache,
        datetime.now(timezone.utc),
        guardrail_cost,
    )
