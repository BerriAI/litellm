import asyncio
from datetime import datetime, timezone
from typing import Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_cost_calc.finalization import FinalizedCost, finalize_cost
from litellm.proxy.hooks.search_free_tier import SharedCounter, allocate_search_free_tier
from litellm.types.utils import CostBreakdown, StandardLoggingPayload


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
        deferred: Final = logging_obj.deferred_success_result
        prepared: Final = logging_obj.prepare_success_logging(
            result=deferred[0] if deferred is not None else response,
            start_time=deferred[1] if deferred is not None else logging_obj.start_time,
            end_time=deferred[2] if deferred is not None else now.replace(tzinfo=None),
            cache_hit=_recorded_cache_hit(logging_obj),
            standard_logging_object=None,
            build_logging_payload=True,
            publish_logging_payload=False,
        )
        payload: Final = prepared[3]
        raw_cost: Final[object] = logging_obj.model_call_details.get("response_cost")
        if payload is None or not isinstance(raw_cost, (int, float)):
            return None
        amount: Final = payload["response_cost"] if guardrail_cost is None else raw_cost + guardrail_cost
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
            _finish_finalization(logging_obj, prepared[2], payload, amount, breakdown, counter, now)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
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
    allocation: Final = (
        await allocate_search_free_tier(payload, amount, response, counter, now)
        if litellm.apply_search_free_tier
        else None
    )
    finalized: Final = finalize_cost(amount, breakdown, allocation.discount if allocation is not None else 0.0)
    logging_obj.finalized_cost = finalized
    return finalized


def _recorded_cache_hit(logging_obj: Logging) -> bool | None:
    """The cache hit the caching handler recorded before deferring, so a cached dict result is not priced."""
    recorded: Final[object] = logging_obj.model_call_details.get("cache_hit")
    return recorded if isinstance(recorded, bool) else None


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
