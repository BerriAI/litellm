import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.finalization import FinalizedCost, finalize_cost
from litellm.types.utils import StandardLoggingPayload


def get_team_cost_margin_percent(metadata: Mapping[str, object]) -> object:
    return metadata.get("user_api_key_team_cost_margin_percent")


def _cost_is_finalized(payload: Mapping[str, object]) -> bool:
    return bool(payload.get("cost_finalized", False))


def apply_team_cost_margin(cost: FinalizedCost, percent: object) -> FinalizedCost:
    if not isinstance(percent, (int, float)) or isinstance(percent, bool):
        return cost
    if not math.isfinite(percent) or not 0 <= percent <= 1000:
        return cost
    if cost.breakdown is not None and "team_cost_margin_percent" in cost.breakdown:
        return cost
    amount: Final = cost.amount * (1 + percent / 100)
    return FinalizedCost(
        amount=amount,
        breakdown=MappingProxyType(
            {
                **(cost.breakdown or {}),
                "team_cost_before_margin": cost.amount,
                "team_cost_margin_percent": float(percent),
                "total_cost": amount,
            }
        ),
        allowance_discount=cost.allowance_discount,
    )


def apply_callback_team_cost_margin(
    payload: StandardLoggingPayload | None, amount: float | None, percent: object
) -> float | None:
    if payload is None or amount is None:
        return amount
    if _cost_is_finalized(payload):
        return amount
    cost: Final = apply_team_cost_margin(finalize_cost(amount, payload["cost_breakdown"]), percent)
    payload["response_cost"] = cost.amount  # rebind-ok: the legacy spend writer consumes this callback payload
    payload["cost_breakdown"] = cost.logging_breakdown()  # rebind-ok: persist reconciliation with the billed cost
    return cost.amount
