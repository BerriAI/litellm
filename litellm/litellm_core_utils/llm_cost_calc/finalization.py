from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter

from litellm.types.utils import CostBreakdown, StandardLoggingPayload, Usage

BreakdownValue: TypeAlias = float | str | None | Mapping[str, float]
_BREAKDOWN_FIELDS: Final = TypeAdapter(Mapping[str, BreakdownValue])
_COST_BREAKDOWN: Final = TypeAdapter(CostBreakdown)


@dataclass(frozen=True, slots=True)
class PreparedCost:
    amount: float
    payload: StandardLoggingPayload
    usage: Usage


@dataclass(frozen=True, slots=True)
class FinalizedCost:
    amount: float
    breakdown: Mapping[str, BreakdownValue] | None
    allowance_discount: float

    def logging_payload(self, payload: StandardLoggingPayload) -> StandardLoggingPayload:
        return {
            **payload,
            "response_cost": self.amount,
            "cost_breakdown": self.logging_breakdown(),
            "cost_finalized": True,
        }

    def logging_breakdown(self) -> CostBreakdown | None:
        return _COST_BREAKDOWN.validate_python(self.breakdown) if self.breakdown is not None else None


def finalize_cost(amount: float, breakdown: CostBreakdown | None, allowance_discount: float = 0.0) -> FinalizedCost:
    discount: Final = min(max(amount, 0.0), max(allowance_discount, 0.0))
    total: Final = amount - discount
    existing: Final = _BREAKDOWN_FIELDS.validate_python(breakdown) if breakdown is not None else None
    adjusted: Final = (
        {
            **(existing or {}),
            "total_cost": total,
            "original_cost": (breakdown or {}).get("original_cost", amount),
            "discount_amount": (breakdown or {}).get("discount_amount", 0.0) + discount,
        }
        if discount > 0
        else existing
    )
    frozen: Final = (
        MappingProxyType(
            {key: MappingProxyType(value) if isinstance(value, Mapping) else value for key, value in adjusted.items()}
        )
        if adjusted is not None
        else None
    )
    return FinalizedCost(amount=total, breakdown=frozen, allowance_discount=discount)
