"""
Team membership table model.

Canonical definition for ``litellm_teammembership``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final

from pydantic import Field

from litellm.models.budget import LiteLLM_BudgetTable, LiteLLM_BudgetTableFull
from litellm.models.budget import active_rollover_credit as calculate_rollover_credit
from litellm.types.llms.base import LiteLLMPydanticObjectBase


def _is_none(value: object) -> bool:
    return value is None


class LiteLLM_TeamMembership(LiteLLMPydanticObjectBase):
    user_id: str
    team_id: str
    budget_id: str | None = None
    spend: float | None = 0.0
    total_spend: float | None = 0.0
    rollover_credit: float | None = Field(default=None, exclude_if=_is_none)
    litellm_budget_table: LiteLLM_BudgetTableFull | LiteLLM_BudgetTable | None = None

    def active_rollover_credit(self) -> float | None:
        budget: Final = self.litellm_budget_table
        if budget is None:
            return None
        return calculate_rollover_credit(
            budget.max_budget, budget.rollover_max_budget, budget.budget_duration, self.rollover_credit
        )

    def effective_budget_with_rollover(self, now: datetime) -> float | None:
        budget: Final = self.litellm_budget_table
        if budget is None:
            return None
        effective: Final = budget.effective_max_budget(now)
        credit: Final = self.active_rollover_credit()
        return effective if credit is None or effective is None else effective + credit

    def rollover_usage(self, now: datetime) -> Mapping[str, float | None]:
        credit: Final = self.active_rollover_credit()
        return MappingProxyType(
            {
                "rollover_credit": credit,
                "effective_max_budget": self.effective_budget_with_rollover(now) if credit is not None else None,
            }
        )

    def safe_get_team_member_rpm_limit(self) -> int | None:
        if self.litellm_budget_table is not None:
            return self.litellm_budget_table.rpm_limit
        return None

    def safe_get_team_member_tpm_limit(self) -> int | None:
        if self.litellm_budget_table is not None:
            return self.litellm_budget_table.tpm_limit
        return None
