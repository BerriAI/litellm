"""
Team membership table model.

Canonical definition for ``litellm_teammembership``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from litellm.models.budget import LiteLLM_BudgetTable, LiteLLM_BudgetTableFull
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_TeamMembership(LiteLLMPydanticObjectBase):
    user_id: str
    team_id: str
    budget_id: str | None = None
    spend: float | None = 0.0
    total_spend: float | None = 0.0
    litellm_budget_table: LiteLLM_BudgetTableFull | LiteLLM_BudgetTable | None = None

    def safe_get_team_member_rpm_limit(self) -> int | None:
        if self.litellm_budget_table is not None:
            return self.litellm_budget_table.rpm_limit
        return None

    def safe_get_team_member_tpm_limit(self) -> int | None:
        if self.litellm_budget_table is not None:
            return self.litellm_budget_table.tpm_limit
        return None
