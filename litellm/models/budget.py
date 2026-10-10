"""
Budget table model.

Canonical definition for ``litellm_budgettable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime, timezone
from typing import Final

from pydantic import ConfigDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_BudgetTable(LiteLLMPydanticObjectBase):
    """Represents user-controllable params for a LiteLLM_BudgetTable record.

    Budget-write paths use `model_fields.keys()` on this class as an allowlist
    for user input. Keep server-managed fields (e.g. `budget_reset_at`) on
    `LiteLLM_BudgetTableFull` so they aren't user-settable.
    """

    budget_id: str | None = None
    soft_budget: float | None = None
    max_budget: float | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    tpd_limit: int | None = None
    model_max_budget: dict | None = None
    budget_duration: str | None = None
    allowed_models: list[str] | None = None  # per-member model scope; empty = inherit team models
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None

    model_config = ConfigDict(protected_namespaces=())

    def active_temp_budget_increase(self, now: datetime) -> float:
        if self.temp_budget_increase is None or self.temp_budget_expiry is None:
            return 0.0
        expiry: Final = (
            self.temp_budget_expiry.replace(tzinfo=timezone.utc)
            if self.temp_budget_expiry.tzinfo is None
            else self.temp_budget_expiry
        )
        return 0.0 if expiry <= now else self.temp_budget_increase

    def effective_max_budget(self, now: datetime) -> float | None:
        if self.max_budget is None:
            return None
        return self.max_budget + self.active_temp_budget_increase(now)


class LiteLLM_BudgetTableFull(LiteLLM_BudgetTable):
    """LiteLLM_BudgetTable + server-managed fields returned on API responses."""

    budget_reset_at: datetime | None = None
    created_at: datetime


class LiteLLM_TeamMemberTable(LiteLLM_BudgetTable):
    """
    Used to track spend of a user_id within a team_id
    """

    spend: float | None = None
    user_id: str | None = None
    team_id: str | None = None
    budget_id: str | None = None

    model_config = ConfigDict(protected_namespaces=())
