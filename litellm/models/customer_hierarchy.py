"""
Customer hierarchy table models.

Canonical definitions for ``litellm_customerorganizationtable``, ``litellm_customerteamtable`` and
``litellm_customermembertable``, as the Rust repositories read them: every scalar column plus the budget rows the
row names, under the Prisma relation names ``budget`` and ``budget_override``.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import ConfigDict

from litellm.models.budget import BudgetLimits
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_CustomerBudgetTable(LiteLLMPydanticObjectBase):
    """A ``LiteLLM_BudgetTable`` row a customer row names. Unlike ``LiteLLM_BudgetTable`` it carries the
    server-managed reset clock and accepts the concurrent ``budget_limits`` only customer hierarchies use."""

    model_config = ConfigDict(frozen=True)

    budget_id: str
    name: str | None = None
    max_budget: float | None = None
    soft_budget: float | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    tpd_limit: int | None = None
    model_max_budget: Mapping[str, object] | None = None
    model_tpm_limit: Mapping[str, int] | None = None
    model_rpm_limit: Mapping[str, int] | None = None
    budget_duration: str | None = None
    budget_limits: BudgetLimits | None = None
    cap_policy: Literal["pause_until_reset", "pause_allow_increase"] | None = None
    budget_reset_at: datetime | None = None
    budget_reset_count: int = 0
    allowed_models: tuple[str, ...] | None = None
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str


class LiteLLM_CustomerOrganizationTable(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    customer_org_id: str
    alias: str | None = None
    spend: float = 0.0
    budget_template_ids: tuple[str, ...] = ()
    budget_id: str | None = None
    default_budget_template_id: str | None = None
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str
    budget: LiteLLM_CustomerBudgetTable | None = None


class LiteLLM_CustomerTeamTable(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    customer_team_id: str
    alias: str | None = None
    parent_org_id: str
    spend: float = 0.0
    spend_reset_window: str | None = None
    budget_id: str | None = None
    budget_override_id: str | None = None
    budget_override_expires_at: datetime | None = None
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str
    budget: LiteLLM_CustomerBudgetTable | None = None
    budget_override: LiteLLM_CustomerBudgetTable | None = None


class LiteLLM_CustomerMemberTable(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    customer_member_id: str
    alias: str | None = None
    parent_org_id: str
    parent_team_ids: tuple[str, ...] = ()
    spend: float = 0.0
    spend_reset_window: str | None = None
    budget_id: str | None = None
    budget_override_id: str | None = None
    budget_override_expires_at: datetime | None = None
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str
    budget: LiteLLM_CustomerBudgetTable | None = None
    budget_override: LiteLLM_CustomerBudgetTable | None = None
