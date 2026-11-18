"""
App table models.

Canonical definitions for ``litellm_apptable``, ``litellm_appcustomerorgtable`` and
``litellm_appcustomermembertable``, as the Rust repositories read them: every scalar column, without relations.
"""

from datetime import datetime

from pydantic import ConfigDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_AppTable(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    app_id: str
    app_group_id: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    alias: str
    spend: float = 0.0
    spend_reset_window: str | None = None
    budget_id: str | None = None
    default_tier_budget_id: str | None = None
    default_customer_member_budget_id: str | None = None
    auto_create_customers: bool = True
    customer_margin: float | None = None
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str


class LiteLLM_AppCustomerOrgTable(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    app_id: str
    customer_org_id: str
    spend: float = 0.0
    spend_reset_window: str | None = None
    tier_budget_id: str | None = None
    budget_id: str | None = None
    budget_override_id: str | None = None
    budget_override_expires_at: datetime | None = None
    customer_member_budget_id: str | None = None


class LiteLLM_AppCustomerMemberTable(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    app_id: str
    customer_member_id: str
    spend: float = 0.0
    spend_reset_window: str | None = None
    budget_override_id: str | None = None
    budget_id: str | None = None
    budget_override_expires_at: datetime | None = None
