from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, BudgetTemplate, TemplatePatch


class AppSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    budget: Budget | None = None
    default_tier_budget_id: str | None = None
    default_customer_member_budget: Budget | None = None
    upsert_policy: Mapping[str, object] | None = None


class AppPatch(AppSettings):
    alias: str | None = Field(default=None, min_length=1, pattern=r"^\S(?:.*\S)?$")


class AppCreate(AppSettings):
    alias: str = Field(min_length=1, pattern=r"^\S(?:.*\S)?$")


class App(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    app_id: str
    alias: str
    spend: float = 0
    spend_reset_window: str | None = None
    budget_id: str | None = None
    tier_budget_ids: tuple[str, ...] = ()
    default_tier_budget_id: str | None = None
    default_customer_member_budget_id: str | None = None
    upsert_policy: Mapping[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    budget: Budget | None = None
    tiers: tuple["AppTier", ...] = ()
    default_customer_member_budget: Budget | None = None


class AppTier(BudgetTemplate):
    model_config = ConfigDict(frozen=True, extra="ignore")

    budget_id: str


class TierPatch(TemplatePatch):
    budget_id: str


class TierDelete(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_id: str


class AppOrganizationPut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tier_budget_id: str | None = None
    budget_override: Budget | None = None
    customer_member_budget: Budget | None = None


class AppMemberPut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_override: Budget | None = None


class AppOrganization(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    app_id: str
    customer_org_id: str
    spend: float = 0
    spend_reset_window: str | None = None
    tier_budget_id: str | None = None
    budget_override_id: str | None = None
    customer_member_budget_id: str | None = None
    effective_budget: Budget | None = None
    effective_customer_member_budget: Budget | None = None


class AppMember(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    app_id: str
    customer_member_id: str
    spend: float = 0
    spend_reset_window: str | None = None
    budget_override_id: str | None = None
    effective_budget: Budget | None = None


@dataclass(frozen=True, slots=True)
class AppFailure:
    status: Literal[400, 403, 404, 409, 422, 503]
    detail: str
