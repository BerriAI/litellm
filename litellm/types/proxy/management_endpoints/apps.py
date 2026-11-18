from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    CustomerBudgetUsage,
)


class AppResourceType(str, Enum):
    TEAM = "team"
    PROJECT = "project"
    VIRTUAL_KEY = "virtual_key"


class AppMemberBudgetSource(str, Enum):
    MEMBER_OVERRIDE = "member_override"
    CUSTOMER_DEFAULT = "customer_default"
    APP_DEFAULT = "app_default"
    NONE = "none"


class AppDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_type: AppResourceType | None = None
    auto_create_customers_without_app: bool = Field(
        default=True,
        description=(
            "Whether a request that belongs to no app may register a customer organization or member the proxy "
            "has never seen. When false, such a request is refused with 403."
        ),
    )


class AppResource(BaseModel):
    model_config = ConfigDict(frozen=True)

    resource_type: AppResourceType
    resource_id: str
    alias: str
    app_id: str | None = None


class AppSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    budget: Budget | None = None
    default_tier_budget_id: str | None = None
    default_customer_member_budget: Budget | None = None
    auto_create_customers: bool | None = Field(
        default=None,
        description=(
            "Whether this app may register a customer organization or member the proxy has never seen. When false, "
            "a request from this app naming an unknown customer is refused with 403. Defaults to true."
        ),
    )
    customer_margin: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description=(
            "Fraction added to the cost of every request this app makes for a customer, e.g. 0.02 adds 2%. It "
            "replaces the gateway-wide customer surcharge for this app, so 0 turns the surcharge off for it. Null "
            "uses the gateway-wide customer surcharge."
        ),
    )


class AppPatch(AppSettings):
    pass


class AppCreate(AppSettings):
    resource_id: str = Field(min_length=1)
    resource_type: AppResourceType | None = None


class App(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    app_id: str
    resource_type: AppResourceType
    resource_id: str
    alias: str
    spend: float = 0
    budget_usage: tuple[CustomerBudgetUsage, ...] | None = Field(
        default=None,
        description="Recorded usage in each effective budget period. Lifetime spend is reported separately in spend.",
    )
    spend_reset_window: str | None = None
    budget_id: str | None = None
    default_tier_budget_id: str | None = None
    default_customer_member_budget_id: str | None = None
    auto_create_customers: bool = True
    customer_margin: float | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    budget: Budget | None = None
    default_customer_member_budget: Budget | None = None


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
    budget_usage: tuple[CustomerBudgetUsage, ...] | None = Field(
        default=None,
        description="Recorded usage in each effective budget period. Lifetime spend is reported separately in spend.",
    )
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
    budget_usage: tuple[CustomerBudgetUsage, ...] | None = Field(
        default=None,
        description="Recorded usage in each effective budget period. Lifetime spend is reported separately in spend.",
    )
    spend_reset_window: str | None = None
    budget_override_id: str | None = None
    effective_budget: Budget | None = None
    budget_source: AppMemberBudgetSource = AppMemberBudgetSource.NONE


@dataclass(frozen=True, slots=True)
class AppFailure:
    status: Literal[400, 403, 404, 409, 422, 503]
    detail: str
