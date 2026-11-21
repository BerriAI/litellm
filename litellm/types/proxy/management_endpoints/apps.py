from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    CustomerBudgetUsage,
)


class AppResourceType(str, Enum):
    TEAM = "team"
    PROJECT = "project"
    VIRTUAL_KEY = "virtual_key"


class AppMemberBudgetSource(str, Enum):
    MEMBER_BUDGET = "member_budget"
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


class AppBudgetSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    budget: Budget | None = None
    budget_id: str | None = Field(
        default=None,
        min_length=1,
        description="Linked budget preset. Omit to keep the current assignment, or send null to remove it. "
        "Send budget instead to create a private custom budget, preserving the preset's untouched limits.",
    )

    @model_validator(mode="after")
    def exclusive_budget(self) -> "AppBudgetSelection":
        if self.budget_id is not None and self.budget is not None:
            raise ValueError("Choose a preset or a custom budget")
        return self


class AppSettings(AppBudgetSelection):
    app_group_id: str | None = Field(
        default=None,
        min_length=1,
        description="Parent app group. Omit to preserve membership, or send null to ungroup the app.",
    )
    default_tier_budget_id: str | None = None
    default_customer_member_budget: Budget | None = None
    default_customer_member_budget_id: str | None = Field(
        default=None,
        description="Linked app × customer member budget group used as this app's per-member default. "
        "Omit to preserve it, or send null to remove it. Send default_customer_member_budget instead for custom limits.",
    )
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

    @model_validator(mode="after")
    def exclusive_member_budget(self) -> "AppSettings":
        if self.default_customer_member_budget_id is not None and self.default_customer_member_budget is not None:
            raise ValueError("Choose a member preset or a custom member budget")
        return self


class AppPatch(AppSettings):
    pass


class AppCreate(AppSettings):
    resource_id: str = Field(min_length=1)
    resource_type: AppResourceType | None = None


class App(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    app_id: str
    app_group_id: str | None = None
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
    budget_id: str | None = None
    budget: Budget | None = None
    budget_override: Budget | None = None
    budget_override_expires_at: AwareDatetime | None = None
    customer_member_budget: Budget | None = None
    customer_member_budget_id: str | None = Field(
        default=None,
        description="Linked app × customer member budget group for members of this customer in this app. "
        "Omit to preserve it, or send null to inherit the app default. Send customer_member_budget for custom limits.",
    )

    @model_validator(mode="after")
    def exclusive_budget(self) -> "AppOrganizationPut":
        if self.budget_id is not None and self.tier_budget_id is not None and self.budget_id != self.tier_budget_id:
            raise ValueError("Choose one budget preset")
        if (self.budget_id is not None or self.tier_budget_id is not None) and self.budget is not None:
            raise ValueError("Choose a preset or a custom budget")
        if self.customer_member_budget_id is not None and self.customer_member_budget is not None:
            raise ValueError("Choose a member preset or a custom member budget")
        return self


class AppMemberPut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_id: str | None = Field(
        default=None,
        description="Linked app × customer member budget group. Omit to preserve it, or send null to inherit defaults. "
        "Send budget instead to fork the preset into private custom limits. Temporary increases remain independent.",
    )
    budget: Budget | None = Field(
        default=None,
        description="Permanent budget for this app and customer member. Copies inherited policy on first edit. "
        "Explicit null fields clear limits. Null budget returns to inheritance. Independent of temporary increases.",
    )
    budget_override: Budget | None = None
    budget_override_expires_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def exclusive_budget(self) -> "AppMemberPut":
        if self.budget_id is not None and self.budget is not None:
            raise ValueError("Choose a preset or a custom budget")
        return self


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
    budget_id: str | None = None
    budget: Budget | None = None
    budget_override: Budget | None = None
    budget_override_expires_at: datetime | None = None
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
    budget_id: str | None = None
    budget_override_expires_at: datetime | None = None
    budget: Budget | None = None
    budget_override: Budget | None = None
    effective_budget: Budget | None = None
    budget_source: AppMemberBudgetSource = AppMemberBudgetSource.NONE


@dataclass(frozen=True, slots=True)
class AppFailure:
    status: Literal[400, 403, 404, 409, 422, 503]
    detail: str
