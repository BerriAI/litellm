from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from litellm.types.proxy.management_endpoints.apps import AppBudgetSelection, AppResourceType
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, CustomerBudgetUsage

GroupName: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class AppGroupBudgetPut(AppBudgetSelection):
    pass


class AppGroupApp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_type: AppResourceType
    resource_id: str = Field(min_length=1)


class AppGroupSettings(AppGroupBudgetPut):
    apps: tuple[AppGroupApp, ...] = Field(
        default=(),
        description="Apps selected by their gateway resource type and ID. When supplied, replaces membership. "
        "Omit to keep membership unchanged. An empty list removes all apps. Existing app budgets are preserved.",
    )
    description: str | None = None
    metadata: Mapping[str, JsonValue] | None = None

    @model_validator(mode="after")
    def unique_apps(self) -> "AppGroupSettings":
        if len(frozenset(self.apps)) != len(self.apps):
            raise ValueError("Each app can only be selected once")
        return self


class AppGroupPatch(AppGroupSettings):
    name: GroupName | None = None

    @model_validator(mode="after")
    def nonnull_name(self) -> "AppGroupPatch":
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("The group name cannot be null")
        return self


class AppGroupCreate(AppGroupSettings):
    name: GroupName


class AppGroup(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    app_group_id: str
    name: str
    description: str | None = None
    metadata: Mapping[str, JsonValue] | None = None
    budget_id: str | None = None
    budget: Budget | None = None
    spend: float = 0
    budget_usage: tuple[CustomerBudgetUsage, ...] = ()
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str


class AppGroupOrganization(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    app_group_id: str
    customer_org_id: str
    budget_id: str | None = None
    budget: Budget | None = None
    spend: float = 0
    budget_usage: tuple[CustomerBudgetUsage, ...] = ()
