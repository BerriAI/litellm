from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, StringConstraints, model_validator

from litellm.types.proxy.management_endpoints.apps import AppBudgetSelection
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, CustomerBudgetUsage

GroupName: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class AppGroupBudgetPut(AppBudgetSelection):
    pass


class AppGroupSettings(AppGroupBudgetPut):
    description: str | None = None
    metadata: Mapping[str, JsonValue] | None = None


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
