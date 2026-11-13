from collections.abc import Mapping
from datetime import datetime
from typing import Final

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, NonNegativeInt, field_validator

from litellm.models.person_budget import (
    PersonBudgetCapPolicy,
    PersonBudgetDuration,
    PersonBudgetLimit,
    PersonBudgetPolicy,
)
from litellm.proxy.common_utils.timezone_utils import budget_duration_error


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_budget: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    soft_budget: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    tpm_limit: int | None = Field(default=None, ge=0)
    rpm_limit: int | None = Field(default=None, ge=0)
    allowed_models: tuple[str, ...] | None = None
    budget_duration: str | None = None
    model_tpm_limit: Mapping[str, NonNegativeInt] | None = None
    model_rpm_limit: Mapping[str, NonNegativeInt] | None = None

    @field_validator("budget_duration")
    @classmethod
    def valid_budget_duration(cls, value: str | None) -> str | None:
        error: Final = budget_duration_error(value)
        if error is not None:
            raise ValueError(error)
        return value


class BudgetTemplate(Budget):
    name: str = Field(min_length=1)
    is_default: bool = False


class MemberBudget(Budget):
    """Personal policy alongside the existing customer-member scalar limits."""

    budget_limits: tuple[PersonBudgetLimit, ...] | None = None
    cap_policy: PersonBudgetCapPolicy | None = None
    temp_budget_increase: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    temp_budget_expiry: AwareDatetime | None = None
    temp_budget_duration: PersonBudgetDuration | None = None

    @field_validator("budget_limits")
    @classmethod
    def unique_durations(cls, value: tuple[PersonBudgetLimit, ...] | None) -> tuple[PersonBudgetLimit, ...] | None:
        return PersonBudgetPolicy(budget_limits=value).budget_limits


class TemplatePatch(Budget):
    name: str | None = Field(default=None, min_length=1)
    is_default: bool | None = None


class OrganizationBudget(BudgetTemplate):
    budget_id: str


class OrganizationPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    budget: Budget | None = None


class OrganizationCreate(OrganizationPatch):
    customer_org_id: str | None = None
    default_budgets: tuple[BudgetTemplate, ...] = ()


class TeamPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    budget: Budget | None = None


class TeamCreate(TeamPatch):
    customer_team_id: str | None = None
    parent_org_id: str


class MemberPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    budget_id: str | None = None
    budget_override: MemberBudget | None = None
    budget_override_expires_at: AwareDatetime | None = None


class MemberCreate(MemberPatch):
    customer_member_id: str | None = None
    parent_team_id: str


class AuditFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    alias: str | None = None
    spend: float = 0.0
    budget_reset_at: datetime | None = None
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str


class Organization(AuditFields):
    member_count: int = Field(default=0, ge=0, description="Current registered members across all customer teams")
    budget_id: str | None = None
    budget: Budget | None = None
    customer_org_id: str
    default_budget_id: str | None = None
    default_budgets: tuple[OrganizationBudget, ...] = ()


class Team(AuditFields):
    customer_team_id: str
    parent_org_id: str
    budget_id: str | None = None
    budget: Budget | None = None


class CustomerMember(AuditFields):
    customer_member_id: str
    parent_team_id: str
    budget_id: str | None = None
    budget_override_id: str | None = None
    budget_override: MemberBudget | None = None
    budget_override_expires_at: datetime | None = None
    effective_budget: MemberBudget
