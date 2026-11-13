from collections.abc import Mapping
from datetime import datetime
from typing import Final, Literal, TypeAlias

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, NonNegativeInt, field_validator

from litellm.models.budget import BudgetLimits
from litellm.proxy.common_utils.timezone_utils import budget_duration_error

CapPolicy: TypeAlias = Literal["pause_until_reset", "pause_allow_increase"]
INCREASE_REQUEST_ROUTE: Final = "/management/v1/customer_budget_increase_requests"
IncreaseRequestStatus: TypeAlias = Literal["pending_review", "approved", "rejected"]


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_budget: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    soft_budget: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    tpm_limit: int | None = Field(default=None, ge=0)
    rpm_limit: int | None = Field(default=None, ge=0)
    allowed_models: tuple[str, ...] | None = None
    budget_duration: str | None = None
    budget_limits: BudgetLimits | None = None
    model_tpm_limit: Mapping[str, NonNegativeInt] | None = None
    model_rpm_limit: Mapping[str, NonNegativeInt] | None = None
    cap_policy: CapPolicy | None = Field(
        default=None,
        description="What a paused member can do at the cap: wait for the reset (default) or request a temporary increase",
    )

    @field_validator("budget_duration")
    @classmethod
    def valid_budget_duration(cls, value: str | None) -> str | None:
        error: Final = budget_duration_error(value)
        if error is not None:
            raise ValueError(error)
        return value


class BudgetTemplate(Budget):
    name: str = Field(min_length=1)


class TemplatePatch(Budget):
    name: str | None = Field(default=None, min_length=1)


class OrganizationBudget(BudgetTemplate):
    budget_id: str


class OrganizationFields(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    budget: Budget | None = None


class OrganizationPatch(OrganizationFields):
    default_budget_template_id: str | None = None


class OrganizationCreate(OrganizationFields):
    customer_org_id: str | None = None
    default_budgets: tuple[BudgetTemplate, ...] = ()
    default_budget_template: str | None = Field(default=None, min_length=1)


class TeamPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    budget: Budget | None = None
    budget_override: Budget | None = None
    budget_override_expires_at: AwareDatetime | None = None


class TeamCreate(TeamPatch):
    customer_team_id: str | None = None
    parent_org_id: str


class _MemberFields(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    budget_id: str | None = None
    budget_override: Budget | None = None
    budget_override_expires_at: AwareDatetime | None = None


class MemberPatch(_MemberFields):
    parent_team_ids: tuple[str, ...] | None = Field(default=None, min_length=1)


class MemberCreate(_MemberFields):
    customer_member_id: str | None = None
    parent_org_id: str | None = Field(default=None, description="Must match the parent teams' organization when set")
    parent_team_ids: tuple[str, ...] = Field(min_length=1, description="Teams the member belongs to, first is default")


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
    default_budget_template_id: str | None = None
    default_budgets: tuple[OrganizationBudget, ...] = ()


class Team(AuditFields):
    customer_team_id: str
    parent_org_id: str
    budget_id: str | None = None
    budget: Budget | None = None
    budget_override_id: str | None = None
    budget_override: Budget | None = None
    budget_override_expires_at: datetime | None = None
    effective_budget: Budget


class CustomerMember(AuditFields):
    customer_member_id: str
    parent_org_id: str
    parent_team_ids: tuple[str, ...]
    budget_id: str | None = None
    budget_override_id: str | None = None
    budget_override: Budget | None = None
    budget_override_expires_at: datetime | None = None
    effective_budget: Budget


class BudgetIncreaseRequestCreate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_member_id: str = Field(max_length=256)
    requested_max_budget: float = Field(ge=0, allow_inf_nan=False)
    expires_at: AwareDatetime = Field(description="When the granted cap ends; a temporary increase always has one")
    reason: str | None = Field(default=None, max_length=1000)


class BudgetIncreaseRejection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    review_notes: str | None = Field(default=None, max_length=1000)


class BudgetIncreaseRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    request_id: str
    customer_member_id: str
    requested_max_budget: float
    expires_at: datetime | None = None
    reason: str | None = None
    status: IncreaseRequestStatus
    submitted_by: str
    submitted_at: datetime
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_notes: str | None = None
