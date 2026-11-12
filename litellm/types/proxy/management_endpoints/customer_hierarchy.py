from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    allowed_models: tuple[str, ...] | None = None
    budget_duration: str | None = None


class BudgetTemplate(Budget):
    is_default: bool = False


class OrganizationBudget(BudgetTemplate):
    budget_id: str


class OrganizationPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    default_budgets: tuple[BudgetTemplate, ...] | None = None


class OrganizationCreate(OrganizationPatch):
    customer_org_id: str | None = None


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
    budget: Budget | None = None


class MemberCreate(MemberPatch):
    customer_member_id: str | None = None
    parent_team_id: str
    budget_template_id: str | None = None


class AuditFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    alias: str | None = None
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str


class Organization(AuditFields):
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
    budget_id: str
    budget: Budget
