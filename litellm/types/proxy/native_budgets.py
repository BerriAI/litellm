from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr


class NativeReseed(BaseModel):
    model_config = ConfigDict(frozen=True)

    counter_key: str
    actual_cost: float


class NativePendingIncrement(BaseModel):
    model_config = ConfigDict(frozen=True)

    counter_key: str
    increment: float


class NativeSettlement(BaseModel):
    model_config = ConfigDict(frozen=True)

    adjustments: tuple[float | None, ...]
    reseed: tuple[NativeReseed, ...]
    pending: tuple[NativePendingIncrement, ...]


PyNumber: TypeAlias = StrictInt | StrictFloat


class _Check(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class KeyMaxCheck(_Check):
    check: Literal["key_max"]
    spend: PyNumber
    max_budget: PyNumber
    token: StrictStr | None = None
    key_alias: StrictStr | None = None
    key_name: StrictStr | None = None
    throttle_pct: PyNumber | None = None


class KeyWindowCheck(_Check):
    check: Literal["key_window"]
    spend: PyNumber
    max_budget: PyNumber
    budget_duration: StrictStr
    token: StrictStr | None = None


class KeySoftCheck(_Check):
    check: Literal["key_soft"]
    spend: PyNumber
    soft_budget: PyNumber | None = None


class TeamMaxCheck(_Check):
    check: Literal["team_max"]
    spend: PyNumber
    max_budget: PyNumber
    team_id: StrictStr | None = None


class TeamWindowCheck(_Check):
    check: Literal["team_window"]
    spend: PyNumber
    max_budget: PyNumber
    budget_duration: StrictStr
    team_id: StrictStr | None = None


class TeamSoftCheck(_Check):
    check: Literal["team_soft"]
    spend: PyNumber | None = None
    soft_budget: PyNumber | None = None


class ProjectMaxCheck(_Check):
    check: Literal["project_max"]
    spend: PyNumber
    max_budget: PyNumber
    project_id: StrictStr | None = None


class ProjectSoftCheck(_Check):
    check: Literal["project_soft"]
    spend: PyNumber | None = None
    soft_budget: PyNumber | None = None


class OrganizationCheck(_Check):
    check: Literal["organization"]
    spend: PyNumber
    max_budget: PyNumber
    org_id: StrictStr


class TagCheck(_Check):
    check: Literal["tag"]
    spend: PyNumber
    max_budget: PyNumber
    tag: StrictStr


class ModelAccessGroupCheck(_Check):
    check: Literal["model_access_group"]
    spend: PyNumber
    max_budget: PyNumber
    group: StrictStr


class UserCheck(_Check):
    check: Literal["user"]
    spend: PyNumber
    max_budget: PyNumber
    user_id: StrictStr | None = None


class TeamMemberCheck(_Check):
    check: Literal["team_member"]
    spend: PyNumber
    max_budget: PyNumber
    user_id: StrictStr | None = None
    team_id: StrictStr | None = None


class TeamMemberPositiveCheck(_Check):
    check: Literal["team_member_positive"]
    spend: PyNumber
    max_budget: PyNumber
    user_id: StrictStr | None = None
    team_id: StrictStr | None = None


class EndUserCheck(_Check):
    check: Literal["end_user"]
    spend: PyNumber
    max_budget: PyNumber
    end_user_id: StrictStr | None = None


class ProxyCheck(_Check):
    check: Literal["proxy"]
    spend: PyNumber
    max_budget: PyNumber


BudgetCheck: TypeAlias = Annotated[
    KeyMaxCheck
    | KeyWindowCheck
    | KeySoftCheck
    | TeamMaxCheck
    | TeamWindowCheck
    | TeamSoftCheck
    | ProjectMaxCheck
    | ProjectSoftCheck
    | OrganizationCheck
    | TagCheck
    | ModelAccessGroupCheck
    | UserCheck
    | TeamMemberCheck
    | TeamMemberPositiveCheck
    | EndUserCheck
    | ProxyCheck,
    Field(discriminator="check"),
]

__all__ = [
    "BudgetCheck",
    "EndUserCheck",
    "KeyMaxCheck",
    "KeySoftCheck",
    "KeyWindowCheck",
    "ModelAccessGroupCheck",
    "NativePendingIncrement",
    "NativeReseed",
    "NativeSettlement",
    "OrganizationCheck",
    "ProjectMaxCheck",
    "ProjectSoftCheck",
    "ProxyCheck",
    "PyNumber",
    "TagCheck",
    "TeamMaxCheck",
    "TeamMemberCheck",
    "TeamMemberPositiveCheck",
    "TeamSoftCheck",
    "TeamWindowCheck",
    "UserCheck",
]
