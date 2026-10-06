from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy._types import TeamMemberBudgetSource


class TeamMemberResponse(BaseModel):
    """One roster entry with the member's spend in the team and the budget row their membership points at.

    The limits are that row's, exactly as `/team/info` returns it under `team_memberships`, so a member
    with no row of their own reads null limits even when `budget_source` is `team_default`.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str | None
    user_email: str | None
    user_alias: str | None
    role: Literal["admin", "user"]
    spend: float
    total_spend: float
    budget_id: str | None
    budget_source: TeamMemberBudgetSource
    max_budget_in_team: float | None
    budget_duration: str | None
    budget_reset_at: datetime | None
    tpm_limit: int | None
    rpm_limit: int | None
    allowed_models: tuple[str, ...]


class TeamMembersQueryParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    q: str | None = Field(
        None, description="Search by `user_id` or `user_email`. Matches any part of the value and ignores case"
    )
    role: Literal["admin", "user"] | None = Field(None, alias="filter[role]", description="Only members with this role")
    role_in: str | None = Field(
        None,
        alias="filter[role][in]",
        description="Only members with one of these roles, comma-separated, as in `admin,user`",
    )
    sort: str | None = Field(
        None,
        description=(
            "`user_alias`, `user_email`, `user_id`, `role`, `spend`, `total_spend`, `max_budget_in_team` or "
            "`budget_reset_at`. Put `-` in front to sort descending. Defaults to the order members were added"
        ),
    )
    page: int = Field(1, ge=1, description="The page to return, starting at 1")
    page_size: int | None = Field(None, ge=1, description="Members per page. Defaults to 50, max 100")
