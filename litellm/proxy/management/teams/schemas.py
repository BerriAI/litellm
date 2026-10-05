from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import TeamMemberBudgetSource


class TeamMemberListItem(BaseModel):
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
