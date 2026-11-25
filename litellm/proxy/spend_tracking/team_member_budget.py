from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.team_membership import LiteLLM_TeamMembership


@dataclass(frozen=True, slots=True)
class TeamMemberBudget:
    max_budget: float
    budget_reset_at: datetime | None


def team_member_budget(
    membership: LiteLLM_TeamMembership | None,
    team_default_budget: LiteLLM_BudgetTable | None,
    now: datetime,
) -> TeamMemberBudget | None:
    from litellm.proxy.auth.auth_checks import (  # noqa: PLC0415  # auth_checks imports this module
        resolve_effective_team_member_budget,
    )

    max_budget: Final = resolve_effective_team_member_budget(
        member_budget=membership.litellm_budget_table if membership is not None else None,
        team_default_budget=team_default_budget,
        now=now,
        membership=membership,
    )
    if max_budget is None:
        return None
    return TeamMemberBudget(
        max_budget=max_budget,
        budget_reset_at=membership.budget_reset_at() if membership is not None else None,
    )


def hours_until(budget_reset_at: datetime, now: datetime) -> float:
    reset_at: Final = (
        budget_reset_at if budget_reset_at.tzinfo is not None else budget_reset_at.replace(tzinfo=timezone.utc)
    )
    return (reset_at - now).total_seconds() / 3600
