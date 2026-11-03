"""Concurrent budget windows scoped to one user within one team."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.duration_parser import get_budget_window_start, get_next_standardized_reset_time
from litellm.models.team import LiteLLM_TeamTable, TeamMemberBudgetDuration, TeamMemberBudgetWindow
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_settings

_MEMBER_ID: Final = TypeAdapter(tuple[str, str])


def member_window_entity_id(user_id: str, team_id: str) -> str:
    return _MEMBER_ID.dump_json((user_id, team_id)).decode()


def parse_member_window_entity_id(entity_id: str) -> tuple[str, str]:
    return _MEMBER_ID.validate_json(entity_id)


def get_team_member_budget_windows(
    team: LiteLLM_TeamTable, membership: LiteLLM_TeamMembership | None
) -> Sequence[TeamMemberBudgetWindow]:
    if membership is not None and membership.budget_limits:
        return membership.budget_limits
    return team.team_member_budget_limits or ()


@dataclass(frozen=True, slots=True)
class MemberBudgetCounter:
    entity_id: str
    duration: TeamMemberBudgetDuration
    start: datetime
    reset_at: datetime

    @property
    def key(self) -> str:
        return f"spend:team_member_window:{self.entity_id}:{self.duration.value}:{int(self.start.timestamp())}"

    @classmethod
    def for_member(
        cls, user_id: str, team_id: str, duration: TeamMemberBudgetDuration, now: datetime | None = None
    ) -> "MemberBudgetCounter":
        settings: Final = get_budget_reset_settings()
        reset_at: Final = get_next_standardized_reset_time(
            duration=duration.value,
            current_time=now if now is not None else datetime.now(timezone.utc),
            timezone_str=settings.timezone,
            reset_time_of_day=settings.reset_time_of_day,
        )
        return cls(
            entity_id=member_window_entity_id(user_id, team_id),
            duration=duration,
            start=get_budget_window_start(duration.value, reset_at),
            reset_at=reset_at,
        )
