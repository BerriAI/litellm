from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.models.team import Member
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustTeamMembershipRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class NewMemberBudget(BaseModel):
    """A budget of the member's own, inserted in the member add's transaction. It holds the columns a budget
    created for a new team member can set, and refuses any other."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_id: str
    created_by: str
    updated_by: str
    allowed_models: tuple[str, ...]
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemberAdded:
    """The membership now stored and the team's roster as the add stored it, the added member last."""

    membership: LiteLLM_TeamMembership
    roster: tuple[Member, ...]


_ROSTER: Final = TypeAdapter(tuple[Member, ...])


MemberNotAdded: TypeAlias = Literal["already_member", "team_missing", "user_missing"]


class TeamMembershipRepository(PrismaTableRepository["prisma_models.LiteLLM_TeamMembership"]):
    table_name = "litellm_teammembership"

    async def find_with_budget(
        self, user_id: str, team_id: str, *, use_writer: bool = False
    ) -> LiteLLM_TeamMembership | None:
        """A user's membership in a team with its member budget, from the writer when asked."""
        found: Final = await rust_database.required_repository(RustTeamMembershipRepository).find_with_budget(
            user_id, team_id, use_writer=use_writer
        )
        return rust_database.decode(LiteLLM_TeamMembership, found)

    async def add_member(
        self, user_id: str, team_id: str, *, budget_id: str | None, own_budget: NewMemberBudget | None
    ) -> MemberAdded | MemberNotAdded:
        """Adds the user to the team as a ``user`` in one transaction under the team's advisory lock: the team on
        the user's ``teams``, ``own_budget`` when the member gets a budget of its own, the membership row linked to
        ``budget_id``, and the user on the team's roster."""
        outcome, membership, roster = await rust_database.required_repository(RustTeamMembershipRepository).add_member(
            user_id,
            team_id,
            budget_id=budget_id,
            own_budget=None
            if own_budget is None
            else (
                own_budget.allowed_models,
                own_budget.budget_duration,
                None if own_budget.budget_reset_at is None else own_budget.budget_reset_at.timestamp(),
                own_budget.created_by,
                own_budget.updated_by,
            ),
        )
        if outcome != "added":
            return outcome
        return MemberAdded(
            LiteLLM_TeamMembership.model_validate_json(membership or "null"), _ROSTER.validate_json(roster or "[]")
        )
