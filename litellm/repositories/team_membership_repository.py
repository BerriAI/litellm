from typing import TYPE_CHECKING, Final

from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustTeamMembershipRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


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
