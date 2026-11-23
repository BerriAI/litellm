from typing import TYPE_CHECKING, Final

from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustEndUserRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class EndUserRepository(PrismaTableRepository["prisma_models.LiteLLM_EndUserTable"]):
    table_name = "litellm_endusertable"

    async def find_with_budget_and_object_permission(self, end_user_id: str) -> LiteLLM_EndUserTable | None:
        """The end user ``end_user_id`` with its budget and object permission."""
        payload: Final = await rust_database.required_repository(
            RustEndUserRepository
        ).find_with_budget_and_object_permission(end_user_id)
        return rust_database.decode(LiteLLM_EndUserTable, payload)

    async def find_restricted_ids(self, limit: int) -> tuple[str, ...]:
        """Ids of end users whose row restricts something auth enforces, at most ``limit`` of them."""
        return tuple(await rust_database.required_repository(RustEndUserRepository).find_restricted_ids(limit))
