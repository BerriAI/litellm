from typing import TYPE_CHECKING, Final

from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustAccessGroupRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class AccessGroupRepository(PrismaTableRepository["prisma_models.LiteLLM_AccessGroupTable"]):
    table_name = "litellm_accessgrouptable"

    async def find_by_name(self, name: str) -> LiteLLM_AccessGroupTable | None:
        row: Final = await self.table.find_unique(where={"access_group_name": name})
        return LiteLLM_AccessGroupTable.model_validate(row, from_attributes=True) if row is not None else None

    async def find_by_access_group_id(
        self, access_group_id: str, *, use_writer: bool = False
    ) -> LiteLLM_AccessGroupTable | None:
        """The access group, from the writer when ``use_writer`` is set."""
        return rust_database.decode(
            LiteLLM_AccessGroupTable,
            await rust_database.required_repository(RustAccessGroupRepository).find_by_access_group_id(
                access_group_id, use_writer=use_writer
            ),
        )
