from typing import TYPE_CHECKING, Final

from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models


class ManagedFileContentRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedFileContentTable"]):
    table_name = "litellm_managedfilecontenttable"

    async def store(self, content: bytes) -> str:
        from prisma import Base64

        row: Final = await self.table.create(data={"content": Base64.encode(content)})
        return row.id

    async def load(self, row_id: str) -> bytes | None:
        row: Final[prisma_models.LiteLLM_ManagedFileContentTable | None] = await self.table.find_unique(
            where={"id": row_id}
        )
        return None if row is None else row.content.decode()

    async def delete(self, row_id: str) -> None:
        from prisma.errors import RecordNotFoundError

        try:
            await self.table.delete(where={"id": row_id})
        except RecordNotFoundError:
            return
