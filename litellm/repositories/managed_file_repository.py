from typing import TYPE_CHECKING, Final

from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.types.llms.openai import OpenAIFileObject

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class ManagedFileRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedFileTable"]):
    table_name = "litellm_managedfiletable"

    async def update_file_object(
        self, unified_file_id: str, file_object: OpenAIFileObject, *, if_file_object: str | None = None
    ) -> bool:
        """Replace a row's file object, only while it still holds ``if_file_object`` (JSON text) when given."""
        updated_rows: Final = await self.table.update_many(
            where={"unified_file_id": unified_file_id}
            if if_file_object is None
            else {"unified_file_id": unified_file_id, "file_object": {"equals": if_file_object}},
            data={"file_object": file_object.model_dump_json()},
        )
        return updated_rows > 0
