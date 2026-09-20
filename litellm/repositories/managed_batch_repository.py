from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.types.utils import LiteLLMBatch

if TYPE_CHECKING:
    from prisma import models as prisma_models


def _batch_of(blob: object) -> LiteLLMBatch:
    return LiteLLMBatch.model_validate_json(blob) if isinstance(blob, str) else LiteLLMBatch.model_validate(blob)


class ManagedBatchRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedObjectTable"]):
    table_name = "litellm_managedobjecttable"

    async def load_batch(self, unified_batch_id: str) -> LiteLLMBatch | None:
        row: Final = await self._find_row(unified_batch_id)
        return None if row is None or not row.file_object else _batch_of(row.file_object)

    async def load_status(self, unified_batch_id: str) -> str | None:
        row: Final = await self._find_row(unified_batch_id)
        return row.status if row is not None else None

    async def compare_and_set(
        self, batch: LiteLLMBatch, unchanged: Mapping[str, object], updated_by: str | None
    ) -> bool:
        updated_rows: Final = await self.table.update_many(
            where={"unified_object_id": batch.id, **unchanged},  # mutable-ok: prisma filters are plain dicts
            data={  # mutable-ok: prisma payloads are plain dicts
                "file_object": batch.model_dump_json(),
                "status": batch.status,
                "updated_by": updated_by,
            },
        )
        return updated_rows > 0

    async def touch(self, unified_batch_id: str, updated_by: str | None) -> None:
        await self.table.update_many(
            where={"unified_object_id": unified_batch_id},  # mutable-ok: prisma filters are plain dicts
            data={"updated_by": updated_by},  # mutable-ok: prisma payloads are plain dicts
        )

    async def _find_row(self, unified_batch_id: str) -> "prisma_models.LiteLLM_ManagedObjectTable | None":
        return await self.table.find_first(
            where={"unified_object_id": unified_batch_id}  # mutable-ok: prisma filters are plain dicts
        )
