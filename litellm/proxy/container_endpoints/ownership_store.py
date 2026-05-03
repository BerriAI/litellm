from typing import Any, Dict, List, Optional, Set

CONTAINER_OBJECT_PURPOSE = "container"


class ContainerOwnershipStore:
    def __init__(self, prisma_client: Any):
        self.prisma_client = prisma_client

    @property
    def _table(self) -> Any:
        return self.prisma_client.db.litellm_managedobjecttable

    async def find_by_model_object_id(self, model_object_id: str) -> Optional[Any]:
        return await self._table.find_unique(where={"model_object_id": model_object_id})

    async def create_owner_record(self, data: Dict[str, Any]) -> None:
        await self._table.create(data=data)

    async def update_owner_record(
        self,
        model_object_id: str,
        data: Dict[str, Any],
    ) -> None:
        await self._table.update(
            where={"model_object_id": model_object_id},
            data=data,
        )

    async def get_owner(self, model_object_id: str) -> Optional[str]:
        row = await self._table.find_first(
            where={
                "model_object_id": model_object_id,
                "file_purpose": CONTAINER_OBJECT_PURPOSE,
            }
        )
        if row is None:
            return None
        return getattr(row, "created_by", None)

    async def list_model_object_ids_for_owners(
        self,
        owner_scopes: List[str],
    ) -> Set[str]:
        rows = await self._table.find_many(
            where={
                "file_purpose": CONTAINER_OBJECT_PURPOSE,
                "created_by": {"in": owner_scopes},
            }
        )
        return {
            row.model_object_id
            for row in rows
            if getattr(row, "model_object_id", None) is not None
        }
