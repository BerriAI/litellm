from typing import Any, Dict, List, Optional


class LiteLLMSkillsStore:
    def __init__(self, prisma_client: Any):
        self.prisma_client = prisma_client

    @property
    def _table(self) -> Any:
        return self.prisma_client.db.litellm_skillstable

    async def create_skill(self, data: Dict[str, Any]) -> Any:
        return await self._table.create(data=data)

    async def list_skills(self, find_many_kwargs: Dict[str, Any]) -> List[Any]:
        return await self._table.find_many(**find_many_kwargs)

    async def find_skill(self, skill_id: str) -> Optional[Any]:
        return await self._table.find_unique(where={"skill_id": skill_id})

    async def delete_skill(self, skill_id: str) -> None:
        await self._table.delete(where={"skill_id": skill_id})
