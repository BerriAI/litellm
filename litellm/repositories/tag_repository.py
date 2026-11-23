from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from litellm.models.tag import LiteLLM_TagTable
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustTagRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class TagRepository(PrismaTableRepository["prisma_models.LiteLLM_TagTable"]):
    table_name = "litellm_tagtable"

    async def names(self, limit: int) -> tuple[str, ...]:
        """Up to ``limit`` tag names."""
        return tuple(await rust_database.required_repository(RustTagRepository).names(limit))

    async def find_many_with_budgets(self, tag_names: Sequence[str]) -> tuple[LiteLLM_TagTable, ...]:
        """The tags among ``tag_names`` that have a row, with their budgets."""
        rows: Final = await rust_database.required_repository(RustTagRepository).find_many_with_budgets(list(tag_names))
        return tuple(LiteLLM_TagTable.model_validate_json(row) for row in rows)
