"""
Budget repository for database operations on LiteLLM_BudgetTable.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, TypeVar

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.repositories.base_repository import BaseRepository
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models


_Row: Final = TypeVar("_Row")


async def create_budget_row(table: TableActions[_Row], data: Mapping[str, object], actor: str) -> _Row:
    return await table.create(data={**data, "created_by": actor, "updated_by": actor})


async def update_budget_row(
    table: TableActions[_Row], budget_id: str, data: Mapping[str, object], actor: str
) -> _Row | None:
    return await table.update(where={"budget_id": budget_id}, data={**data, "updated_by": actor})


class _BudgetTable(PrismaTableRepository["prisma_models.LiteLLM_BudgetTable"]):
    table_name = "litellm_budgettable"


class BudgetRepository(BaseRepository[LiteLLM_BudgetTable]):
    """Repository for budget database operations."""

    @property
    def table(self) -> TableActions["prisma_models.LiteLLM_BudgetTable"]:
        return _BudgetTable(self._prisma_client).table

    @property
    def model_class(self) -> type[LiteLLM_BudgetTable]:
        return LiteLLM_BudgetTable

    async def find_by_id(self, budget_id: str, id_field: str = "budget_id") -> LiteLLM_BudgetTable | None:
        return await super().find_by_id(budget_id, id_field)

    async def create_budget(
        self,
        created_by: str,
        max_budget: float | None = None,
        soft_budget: float | None = None,
        max_parallel_requests: int | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        model_max_budget: Mapping[str, object] | None = None,
        budget_duration: str | None = None,
        allowed_models: list[str] | None = None,
    ) -> LiteLLM_BudgetTable:
        """Create a new budget record."""
        data: Final[dict[str, object]] = {
            "created_by": created_by,
            "updated_by": created_by,
        }
        if max_budget is not None:
            data["max_budget"] = max_budget
        if soft_budget is not None:
            data["soft_budget"] = soft_budget
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if model_max_budget is not None:
            data["model_max_budget"] = model_max_budget
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if allowed_models is not None:
            data["allowed_models"] = allowed_models

        record: Final = await create_budget_row(self.table, data, created_by)
        model: Final = self._to_model(record)
        assert model is not None
        return model

    async def update_budget(
        self,
        budget_id: str,
        updated_by: str,
        max_budget: float | None = None,
        soft_budget: float | None = None,
        max_parallel_requests: int | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        model_max_budget: Mapping[str, object] | None = None,
        budget_duration: str | None = None,
        allowed_models: list[str] | None = None,
    ) -> LiteLLM_BudgetTable | None:
        """Update an existing budget record."""
        data: Final[dict[str, object]] = {"updated_by": updated_by}
        if max_budget is not None:
            data["max_budget"] = max_budget
        if soft_budget is not None:
            data["soft_budget"] = soft_budget
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if model_max_budget is not None:
            data["model_max_budget"] = model_max_budget
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if allowed_models is not None:
            data["allowed_models"] = allowed_models

        record: Final = await update_budget_row(self.table, budget_id, data, updated_by)
        return self._to_model(record)

    async def delete_budget(self, budget_id: str) -> LiteLLM_BudgetTable | None:
        """Delete a budget record."""
        return await self.delete(budget_id, id_field="budget_id")
