"""
Budget repository for database operations on LiteLLM_BudgetTable.
"""

from collections.abc import Sequence
from typing import Any, Final

from litellm.models.budget import LiteLLM_BudgetTable, LiteLLM_BudgetTableFull
from litellm.repositories.base_repository import BaseRepository, DbRecord, record_to_dict


class BudgetRepository(BaseRepository[LiteLLM_BudgetTable]):
    """Repository for budget database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_budgettable

    @property
    def model_class(self) -> type[LiteLLM_BudgetTable]:
        return LiteLLM_BudgetTable

    async def find_by_id(self, budget_id: str, id_field: str = "budget_id") -> LiteLLM_BudgetTable | None:
        return await super().find_by_id(budget_id, id_field)

    async def find_full_by_ids(self, budget_ids: Sequence[str]) -> tuple[LiteLLM_BudgetTableFull, ...]:
        """Reset schedules are server-managed, so they are absent from the model the generic finders return."""
        records: Final[Sequence[DbRecord]] = await self.table.find_many(
            where={"budget_id": {"in": list(budget_ids)}}  # mutable-ok: prisma builds its query from plain dicts
        )
        return tuple(LiteLLM_BudgetTableFull.model_validate(record_to_dict(record)) for record in records)

    async def create_budget(
        self,
        created_by: str,
        max_budget: float | None = None,
        soft_budget: float | None = None,
        max_parallel_requests: int | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        model_max_budget: dict[str, Any] | None = None,
        budget_duration: str | None = None,
        allowed_models: list[str] | None = None,
    ) -> LiteLLM_BudgetTable:
        """Create a new budget record."""
        data: Final[dict[str, Any]] = {
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

        return await self.create(data)

    async def update_budget(
        self,
        budget_id: str,
        updated_by: str,
        max_budget: float | None = None,
        soft_budget: float | None = None,
        max_parallel_requests: int | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        model_max_budget: dict[str, Any] | None = None,
        budget_duration: str | None = None,
        allowed_models: list[str] | None = None,
    ) -> LiteLLM_BudgetTable | None:
        """Update an existing budget record."""
        data: Final[dict[str, Any]] = {"updated_by": updated_by}
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

        return await self.update(budget_id, data, id_field="budget_id")

    async def delete_budget(self, budget_id: str) -> LiteLLM_BudgetTable | None:
        """Delete a budget record."""
        return await self.delete(budget_id, id_field="budget_id")
