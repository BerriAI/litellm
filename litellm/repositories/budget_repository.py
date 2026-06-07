"""
Budget repository for database operations on LiteLLM_BudgetTable.
"""

from typing import Any, Dict, List, Optional, Type

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.repositories.base_repository import BaseRepository


class BudgetRepository(BaseRepository[LiteLLM_BudgetTable]):
    """Repository for budget database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_budgettable

    @property
    def model_class(self) -> Type[LiteLLM_BudgetTable]:
        return LiteLLM_BudgetTable

    async def find_by_id(
        self, budget_id: str, id_field: str = "budget_id"
    ) -> Optional[LiteLLM_BudgetTable]:
        return await super().find_by_id(budget_id, id_field)

    async def create_budget(
        self,
        created_by: str,
        max_budget: Optional[float] = None,
        soft_budget: Optional[float] = None,
        max_parallel_requests: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        model_max_budget: Optional[Dict[str, Any]] = None,
        budget_duration: Optional[str] = None,
        allowed_models: Optional[List[str]] = None,
    ) -> LiteLLM_BudgetTable:
        """Create a new budget record."""
        data: Dict[str, Any] = {
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
        max_budget: Optional[float] = None,
        soft_budget: Optional[float] = None,
        max_parallel_requests: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        model_max_budget: Optional[Dict[str, Any]] = None,
        budget_duration: Optional[str] = None,
        allowed_models: Optional[List[str]] = None,
    ) -> Optional[LiteLLM_BudgetTable]:
        """Update an existing budget record."""
        data: Dict[str, Any] = {"updated_by": updated_by}
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

    async def delete_budget(self, budget_id: str) -> Optional[LiteLLM_BudgetTable]:
        """Delete a budget record."""
        return await self.delete(budget_id, id_field="budget_id")
