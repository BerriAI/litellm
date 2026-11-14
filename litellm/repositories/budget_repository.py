"""
Budget repository for database operations on LiteLLM_BudgetTable.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
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


def _window_name(row: "prisma_models.LiteLLM_BudgetTable") -> str:
    closes_at: Final = row.budget_reset_at.isoformat() if row.budget_reset_at is not None else "unset"
    return f"{row.budget_id}:{row.created_at.isoformat()}:{row.budget_reset_count}:{closes_at}"


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

    async def due_tier_windows(self, budget_ids: Sequence[str], due_before: datetime) -> Mapping[str, str]:
        """The name of the window each listed tier still due before ``due_before``
        closes: the tier, when it was created, how many windows it has closed
        before, and when this one closes. An admin moving the reset time or
        recreating the tier opens a new window even when no advance ran. Read
        on the writer, so a lagging replica cannot show a tier another sweep
        already advanced as still due."""
        rows: Final = await _BudgetTable(self._prisma_client, use_writer=True).table.find_many(
            where={
                # bounded-ok: callers pass one get_data page of at most RESET_BUDGET_JOB_BATCH_SIZE tiers
                "budget_id": {"in": [*budget_ids]},
                "OR": [{"budget_reset_at": None}, {"budget_reset_at": {"lt": due_before}}],
            }
        )
        return MappingProxyType({row.budget_id: _window_name(row) for row in rows})

    async def delete_budget(self, budget_id: str) -> LiteLLM_BudgetTable | None:
        """Delete a budget record."""
        return await self.delete(budget_id, id_field="budget_id")
