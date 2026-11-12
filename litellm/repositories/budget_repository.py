"""
Budget repository for database operations on LiteLLM_BudgetTable.
"""

from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from litellm.litellm_core_utils.core_helpers import require_attributes
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.repositories.base_repository import BaseRepository
from litellm.repositories.prisma_protocols import TableActions, UpdateManyTable
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models


class _BudgetTable(PrismaTableRepository["prisma_models.LiteLLM_BudgetTable"]):
    table_name = "litellm_budgettable"


class _TierTransaction(Protocol):
    @property
    def litellm_budgettable(self) -> TableActions["prisma_models.LiteLLM_BudgetTable"]: ...

    @property
    def litellm_endusertable(self) -> UpdateManyTable: ...


class _TierDatabase(Protocol):
    def tx(self, *, timeout: timedelta) -> AbstractAsyncContextManager[_TierTransaction]: ...


class _TierWriterClient(Protocol):
    @property
    def writer_db(self) -> _TierDatabase: ...


def _reopens_closed_window(stored: datetime | None, written: datetime | None) -> bool:
    return written is not None and written != stored and written <= datetime.now(timezone.utc)


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

        return await self.update(budget_id, data, id_field="budget_id")

    async def update_tier(
        self, budget_id: str, data: Mapping[str, object]
    ) -> "prisma_models.LiteLLM_BudgetTable | None":
        """Moving the tier's reset time back to a window its end users were
        already reset in clears their stamp, so that window resets them again.
        Rewriting the stored time keeps the stamps a failed advance's retry needs.
        The write and the clear commit together, so repeating a failed request
        still finds the old time and clears the stamps."""
        if "budget_reset_at" not in data:
            return await self.table.update(where={"budget_id": budget_id}, data=data)
        writer: Final = require_attributes(self._connected_client(), _TierWriterClient, "writer_db").writer_db
        async with writer.tx(timeout=timedelta(seconds=60)) as transaction:
            before: Final = await transaction.litellm_budgettable.find_unique(where={"budget_id": budget_id})
            after: Final = await transaction.litellm_budgettable.update(where={"budget_id": budget_id}, data=data)
            if before is None or after is None:
                return after
            if not _reopens_closed_window(before.budget_reset_at, after.budget_reset_at):
                return after
            await transaction.litellm_endusertable.update_many(
                where={"budget_id": budget_id, "spend_reset_window": after.budget_reset_at},
                data={"spend_reset_window": None},
            )
        return after

    async def pin_unset_reset_times(self, budget_ids: Sequence[str], at: datetime) -> Mapping[str, datetime]:
        """Set ``budget_reset_at`` to ``at`` on the tiers that have none, then
        return every tier's stored reset time. Both run on the writer, so a
        lagging replica cannot hide the pin. Only empty reset times are written,
        so a tier another sweep pinned or advanced first returns that value."""
        writer: Final = _BudgetTable(self._prisma_client, use_writer=True).table
        await writer.update_many(
            # bounded-ok: callers pass one get_data page of at most RESET_BUDGET_JOB_BATCH_SIZE tiers
            where={"budget_id": {"in": [*budget_ids]}, "budget_reset_at": None},
            data={"budget_reset_at": at},
        )
        # bounded-ok: callers pass one get_data page of at most RESET_BUDGET_JOB_BATCH_SIZE tiers
        rows: Final = await writer.find_many(where={"budget_id": {"in": [*budget_ids]}})
        return MappingProxyType({row.budget_id: row.budget_reset_at for row in rows if row.budget_reset_at is not None})

    async def delete_budget(self, budget_id: str) -> LiteLLM_BudgetTable | None:
        """Delete a budget record."""
        return await self.delete(budget_id, id_field="budget_id")
