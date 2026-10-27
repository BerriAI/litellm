"""
Budget table model.

Canonical definition for ``litellm_budgettable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class _MetadataTempBudgetIncrease(BaseModel):
    model_config = ConfigDict(frozen=True)

    increase: float = Field(allow_inf_nan=False)
    expiry: datetime


def active_temp_budget_increase(increase: float | None, expiry: datetime | None, now: datetime) -> float:
    if increase is None or expiry is None:
        return 0.0
    normalized_expiry: Final = expiry.replace(tzinfo=timezone.utc) if expiry.tzinfo is None else expiry
    return 0.0 if normalized_expiry <= now else increase


def metadata_temp_budget_increase(metadata: Mapping[str, object] | None, now: datetime) -> float:
    if metadata is None:
        return 0.0
    try:
        parsed: Final = _MetadataTempBudgetIncrease.model_validate(
            {
                "increase": metadata.get("temp_budget_increase"),
                "expiry": metadata.get("temp_budget_expiry"),
            }
        )
    except (TypeError, ValidationError):
        return 0.0
    return active_temp_budget_increase(increase=parsed.increase, expiry=parsed.expiry, now=now)


class LiteLLM_BudgetTable(LiteLLMPydanticObjectBase):
    """Represents user-controllable params for a LiteLLM_BudgetTable record.

    Budget-write paths use `model_fields.keys()` on this class as an allowlist
    for user input. Keep server-managed fields (e.g. `budget_reset_at`) on
    `LiteLLM_BudgetTableFull` so they aren't user-settable.
    """

    budget_id: str | None = None
    soft_budget: float | None = None
    max_budget: float | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    tpd_limit: int | None = None
    model_max_budget: dict | None = None
    budget_duration: str | None = None
    allowed_models: list[str] | None = None  # per-member model scope; empty = inherit team models
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None

    model_config = ConfigDict(protected_namespaces=())

    def active_temp_budget_increase(self, now: datetime) -> float:
        return active_temp_budget_increase(
            increase=self.temp_budget_increase,
            expiry=self.temp_budget_expiry,
            now=now,
        )

    def effective_max_budget(self, now: datetime) -> float | None:
        if self.max_budget is None:
            return None
        return self.max_budget + self.active_temp_budget_increase(now)


class LiteLLM_BudgetTableFull(LiteLLM_BudgetTable):
    """LiteLLM_BudgetTable + server-managed fields returned on API responses."""

    budget_reset_at: datetime | None = None
    created_at: datetime


class LiteLLM_TeamMemberTable(LiteLLM_BudgetTable):
    """
    Used to track spend of a user_id within a team_id
    """

    spend: float | None = None
    user_id: str | None = None
    team_id: str | None = None
    budget_id: str | None = None

    model_config = ConfigDict(protected_namespaces=())
