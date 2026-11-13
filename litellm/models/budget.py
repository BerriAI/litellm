"""
Budget table model.

Canonical definition for ``litellm_budgettable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final, Literal

from pydantic import ConfigDict, field_validator

from litellm.models.team import BudgetLimitEntry
from litellm.types.llms.base import LiteLLMPydanticObjectBase


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
    model_tpm_limit: Mapping[str, int] | None = None
    model_rpm_limit: Mapping[str, int] | None = None
    budget_duration: str | None = None
    allowed_models: list[str] | None = None  # per-member model scope; empty = inherit team models
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None
    temp_budget_duration: Literal["1mo", "12mo"] | None = None
    budget_limits: tuple[BudgetLimitEntry, ...] | None = None
    is_profile: bool | None = None
    cap_policy: Literal["pause_allow_increase", "pause_until_reset"] | None = None

    model_config = ConfigDict(protected_namespaces=())

    @field_validator("budget_limits")
    @classmethod
    def valid_budget_limits(cls, value: tuple[BudgetLimitEntry, ...] | None) -> tuple[BudgetLimitEntry, ...] | None:
        return validate_person_budget_limits(value)

    def active_temp_budget_increase(self, now: datetime) -> float:
        if self.temp_budget_increase is None or self.temp_budget_expiry is None:
            return 0.0
        expiry: Final = (
            self.temp_budget_expiry.replace(tzinfo=timezone.utc)
            if self.temp_budget_expiry.tzinfo is None
            else self.temp_budget_expiry
        )
        return 0.0 if expiry <= now else self.temp_budget_increase

    def effective_max_budget(self, now: datetime) -> float | None:
        if self.max_budget is None:
            return None
        return self.max_budget + (self.active_temp_budget_increase(now) if self.temp_budget_duration is None else 0.0)


def validate_person_budget_limits(
    limits: tuple[BudgetLimitEntry, ...] | None,
) -> tuple[BudgetLimitEntry, ...] | None:
    import math

    if limits is None:
        return None
    durations: Final = tuple(limit.budget_duration for limit in limits)
    if len(set(durations)) != len(durations):
        raise ValueError("Each budget duration may appear only once")
    for limit in limits:
        if limit.budget_duration not in ("1mo", "12mo"):
            raise ValueError("Personal budget limits support calendar month (1mo) and calendar year (12mo)")
        if not math.isfinite(limit.max_budget) or limit.max_budget < 0:
            raise ValueError("Budget limits must be finite, non-negative amounts")
        if limit.reset_at is not None:
            raise ValueError("Personal budget windows reset automatically at calendar boundaries")
    return limits


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


CUSTOMER_ONLY_BUDGET_FIELDS: Final = frozenset({"model_tpm_limit", "model_rpm_limit"})
