"""
Budget table model.

Canonical definition for ``litellm_budgettable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from math import isfinite
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

from pydantic import AfterValidator, ConfigDict, Field, model_validator

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.llms.base import LiteLLMPydanticObjectBase

if TYPE_CHECKING:
    from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings


BudgetRolloverCap: TypeAlias = Annotated[
    float,
    Field(gt=0, allow_inf_nan=False, description="Maximum allowance including unused budget carried forward, in USD."),
]


def active_rollover_credit(
    base: float | None, ceiling: float | None, duration: str | None, credit: float | None
) -> float | None:
    if base is None or ceiling is None or not duration:
        return None
    if not 0 < base < ceiling or not isfinite(ceiling):
        return None
    return min(max(0.0, credit or 0.0), ceiling - base)


def _is_none(value: object) -> bool:
    return value is None


class BudgetLimit(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    budget_duration: str = Field(description="Positive duration such as '1h', '7d', '1mo' or '12mo'.")
    max_budget: float
    reset_at: datetime | None = None
    window_start: datetime | None = Field(
        default=None,
        description="Server-managed start of the current spend window.",
        exclude_if=_is_none,
    )


def validate_budget_limits(limits: Sequence[BudgetLimit]) -> Sequence[BudgetLimit]:
    from litellm.proxy.common_utils.timezone_utils import budget_duration_error

    if len({limit.budget_duration for limit in limits}) != len(limits):
        raise ValueError("Each budget duration may appear only once")
    for limit in limits:
        if (error := budget_duration_error(limit.budget_duration)) is not None:
            raise ValueError(error)
        if not 0 <= limit.max_budget < float("inf"):
            raise ValueError("Window max_budget must be finite and non-negative")
    return limits


BudgetLimits: TypeAlias = Annotated[tuple[BudgetLimit, ...], AfterValidator(validate_budget_limits)]
BudgetLimitList: TypeAlias = Annotated[list[BudgetLimit], AfterValidator(validate_budget_limits)]


def carry_budget_limit_resets(
    limits: tuple[BudgetLimit, ...], previous: tuple[BudgetLimit, ...]
) -> tuple[BudgetLimit, ...]:
    stored: Final = {limit.budget_duration: limit for limit in previous}
    return tuple(_carried_schedule(limit, stored.get(limit.budget_duration)) for limit in limits)


def _carried_schedule(limit: BudgetLimit, stored: BudgetLimit | None) -> BudgetLimit:
    return limit.model_copy(
        update={
            "reset_at": stored.reset_at if stored is not None else limit.reset_at,
            "window_start": stored.window_start if stored is not None else None,
        }
    )


def initialize_budget_limits(
    limits: tuple[BudgetLimit, ...], settings: "BudgetResetSettings", now: datetime
) -> tuple[BudgetLimit, ...]:
    from litellm.proxy.common_utils.timezone_utils import get_budget_window_reset_fields

    return tuple(
        limit
        if limit.reset_at is not None
        else BudgetLimit.model_validate(
            {**limit.model_dump(), **get_budget_window_reset_fields(limit.budget_duration, settings=settings, now=now)}
        )
        for limit in limits
    )


class LiteLLM_BudgetTable(LiteLLMPydanticObjectBase):
    """Represents user-controllable params for a LiteLLM_BudgetTable record.

    Budget-write paths use `model_fields.keys()` on this class as an allowlist
    for user input. Keep server-managed fields (e.g. `budget_reset_at`) on
    `LiteLLM_BudgetTableFull` so they aren't user-settable.
    """

    budget_id: str | None = None
    soft_budget: float | None = None
    max_budget: float | None = None
    rollover_max_budget: float | None = Field(default=None, exclude_if=_is_none)
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

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def reject_concurrent_windows(cls, value: object) -> object:
        values: Final = as_str_mapping(value)
        if values is not None and values.get("budget_limits"):
            raise ValueError("Concurrent budget windows are supported only by customer hierarchies")
        return value

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


CUSTOMER_ONLY_BUDGET_FIELDS: Final = frozenset({"model_tpm_limit", "model_rpm_limit"})
