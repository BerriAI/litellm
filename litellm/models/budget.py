"""
Budget table model.

Canonical definition for ``litellm_budgettable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import AfterValidator, ConfigDict, Field, model_validator

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.llms.base import LiteLLMPydanticObjectBase

if TYPE_CHECKING:
    from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings


class BudgetLimit(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)

    budget_duration: Literal["1mo", "12mo"]
    max_budget: float = Field(ge=0, allow_inf_nan=False)
    reset_at: datetime | None = None


def unique_budget_limits(limits: tuple[BudgetLimit, ...]) -> tuple[BudgetLimit, ...]:
    if len({limit.budget_duration for limit in limits}) != len(limits):
        raise ValueError("Each budget duration may appear only once")
    return limits


BudgetLimits: TypeAlias = Annotated[tuple[BudgetLimit, ...], AfterValidator(unique_budget_limits)]


def initialize_budget_limits(
    limits: tuple[BudgetLimit, ...], settings: "BudgetResetSettings", now: datetime
) -> tuple[BudgetLimit, ...]:
    from litellm.proxy.common_utils.timezone_utils import compute_budget_reset_at

    return tuple(
        limit
        if limit.reset_at is not None
        else limit.model_copy(update={"reset_at": compute_budget_reset_at(limit.budget_duration, settings, now)})
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
