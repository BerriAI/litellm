from datetime import datetime, timezone
from typing import Final, Literal, TypeAlias

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

PersonBudgetDuration: TypeAlias = Literal["1mo", "12mo"]
PersonBudgetCapPolicy: TypeAlias = Literal["pause_allow_increase", "pause_until_reset"]
PersonBudgetSubject: TypeAlias = Literal["user", "customer_member"]


class PersonBudgetLimit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_duration: PersonBudgetDuration
    max_budget: float = Field(ge=0, allow_inf_nan=False)


class PersonBudgetPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    budget_limits: tuple[PersonBudgetLimit, ...] | None = None
    cap_policy: PersonBudgetCapPolicy | None = None
    temp_budget_increase: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    temp_budget_expiry: AwareDatetime | None = None
    temp_budget_duration: PersonBudgetDuration | None = None

    @field_validator("budget_limits")
    @classmethod
    def unique_durations(cls, value: tuple[PersonBudgetLimit, ...] | None) -> tuple[PersonBudgetLimit, ...] | None:
        if value is not None and len({limit.budget_duration for limit in value}) != len(value):
            raise ValueError("Each personal budget duration may appear only once")
        return value

    def increase_for(self, duration: PersonBudgetDuration, now: datetime) -> float:
        if self.temp_budget_duration != duration or self.temp_budget_expiry is None:
            return 0.0
        if self.temp_budget_expiry <= now:
            return 0.0
        return self.temp_budget_increase or 0.0


class PersonBudgetDefinition(PersonBudgetPolicy):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    budget_id: str
    is_profile: bool = False
    updated_at: datetime | None = None


class PersonBudgetAssignmentRecord(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    budget_id: str | None = None
    budget_override_id: str | None = None
    budget_override_expires_at: datetime | None = None
    budget: PersonBudgetDefinition | None = None
    budget_override: PersonBudgetDefinition | None = None
    updated_at: datetime | None = None

    def override_is_active(self, now: datetime) -> bool:
        expiry: Final = self.budget_override_expires_at
        utc_expiry: Final = (
            expiry.replace(tzinfo=timezone.utc) if expiry is not None and expiry.tzinfo is None else expiry
        )
        return utc_expiry is None or utc_expiry > now

    def effective_budget(self, now: datetime) -> PersonBudgetPolicy:
        inherited: Final = self.budget or PersonBudgetPolicy()
        override: Final = self.budget_override if self.override_is_active(now) else None
        selected: Final = override or inherited
        limits: Final = {
            **{limit.budget_duration: limit for limit in inherited.budget_limits or ()},
            **(
                {limit.budget_duration: limit for limit in override.budget_limits or ()} if override is not None else {}
            ),
        }
        return PersonBudgetPolicy(
            budget_limits=tuple(limits.values()) or None,
            cap_policy=selected.cap_policy if selected.cap_policy is not None else inherited.cap_policy,
            temp_budget_increase=selected.temp_budget_increase,
            temp_budget_expiry=selected.temp_budget_expiry,
            temp_budget_duration=selected.temp_budget_duration,
        )
