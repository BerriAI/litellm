"""Daily usage alert rules and incidents shared by the API and evaluator."""

from datetime import date, datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

UsageAlertScope: TypeAlias = Literal["deployment", "key", "user", "team", "model"]
UsageAlertMetric: TypeAlias = Literal["spend", "requests", "latency", "errors", "error_rate"]
UsageAlertDirection: TypeAlias = Literal["increase", "decrease"]
UsageAlertStatus: TypeAlias = Literal["open", "resolved", "dismissed"]
UsageAlertChannel: TypeAlias = Literal["email", "slack"]
UsageAlertEvaluationStatus: TypeAlias = Literal["pending", "complete", "insufficient_coverage", "insufficient_history"]
UsageAlertRuleKind: TypeAlias = Literal["all", "default", "custom"]


class AlertRuleInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=200, pattern=r"^[^\x00-\x1f\x7f]+$")
    scope_type: UsageAlertScope
    scope_id: str | None = Field(default=None, min_length=1, max_length=1024)
    metric: UsageAlertMetric
    direction: UsageAlertDirection
    multiplier: float = Field(default=3, gt=1, le=1000)
    baseline_days: int = Field(default=7, ge=1, le=30)
    min_history_days: int = Field(default=3, ge=1, le=30)
    min_requests: int = Field(default=100, ge=0)
    min_value: float = Field(default=0, ge=0, description="Minimum absolute change in the metric's units")
    enabled: bool = True
    notification_channels: tuple[UsageAlertChannel, ...] = Field(default=(), max_length=2)
    email_recipients: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        if self.min_history_days > self.baseline_days:
            raise ValueError("min_history_days must not exceed baseline_days")
        if self.scope_type == "deployment" and self.scope_id is not None:
            raise ValueError("Deployment alerts cover the whole gateway; omit scope_id")
        if (
            self.scope_type == "key"
            and self.scope_id is not None
            and self.scope_id.removeprefix("raw:").startswith("sk-")
        ):
            raise ValueError("Use the key's reporting ID, never the secret key")
        if len(set(self.notification_channels)) != len(self.notification_channels):
            raise ValueError("notification_channels must be unique")
        if "email" in self.notification_channels and not self.email_recipients:
            raise ValueError("Email notifications require at least one recipient")
        if any(
            "@" not in email or "\n" in email or "\r" in email or len(email) > 254 for email in self.email_recipients
        ):
            raise ValueError("Invalid email recipient")
        return self


class AlertRule(AlertRuleInput):
    overrides_defaults: bool = Field(default=False, exclude=True)
    id: str
    created_at: datetime
    updated_at: datetime
    created_by: str | None
    last_evaluated_day: date | None = None
    evaluation_status: UsageAlertEvaluationStatus = "pending"
    is_default: bool = False


class PerformanceSignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    id: str
    rule_id: str
    rule_name: str
    scope_type: UsageAlertScope
    scope_id: str | None
    metric: UsageAlertMetric
    direction: UsageAlertDirection
    status: UsageAlertStatus
    day: date
    baseline_start: date
    baseline_end: date
    baseline_sample_days: int | None = Field(default=None, ge=1, le=30)
    observed_value: float
    baseline_value: float
    threshold: float
    first_seen: datetime
    last_seen: datetime
    activity_path: str


class UsageAlertEvaluationCounts(BaseModel, frozen=True):
    pending: int = 0
    complete: int = 0
    insufficient_coverage: int = 0
    insufficient_history: int = 0


class UsageAlertDefaultSummary(BaseModel, frozen=True):
    total: int = 0
    enabled: int = 0
    evaluation_status_counts: UsageAlertEvaluationCounts = Field(default_factory=UsageAlertEvaluationCounts)


class AlertRuleList(BaseModel, frozen=True):
    data: tuple[AlertRule, ...]
    total: int
    limit: int
    offset: int
    default_summary: UsageAlertDefaultSummary = Field(default_factory=UsageAlertDefaultSummary)


class PerformanceSignalList(BaseModel, frozen=True):
    data: tuple[PerformanceSignal, ...]
    total: int
    limit: int
    offset: int


class AlertModelOption(BaseModel, frozen=True):
    scope_id: str
    label: str


class AlertModelList(BaseModel, frozen=True):
    data: tuple[AlertModelOption, ...]
    limit: int
    next_cursor: str | None
    start_date: date
    end_date: date


class UsageAlertObservation(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    entity_id: str | None
    day: date
    observed: float
    baseline: float
    baseline_sample_days: int | None = Field(default=None, ge=1, le=30)
    threshold: float
    breached: bool
