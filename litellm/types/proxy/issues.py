from datetime import date, datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from litellm.types.proxy.management_endpoints.common_daily_activity import (
    BoundedDailyActivityResponse,
)
from litellm.types.proxy.model_deprecation import DeprecationAlertState, ModelDeprecationInfo
from litellm.types.proxy.usage_alert_defaults import PerformanceIssueFamily
from litellm.types.proxy.usage_alerts import (
    AlertRule,
    PerformanceSignal,
    UsageAlertChannel,
    UsageAlertDirection,
    UsageAlertEvaluationCounts,
    UsageAlertMetric,
    UsageAlertMetricPoint,
    UsageAlertStatus,
)

AlertActivityResponse: TypeAlias = BoundedDailyActivityResponse

AlertKind: TypeAlias = Literal["performance", "model_deprecation"]
AlertFamily: TypeAlias = Literal["retirement", "latency", "reliability", "spend", "traffic"]
AlertScope: TypeAlias = Literal["deployment", "key", "user", "team", "model", "provider"]
AlertGroupingReason: TypeAlias = Literal["same_resource", "dominant_key", "provider_cohort", "model_deprecation"]
AlertSignalGroupingReason: TypeAlias = Literal[
    "same_resource", "same_key_traffic", "dominant_key", "provider_cohort", "volume_explains_spend", "model_deprecation"
]


class PerformanceMonitoringUpdate(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool


class PerformanceMonitoring(BaseModel, frozen=True):
    enabled: bool = False
    incidents_used: int = Field(default=0, ge=0, le=5, serialization_alias="alerts_used")
    incident_limit: int | None = Field(default=5, serialization_alias="alert_limit")

    @property
    def can_detect(self) -> bool:
        return self.enabled and (self.incident_limit is None or self.incidents_used < self.incident_limit)


class AlertModelPair(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=1024)


class AlertCohort(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scope_type: AlertScope
    scope_id: str | None = Field(default=None, min_length=1, max_length=1024)
    model_pairs: tuple[AlertModelPair, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if (self.scope_type == "deployment") != (self.scope_id is None):
            raise ValueError("Only a deployment cohort omits scope_id")
        if self.scope_type == "key" and self.scope_id and self.scope_id.removeprefix("raw:").startswith("sk-"):
            raise ValueError("Use a reporting key ID, never a secret key")
        if len(frozenset((pair.provider, pair.model) for pair in self.model_pairs)) != len(self.model_pairs):
            raise ValueError("model_pairs must be unique")
        return self


class AlertComparison(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    metric: UsageAlertMetric
    direction: UsageAlertDirection
    observed_value: float = Field(ge=0)
    baseline_value: float = Field(ge=0)
    threshold: float | None = None
    baseline_start: date
    baseline_end: date
    baseline_sample_days: int | None = Field(default=None, ge=1, le=30)
    history: tuple[UsageAlertMetricPoint, ...] = Field(default=(), max_length=31)

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.baseline_start > self.baseline_end:
            raise ValueError("baseline_start must not follow baseline_end")
        if self.metric == "error_rate" and (self.observed_value > 100 or self.baseline_value > 100):
            raise ValueError("error_rate uses percentages from 0 to 100")
        return self


class AlertImpact(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    keys: int | None = Field(default=None, ge=0)
    users: int | None = Field(default=None, ge=0)
    teams: int | None = Field(default=None, ge=0)
    models: int | None = Field(default=None, ge=0)
    requests: int | None = Field(default=None, ge=0)
    measured_keys: int | None = Field(default=None, ge=0)
    measured_models: int | None = Field(default=None, ge=0)


class AlertDeprecation(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelDeprecationInfo
    usage_model: str = Field(min_length=1, max_length=1024)
    requests: int = Field(ge=0)
    last_used_date: date
    last_day_requests: int = Field(ge=0)
    state: DeprecationAlertState
    daily_requests: tuple[UsageAlertMetricPoint, ...] = Field(default=(), max_length=30)


class AlertInput(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    cohort_key: str = Field(min_length=1, max_length=200)
    kind: AlertKind
    family: AlertFamily
    title: str = Field(min_length=1, max_length=2048)
    cohort: AlertCohort
    observed_on: date
    observed_at: datetime
    range_start: date
    range_end: date
    grouping_reason: AlertGroupingReason
    explanation: str = Field(default="", max_length=2000)
    comparisons: tuple[AlertComparison, ...] = Field(default=(), max_length=10)
    impact: AlertImpact = Field(default_factory=AlertImpact)
    deprecation: AlertDeprecation | None = None

    @model_validator(mode="after")
    def validate_issue(self) -> Self:
        if self.range_start > self.range_end:
            raise ValueError("range_start must not follow range_end")
        if (self.kind == "model_deprecation") != (self.deprecation is not None):
            raise ValueError("Model deprecation alerts require deprecation details")
        if (self.kind == "model_deprecation") != (self.family == "retirement"):
            raise ValueError("Only model deprecation alerts use the retirement family")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must include a timezone")
        return self


class Alert(AlertInput, frozen=True):
    status: UsageAlertStatus
    first_seen: datetime
    last_seen: datetime
    updated_at: datetime
    scope_label: str | None = None
    active: bool = True
    can_update_status: bool | None = Field(
        default=None, description="Current reader's status permission, evaluated by the alert detail endpoint"
    )
    can_view_activity: bool | None = Field(
        default=None, description="Current reader's permission to inspect Activity, keys, users and impact"
    )


class AlertList(BaseModel, frozen=True):
    data: tuple[Alert, ...]
    total: int
    limit: int = Field(description="Effective page size, reduced for large records. Continue at offset + limit.")
    offset: int


class AlertUpdate(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: UsageAlertStatus
    expected_status: UsageAlertStatus


class AlertSignalInput(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["performance", "model_deprecation"]
    source_id: str = Field(min_length=1, max_length=200)
    performance: PerformanceSignal | None = None
    deprecation: AlertDeprecation | None = None
    grouping_reason: AlertSignalGroupingReason = "same_resource"
    explanation: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_signal(self) -> Self:
        if self.source == "performance":
            if self.performance is None or self.deprecation is not None or self.performance.id != self.source_id:
                raise ValueError("Performance signals require their matching evidence")
        elif self.deprecation is None or self.performance is not None:
            raise ValueError("Deprecation signals require deprecation details")
        return self


class AlertSignal(AlertSignalInput, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    issue_id: str = Field(alias="alert_id")


class AlertSignalList(BaseModel, frozen=True):
    data: tuple[AlertSignal, ...]
    total: int
    limit: int
    offset: int


class AlertProjectionSignal(BaseModel, frozen=True):
    alert: PerformanceSignal
    rule: AlertRule
    evaluation_day: date


class AlertDeliveryInput(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    issue_id: str = Field(min_length=1, max_length=200)
    channel: UsageAlertChannel
    email_recipients: tuple[str, ...] = Field(default=(), max_length=20)
    slack_destination: str | None = Field(default=None, pattern="^[a-f0-9]{64}$")
    parent_delivery_id: str | None = Field(default=None, min_length=1, max_length=200)


class AlertDeliveryBinding(BaseModel, frozen=True):
    delivery: AlertDeliveryInput
    rule_id: str
    rule_revision: str
    source_id: str


class AlertDelivery(AlertDeliveryInput, frozen=True):
    attempts: int = 0
    next_attempt: datetime | None = None
    last_error: Literal["delivery_failed", "transport_not_configured"] | None = None


class AlertPendingDelivery(BaseModel, frozen=True):
    delivery: AlertDelivery
    issue: Alert
    rule: AlertRule
    alert: PerformanceSignal


class AlertFamilyPolicy(BaseModel, frozen=True):
    direction: UsageAlertDirection
    enabled: bool | None
    multiplier: float | None
    min_value: float | None
    min_requests: int | None
    baseline_days: int | None
    min_history_days: int | None
    mixed: bool
    total_rules: int
    enabled_rules: int


class AlertFamilySetting(BaseModel, frozen=True):
    family: PerformanceIssueFamily
    enabled: bool | None
    total_rules: int
    enabled_rules: int
    policies: tuple[AlertFamilyPolicy, ...]
    evaluation_status_counts: UsageAlertEvaluationCounts = Field(default_factory=UsageAlertEvaluationCounts)


class AlertFamilyList(BaseModel, frozen=True):
    data: tuple[AlertFamilySetting, ...]
    projection_pending: bool = False


class AlertFamilyPolicyUpdate(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: UsageAlertDirection
    enabled: bool | None = None
    multiplier: float | None = Field(default=None, gt=1, le=1000)
    min_value: float | None = Field(default=None, ge=0)
    min_requests: int | None = Field(default=None, ge=0)
    baseline_days: int | None = Field(default=None, ge=1, le=30)
    min_history_days: int | None = Field(default=None, ge=1, le=30)


class AlertFamilyUpdate(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool | None = None
    policies: tuple[AlertFamilyPolicyUpdate, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def validate_update(self) -> Self:
        if len(frozenset(policy.direction for policy in self.policies)) != len(self.policies):
            raise ValueError("Each direction can occur once")
        if self.enabled is None and not self.policies:
            raise ValueError("Specify enabled or a policy update")
        return self
