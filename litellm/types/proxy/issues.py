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
    UsageAlert,
    UsageAlertChannel,
    UsageAlertDirection,
    UsageAlertEvaluationCounts,
    UsageAlertMetric,
    UsageAlertMetricPoint,
    UsageAlertRule,
    UsageAlertStatus,
)

IssueActivityResponse: TypeAlias = BoundedDailyActivityResponse

IssueKind: TypeAlias = Literal["performance", "model_deprecation"]
IssueFamily: TypeAlias = Literal["retirement", "latency", "reliability", "spend", "traffic"]
IssueScope: TypeAlias = Literal["deployment", "key", "user", "team", "model", "provider"]
IssueGroupingReason: TypeAlias = Literal["same_resource", "dominant_key", "provider_cohort", "model_deprecation"]
IssueSignalGroupingReason: TypeAlias = Literal[
    "same_resource", "same_key_traffic", "dominant_key", "provider_cohort", "volume_explains_spend", "model_deprecation"
]


class PerformanceMonitoringUpdate(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool


class PerformanceMonitoring(BaseModel, frozen=True):
    enabled: bool = False
    incidents_used: int = Field(default=0, ge=0, le=5)
    incident_limit: int | None = 5

    @property
    def can_detect(self) -> bool:
        return self.enabled and (self.incident_limit is None or self.incidents_used < self.incident_limit)


class IssueModelPair(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=1024)


class IssueCohort(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scope_type: IssueScope
    scope_id: str | None = Field(default=None, min_length=1, max_length=1024)
    model_pairs: tuple[IssueModelPair, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if (self.scope_type == "deployment") != (self.scope_id is None):
            raise ValueError("Only a deployment cohort omits scope_id")
        if self.scope_type == "key" and self.scope_id and self.scope_id.removeprefix("raw:").startswith("sk-"):
            raise ValueError("Use a reporting key ID, never a secret key")
        if len(frozenset((pair.provider, pair.model) for pair in self.model_pairs)) != len(self.model_pairs):
            raise ValueError("model_pairs must be unique")
        return self


class IssueComparison(BaseModel, frozen=True):
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


class IssueImpact(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    keys: int | None = Field(default=None, ge=0)
    users: int | None = Field(default=None, ge=0)
    teams: int | None = Field(default=None, ge=0)
    models: int | None = Field(default=None, ge=0)
    requests: int | None = Field(default=None, ge=0)
    measured_keys: int | None = Field(default=None, ge=0)
    measured_models: int | None = Field(default=None, ge=0)


class IssueDeprecation(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelDeprecationInfo
    usage_model: str = Field(min_length=1, max_length=1024)
    requests: int = Field(ge=0)
    last_used_date: date
    last_day_requests: int = Field(ge=0)
    state: DeprecationAlertState
    daily_requests: tuple[UsageAlertMetricPoint, ...] = Field(default=(), max_length=30)


class IssueInput(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    cohort_key: str = Field(min_length=1, max_length=200)
    kind: IssueKind
    family: IssueFamily
    title: str = Field(min_length=1, max_length=2048)
    cohort: IssueCohort
    observed_on: date
    observed_at: datetime
    range_start: date
    range_end: date
    grouping_reason: IssueGroupingReason
    explanation: str = Field(default="", max_length=2000)
    comparisons: tuple[IssueComparison, ...] = Field(default=(), max_length=10)
    impact: IssueImpact = Field(default_factory=IssueImpact)
    deprecation: IssueDeprecation | None = None

    @model_validator(mode="after")
    def validate_issue(self) -> Self:
        if self.range_start > self.range_end:
            raise ValueError("range_start must not follow range_end")
        if (self.kind == "model_deprecation") != (self.deprecation is not None):
            raise ValueError("Model deprecation issues require deprecation details")
        if (self.kind == "model_deprecation") != (self.family == "retirement"):
            raise ValueError("Only model deprecation issues use the retirement family")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must include a timezone")
        return self


class Issue(IssueInput, frozen=True):
    status: UsageAlertStatus
    first_seen: datetime
    last_seen: datetime
    updated_at: datetime
    scope_label: str | None = None
    active: bool = True
    can_update_status: bool | None = Field(
        default=None, description="Current reader's status permission, evaluated by the issue-detail endpoint"
    )
    can_view_activity: bool | None = Field(
        default=None, description="Current reader's permission to inspect Activity, keys, users and impact"
    )


class IssueList(BaseModel, frozen=True):
    data: tuple[Issue, ...]
    total: int
    limit: int = Field(description="Effective page size, reduced for large records. Continue at offset + limit.")
    offset: int


class IssueUpdate(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: UsageAlertStatus
    expected_status: UsageAlertStatus


class IssueSignalInput(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["usage_alert", "model_deprecation"]
    source_id: str = Field(min_length=1, max_length=200)
    usage_alert: UsageAlert | None = None
    deprecation: IssueDeprecation | None = None
    grouping_reason: IssueSignalGroupingReason = "same_resource"
    explanation: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_signal(self) -> Self:
        if self.source == "usage_alert":
            if self.usage_alert is None or self.deprecation is not None or self.usage_alert.id != self.source_id:
                raise ValueError("Usage signals require their matching source alert")
        elif self.deprecation is None or self.usage_alert is not None:
            raise ValueError("Deprecation signals require deprecation details")
        return self


class IssueSignal(IssueSignalInput, frozen=True):
    issue_id: str


class IssueSignalList(BaseModel, frozen=True):
    data: tuple[IssueSignal, ...]
    total: int
    limit: int
    offset: int


class IssueProjectionSignal(BaseModel, frozen=True):
    alert: UsageAlert
    rule: UsageAlertRule
    evaluation_day: date


class IssueDeliveryInput(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    issue_id: str = Field(min_length=1, max_length=200)
    channel: UsageAlertChannel
    email_recipients: tuple[str, ...] = Field(default=(), max_length=20)


class IssueDeliveryBinding(BaseModel, frozen=True):
    delivery: IssueDeliveryInput
    rule_id: str
    rule_revision: str
    source_id: str


class IssueDelivery(IssueDeliveryInput, frozen=True):
    attempts: int = 0
    next_attempt: datetime | None = None
    last_error: Literal["delivery_failed", "transport_not_configured"] | None = None


class IssuePendingDelivery(BaseModel, frozen=True):
    delivery: IssueDelivery
    issue: Issue
    rule: UsageAlertRule
    alert: UsageAlert


class IssueFamilyPolicy(BaseModel, frozen=True):
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


class IssueFamilySetting(BaseModel, frozen=True):
    family: PerformanceIssueFamily
    enabled: bool | None
    total_rules: int
    enabled_rules: int
    policies: tuple[IssueFamilyPolicy, ...]
    evaluation_status_counts: UsageAlertEvaluationCounts = Field(default_factory=UsageAlertEvaluationCounts)


class IssueFamilyList(BaseModel, frozen=True):
    data: tuple[IssueFamilySetting, ...]
    projection_pending: bool = False


class IssueFamilyPolicyUpdate(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: UsageAlertDirection
    enabled: bool | None = None
    multiplier: float | None = Field(default=None, gt=1, le=1000)
    min_value: float | None = Field(default=None, ge=0)
    min_requests: int | None = Field(default=None, ge=0)
    baseline_days: int | None = Field(default=None, ge=1, le=30)
    min_history_days: int | None = Field(default=None, ge=1, le=30)


class IssueFamilyUpdate(BaseModel, frozen=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool | None = None
    policies: tuple[IssueFamilyPolicyUpdate, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def validate_update(self) -> Self:
        if len(frozenset(policy.direction for policy in self.policies)) != len(self.policies):
            raise ValueError("Each direction can occur once")
        if self.enabled is None and not self.policies:
            raise ValueError("Specify enabled or a policy update")
        return self
