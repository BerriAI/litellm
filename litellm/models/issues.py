from datetime import date, datetime
from typing import Literal, Protocol

from litellm.models.team import LiteLLM_TeamTable
from litellm.types.proxy.issues import (
    Alert,
    AlertActivityResponse,
    AlertCohort,
    AlertDeliveryBinding,
    AlertFamily,
    AlertFamilyList,
    AlertFamilySetting,
    AlertFamilyUpdate,
    AlertImpact,
    AlertInput,
    AlertKind,
    AlertList,
    AlertPendingDelivery,
    AlertProjectionSignal,
    AlertScope,
    AlertSignal,
    AlertSignalList,
    PerformanceMonitoring,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailyActivityKeyPageResponse,
    DailyActivityUserPageResponse,
)
from litellm.types.proxy.usage_alert_defaults import PerformanceIssueFamily
from litellm.types.proxy.usage_alerts import UsageAlertMetric, UsageAlertStatus


class IssueStore(Protocol):
    async def performance_monitoring(self, enterprise: bool = False) -> PerformanceMonitoring: ...

    async def set_performance_monitoring(self, enabled: bool, enterprise: bool = False) -> PerformanceMonitoring: ...

    async def publish_performance_issues(
        self, inputs: tuple[AlertInput, ...], enterprise: bool
    ) -> tuple[Alert, ...]: ...

    async def get_issue(self, issue_id: str) -> Alert | None: ...

    async def team_activity_access(self, team_id: str, user_id: str) -> LiteLLM_TeamTable | None: ...

    async def list_issues(
        self,
        kind: AlertKind | None = None,
        family: AlertFamily | None = None,
        status: UsageAlertStatus | None = None,
        scope_type: AlertScope | None = None,
        scope_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> AlertList: ...

    async def set_status(
        self,
        issue_id: str,
        status: UsageAlertStatus,
        expected_status: UsageAlertStatus,
        actor: str | None = None,
        authorized_scopes: tuple[AlertCohort, ...] | None = None,
    ) -> Alert | None: ...

    async def status_scopes(self, issue_id: str) -> tuple[AlertCohort, ...]: ...

    async def upsert_issues(
        self,
        inputs: tuple[AlertInput, ...],
        reopen: bool = False,
        status: UsageAlertStatus | None = None,
    ) -> tuple[Alert, ...]: ...

    async def find_active_issues(self, cohort_keys: tuple[str, ...]) -> tuple[Alert, ...]: ...

    async def add_signal_batch(self, signals: tuple[AlertSignal, ...]) -> None: ...

    async def list_signals(
        self, issue_id: str, limit: int = 50, offset: int = 0, visible_scope: AlertCohort | None = None
    ) -> AlertSignalList: ...

    async def active_signals(
        self,
        day: date,
        limit: int = 500,
        cursor: str | None = None,
        scope_type: AlertScope | None = None,
        metric: UsageAlertMetric | None = None,
        scope_ids: tuple[str, ...] | None = None,
        completed_only: bool = False,
    ) -> tuple[AlertProjectionSignal, ...]: ...

    async def active_resource_issue_ids(self, cursor: str | None = None) -> tuple[str, ...]: ...

    async def recover_issues(self, day: date, issue_ids: tuple[str, ...]) -> int: ...

    async def active_provider_issues(self, limit: int = 500, cursor: str | None = None) -> tuple[Alert, ...]: ...

    async def recover_issue_ids(self, ids: tuple[str, ...], observed_at: datetime) -> int: ...

    async def projection_completed(self, day: date, version: str) -> bool: ...

    async def complete_projection(self, day: date, version: str) -> None: ...

    async def finish_deprecation_projection(self, observed_at: datetime) -> int: ...

    async def family_settings(self) -> AlertFamilyList: ...

    async def update_family(
        self, family: PerformanceIssueFamily, data: AlertFamilyUpdate
    ) -> AlertFamilySetting | None: ...

    async def queue_deliveries(self, bindings: tuple[AlertDeliveryBinding, ...]) -> None: ...

    async def pending_deliveries(self, limit: int = 100) -> tuple[AlertPendingDelivery, ...]: ...

    async def pending_delivery(self, delivery_id: str) -> AlertPendingDelivery | None: ...

    async def reserve_delivery(self, delivery_id: str) -> bool: ...

    async def mark_delivery_sent(self, delivery_id: str) -> None: ...

    async def defer_delivery(
        self, delivery_id: str, error: Literal["delivery_failed", "transport_not_configured"]
    ) -> None: ...

    async def prune(self, retention_days: int = 90, limit: int = 500) -> bool: ...


class IssueActivityStore(Protocol):
    async def daily_activity(
        self, cohort: AlertCohort, start: date, end: date, model_limit: int = 20
    ) -> AlertActivityResponse: ...

    async def keys(
        self, cohort: AlertCohort, start: date, end: date, limit: int, offset: int
    ) -> DailyActivityKeyPageResponse: ...

    async def users(
        self, cohort: AlertCohort, start: date, end: date, limit: int, offset: int
    ) -> DailyActivityUserPageResponse: ...

    async def impact(self, cohort: AlertCohort, start: date, end: date) -> AlertImpact: ...
