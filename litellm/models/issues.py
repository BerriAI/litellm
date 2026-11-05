from datetime import date, datetime
from typing import Literal, Protocol

from litellm.types.proxy.issues import (
    Issue,
    IssueCohort,
    IssueDeliveryBinding,
    IssueDeliveryInput,
    IssueFamily,
    IssueFamilyList,
    IssueFamilySetting,
    IssueFamilyUpdate,
    IssueImpact,
    IssueInput,
    IssueKind,
    IssueList,
    IssuePendingDelivery,
    IssueProjectionSignal,
    IssueScope,
    IssueSignal,
    IssueSignalInput,
    IssueSignalList,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailyActivityKeyPageResponse,
    DailyActivityUserPageResponse,
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.usage_alert_defaults import PerformanceIssueFamily
from litellm.types.proxy.usage_alerts import UsageAlertMetric, UsageAlertStatus


class IssueStore(Protocol):
    async def get_issue(self, issue_id: str) -> Issue | None: ...

    async def list_issues(
        self,
        kind: IssueKind | None = None,
        family: IssueFamily | None = None,
        status: UsageAlertStatus | None = None,
        scope_type: IssueScope | None = None,
        scope_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> IssueList: ...

    async def set_status(
        self,
        issue_id: str,
        status: UsageAlertStatus,
        expected_status: UsageAlertStatus,
        actor: str | None = None,
    ) -> Issue | None: ...

    async def upsert_issue(
        self,
        data: IssueInput,
        reopen: bool = False,
        status: UsageAlertStatus | None = None,
    ) -> Issue: ...

    async def upsert_issues(
        self,
        inputs: tuple[IssueInput, ...],
        reopen: bool = False,
        status: UsageAlertStatus | None = None,
    ) -> tuple[Issue, ...]: ...

    async def find_active_issue(self, cohort_key: str) -> Issue | None: ...

    async def find_active_issues(self, cohort_keys: tuple[str, ...]) -> tuple[Issue, ...]: ...

    async def add_signals(self, issue_id: str, signals: tuple[IssueSignalInput, ...]) -> None: ...

    async def add_signal_batch(self, signals: tuple[IssueSignal, ...]) -> None: ...

    async def list_signals(
        self, issue_id: str, limit: int = 50, offset: int = 0, visible_scope: IssueCohort | None = None
    ) -> IssueSignalList: ...

    async def active_signals(
        self,
        day: date,
        limit: int = 500,
        cursor: str | None = None,
        scope_type: IssueScope | None = None,
        metric: UsageAlertMetric | None = None,
        scope_ids: tuple[str, ...] | None = None,
    ) -> tuple[IssueProjectionSignal, ...]: ...

    async def recover_issues(self, day: date) -> int: ...

    async def active_provider_issues(self, limit: int = 500, cursor: str | None = None) -> tuple[Issue, ...]: ...

    async def recover_issue_ids(self, ids: tuple[str, ...], observed_at: datetime) -> int: ...

    async def projection_completed(self, day: date, version: str) -> bool: ...

    async def complete_projection(self, day: date, version: str) -> None: ...

    async def finish_deprecation_projection(self, observed_at: datetime) -> int: ...

    async def family_settings(self) -> IssueFamilyList: ...

    async def update_family(
        self, family: PerformanceIssueFamily, data: IssueFamilyUpdate
    ) -> IssueFamilySetting | None: ...

    async def queue_delivery(
        self,
        data: IssueDeliveryInput,
        rule_id: str,
        rule_revision: str,
        source_id: str,
    ) -> None: ...

    async def queue_deliveries(self, bindings: tuple[IssueDeliveryBinding, ...]) -> None: ...

    async def pending_deliveries(self, limit: int = 100) -> tuple[IssuePendingDelivery, ...]: ...

    async def pending_delivery(self, delivery_id: str) -> IssuePendingDelivery | None: ...

    async def mark_delivery_sent(self, delivery_id: str) -> None: ...

    async def defer_delivery(
        self, delivery_id: str, error: Literal["delivery_failed", "transport_not_configured"]
    ) -> None: ...

    async def prune(self, retention_days: int = 90, limit: int = 500) -> None: ...


class IssueActivityStore(Protocol):
    async def daily_activity(self, cohort: IssueCohort, start: date, end: date) -> SpendAnalyticsPaginatedResponse: ...

    async def keys(
        self, cohort: IssueCohort, start: date, end: date, limit: int, offset: int
    ) -> DailyActivityKeyPageResponse: ...

    async def users(
        self, cohort: IssueCohort, start: date, end: date, limit: int, offset: int
    ) -> DailyActivityUserPageResponse: ...

    async def impact(self, cohort: IssueCohort, start: date, end: date) -> IssueImpact: ...
