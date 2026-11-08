from datetime import date
from typing import Protocol

from litellm.types.proxy.usage_alerts import (
    AlertModelList,
    AlertRule,
    AlertRuleInput,
    AlertRuleList,
    PerformanceSignal,
    PerformanceSignalList,
    UsageAlertMetric,
    UsageAlertRuleKind,
    UsageAlertScope,
    UsageAlertStatus,
)


class UsageAlertStore(Protocol):
    async def get_rule(self, rule_id: str) -> AlertRule | None: ...

    async def save_rule(
        self,
        data: AlertRuleInput,
        created_by: str | None,
        existing: AlertRule | None = None,
        *,
        overrides_defaults: bool = False,
    ) -> AlertRule | None: ...

    async def delete_rule(self, rule_id: str, *, existing: AlertRule | None = None) -> bool: ...

    async def list_rules(
        self,
        scope_type: UsageAlertScope | None = None,
        scope_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        kind: UsageAlertRuleKind = "all",
    ) -> AlertRuleList: ...

    async def get_alert(self, alert_id: str) -> PerformanceSignal | None: ...

    async def list_alerts(
        self,
        scope_type: UsageAlertScope | None = None,
        scope_id: str | None = None,
        metric: UsageAlertMetric | None = None,
        status: UsageAlertStatus | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PerformanceSignalList: ...

    async def set_status(self, alert_id: str, status: UsageAlertStatus) -> PerformanceSignal | None: ...

    async def list_reported_models(self, search: str | None, cursor: str | None, limit: int) -> AlertModelList: ...

    async def key_ownership(self, key_id: str) -> tuple[str | None, str | None] | None: ...

    async def team_role(self, team_id: str, user_id: str) -> str | None: ...
