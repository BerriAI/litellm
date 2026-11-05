from datetime import date
from typing import Protocol

from litellm.types.proxy.usage_alerts import (
    UsageAlert,
    UsageAlertGroupList,
    UsageAlertGroupUpdate,
    UsageAlertGroupUpdateResult,
    UsageAlertList,
    UsageAlertMetric,
    UsageAlertModelList,
    UsageAlertRule,
    UsageAlertRuleInput,
    UsageAlertRuleKind,
    UsageAlertRuleList,
    UsageAlertScope,
    UsageAlertStatus,
)


class UsageAlertStore(Protocol):
    async def get_rule(self, rule_id: str) -> UsageAlertRule | None: ...

    async def save_rule(
        self, data: UsageAlertRuleInput, created_by: str | None, existing: UsageAlertRule | None = None
    ) -> UsageAlertRule | None: ...

    async def delete_rule(self, rule_id: str, *, existing: UsageAlertRule | None = None) -> bool: ...

    async def list_rules(
        self,
        scope_type: UsageAlertScope | None = None,
        scope_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        kind: UsageAlertRuleKind = "all",
    ) -> UsageAlertRuleList: ...

    async def get_alert(self, alert_id: str) -> UsageAlert | None: ...

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
    ) -> UsageAlertList: ...

    async def set_status(self, alert_id: str, status: UsageAlertStatus) -> UsageAlert | None: ...

    async def list_groups(
        self,
        scope_type: UsageAlertScope | None = None,
        scope_id: str | None = None,
        status: UsageAlertStatus | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> UsageAlertGroupList: ...

    async def set_group_status(self, data: UsageAlertGroupUpdate) -> UsageAlertGroupUpdateResult: ...

    async def list_reported_models(self, search: str | None, cursor: str | None, limit: int) -> UsageAlertModelList: ...

    async def key_ownership(self, key_id: str) -> tuple[str | None, str | None] | None: ...

    async def team_role(self, team_id: str, user_id: str) -> str | None: ...
