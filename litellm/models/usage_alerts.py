from datetime import date
from typing import Protocol

from litellm.types.proxy.usage_alerts import (
    UsageAlert,
    UsageAlertList,
    UsageAlertMetric,
    UsageAlertRule,
    UsageAlertRuleInput,
    UsageAlertRuleList,
    UsageAlertScope,
    UsageAlertStatus,
)


class UsageAlertStore(Protocol):
    async def get_rule(self, rule_id: str) -> UsageAlertRule | None: ...

    async def save_rule(
        self, data: UsageAlertRuleInput, created_by: str | None, existing: UsageAlertRule | None = None
    ) -> UsageAlertRule | None: ...

    async def delete_rule(self, rule_id: str) -> None: ...

    async def list_rules(
        self, scope_type: UsageAlertScope | None = None, scope_id: str | None = None, limit: int = 50, offset: int = 0
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

    async def key_ownership(self, key_id: str) -> tuple[str | None, str | None] | None: ...

    async def team_role(self, team_id: str, user_id: str) -> str | None: ...
