from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, get_args

import pytest

from litellm.repositories.issue_families import family_settings
from litellm.types.proxy.usage_alerts import (
    UsageAlertEvaluationStatus,
    UsageAlertMetric,
    AlertRule,
    UsageAlertScope,
)


@dataclass
class RuleRows:
    rules: tuple[AlertRule, ...]
    calls: int = 0

    async def query_raw(self, query: str, *args: object) -> object:
        self.calls += 1
        if "SELECT EXISTS" in query:
            return ({"pending": False},)
        return tuple({"record": rule.model_dump(mode="json")} for rule in self.rules)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", get_args(UsageAlertEvaluationStatus))
async def test_family_counts_only_its_enabled_rules_without_an_extra_query(status: UsageAlertEvaluationStatus) -> None:
    timestamp: Final = datetime(2026, 11, 5, tzinfo=timezone.utc)
    inputs: Final[tuple[tuple[UsageAlertScope, UsageAlertMetric, bool], ...]] = (
        ("key", "spend", True),
        ("team", "spend", True),
        ("user", "spend", False),
        ("model", "latency", True),
    )
    rules: Final = tuple(
        AlertRule(
            id=f"test-{scope}-{metric}",
            name="Daily monitoring",
            scope_type=scope,
            metric=metric,
            direction="increase",
            enabled=enabled,
            evaluation_status=status,
            created_at=timestamp,
            updated_at=timestamp,
            created_by="test-admin",
        )
        for scope, metric, enabled in inputs
    )

    database: Final = RuleRows(rules)
    result: Final = next(setting for setting in (await family_settings(database)).data if setting.family == "spend")

    assert database.calls == 2  # existing rule rows plus the separate projection-backlog read
    assert result.enabled_rules == 2
    assert result.total_rules == 3
    assert result.evaluation_status_counts.model_dump() == {
        **dict.fromkeys(get_args(UsageAlertEvaluationStatus), 0),
        status: 2,
    }
