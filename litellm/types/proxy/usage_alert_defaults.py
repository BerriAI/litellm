from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from uuid import NAMESPACE_URL, uuid5

from litellm.types.proxy.usage_alerts import (
    UsageAlertChannel,
    UsageAlertDirection,
    UsageAlertMetric,
    UsageAlertRule,
    UsageAlertRuleInput,
    UsageAlertScope,
)

PerformanceIssueFamily: TypeAlias = Literal["spend", "latency", "reliability", "traffic"]
USAGE_ALERT_SCOPES: Final[tuple[UsageAlertScope, ...]] = ("deployment", "key", "user", "team", "model")


@dataclass(frozen=True, slots=True)
class DefaultUsagePolicy:
    family: PerformanceIssueFamily
    metric: UsageAlertMetric
    direction: UsageAlertDirection
    multiplier: float
    min_value: float
    min_requests: int = 0
    min_history_days: int = 3


DEFAULT_USAGE_POLICIES: Final = (
    DefaultUsagePolicy("spend", "spend", "increase", 3, 10),
    DefaultUsagePolicy("latency", "latency", "increase", 3, 500, 100),
    DefaultUsagePolicy("reliability", "error_rate", "increase", 3, 2, 500),
    DefaultUsagePolicy("traffic", "requests", "increase", 10, 1000),
    DefaultUsagePolicy("traffic", "requests", "decrease", 5, 0, 500, 7),
)


def default_usage_rule_id(scope: UsageAlertScope, metric: UsageAlertMetric, direction: UsageAlertDirection) -> str:
    return str(uuid5(NAMESPACE_URL, f"litellm:usage-alert:{scope}:{metric}:{direction}"))


_DEFAULT_USAGE_IDENTITIES: Final = MappingProxyType(
    {
        default_usage_rule_id(scope, policy.metric, policy.direction): (scope, policy.metric, policy.direction)
        for scope, policy in product(USAGE_ALERT_SCOPES, DEFAULT_USAGE_POLICIES)
    }
)
DEFAULT_USAGE_RULE_IDS: Final = frozenset(_DEFAULT_USAGE_IDENTITIES)


def is_default_usage_rule(rule: UsageAlertRule) -> bool:
    return _DEFAULT_USAGE_IDENTITIES.get(rule.id) == (rule.scope_type, rule.metric, rule.direction)


def default_usage_rules(
    channels: tuple[UsageAlertChannel, ...] = (), *, legacy_user_spend_alerts: bool = False
) -> tuple[UsageAlertRuleInput, ...]:
    return tuple(
        UsageAlertRuleInput(
            name=f"Daily {scope} {policy.metric.replace('_', ' ')} {policy.direction}",
            scope_type=scope,
            metric=policy.metric,
            direction=policy.direction,
            multiplier=policy.multiplier,
            min_value=policy.min_value,
            min_requests=policy.min_requests,
            min_history_days=policy.min_history_days,
            notification_channels=(
                ()
                if legacy_user_spend_alerts
                and (scope, policy.metric, policy.direction) == ("user", "spend", "increase")
                else channels
            ),
        )
        for scope, policy in product(USAGE_ALERT_SCOPES, DEFAULT_USAGE_POLICIES)
    )
