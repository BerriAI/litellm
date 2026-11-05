from typing import Final

import pytest
from pydantic import ValidationError

from litellm.types.proxy.usage_alerts import UsageAlertGroupUpdate, UsageAlertRuleInput

RULE: Final = {
    "name": "Daily spend",
    "scope_type": "team",
    "scope_id": "qa-team",
    "metric": "spend",
    "direction": "increase",
}


@pytest.mark.parametrize(
    "changes",
    [
        {"multiplier": 1},
        {"multiplier": float("inf")},
        {"min_value": float("nan")},
        {"baseline_days": 2, "min_history_days": 3},
        {"baseline_days": 31},
        {"scope_type": "deployment"},
        {"scope_type": "key", "scope_id": "sk-test-placeholder"},
        {"notification_channels": ["email"]},
        {"notification_channels": ["slack", "slack"]},
        {"email_recipients": ["ops@example.com\r\nBcc: other@example.com"]},
    ],
)
def test_invalid_rules_cannot_be_stored(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        UsageAlertRuleInput.model_validate({**RULE, **changes})


def test_zero_request_floor_and_daily_drop_multiplier_are_supported() -> None:
    rule: Final = UsageAlertRuleInput.model_validate(
        {**RULE, "metric": "requests", "direction": "decrease", "multiplier": 2, "min_requests": 0}
    )
    assert rule.direction == "decrease"
    assert rule.multiplier == 2
    assert rule.min_requests == 0


@pytest.mark.parametrize(
    "changes",
    [{"scope_id": None}, {"scope_type": "deployment"}, {"status": "open"}, {"status": "unknown"}],
)
def test_group_updates_require_exact_resource_and_a_different_valid_status(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        UsageAlertGroupUpdate.model_validate(
            {
                "scope_type": "team",
                "scope_id": "team-example",
                "day": "2026-11-04",
                "expected_status": "open",
                "status": "resolved",
                **changes,
            }
        )
