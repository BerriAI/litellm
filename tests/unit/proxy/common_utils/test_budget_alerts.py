import pytest

from litellm.proxy.common_utils.budget_alerts import parse_budget_alert_recipients


@pytest.mark.parametrize(
    "raw, expected",
    [
        ([" finance@example.com ", "", "ops@internal"], ["finance@example.com", "ops@internal"]),
        (" finance@example.com, ,ops@internal ", ["finance@example.com", "ops@internal"]),
        ([None, 42, "ops@example.com"], ["ops@example.com"]),
        (None, []),
        (" , , ", []),
    ],
)
def test_parse_budget_alert_recipients(raw: object, expected: list[str]) -> None:
    assert parse_budget_alert_recipients(raw) == expected
