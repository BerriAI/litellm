import uuid
from datetime import date, timedelta
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows

EXPECTED_SPEND: Final = 20 * 0.001 + 20 * 0.002


def _logged_request(gateway: Gateway, scenario: Scenario) -> tuple[str, str, str]:
    model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002, num_retries=0)
    key: Final = scenario.key(models=[model])
    body: Final = gateway.chat(model, key=key, text=uuid.uuid4().hex)
    request_id: Final = string_value(body["id"])
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT to_char("startTime", \'YYYY-MM-DD\') AS day, spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (request_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert float(rows[0]["spend"]) == pytest.approx(EXPECTED_SPEND)
    return key, request_id, string_value(rows[0]["day"])


def _spend(row: dict[str, JsonValue]) -> float:
    value: Final = row["spend"]
    assert isinstance(value, (int, float)), f"spend is not numeric: {row}"
    return float(value)


@pytest.mark.covers("quota_management.spend_tracking.date_range.includes_end_date")
def test_single_day_window_returns_that_days_summary_and_rows(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key, request_id, day = _logged_request(gateway, scenario)
        window: Final = {"start_date": day, "end_date": day, "api_key": key}
        summary: Final = gateway.request("GET", "/spend/logs", params=window)
        assert summary.status_code == 200, summary.text
        days: Final = tuple(object_value(row) for row in summary.json())
        assert [row["startTime"] for row in days] == [day], summary.text
        assert _spend(days[0]) == pytest.approx(EXPECTED_SPEND), summary.text
        raw: Final = gateway.request("GET", "/spend/logs", params={**window, "summarize": "false"})
        assert raw.status_code == 200, raw.text
        assert [object_value(row)["request_id"] for row in raw.json()] == [request_id], raw.text


@pytest.mark.covers("quota_management.spend_tracking.date_range.includes_end_date")
def test_multi_day_window_ends_with_the_end_days_real_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key, _, day = _logged_request(gateway, scenario)
        start: Final = (date.fromisoformat(day) - timedelta(days=2)).isoformat()
        summary: Final = gateway.request(
            "GET", "/spend/logs", params={"start_date": start, "end_date": day, "api_key": key}
        )
        assert summary.status_code == 200, summary.text
        days: Final = tuple(object_value(row) for row in summary.json())
        assert days, summary.text
        assert days[-1]["startTime"] == day, summary.text
        assert _spend(days[-1]) == pytest.approx(EXPECTED_SPEND), summary.text
