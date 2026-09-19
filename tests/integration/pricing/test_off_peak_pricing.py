import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows

STANDARD_INPUT_RATE: Final = 0.001
STANDARD_OUTPUT_RATE: Final = 0.002
OFF_PEAK_INPUT_RATE: Final = 0.0001
OFF_PEAK_OUTPUT_RATE: Final = 0.0002


def off_peak_window(start_offset_hours: int, end_offset_hours: int) -> Mapping[str, JsonValue]:
    now: Final = datetime.now(timezone.utc)
    start: Final = now + timedelta(hours=start_offset_hours)
    end: Final = now + timedelta(hours=end_offset_hours)
    return {
        "hours_utc": f"{start:%H:%M}-{end:%H:%M}",
        "input_cost_per_token": OFF_PEAK_INPUT_RATE,
        "output_cost_per_token": OFF_PEAK_OUTPUT_RATE,
    }


def billed_model(scenario: Scenario, off_peak: Mapping[str, JsonValue]) -> str:
    return scenario.model(
        input_cost_per_token=STANDARD_INPUT_RATE,
        output_cost_per_token=STANDARD_OUTPUT_RATE,
        model_info={"off_peak_pricing": dict(off_peak)},
    )


def assert_chat_bills_rates(gateway: Gateway, model: str, input_rate: float, output_rate: float) -> None:
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "off peak control"}]}
    )
    assert response.status_code == 200, response.text
    expected: Final = 20 * input_rate + 20 * output_rate
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6)
    request_id: Final = string_value(object_value(response.json())["id"])
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, metadata, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (request_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["prompt_tokens"] == 20
    assert rows[0]["completion_tokens"] == 20
    assert float(rows[0]["spend"]) == pytest.approx(expected, rel=1e-6)
    metadata: Final = rows[0]["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
    breakdown: Final = object_value(parsed["cost_breakdown"])
    assert float(breakdown["input_cost"]) == pytest.approx(20 * input_rate, rel=1e-6)
    assert float(breakdown["output_cost"]) == pytest.approx(20 * output_rate, rel=1e-6)


@pytest.mark.covers("quota_management.spend_tracking.off_peak_pricing.open_window_bills_off_peak_rates")
def test_open_off_peak_window_bills_off_peak_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = billed_model(scenario, off_peak_window(-1, 1))
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list)
        matching: Final = tuple(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model)
        assert len(matching) == 1
        info: Final = object_value(matching[0]["model_info"])
        off_peak: Final = object_value(info["off_peak_pricing"])
        assert off_peak["input_cost_per_token"] == OFF_PEAK_INPUT_RATE
        assert off_peak["output_cost_per_token"] == OFF_PEAK_OUTPUT_RATE
        assert_chat_bills_rates(gateway, model, OFF_PEAK_INPUT_RATE, OFF_PEAK_OUTPUT_RATE)


@pytest.mark.covers("quota_management.spend_tracking.off_peak_pricing.closed_window_bills_standard_rates")
def test_closed_off_peak_window_bills_standard_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = billed_model(scenario, off_peak_window(2, 3))
        assert_chat_bills_rates(gateway, model, STANDARD_INPUT_RATE, STANDARD_OUTPUT_RATE)
