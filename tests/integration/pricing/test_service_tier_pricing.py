import json
from typing import Final

import httpx
import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows

STANDARD_INPUT_RATE: Final = 0.001
STANDARD_OUTPUT_RATE: Final = 0.002
ULTRAFAST_INPUT_RATE: Final = 0.01
ULTRAFAST_OUTPUT_RATE: Final = 0.02


def assert_chat_bills_rates(
    gateway: Gateway, model: str, service_tier: str | None, input_rate: float, output_rate: float
) -> None:
    with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        upstream.get("/__observations").raise_for_status()
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"service tier {service_tier} control"}],
                **({} if service_tier is None else {"service_tier": service_tier}),
            },
        )
        assert response.status_code == 200, response.text
        expected: Final = 20 * input_rate + 20 * output_rate
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6), response.text
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert isinstance(observations, list)
        assert len(observations) == 1
        body: Final = object_value(object_value(observations[0])["body"])
        assert body == {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"service tier {service_tier} control"}],
            **({} if service_tier is None else {"service_tier": service_tier}),
        }, response.text
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


@pytest.mark.covers("quota_management.spend_tracking.service_tier_pricing.ultrafast_bills_ultrafast_rates")
def test_ultrafast_service_tier_bills_ultrafast_rates_and_keeps_pricing_off_the_wire(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            input_cost_per_token=STANDARD_INPUT_RATE,
            output_cost_per_token=STANDARD_OUTPUT_RATE,
            input_cost_per_token_ultrafast=ULTRAFAST_INPUT_RATE,
            output_cost_per_token_ultrafast=ULTRAFAST_OUTPUT_RATE,
        )
        assert_chat_bills_rates(gateway, model, "ultrafast", ULTRAFAST_INPUT_RATE, ULTRAFAST_OUTPUT_RATE)
        assert_chat_bills_rates(gateway, model, None, STANDARD_INPUT_RATE, STANDARD_OUTPUT_RATE)
