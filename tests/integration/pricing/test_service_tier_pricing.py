from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.database import read_rows


@pytest.mark.covers("quota_management.spend_tracking.service_tier.priority_request_priced_as_priority")
@pytest.mark.timeout(180)
def test_databricks_priority_request_is_forwarded_and_charged_at_priority_rates(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(
            model="databricks/dbrx-instruct",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            input_cost_per_token_priority=0.003,
            output_cost_per_token_priority=0.004,
        )
        upstream.get("/__observations").raise_for_status()
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "priority price control"}],
                "service_tier": "priority",
            },
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        observed: Final = upstream.get("/__observations").json()["requests"]
        assert len(observed) == 1, observed
        assert observed[0]["body"]["service_tier"] == "priority", observed
        expected_spend: Final = 20 * 0.003 + 20 * 0.004
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected_spend)
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (body["id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(expected_spend), rows
