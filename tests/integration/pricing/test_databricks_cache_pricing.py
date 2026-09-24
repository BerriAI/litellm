import json
import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse

INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CACHE_CREATION_RATE: Final = 0.004
CACHE_READ_RATE: Final = 0.0001
UNCACHED_PROMPT_TOKENS: Final = 1000
CACHE_CREATION_TOKENS: Final = 2000
CACHE_READ_TOKENS: Final = 8000
PROMPT_TOKENS: Final = UNCACHED_PROMPT_TOKENS + CACHE_CREATION_TOKENS + CACHE_READ_TOKENS
COMPLETION_TOKENS: Final = 500


def databricks_cached_response() -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "databricks-claude-integration",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "cached reply"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                "cache_creation_input_tokens": CACHE_CREATION_TOKENS,
                "cache_read_input_tokens": CACHE_READ_TOKENS,
            },
        },
    )


@pytest.mark.covers("pricing.databricks.cached_prompt_tokens_bill_at_cache_rates")
def test_databricks_cached_prompt_tokens_bill_at_cache_rates_not_input_rate(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"databricks-cache-{uuid.uuid4().hex[:12]}"
        handle: Final = register_scenario(scenario_id, databricks_cached_response())
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="databricks/databricks-claude-integration",
            api_base=handle.api_base(),
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
            cache_creation_input_token_cost=CACHE_CREATION_RATE,
            cache_read_input_token_cost=CACHE_READ_RATE,
        )
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "cache control"}]}
        )
        assert response.status_code == 200, response.text
        expected_prompt_cost: Final = (
            UNCACHED_PROMPT_TOKENS * INPUT_RATE
            + CACHE_CREATION_TOKENS * CACHE_CREATION_RATE
            + CACHE_READ_TOKENS * CACHE_READ_RATE
        )
        expected_completion_cost: Final = COMPLETION_TOKENS * OUTPUT_RATE
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(
            expected_prompt_cost + expected_completion_cost, rel=1e-6
        ), response.text
        request_id: Final = string_value(object_value(response.json())["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, metadata, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
                "WHERE request_id = %s",
                (request_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["prompt_tokens"] == PROMPT_TOKENS
        assert rows[0]["completion_tokens"] == COMPLETION_TOKENS
        assert float(rows[0]["spend"]) == pytest.approx(expected_prompt_cost + expected_completion_cost, rel=1e-6)
        metadata: Final = rows[0]["metadata"]
        parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
        breakdown: Final = object_value(parsed["cost_breakdown"])
        assert float(breakdown["input_cost"]) == pytest.approx(expected_prompt_cost, rel=1e-6)
        assert float(breakdown["output_cost"]) == pytest.approx(expected_completion_cost, rel=1e-6)
