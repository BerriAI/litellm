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
UNCACHED_INPUT_TOKENS: Final = 1000
CACHE_WRITE_TOKENS: Final = 2000
CACHED_TOKENS: Final = 8000
INPUT_TOKENS: Final = UNCACHED_INPUT_TOKENS + CACHE_WRITE_TOKENS + CACHED_TOKENS
OUTPUT_TOKENS: Final = 500


def responses_cache_write_response() -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$REQUEST_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": INPUT_TOKENS,
                "output_tokens": OUTPUT_TOKENS,
                "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
                "input_tokens_details": {
                    "cached_tokens": CACHED_TOKENS,
                    "cache_write_tokens": CACHE_WRITE_TOKENS,
                },
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    )


@pytest.mark.covers("spend.responses_api.cache_write_tokens_itemized_as_cache_creation_cost")
def test_responses_cache_write_tokens_are_itemized_as_cache_creation_cost(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"responses-cache-write-{uuid.uuid4().hex[:12]}"
        handle: Final = register_scenario(scenario_id, responses_cache_write_response())
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="openai/gpt-5.6",
            api_base=handle.api_base(),
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
            cache_creation_input_token_cost=CACHE_CREATION_RATE,
            cache_read_input_token_cost=CACHE_READ_RATE,
        )
        response: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": "cache write control"})
        assert response.status_code == 200, response.text
        expected_cache_creation_cost: Final = CACHE_WRITE_TOKENS * CACHE_CREATION_RATE
        expected_cache_read_cost: Final = CACHED_TOKENS * CACHE_READ_RATE
        expected_input_cost: Final = (
            UNCACHED_INPUT_TOKENS * INPUT_RATE + expected_cache_creation_cost + expected_cache_read_cost
        )
        expected_output_cost: Final = OUTPUT_TOKENS * OUTPUT_RATE
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
        assert rows[0]["prompt_tokens"] == INPUT_TOKENS, response.text
        assert rows[0]["completion_tokens"] == OUTPUT_TOKENS, response.text
        assert float(rows[0]["spend"]) == pytest.approx(expected_input_cost + expected_output_cost, rel=1e-6), (
            response.text
        )
        metadata: Final = rows[0]["metadata"]
        parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
        breakdown: Final = object_value(parsed["cost_breakdown"])
        assert breakdown.get("cache_creation_cost") == pytest.approx(expected_cache_creation_cost, rel=1e-6), breakdown
        assert breakdown.get("cache_read_cost") == pytest.approx(expected_cache_read_cost, rel=1e-6), breakdown
        assert float(breakdown["input_cost"]) == pytest.approx(expected_input_cost, rel=1e-6), breakdown
        assert float(breakdown["output_cost"]) == pytest.approx(expected_output_cost, rel=1e-6), breakdown
