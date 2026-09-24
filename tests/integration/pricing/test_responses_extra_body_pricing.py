import json
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse

STANDARD_INPUT_RATE: Final = 0.001
STANDARD_OUTPUT_RATE: Final = 0.002
FLEX_INPUT_RATE: Final = 0.0003
FLEX_OUTPUT_RATE: Final = 0.0004
INPUT_TOKENS: Final = 100
OUTPUT_TOKENS: Final = 50
OPENAI_UPSTREAM_MODEL: Final = "gpt-4o-mini"
SAIL_UPSTREAM_MODEL: Final = "integration-sail-model"


def scripted_response(upstream_model: str) -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": upstream_model,
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
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    )


def assert_responses_bills_rates(
    gateway: Gateway,
    handle: ScenarioHandle,
    model: str,
    extra_body: Mapping[str, JsonValue],
    input_rate: float,
    output_rate: float,
    *,
    upstream_model: str,
) -> None:
    prompt: Final = f"extra body {json.dumps(dict(extra_body), sort_keys=True)} control {uuid.uuid4().hex}"
    with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        upstream.get("/__observations").raise_for_status()
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": prompt, "extra_body": dict(extra_body)}
        )
        assert response.status_code == 200, response.text
        expected: Final = INPUT_TOKENS * input_rate + OUTPUT_TOKENS * output_rate
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6), response.text
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert isinstance(observations, list)
        assert len(observations) == 1, observations
        observed: Final = object_value(observations[0])
        assert observed["path"] == f"/{handle.scenario_id}/responses"
        assert observed["body"] == {"model": upstream_model, "input": prompt, **extra_body}, response.text
    request_id: Final = string_value(JSON_OBJECT.validate_json(response.content)["id"])
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, metadata, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (request_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["prompt_tokens"] == INPUT_TOKENS
    assert rows[0]["completion_tokens"] == OUTPUT_TOKENS
    assert float(rows[0]["spend"]) == pytest.approx(expected, rel=1e-6)
    metadata: Final = rows[0]["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
    breakdown: Final = object_value(parsed["cost_breakdown"])
    assert float(breakdown["input_cost"]) == pytest.approx(INPUT_TOKENS * input_rate, rel=1e-6)
    assert float(breakdown["output_cost"]) == pytest.approx(OUTPUT_TOKENS * output_rate, rel=1e-6)


def test_responses_extra_body_on_an_openai_shaped_deployment_reaches_the_wire_and_bills_base_rates(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(
            f"responses-extra-body-{uuid.uuid4().hex[:12]}", scripted_response(OPENAI_UPSTREAM_MODEL)
        )
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            api_base=handle.api_base(),
            input_cost_per_token=STANDARD_INPUT_RATE,
            output_cost_per_token=STANDARD_OUTPUT_RATE,
            input_cost_per_token_flex=FLEX_INPUT_RATE,
            output_cost_per_token_flex=FLEX_OUTPUT_RATE,
        )
        assert_responses_bills_rates(
            gateway,
            handle,
            model,
            {"metadata": {"completion_window": "flex", "trace": "extra-body-control"}, "custom_flag": True},
            STANDARD_INPUT_RATE,
            STANDARD_OUTPUT_RATE,
            upstream_model=OPENAI_UPSTREAM_MODEL,
        )


def test_responses_extra_body_completion_window_on_a_sail_deployment_bills_the_window(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = register_scenario(
            f"responses-sail-window-{uuid.uuid4().hex[:12]}", scripted_response(SAIL_UPSTREAM_MODEL)
        )
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model=f"sail/{SAIL_UPSTREAM_MODEL}",
            api_base=handle.api_base(),
            input_cost_per_token=STANDARD_INPUT_RATE,
            output_cost_per_token=STANDARD_OUTPUT_RATE,
            input_cost_per_token_flex=FLEX_INPUT_RATE,
            output_cost_per_token_flex=FLEX_OUTPUT_RATE,
        )
        assert_responses_bills_rates(
            gateway,
            handle,
            model,
            {"metadata": {"completion_window": "flex"}},
            FLEX_INPUT_RATE,
            FLEX_OUTPUT_RATE,
            upstream_model=SAIL_UPSTREAM_MODEL,
        )
        assert_responses_bills_rates(
            gateway,
            handle,
            model,
            {"metadata": {"completion_window": "asap"}},
            STANDARD_INPUT_RATE,
            STANDARD_OUTPUT_RATE,
            upstream_model=SAIL_UPSTREAM_MODEL,
        )
