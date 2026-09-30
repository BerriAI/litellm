import json
import uuid
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse

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


LONG_CONTEXT_PRICING: Final[dict[str, JsonValue]] = {
    "input_cost_per_token": 1e-06,
    "output_cost_per_token": 2e-06,
    "cache_read_input_token_cost": 1e-07,
    "input_cost_per_token_above_272k_tokens": 3e-06,
    "output_cost_per_token_above_272k_tokens": 4e-06,
    "cache_read_input_token_cost_above_272k_tokens": 3e-07,
    "input_cost_per_token_ultrafast": 1e-05,
    "output_cost_per_token_ultrafast": 2e-05,
    "cache_read_input_token_cost_ultrafast": 1e-06,
    "input_cost_per_token_above_272k_tokens_ultrafast": 5e-05,
    "output_cost_per_token_above_272k_tokens_ultrafast": 6e-05,
    "cache_read_input_token_cost_above_272k_tokens_ultrafast": 5e-06,
    "cache_creation_input_token_cost_above_272k_tokens_ultrafast": 6e-06,
}
LONG_PROMPT_TOKENS: Final = 300_000
SHORT_PROMPT_TOKENS: Final = 1_000
CACHED_TOKENS: Final = 400
COMPLETION_TOKENS: Final = 1_000


def _chat_response(service_tier: str | None, prompt_tokens: int) -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$UNIQUE_ID",
            "object": "chat.completion",
            "created": 1,
            "model": "integration-ultrafast-long-context",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "long context answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": prompt_tokens + COMPLETION_TOKENS,
                "prompt_tokens_details": {"cached_tokens": CACHED_TOKENS},
            },
            **({} if service_tier is None else {"service_tier": service_tier}),
        },
    )


def _responses_response(service_tier: str | None, prompt_tokens: int) -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$UNIQUE_ID",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "integration-ultrafast-long-context",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$UNIQUE_ID",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "long context answer", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": COMPLETION_TOKENS,
                "total_tokens": prompt_tokens + COMPLETION_TOKENS,
                "input_tokens_details": {"cached_tokens": CACHED_TOKENS},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            **({} if service_tier is None else {"service_tier": service_tier}),
        },
    )


def _surface_response(
    surface: Literal["chat", "responses"], service_tier: str | None, prompt_tokens: int
) -> JsonResponse:
    match surface:
        case "chat":
            return _chat_response(service_tier, prompt_tokens)
        case "responses":
            return _responses_response(service_tier, prompt_tokens)


def _surface_request(
    surface: Literal["chat", "responses"], scenario_id: str, model: str, service_tier: str | None
) -> tuple[str, dict[str, JsonValue], str]:
    match surface:
        case "chat":
            return (
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "long context ultrafast control"}],
                    **({} if service_tier is None else {"service_tier": service_tier}),
                },
                f"/{scenario_id}/chat/completions",
            )
        case "responses":
            return (
                "/v1/responses",
                {
                    "model": model,
                    "input": "long context ultrafast control",
                    **({} if service_tier is None else {"service_tier": service_tier}),
                },
                f"/{scenario_id}/responses",
            )


@pytest.mark.parametrize(
    ("service_tier", "prompt_tokens", "input_rate", "cache_read_rate", "output_rate"),
    (
        ("ultrafast", LONG_PROMPT_TOKENS, 5e-05, 5e-06, 6e-05),
        ("ultrafast", SHORT_PROMPT_TOKENS, 1e-05, 1e-06, 2e-05),
        (None, LONG_PROMPT_TOKENS, 3e-06, 3e-07, 4e-06),
    ),
    ids=("ultrafast_above_272k", "ultrafast_below_272k", "standard_above_272k"),
)
@pytest.mark.parametrize("surface", ("chat", "responses"), ids=("chat", "responses"))
def test_ultrafast_long_context_prompt_bills_ultrafast_long_context_rates(
    gateway: Gateway,
    surface: Literal["chat", "responses"],
    service_tier: str | None,
    prompt_tokens: int,
    input_rate: float,
    cache_read_rate: float,
    output_rate: float,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"ultrafast-long-context-{uuid.uuid4().hex}"
        handle: Final = register_scenario(
            scenario_id, _surface_response(surface, service_tier, prompt_tokens)
        )
        scenario.cleanups.callback(delete_scenario, handle)
        key: Final = scenario.key()
        model: Final = scenario.model(
            model=f"openai/integration-ultrafast-long-context-{uuid.uuid4().hex}",
            api_key=scenario_id,
            api_base=handle.api_base(),
            **LONG_CONTEXT_PRICING,
        )
        request_path, request_body, expected_upstream_path = _surface_request(surface, scenario_id, model, service_tier)
        with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
            upstream.get("/__observations").raise_for_status()
            response: Final = gateway.request(
                "POST",
                request_path,
                request_body,
                key=key,
            )
            observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert response.status_code == 200, response.text
        expected_input: Final = (prompt_tokens - CACHED_TOKENS) * input_rate + CACHED_TOKENS * cache_read_rate
        expected_output: Final = COMPLETION_TOKENS * output_rate
        expected: Final = expected_input + expected_output
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6), response.text
        request_id: Final = string_value(object_value(response.json())["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, metadata, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (request_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["prompt_tokens"] == prompt_tokens
        assert rows[0]["completion_tokens"] == COMPLETION_TOKENS
        assert float(rows[0]["spend"]) == pytest.approx(expected, rel=1e-6)
        metadata: Final = rows[0]["metadata"]
        parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
        breakdown: Final = object_value(parsed["cost_breakdown"])
        assert float(breakdown["input_cost"]) == pytest.approx(expected_input, rel=1e-6)
        assert float(breakdown["output_cost"]) == pytest.approx(expected_output, rel=1e-6)
        assert isinstance(observations, list)
        assert len(observations) == 1
        observation: Final = object_value(observations[0])
        upstream_path: Final = string_value(observation["path"])
        assert upstream_path == expected_upstream_path, upstream_path
        body: Final = object_value(observation["body"])
        assert body.get("service_tier") == service_tier, body
        assert not set(LONG_CONTEXT_PRICING).intersection(body), body


BUNDLED_COST_MAP: Final = (
    Path(__file__).resolve().parents[3] / "litellm" / "model_prices_and_context_window_backup.json"
)
CUSTOM_STANDARD_INPUT_RATE: Final = 0.001
CUSTOM_STANDARD_OUTPUT_RATE: Final = 0.002


def _bundled_rate(model: str, field: str) -> float:
    rate: Final = object_value(JSON_OBJECT.validate_json(BUNDLED_COST_MAP.read_bytes())[model])[field]
    assert isinstance(rate, float) and rate > 0, f"{model}.{field} in {BUNDLED_COST_MAP.name}: {rate}"
    return rate


@pytest.mark.parametrize(
    ("service_tier", "input_field", "output_field"),
    (
        ("ultrafast", "input_cost_per_token_ultrafast", "output_cost_per_token_ultrafast"),
        (None, None, None),
    ),
    ids=("ultrafast", "standard"),
)
def test_custom_standard_rates_bill_served_ultrafast_tier_at_the_catalog_tier_rate(
    gateway: Gateway, service_tier: str | None, input_field: str | None, output_field: str | None
) -> None:
    input_rate: Final = CUSTOM_STANDARD_INPUT_RATE if input_field is None else _bundled_rate("gpt-6-astra", input_field)
    output_rate: Final = (
        CUSTOM_STANDARD_OUTPUT_RATE if output_field is None else _bundled_rate("gpt-6-astra", output_field)
    )
    with gateway.scenario() as scenario:
        scenario_id: Final = f"custom-standard-ultrafast-{uuid.uuid4().hex}"
        handle: Final = register_scenario(
            scenario_id,
            JsonResponse(
                content_type="application/json",
                body={
                    "id": "chatcmpl-$UNIQUE_ID",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-6-astra",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
                    **({} if service_tier is None else {"service_tier": service_tier}),
                },
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="openai/gpt-6-astra",
            api_key=scenario_id,
            api_base=handle.api_base(),
            input_cost_per_token=CUSTOM_STANDARD_INPUT_RATE,
            output_cost_per_token=CUSTOM_STANDARD_OUTPUT_RATE,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "OK"}],
                **({} if service_tier is None else {"service_tier": service_tier}),
            },
            key=scenario.key(),
        )
        assert response.status_code == 200, response.text
        expected: Final = 1000 * input_rate + 100 * output_rate
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6), response.text
        request_id: Final = string_value(object_value(response.json())["id"])
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(expected, rel=1e-6), rows
