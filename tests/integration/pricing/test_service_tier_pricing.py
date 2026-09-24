import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows

STANDARD_INPUT_RATE: Final = 0.001
STANDARD_OUTPUT_RATE: Final = 0.002
ULTRAFAST_INPUT_RATE: Final = 0.01
ULTRAFAST_OUTPUT_RATE: Final = 0.02
BALANCED_INPUT_RATE: Final = 0.005
BALANCED_OUTPUT_RATE: Final = 0.006
FLEX_INPUT_RATE: Final = 0.0003
FLEX_OUTPUT_RATE: Final = 0.0004
OPENAI_UPSTREAM_MODEL: Final = "gpt-4o-mini"
SAIL_UPSTREAM_MODEL: Final = "integration-sail-model"
JSON_PROVIDER_WITHOUT_SPECIAL_HANDLING: Final = "cognition"
NO_FIELDS: Final[Mapping[str, JsonValue]] = MappingProxyType({})


def tier_fields(service_tier: str | None) -> Mapping[str, JsonValue]:
    return NO_FIELDS if service_tier is None else MappingProxyType({"service_tier": service_tier})


def assert_chat_bills_rates(
    gateway: Gateway,
    model: str,
    service_tier: str | None,
    input_rate: float,
    output_rate: float,
    *,
    request_fields: Mapping[str, JsonValue] = NO_FIELDS,
    upstream_model: str = OPENAI_UPSTREAM_MODEL,
    upstream_fields: Mapping[str, JsonValue] | None = None,
) -> None:
    content: Final = f"service tier {service_tier} {json.dumps(dict(request_fields), sort_keys=True)} control"
    messages: Final[tuple[JsonValue, ...]] = ({"role": "user", "content": content},)
    with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        upstream.get("/__observations").raise_for_status()
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": list(messages), **tier_fields(service_tier), **request_fields},
        )
        assert response.status_code == 200, response.text
        expected: Final = 20 * input_rate + 20 * output_rate
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6), response.text
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert isinstance(observations, list)
        assert len(observations) == 1, observations
        body: Final = object_value(object_value(observations[0])["body"])
        assert body == {
            "model": upstream_model,
            "messages": list(messages),
            **(tier_fields(service_tier) if upstream_fields is None else upstream_fields),
        }, response.text
    request_id: Final = string_value(JSON_OBJECT.validate_json(response.content)["id"])
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


def test_balanced_service_tier_bills_balanced_rates_and_keeps_pricing_off_the_wire(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            input_cost_per_token=STANDARD_INPUT_RATE,
            output_cost_per_token=STANDARD_OUTPUT_RATE,
            input_cost_per_token_balanced=BALANCED_INPUT_RATE,
            output_cost_per_token_balanced=BALANCED_OUTPUT_RATE,
        )
        assert_chat_bills_rates(gateway, model, "balanced", BALANCED_INPUT_RATE, BALANCED_OUTPUT_RATE)
        assert_chat_bills_rates(gateway, model, None, STANDARD_INPUT_RATE, STANDARD_OUTPUT_RATE)


def test_balanced_and_flex_on_a_base_rate_only_model_bill_base_rates_and_pass_the_tier_through(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            input_cost_per_token=STANDARD_INPUT_RATE, output_cost_per_token=STANDARD_OUTPUT_RATE
        )
        assert_chat_bills_rates(gateway, model, "balanced", STANDARD_INPUT_RATE, STANDARD_OUTPUT_RATE)
        assert_chat_bills_rates(gateway, model, "flex", STANDARD_INPUT_RATE, STANDARD_OUTPUT_RATE)


@pytest.mark.parametrize("provider_prefix", ["openai", JSON_PROVIDER_WITHOUT_SPECIAL_HANDLING])
def test_extra_body_metadata_on_an_openai_shaped_deployment_replaces_the_wire_metadata_whole(
    gateway: Gateway, provider_prefix: str
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"{provider_prefix}/{OPENAI_UPSTREAM_MODEL}",
            input_cost_per_token=STANDARD_INPUT_RATE,
            output_cost_per_token=STANDARD_OUTPUT_RATE,
        )
        assert_chat_bills_rates(
            gateway,
            model,
            None,
            STANDARD_INPUT_RATE,
            STANDARD_OUTPUT_RATE,
            request_fields={"metadata": {"a": 1, "b": 2}, "extra_body": {"metadata": {"b": 3, "c": 4}}},
            upstream_fields={"metadata": {"b": 3, "c": 4}},
        )


def test_sail_flex_sends_completion_window_without_service_tier_and_bills_flex_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"sail/{SAIL_UPSTREAM_MODEL}",
            input_cost_per_token=STANDARD_INPUT_RATE,
            output_cost_per_token=STANDARD_OUTPUT_RATE,
            input_cost_per_token_flex=FLEX_INPUT_RATE,
            output_cost_per_token_flex=FLEX_OUTPUT_RATE,
        )
        assert_chat_bills_rates(
            gateway,
            model,
            "flex",
            FLEX_INPUT_RATE,
            FLEX_OUTPUT_RATE,
            upstream_model=SAIL_UPSTREAM_MODEL,
            upstream_fields={"metadata": {"completion_window": "flex"}},
        )
        assert_chat_bills_rates(
            gateway,
            model,
            "flex",
            STANDARD_INPUT_RATE,
            STANDARD_OUTPUT_RATE,
            request_fields={"extra_body": {"metadata": {"completion_window": "asap"}}},
            upstream_model=SAIL_UPSTREAM_MODEL,
            upstream_fields={"metadata": {"completion_window": "asap"}},
        )
        assert_chat_bills_rates(
            gateway,
            model,
            None,
            STANDARD_INPUT_RATE,
            STANDARD_OUTPUT_RATE,
            upstream_model=SAIL_UPSTREAM_MODEL,
        )
