from __future__ import annotations

import asyncio
import os
import uuid
from typing import Final

import pytest
from openai import OpenAI
from pydantic import JsonValue

import tests.integration._support.service_tier_pricing as pricing
from tests.integration._support.client import Gateway, eventually, object_value, string_value


@pytest.mark.parametrize(
    ("service_tier", "rate_tier", "prompt_tokens", "completion_tokens", "cached_tokens"),
    pricing.SERVICE_TIER_CASES,
    ids=("priority", "flex", "ultrafast", "fast-alias-to-priority"),
)
def test_raw_chat_tier_rates_follow_the_catalog_for_custom_deployments(
    gateway: Gateway,
    service_tier: str,
    rate_tier: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response(
                service_tier, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
            identifier=f"tier-matrix-chat-{service_tier}-billed-as-{rate_tier}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        payload: Final = pricing.chat_payload(service_tier)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **payload},
            key=key,
        )
        pricing.assert_success(
            gateway,
            outcome,
            service_tier=service_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            expected_cost=pricing.expected_cost(rate_tier, prompt_tokens, completion_tokens),
        )


@pytest.mark.parametrize(
    "service_tier",
    ("balanced", None),
    ids=("balanced", "no-tier"),
)
def test_balanced_and_no_tier_keep_custom_standard_rates(
    gateway: Gateway,
    service_tier: str | None,
) -> None:
    scenario_name: Final = "balanced" if service_tier is not None else "no-tier"
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response(service_tier),
            identifier=f"tier-matrix-chat-{scenario_name}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload(service_tier)},
            key=key,
        )
        pricing.assert_success(
            gateway,
            outcome,
            service_tier=service_tier,
            expected_cost=pricing.expected_cost(
                service_tier, pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
            ),
        )


def test_openai_sdk_sync_priority_chat_uses_inherited_catalog_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-sdk-sync-chat-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_sync_sdk(
            os.environ["INTEGRATION_PROXY_URL"],
            key=key,
            model=model,
            service_tier="priority",
            stream=False,
        )
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.expected_cost(
                "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
            ),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_openai_sdk_async_priority_chat_uses_inherited_catalog_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-sdk-async-chat-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = asyncio.run(
            pricing._invoke_async_sdk(
                os.environ["INTEGRATION_PROXY_URL"],
                key=key,
                model=model,
                service_tier="priority",
            )
        )
        assert outcome.status == 200, outcome.text
        assert outcome.request_id is not None, outcome
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.expected_cost(
                "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
            ),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_openai_sdk_streaming_priority_chat_consumes_usage_and_logs_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_sse_response(),
            identifier=f"tier-matrix-sdk-stream-chat-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_sync_sdk(
            os.environ["INTEGRATION_PROXY_URL"],
            key=key,
            model=model,
            service_tier="priority",
            stream=True,
        )
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.expected_cost(
                "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
            ),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
            allow_missing_header=True,
        )
        pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_raw_responses_priority_uses_inherited_catalog_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.responses_json_response("priority"),
            identifier=f"tier-matrix-raw-responses-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_RESPONSES_PATH,
            payload={
                "model": model,
                "input": "service tier matrix request",
                "service_tier": "priority",
                "cache": {"no-cache": True},
            },
            key=key,
            surface="responses",
        )
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.expected_cost(
                "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
            ),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_openai_sdk_streaming_responses_priority_consumes_full_stream(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.responses_sse_response(),
            identifier=f"tier-matrix-sdk-stream-responses-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        with OpenAI(
            api_key=key,
            base_url=f"{os.environ['INTEGRATION_PROXY_URL'].rstrip('/')}/v1",
        ) as client:
            with client.responses.with_streaming_response.create(
                model=model,
                input="service tier matrix request",
                service_tier="priority",
                stream=True,
                extra_body={"cache": {"no-cache": True}},
            ) as response:
                stream_text: Final = "\n".join(response.iter_lines())
                outcome: Final = pricing.stream_outcome(
                    status=response.status_code,
                    headers=response.headers,
                    text=stream_text,
                    surface="responses",
                )
        expected_cost: Final = pricing.expected_cost(
            "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
        )
        pricing.assert_cost_header(
            outcome,
            expected_cost=expected_cost,
            allow_missing_header=True,
        )
        rows: Final = eventually(
            lambda: pricing.spend_rows_for_key(key),
            lambda current: len(current) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "success", row
        assert float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6), row
        assert int(str(row["prompt_tokens"])) == pricing.MATRIX_PROMPT_TOKENS, row
        assert int(str(row["completion_tokens"])) == pricing.MATRIX_COMPLETION_TOKENS, row
        pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
def test_messages_endpoint_routes_to_responses_upstream_and_bills_custom_standard(
    gateway: Gateway,
    stream: bool,
) -> None:
    mode: Final = "streaming" if stream else "non-streaming"
    response: Final = (
        pricing.responses_sse_response(service_tier=None) if stream else pricing.responses_json_response(None)
    )
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=response,
            identifier=f"tier-matrix-messages-{mode}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        payload: dict[str, JsonValue] = {
            "model": model,
            "max_tokens": pricing.MATRIX_COMPLETION_TOKENS,
            "messages": [{"role": "user", "content": f"service tier matrix request {scenario_id}"}],
            "cache": {"no-cache": True},
        }
        if stream:
            payload["stream"] = True
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_MESSAGES_PATH,
            payload=payload,
            key=key,
            stream=stream,
            surface="messages",
        )
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.expected_cost(None, pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
            allow_missing_header=stream,
        )
        observed_request: Final = pricing.assert_upstream_request(gateway)
        observed_path: Final = string_value(observed_request["path"])
        assert observed_path == f"/{scenario_id}/responses", observed_request
        observed: Final = object_value(observed_request["body"])
        assert "service_tier" not in observed
