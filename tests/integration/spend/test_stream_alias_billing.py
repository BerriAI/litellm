"""A streamed alias never replaces the deployment's model for pricing (LIT-9065).

The proxy shows the client's alias on every streamed chunk, but the chunks kept for end-of-stream cost calculation
keep the deployment's model. "claude-opus-4.8-<digits>" is no cost-map key and only matches the claude capability
rules, whose model info carries no prices, so a stream through that alias must bill exactly what the plain alias
"integration-<hex>" bills at the same deployment rates, and the client must still see the alias it asked for
"""

import json
from collections.abc import Callable
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue


def _sse_event(name: str, payload: dict[str, JsonValue]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _anthropic_reply(request: Request) -> Reply:
    assert request.target.endswith("/v1/messages"), request.target
    body: Final = json.loads(request.body)
    assert body["model"] == "claude-opus-4-8", body
    if body.get("stream") is not True:
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg_{uuid4().hex[:12]}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-8",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 30, "output_tokens": 40},
                }
            ).encode()
        )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{uuid4().hex[:12]}",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-4-8",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 30, "output_tokens": 1},
                    },
                },
            ),
            _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            _sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 40},
                },
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        ),
    )


def _deployment(
    scenario: Scenario,
    model_name: str,
    litellm_params: dict[str, JsonValue],
    model_info: dict[str, JsonValue] | None = None,
) -> str:
    created: Final = scenario.gateway.post(
        "/model/new", {"model_name": model_name, "litellm_params": litellm_params, "model_info": model_info or {}}
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return model_name


def _streamed_spend(gateway: Gateway, scenario: Scenario, model: str, content: str) -> dict[str, JsonValue]:
    key: Final = scenario.key(models=[model])
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        key=key,
    )
    assert response.status_code == 200, response.text
    chunks: Final = tuple(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert chunks and {chunk["model"] for chunk in chunks} == {model}, response.text
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _listed_deployments(gateway: Gateway, model_name: str) -> tuple[dict[str, JsonValue], ...]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    return tuple(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model_name)


def _deployment_pricing(gateway: Gateway, model_name: str) -> dict[str, JsonValue]:
    listed: Final = eventually(lambda: _listed_deployments(gateway, model_name), lambda found: len(found) == 1)
    return object_value(listed[0]["model_info"])


_BACKENDS: Final = (
    pytest.param(
        lambda _: {"model": "vertex_ai/claude-opus-4-8@default", "mock_response": "hi"},
        id="vertex-mock-response",
    ),
    pytest.param(
        lambda wire_url: {
            "model": "anthropic/claude-opus-4-8",
            "api_key": "integration-provider-key",
            "api_base": wire_url,
        },
        id="anthropic-upstream",
    ),
)


@pytest.mark.covers("quota_management.spend_tracking.streamed_alias_bills_deployment_price")
@pytest.mark.parametrize("litellm_params", _BACKENDS)
@pytest.mark.timeout(180)
def test_streamed_alias_matching_a_capability_rule_bills_the_deployment_price(
    gateway: Gateway, litellm_params: Callable[[str], dict[str, JsonValue]]
) -> None:
    with wire_server(_anthropic_reply) as wire, gateway.scenario() as scenario:
        content: Final = f"alias billing {uuid4().hex}"
        plain_alias: Final = f"integration-{uuid4().hex}"
        rule_alias: Final = f"claude-opus-4.8-{uuid4().int % 10**8:08d}"
        exact_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, plain_alias, litellm_params(wire.url)), content
        )
        alias_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, rule_alias, litellm_params(wire.url)), content
        )

        for model_name, row in ((plain_alias, exact_row), (rule_alias, alias_row)):
            pricing: Final = _deployment_pricing(gateway, model_name)
            input_rate: Final = float(str(pricing["input_cost_per_token"]))
            output_rate: Final = float(str(pricing["output_cost_per_token"]))
            uplift: Final = float(str(pricing["regional_endpoint_uplift_multiplier"] or 1))
            assert input_rate > 0 and output_rate > 0, pricing
            assert float(str(row["spend"])) == pytest.approx(
                uplift
                * (float(str(row["prompt_tokens"])) * input_rate + float(str(row["completion_tokens"])) * output_rate)
            ), (model_name, row, pricing)

