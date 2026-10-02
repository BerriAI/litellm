"""Streamed /chat/completions bills the deployment's priced model whatever the model_name alias looks like.

The proxy restamps every streamed chunk with the client's alias, so end-of-stream cost calculation can see
"claude-opus-4.8" before the deployment's model. That name is no cost-map key but matches the claude capability
generalization rules, whose model info carries no prices, so the dotted alias must still bill exactly what the
exact-key alias "claude-opus-4-8" bills for the same usage (LIT-9065)
"""

import json
from collections.abc import Callable
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

CAPABILITY_RULE_ALIAS: Final = "claude-opus-4.8"
EXACT_KEY_ALIAS: Final = "claude-opus-4-8"


def _sse_event(name: str, payload: dict[str, JsonValue]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _anthropic_stream(request: Request) -> Reply:
    assert request.target.endswith("/v1/messages"), request.target
    body: Final = json.loads(request.body)
    assert body["model"] == "claude-opus-4-8" and body["stream"] is True, body
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


def _deployment(scenario: Scenario, model_name: str, litellm_params: dict[str, JsonValue]) -> str:
    created: Final = scenario.gateway.post(
        "/model/new", {"model_name": model_name, "litellm_params": litellm_params, "model_info": {}}
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
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _deployment_pricing(gateway: Gateway, model_name: str) -> dict[str, JsonValue]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    target: Final = next(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model_name)
    return object_value(target["model_info"])


@pytest.mark.parametrize(
    "litellm_params",
    (
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
    ),
)
@pytest.mark.timeout(180)
def test_streamed_alias_matching_a_capability_rule_bills_the_deployment_price(
    gateway: Gateway, litellm_params: Callable[[str], dict[str, JsonValue]]
) -> None:
    with wire_server(_anthropic_stream) as wire, gateway.scenario() as scenario:
        content: Final = f"alias billing {uuid4().hex}"
        exact_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, EXACT_KEY_ALIAS, litellm_params(wire.url)), content
        )
        alias_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, CAPABILITY_RULE_ALIAS, litellm_params(wire.url)), content
        )

        for model_name, row in ((EXACT_KEY_ALIAS, exact_row), (CAPABILITY_RULE_ALIAS, alias_row)):
            pricing: Final = _deployment_pricing(gateway, model_name)
            input_rate: Final = float(str(pricing["input_cost_per_token"]))
            output_rate: Final = float(str(pricing["output_cost_per_token"]))
            uplift: Final = float(str(pricing["regional_endpoint_uplift_multiplier"] or 1))
            assert input_rate > 0 and output_rate > 0, pricing
            assert float(str(row["spend"])) == pytest.approx(
                uplift
                * (float(str(row["prompt_tokens"])) * input_rate + float(str(row["completion_tokens"])) * output_rate)
            ), (model_name, row, pricing)
