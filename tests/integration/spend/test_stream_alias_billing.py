"""Streamed /chat/completions bills the deployment's priced model whatever the model_name alias looks like.

The proxy restamps every streamed chunk with the client's alias, so end-of-stream cost calculation can see
"claude-opus-4.8-<digits>" before the deployment's model. That name is no cost-map key but matches the claude
capability generalization rules, whose model info carries no prices, so the dotted alias must bill exactly what
the plain alias "integration-<hex>" bills at the same deployment rates (LIT-9065)
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

import litellm


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


def _anthropic_stream_haiku(request: Request) -> Reply:
    assert request.target.endswith("/v1/messages"), request.target
    body: Final = json.loads(request.body)
    assert body["model"] == "claude-haiku-4-5" and body["stream"] is True, body
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
                        "model": "claude-haiku-4-5",
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


def _deployment_with_model_info(
    scenario: Scenario, model_name: str, litellm_params: dict[str, JsonValue], model_info: dict[str, JsonValue]
) -> str:
    created: Final = scenario.gateway.post(
        "/model/new", {"model_name": model_name, "litellm_params": litellm_params, "model_info": model_info}
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
        plain_alias: Final = f"integration-{uuid4().hex}"
        rule_alias: Final = f"claude-opus-4.8-{uuid4().int % 10**8:08d}"
        exact_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, plain_alias, litellm_params(wire.url)), content
        )
        alias_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, rule_alias, litellm_params(wire.url)), content
        )

        assert alias_row["prompt_tokens"] == exact_row["prompt_tokens"], (plain_alias, rule_alias)
        assert alias_row["completion_tokens"] == exact_row["completion_tokens"], (plain_alias, rule_alias)
        assert float(str(alias_row["spend"])) == pytest.approx(float(str(exact_row["spend"]))), (
            plain_alias,
            rule_alias,
        )
        assert float(str(alias_row["spend"])) > 0, (rule_alias, alias_row)


@pytest.mark.timeout(180)
def test_streamed_deployment_with_rule_only_base_model_bills_the_deployment_price(gateway: Gateway) -> None:
    """A deployment whose model_info.base_model only matches a capability rule still bills
    at the deployment model's rates, same as a plain deployment on the same model."""
    with wire_server(_anthropic_stream_haiku) as wire, gateway.scenario() as scenario:
        litellm_params: Final[dict[str, JsonValue]] = {
            "model": "anthropic/claude-haiku-4-5",
            "api_key": "integration-provider-key",
            "api_base": wire.url,
        }
        content: Final = f"base_model billing {uuid4().hex}"
        control_row: Final = _streamed_spend(
            gateway, scenario, _deployment(scenario, f"integration-{uuid4().hex}", litellm_params), content
        )
        base_rule_row: Final = _streamed_spend(
            gateway,
            scenario,
            _deployment_with_model_info(
                scenario, f"haiku-base-rule-{uuid4().hex[:8]}", litellm_params, {"base_model": "claude-opus-9"}
            ),
            content,
        )

        row: Final = litellm.model_cost["claude-haiku-4-5"]
        for spend_row in (control_row, base_rule_row):
            expected: Final = int(str(spend_row["prompt_tokens"])) * float(str(row["input_cost_per_token"])) + int(
                str(spend_row["completion_tokens"])
            ) * float(str(row["output_cost_per_token"]))
            assert expected > 0, spend_row
            assert float(str(spend_row["spend"])) == pytest.approx(expected), spend_row
        assert float(str(base_rule_row["spend"])) == pytest.approx(float(str(control_row["spend"]))), (
            control_row,
            base_rule_row,
        )
