import json
import uuid
from dataclasses import dataclass
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.wire import Reply, Request, Wire, wire_server

BACKEND: Final = "zai-org/GLM-5.3"
PROMPT_TOKENS: Final = 1000
CACHED_TOKENS: Final = 400
COMPLETION_TOKENS: Final = 200

PRICES: Final[dict[str, JsonValue]] = {
    "input_cost_per_token": 3e-06,
    "output_cost_per_token": 9e-06,
    "cache_read_input_token_cost": 1e-06,
    "input_cost_per_token_balanced": 2e-06,
    "output_cost_per_token_balanced": 6e-06,
    "cache_read_input_token_cost_balanced": 7e-07,
    "input_cost_per_token_flex": 1e-06,
    "output_cost_per_token_flex": 3e-06,
    "cache_read_input_token_cost_flex": 4e-07,
}
BASE_COST: Final = 600 * 3e-06 + 400 * 1e-06 + 200 * 9e-06
BALANCED_COST: Final = 600 * 2e-06 + 400 * 7e-07 + 200 * 6e-06
FLEX_COST: Final = 600 * 1e-06 + 400 * 4e-07 + 200 * 3e-06


def _chat_reply(_request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": BACKEND,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                    "prompt_tokens_details": {"cached_tokens": CACHED_TOKENS},
                },
            }
        ).encode()
    )


def _responses_reply(_request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": BACKEND,
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": PROMPT_TOKENS,
                    "output_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                    "input_tokens_details": {"cached_tokens": CACHED_TOKENS},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        ).encode()
    )


def _messages_reply(_request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": BACKEND,
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS},
            }
        ).encode()
    )


def _upstream_body(request: Request) -> dict[str, JsonValue]:
    return object_value(json.loads(request.body))


def _sail_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model=f"sail/{BACKEND}", api_base=wire.url, api_key="synthetic-sail-key", **PRICES)


def _assert_spend_row(gateway: Gateway, response_id: str, expected_cost: float) -> None:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, custom_llm_provider, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
            "WHERE request_id = %s",
            (response_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["custom_llm_provider"] == "sail", rows
    assert rows[0]["prompt_tokens"] == PROMPT_TOKENS, rows
    assert rows[0]["completion_tokens"] == COMPLETION_TOKENS, rows
    assert float(rows[0]["spend"]) == pytest.approx(expected_cost, rel=1e-6), rows


@dataclass(frozen=True, slots=True)
class Served:
    upstream_body: dict[str, JsonValue]
    header_cost: float
    response_id: str


def _chat_through_sail(gateway: Gateway, extra: dict[str, JsonValue]) -> Served:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        model: Final = _sail_model(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "Reply with one word."}], **extra},
        )
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", "/chat/completions")]
    return Served(
        upstream_body=_upstream_body(received[0]),
        header_cost=float(response.headers["x-litellm-response-cost"]),
        response_id=string_value(object_value(response.json())["id"]),
    )


@pytest.mark.parametrize(
    ("service_tier", "completion_window", "expected_cost"),
    (
        ("balanced", "balanced", BALANCED_COST),
        ("flex", "flex", FLEX_COST),
        ("auto", None, BASE_COST),
    ),
    ids=("balanced", "flex", "auto"),
)
def test_chat_service_tier_is_sent_as_completion_window_and_billed_at_that_window(
    gateway: Gateway, service_tier: str, completion_window: str | None, expected_cost: float
) -> None:
    served: Final = _chat_through_sail(gateway, {"service_tier": service_tier})

    assert served.upstream_body == {
        "model": BACKEND,
        "messages": [{"role": "user", "content": "Reply with one word."}],
        **({} if completion_window is None else {"metadata": {"completion_window": completion_window}}),
    }, served.upstream_body
    assert served.header_cost == pytest.approx(expected_cost, rel=1e-6)
    _assert_spend_row(gateway, served.response_id, expected_cost)


@pytest.mark.parametrize("service_tier", ("bogus", 5), ids=("string", "integer"))
def test_chat_unknown_service_tier_under_drop_params_is_dropped_and_billed_at_the_base_rate(
    gateway: Gateway, service_tier: str | int
) -> None:
    served: Final = _chat_through_sail(gateway, {"service_tier": service_tier, "drop_params": True})

    assert served.upstream_body == {
        "model": BACKEND,
        "messages": [{"role": "user", "content": "Reply with one word."}],
    }, served.upstream_body
    assert served.header_cost == pytest.approx(BASE_COST, rel=1e-6)
    _assert_spend_row(gateway, served.response_id, BASE_COST)


def test_responses_caller_completion_window_reaches_sail_and_is_billed_at_that_window(gateway: Gateway) -> None:
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = _sail_model(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": "Reply with one word.", "metadata": {"completion_window": "balanced"}},
        )
        assert response.status_code == 200, response.text
        received: Final = wire.drain()

    assert [(request.method, request.target) for request in received] == [("POST", "/responses")]
    assert _upstream_body(received[0]) == {
        "model": BACKEND,
        "input": "Reply with one word.",
        "metadata": {"completion_window": "balanced"},
    }, received[0].body
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(BALANCED_COST, rel=1e-6), response.text
    _assert_spend_row(gateway, string_value(object_value(response.json())["id"]), BALANCED_COST)


def test_messages_call_to_sail_returns_an_anthropic_message(gateway: Gateway) -> None:
    with wire_server(_messages_reply) as wire, gateway.scenario() as scenario:
        model: Final = _sail_model(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": "Reply with one word."}]},
        )
        assert response.status_code == 200, response.text
        received: Final = wire.drain()

    assert [(request.method, request.target) for request in received] == [("POST", "/v1/messages")], received
    assert _upstream_body(received[0]) == {
        "model": BACKEND,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Reply with one word."}],
        "stream": False,
    }, received[0].body
    payload: Final = object_value(response.json())
    assert payload["type"] == "message" and payload["role"] == "assistant", response.text
    assert payload["content"] == [{"type": "text", "text": "OK"}], response.text
    assert object_value(payload["usage"])["output_tokens"] == COMPLETION_TOKENS, response.text
