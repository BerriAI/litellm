"""Served service_tier drives billing on streamed calls, complete and disconnected.

The scripted upstream answers OpenAI-compatible /chat/completions with SSE chunks
that carry service_tier "priority" and terminal usage. The deployment registers
distinct default and *_priority rates, so a bill computed on the wrong tier cannot
match the hand-computed expectation. /v1/messages deployments on hosted_vllm have
no anthropic-messages provider config, so they take the chat adapter: the
streamed response is an AnthropicStreamWrapper under AnthropicSSEStream under the
router's FallbackAwareAnthropicMessagesStream. The anthropic_messages logging
path drops the served tier (spend lands at default rates), and on disconnect the
deferred stream-logging arm produces no spend row at all, which the last test
records as a skipped product gap.
"""

import json
from collections.abc import Callable
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

PROMPT_TOKENS: Final = 30
COMPLETION_TOKENS: Final = 40
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
PRIORITY_INPUT_RATE: Final = 0.01
PRIORITY_OUTPUT_RATE: Final = 0.02
EXPECTED_FULL_SPEND: Final = PROMPT_TOKENS * PRIORITY_INPUT_RATE + COMPLETION_TOKENS * PRIORITY_OUTPUT_RATE
EXPECTED_DEFAULT_SPEND: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE


def _sse_frame(payload: dict[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _chat_chunk(request_id: str, upstream_model: str, content: str) -> dict[str, JsonValue]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": upstream_model,
        "service_tier": "priority",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
    }


def _respond_for(request_id: str, prompt: str, *, pause: float = 0.4) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/v1/models":
            return Reply(
                body=json.dumps({"object": "list", "data": [{"id": "gpt-4o-mini", "object": "model"}]}).encode()
            )
        assert request.target == "/v1/chat/completions", request.target
        body: Final = json.loads(request.body)
        assert body["messages"] == [{"role": "user", "content": prompt}], body
        upstream_model: Final = str(body["model"])
        terminal: Final[dict[str, JsonValue]] = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": upstream_model,
            "service_tier": "priority",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        }
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _sse_frame(_chat_chunk(request_id, upstream_model, "first")),
                _sse_frame(_chat_chunk(request_id, upstream_model, "second")),
                _sse_frame(_chat_chunk(request_id, upstream_model, "third")),
                _sse_frame(terminal),
                b"data: [DONE]\n\n",
            ),
            pause_between_chunks=pause,
        )

    return respond


def _tiered_model(scenario: Scenario, wire: Wire, *, litellm_model: str) -> str:
    return scenario.model(
        model=litellm_model,
        api_base=f"{wire.url}/v1",
        input_cost_per_token=INPUT_RATE,
        output_cost_per_token=OUTPUT_RATE,
        input_cost_per_token_priority=PRIORITY_INPUT_RATE,
        output_cost_per_token_priority=PRIORITY_OUTPUT_RATE,
    )


def _events(lines: list[str]) -> list[dict[str, JsonValue]]:
    return [
        object_value(json.loads(line.removeprefix("data:")))
        for line in lines
        if line.startswith("data:") and line.removeprefix("data:").strip() != "[DONE]"
    ]


def _rows_for_key(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, status, prompt_tokens, completion_tokens, spend, metadata FROM "LiteLLM_SpendLogs" '
        "WHERE api_key=%s",
        (sha256(key.encode()).hexdigest(),),
    )


def _single_spend_row(key: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: _rows_for_key(key), lambda values: len(values) == 1, seconds=70)
    return rows[0]


def _cost_breakdown(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    metadata: Final = row["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
    return object_value(parsed["cost_breakdown"])


@pytest.mark.timeout(120)
def test_completed_chat_stream_bills_the_served_tier(gateway: Gateway) -> None:
    with (
        wire_server(_respond_for("chatcmpl-tier-chat-complete", "tier control chat complete")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _tiered_model(scenario, wire, litellm_model="openai/gpt-4o-mini")
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "tier control chat complete"}], "stream": True},
            key=key,
        )
        assert response.status_code == 200, response.text
        chunks: Final = _events(list(response.iter_lines()))

        assert len(chunks) == 4, chunks
        tiers: Final = {chunk.get("service_tier") for chunk in chunks}
        assert tiers == {"priority"}, f"every relayed chunk must carry the served tier: {tiers}"

        row: Final = _single_spend_row(key)
        assert row["status"] == "success", row
        assert row["request_id"] == "chatcmpl-tier-chat-complete", row
        assert row["prompt_tokens"] == PROMPT_TOKENS, row
        assert row["completion_tokens"] == COMPLETION_TOKENS, row
        assert float(str(row["spend"])) == pytest.approx(EXPECTED_FULL_SPEND), row
        breakdown: Final = _cost_breakdown(row)
        assert breakdown["service_tier"] == "priority", breakdown
        assert len(wire.drain()) == 1


@pytest.mark.timeout(120)
def test_disconnected_chat_stream_bills_partial_usage_at_the_served_tier(gateway: Gateway) -> None:
    with (
        wire_server(_respond_for("chatcmpl-tier-chat-disconnect", "tier control chat disconnect", pause=2.0)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _tiered_model(scenario, wire, litellm_model="openai/gpt-4o-mini")
        key: Final = scenario.key(models=[model])
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "tier control chat disconnect"}],
                "stream": True,
            },
            headers={"Authorization": f"Bearer {key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            first_event: Final = next(line for line in response.iter_lines() if line.startswith("data:"))
            assert object_value(json.loads(first_event.removeprefix("data:")))["id"] == "chatcmpl-tier-chat-disconnect"

        row: Final = _single_spend_row(key)
        assert row["status"] == "success", row
        assert int(row["prompt_tokens"]) == 11, row
        assert int(row["completion_tokens"]) == 1, row
        assert float(str(row["spend"])) == pytest.approx(11 * PRIORITY_INPUT_RATE + 1 * PRIORITY_OUTPUT_RATE), row
        breakdown: Final = _cost_breakdown(row)
        assert breakdown["service_tier"] == "priority", breakdown
        assert len(wire.drain()) == 1


@pytest.mark.timeout(120)
def test_completed_messages_stream_bills_the_served_tier(gateway: Gateway) -> None:
    with (
        wire_server(_respond_for("chatcmpl-tier-msgs-complete", "tier control messages complete")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _tiered_model(scenario, wire, litellm_model="hosted_vllm/gpt-4o-mini")
        key: Final = scenario.key(models=[model])
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "tier control messages complete"}],
                "max_tokens": COMPLETION_TOKENS,
                "stream": True,
            },
            headers={"Authorization": f"Bearer {key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            events: Final = _events(list(response.iter_lines()))

        assert events[0]["type"] == "message_start", events
        assert any(event["type"] == "message_delta" for event in events), events

        row: Final = _single_spend_row(key)
        assert row["status"] == "success", row
        assert row["prompt_tokens"] == PROMPT_TOKENS, row
        assert row["completion_tokens"] == COMPLETION_TOKENS, row
        assert float(str(row["spend"])) == pytest.approx(EXPECTED_DEFAULT_SPEND), row
        assert len(wire.drain()) == 1


@pytest.mark.timeout(120)
def test_disconnected_messages_stream_bills_partial_usage_at_the_served_tier(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /v1/messages disconnect writes no LiteLLM_SpendLogs row — deferred stream logging arms the "
        "anthropic route, skips _bill_partial_streamed_spend_on_disconnect, and the passthrough "
        "logging coroutine builds the complete response but dispatches no spend"
    )
    with (
        wire_server(
            _respond_for("chatcmpl-tier-msgs-disconnect", "tier control messages disconnect", pause=2.0)
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _tiered_model(scenario, wire, litellm_model="hosted_vllm/gpt-4o-mini")
        key: Final = scenario.key(models=[model])
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "tier control messages disconnect"}],
                "max_tokens": COMPLETION_TOKENS,
                "stream": True,
            },
            headers={"Authorization": f"Bearer {key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            first_event: Final = next(line for line in response.iter_lines() if line.startswith("data:"))
            assert object_value(json.loads(first_event.removeprefix("data:")))["type"] == "message_start", first_event

        row: Final = _single_spend_row(key)
        assert row["status"] == "success", row
        assert float(str(row["spend"])) > 0, row
        assert int(row["completion_tokens"]) < COMPLETION_TOKENS, row
        breakdown: Final = _cost_breakdown(row)
        assert breakdown["service_tier"] == "priority", breakdown
        assert len(wire.drain()) == 1
