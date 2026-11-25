import json
import uuid
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

ANTHROPIC_MODEL: Final = "claude-haiku-4-5"
UNCACHED_INPUT_TOKENS: Final = 30
CACHE_READ_TOKENS: Final = 2000
CACHE_WRITE_TOKENS: Final = 100
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CACHE_READ_RATE: Final = 0.0001
CACHE_WRITE_RATE: Final = 0.004


def _anthropic_event(payload: dict[str, JsonValue]) -> bytes:
    return f"event: {payload['type']}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _message_start(message_id: str) -> bytes:
    return _anthropic_event(
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": ANTHROPIC_MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": UNCACHED_INPUT_TOKENS,
                    "cache_read_input_tokens": CACHE_READ_TOKENS,
                    "cache_creation_input_tokens": CACHE_WRITE_TOKENS,
                    "output_tokens": 1,
                },
            },
        }
    ) + _anthropic_event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})


def _text_delta(text: str) -> bytes:
    return _anthropic_event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})


def _terminal_usage() -> bytes:
    return (
        _anthropic_event({"type": "content_block_stop", "index": 0})
        + _anthropic_event({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 400}})
        + _anthropic_event({"type": "message_stop"})
    )


@pytest.mark.timeout(120)
def test_a_responses_stream_cut_mid_answer_bills_the_input_and_cache_tokens_anthropic_reported(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.target.endswith("/v1/messages"), request.target
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _message_start(f"msg_{uuid.uuid4().hex}") + _text_delta("first"),
                _text_delta(" second"),
                _text_delta(" third"),
                _terminal_usage(),
            ),
            pause_between_chunks=1.0,
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{ANTHROPIC_MODEL}",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
            cache_read_input_token_cost=CACHE_READ_RATE,
            cache_creation_input_token_cost=CACHE_WRITE_RATE,
        )
        key: Final = scenario.key(models=[model])
        with gateway.client.stream(
            "POST",
            "/v1/responses",
            json={"model": model, "input": "disconnect control", "stream": True},
            headers={"Authorization": f"Bearer {key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            first_delta: Final = next(
                line for line in response.iter_lines() if line.startswith("data:") and "output_text.delta" in line
            )
            assert json.loads(first_delta.removeprefix("data:"))["delta"] == "first", first_delta

        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, prompt_tokens, completion_tokens, spend, metadata FROM "LiteLLM_SpendLogs" '
                "WHERE api_key=%s",
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "success", row
        assert row["prompt_tokens"] == UNCACHED_INPUT_TOKENS + CACHE_READ_TOKENS + CACHE_WRITE_TOKENS, row
        assert float(str(row["spend"])) == pytest.approx(
            UNCACHED_INPUT_TOKENS * INPUT_RATE
            + CACHE_READ_TOKENS * CACHE_READ_RATE
            + CACHE_WRITE_TOKENS * CACHE_WRITE_RATE
            + int(str(row["completion_tokens"])) * OUTPUT_RATE
        ), row
        metadata: Final = row["metadata"]
        parsed: Final = object_value(json.loads(metadata) if isinstance(metadata, str) else metadata)
        error_information: Final = object_value(parsed["error_information"])
        assert (error_information["error_code"], error_information["error_class"]) == ("499", "ClientDisconnected")
        assert len(wire.drain()) == 1
