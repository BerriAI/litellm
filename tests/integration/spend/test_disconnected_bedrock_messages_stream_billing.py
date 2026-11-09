import base64
import json
import uuid
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

BEDROCK_MODEL: Final = "anthropic.claude-haiku-4-5-20251001-v1:0"
INPUT_TOKENS: Final = 30
FULL_OUTPUT_TOKENS: Final = 412
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002


def _invoke_chunk(payload: dict[str, JsonValue]) -> bytes:
    encoded: Final = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    return _aws_event_frame("chunk", {"bytes": encoded}, "", "")


def _message_start(message_id: str) -> bytes:
    return _invoke_chunk(
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": BEDROCK_MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 0},
            },
        }
    ) + _invoke_chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})


def _text_delta(text: str) -> bytes:
    return _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})


def _terminal_usage() -> bytes:
    return (
        _invoke_chunk({"type": "content_block_stop", "index": 0})
        + _invoke_chunk(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": FULL_OUTPUT_TOKENS},
            }
        )
        + _invoke_chunk({"type": "message_stop"})
    )


@pytest.mark.covers("spend.anthropic_messages_stream.client_disconnect_bills_terminal_bedrock_usage")
@pytest.mark.timeout(120)
def test_client_disconnect_mid_bedrock_messages_stream_still_bills_terminal_usage(gateway: Gateway) -> None:
    message_id: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.target == f"/model/{BEDROCK_MODEL}/invoke-with-response-stream", request.target
        assert json.loads(request.body)["messages"] == [{"role": "user", "content": "disconnect control"}], request.body
        return Reply(
            content_type="application/vnd.amazon.eventstream",
            chunks=(
                _message_start(message_id) + _text_delta("first"),
                _text_delta("second"),
                _text_delta("third"),
                _terminal_usage(),
            ),
            pause_between_chunks=0.5,
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{BEDROCK_MODEL}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
        )
        key: Final = scenario.key(models=[model])
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "disconnect control"}],
                "max_tokens": FULL_OUTPUT_TOKENS,
                "stream": True,
            },
            headers={"Authorization": f"Bearer {key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            first_event: Final = next(line for line in response.iter_lines() if line.startswith("data:"))
            assert json.loads(first_event.removeprefix("data:"))["type"] == "message_start", first_event

        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, status, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" '
                "WHERE api_key=%s",
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["request_id"] == message_id, rows
        assert rows[0]["status"] == "success", rows
        assert rows[0]["prompt_tokens"] == INPUT_TOKENS, rows
        assert rows[0]["completion_tokens"] == FULL_OUTPUT_TOKENS, rows
        assert float(str(rows[0]["spend"])) == pytest.approx(
            INPUT_TOKENS * INPUT_RATE + FULL_OUTPUT_TOKENS * OUTPUT_RATE
        ), rows
        assert len(wire.drain()) == 1
