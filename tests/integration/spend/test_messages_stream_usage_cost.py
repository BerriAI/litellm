import base64
import json
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

BEDROCK_MODEL: Final = "anthropic.claude-haiku-4-5-20251001-v1:0"
INPUT_TOKENS: Final = 30
CACHE_READ_TOKENS: Final = 900
CACHE_CREATION_TOKENS: Final = 400
OUTPUT_TOKENS: Final = 57
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CACHE_READ_RATE: Final = 0.0001
CACHE_CREATION_RATE: Final = 0.00125
STREAMED_USAGE: Final = TypeAdapter(dict[str, float])
EXPECTED_SPEND: Final = (
    INPUT_TOKENS * INPUT_RATE
    + CACHE_READ_TOKENS * CACHE_READ_RATE
    + CACHE_CREATION_TOKENS * CACHE_CREATION_RATE
    + OUTPUT_TOKENS * OUTPUT_RATE
)


def _invoke_chunk(payload: dict[str, JsonValue]) -> bytes:
    encoded: Final = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    return _aws_event_frame("chunk", {"bytes": encoded}, "", "")


def _stream(message_id: str) -> bytes:
    return (
        _invoke_chunk(
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
                    "usage": {
                        "input_tokens": INPUT_TOKENS,
                        "cache_read_input_tokens": CACHE_READ_TOKENS,
                        "cache_creation_input_tokens": CACHE_CREATION_TOKENS,
                        "output_tokens": 0,
                    },
                },
            }
        )
        + _invoke_chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        + _invoke_chunk(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "cached answer"}}
        )
        + _invoke_chunk({"type": "content_block_stop", "index": 0})
        + _invoke_chunk(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "input_tokens": INPUT_TOKENS,
                    "cache_read_input_tokens": CACHE_READ_TOKENS,
                    "cache_creation_input_tokens": CACHE_CREATION_TOKENS,
                    "output_tokens": OUTPUT_TOKENS,
                },
            }
        )
        + _invoke_chunk({"type": "message_stop"})
    )


def _proxy_config(directory: Path, model: str, upstream_url: str) -> Path:
    config: Final = directory / "streamed_usage_cost_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": f"bedrock/invoke/{BEDROCK_MODEL}",
                            "api_base": upstream_url,
                            "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                            "aws_secret_access_key": "scripted-secret",
                            "aws_region_name": "us-east-1",
                            "input_cost_per_token": INPUT_RATE,
                            "output_cost_per_token": OUTPUT_RATE,
                            "cache_read_input_token_cost": CACHE_READ_RATE,
                            "cache_creation_input_token_cost": CACHE_CREATION_RATE,
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "disable_spend_logs": False,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
                "litellm_settings": {"include_cost_in_streaming_usage": True},
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _data_events(body: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(json.loads(line.removeprefix("data:")) for line in body.splitlines() if line.startswith("data:"))


@pytest.mark.covers("spend.anthropic_messages_stream.streamed_usage_cost_equals_recorded_spend")
@pytest.mark.timeout(180)
def test_bedrock_messages_stream_usage_cost_matches_recorded_spend_with_custom_cache_rates(
    gateway: Gateway, tmp_path: Path
) -> None:
    message_id: Final = f"msg_{uuid.uuid4().hex}"
    model: Final = f"{BEDROCK_MODEL}-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.target == f"/model/{BEDROCK_MODEL}/invoke-with-response-stream", request.target
        assert json.loads(request.body)["messages"] == [{"role": "user", "content": "cached cost control"}], (
            request.body
        )
        return Reply(content_type="application/vnd.amazon.eventstream", chunks=(_stream(message_id),))

    with wire_server(respond) as wire:
        config: Final = _proxy_config(tmp_path, model, wire.url)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            response: Final = candidate.request(
                "POST",
                "/v1/messages",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "cached cost control"}],
                    "max_tokens": OUTPUT_TOKENS,
                    "stream": True,
                },
            )
            assert response.status_code == 200, response.text
            message_delta: Final = next(
                event for event in _data_events(response.text) if event["type"] == "message_delta"
            )
            streamed_usage: Final = STREAMED_USAGE.validate_python(message_delta["usage"])
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT status, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" '
                    "WHERE request_id=%s",
                    (message_id,),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
        assert rows[0]["status"] == "success", rows
        assert rows[0]["prompt_tokens"] == INPUT_TOKENS + CACHE_READ_TOKENS + CACHE_CREATION_TOKENS, rows
        assert rows[0]["completion_tokens"] == OUTPUT_TOKENS, rows
        recorded_spend: Final = float(str(rows[0]["spend"]))
        assert recorded_spend == pytest.approx(EXPECTED_SPEND), rows
        assert streamed_usage == {
            "input_tokens": INPUT_TOKENS,
            "cache_read_input_tokens": CACHE_READ_TOKENS,
            "cache_creation_input_tokens": CACHE_CREATION_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "cost": pytest.approx(recorded_spend),
        }, (streamed_usage, rows, response.text)
        assert len(wire.drain()) == 1
