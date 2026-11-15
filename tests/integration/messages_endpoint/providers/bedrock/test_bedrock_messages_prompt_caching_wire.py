import base64
import json
import uuid
from typing import Final

import anthropic
import httpx
import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL_ID: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_TOKEN: Final = "synthetic-bedrock-bearer"
_PROMPT: Final = "Reply with exactly the word PONG"
_INPUT_TOKENS: Final = 10
_OUTPUT_TOKENS: Final = 4
_CACHE_READ_TOKENS: Final = 900
_CACHE_WRITE_TOKENS: Final = 300
_INPUT_RATE: Final = 0.001
_OUTPUT_RATE: Final = 0.002
_CACHE_READ_RATE: Final = 0.0001
_CACHE_WRITE_RATE: Final = 0.0015
_EXPECTED_SPEND: Final = (
    _INPUT_TOKENS * _INPUT_RATE
    + _CACHE_READ_TOKENS * _CACHE_READ_RATE
    + _CACHE_WRITE_TOKENS * _CACHE_WRITE_RATE
    + _OUTPUT_TOKENS * _OUTPUT_RATE
)
_PROMPT_TOKENS: Final = _INPUT_TOKENS + _CACHE_READ_TOKENS + _CACHE_WRITE_TOKENS
_USAGE: Final = {
    "input_tokens": _INPUT_TOKENS,
    "output_tokens": _OUTPUT_TOKENS,
    "cache_read_input_tokens": _CACHE_READ_TOKENS,
    "cache_creation_input_tokens": _CACHE_WRITE_TOKENS,
}
_USER_CONTENT: Final = [{"type": "text", "text": _PROMPT, "cache_control": cc.CACHE}]
_CACHED_SYSTEM_BLOCKS: Final = [block for block in cc.system_blocks() if "cache_control" in block]


def _assert_claude_code_breakpoints_reach_bedrock(request: Request, target: str) -> dict[str, JsonValue]:
    assert request.method == "POST" and request.target == target, request.target
    assert request.headers["authorization"] == f"Bearer {_TOKEN}"
    body: Final = cc.JSON_OBJECT.validate_json(request.body)
    assert body["anthropic_version"] == "bedrock-2023-05-31", body
    assert body["system"] == _CACHED_SYSTEM_BLOCKS, f"billing-header block dropped, breakpoints kept: {body}"
    assert body["messages"] == [{"role": "user", "content": _USER_CONTENT}], body
    return body


def _invoke_chunk(event: dict[str, JsonValue]) -> bytes:
    return _aws_event_frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}, "sc", "u")


def _stream_frames(message_id: str) -> tuple[bytes, ...]:
    return (
        _invoke_chunk(
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-sonnet-4-5-20250929",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {**_USAGE, "output_tokens": 1},
                },
            }
        ),
        _invoke_chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "PONG"}}),
        _invoke_chunk({"type": "content_block_stop", "index": 0}),
        _invoke_chunk(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": _OUTPUT_TOKENS},
            }
        ),
        _invoke_chunk({"type": "message_stop"}),
    )


def _spend_row(message_id: str) -> dict[str, object]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (message_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _client(gateway: Gateway, transport: httpx.Client) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=gateway.key, base_url=str(gateway.client.base_url), max_retries=0, http_client=transport
    )


def test_bedrock_claude_cache_control_reaches_invoke_and_cache_tokens_are_reported_and_charged(
    gateway: Gateway,
) -> None:
    message_id: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        _assert_claude_code_breakpoints_reach_bedrock(request, f"/model/{_MODEL_ID}/invoke")
        return Reply(
            body=cc.message_reply(
                message_id, "claude-sonnet-4-5-20250929", ({"type": "text", "text": "PONG"},), dict(_USAGE)
            )
        )

    with (
        wire_server(respond) as wire,
        gateway.scenario() as scenario,
        httpx.Client(timeout=15, trust_env=False) as transport,
    ):
        model: Final = scenario.model(
            model=f"bedrock/invoke/{_MODEL_ID}",
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            api_base=wire.url,
            input_cost_per_token=_INPUT_RATE,
            output_cost_per_token=_OUTPUT_RATE,
            cache_read_input_token_cost=_CACHE_READ_RATE,
            cache_creation_input_token_cost=_CACHE_WRITE_RATE,
        )
        final: Final = _client(gateway, transport).messages.create(
            model=model,
            max_tokens=32,
            system=list(cc.system_blocks()),
            messages=[{"role": "user", "content": _USER_CONTENT}],
        )
        assert len(wire.drain()) == 1
        assert final.id == message_id, final
        assert [(block.type, block.text) for block in final.content] == [("text", "PONG")], final
        assert final.usage.model_dump(exclude_none=True, include=set(_USAGE)) == _USAGE, final.usage
        row: Final = _spend_row(message_id)

    assert float(str(row["spend"])) == pytest.approx(_EXPECTED_SPEND), row
    assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_TOKENS, _OUTPUT_TOKENS), row


def test_bedrock_claude_stream_cache_tokens_from_message_start_are_reported_and_charged(gateway: Gateway) -> None:
    message_id: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        body: Final = _assert_claude_code_breakpoints_reach_bedrock(
            request, f"/model/{_MODEL_ID}/invoke-with-response-stream"
        )
        assert "stream" not in body, body
        return Reply(chunks=_stream_frames(message_id), content_type="application/vnd.amazon.eventstream")

    with (
        wire_server(respond) as wire,
        gateway.scenario() as scenario,
        httpx.Client(timeout=15, trust_env=False) as transport,
    ):
        model: Final = scenario.model(
            model=f"bedrock/invoke/{_MODEL_ID}",
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            api_base=wire.url,
            input_cost_per_token=_INPUT_RATE,
            output_cost_per_token=_OUTPUT_RATE,
            cache_read_input_token_cost=_CACHE_READ_RATE,
            cache_creation_input_token_cost=_CACHE_WRITE_RATE,
        )
        with _client(gateway, transport).messages.stream(
            model=model,
            max_tokens=32,
            system=list(cc.system_blocks()),
            messages=[{"role": "user", "content": _USER_CONTENT}],
        ) as stream:
            deltas: Final = tuple(stream.text_stream)
            final: Final = stream.get_final_message()
        assert len(wire.drain()) == 1
        assert deltas == ("PONG",)
        assert final.id == message_id, final
        assert final.usage.model_dump(exclude_none=True, include=set(_USAGE)) == _USAGE, final.usage
        row: Final = _spend_row(message_id)

    assert float(str(row["spend"])) == pytest.approx(_EXPECTED_SPEND), row
    assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_TOKENS, _OUTPUT_TOKENS), row
