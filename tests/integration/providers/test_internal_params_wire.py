import asyncio
import base64
import json
import os
import struct
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import litellm
import pytest
from integration._support.upstream import INTERNAL_FIELDS
from integration._support.wire import Reply, Request, wire_server
from tests._support.stream_chunk_size import keys_at_every_depth, record_litellm_params, recorded_stream_chunk_size

TEXT: Final = "wire control"
OPENAI_RESPONSE: Final = {
    "id": "chatcmpl-wire",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4.1-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": TEXT}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
}
ANTHROPIC_RESPONSE: Final = {
    "id": "msg_wire",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": TEXT}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 4},
}
GEMINI_RESPONSE: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": TEXT}]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
}
CONVERSE_RESPONSE: Final = {
    "output": {"message": {"role": "assistant", "content": [{"text": TEXT}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
    "metrics": {"latencyMs": 1},
}
OPENAI_STREAM_CHUNKS: Final = (
    {
        "id": "chatcmpl-wire",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": TEXT}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-wire",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
)
ANTHROPIC_STREAM_EVENTS: Final = (
    {
        "type": "message_start",
        "message": {
            "id": "msg_wire",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": TEXT}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
    {"type": "message_stop"},
)
GEMINI_STREAM_CHUNKS: Final = (
    {"candidates": [{"content": {"role": "model", "parts": [{"text": TEXT}]}, "index": 0}]},
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
    },
)
CONVERSE_STREAM_EVENTS: Final = (
    ("contentBlockDelta", {"delta": {"text": TEXT}, "contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    ("metadata", {"usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14}, "metrics": {"latencyMs": 1}}),
)
NON_STREAM_BODIES: Final = {
    "openai": OPENAI_RESPONSE,
    "azure": OPENAI_RESPONSE,
    "anthropic": ANTHROPIC_RESPONSE,
    "gemini": GEMINI_RESPONSE,
    "converse": CONVERSE_RESPONSE,
    "invoke": ANTHROPIC_RESPONSE,
}
PROVIDERS: Final = ("openai", "azure", "anthropic", "gemini", "converse", "invoke")


def _aws_string_header(name: str, value: str) -> bytes:
    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return struct.pack("!B", len(name_bytes)) + name_bytes + b"\x07" + struct.pack("!H", len(value_bytes)) + value_bytes


def _aws_event_frame(event_type: str, payload: Mapping[str, object]) -> bytes:
    body: Final = json.dumps(payload, separators=(",", ":")).encode()
    headers: Final = (
        _aws_string_header(":event-type", event_type)
        + _aws_string_header(":content-type", "application/json")
        + _aws_string_header(":message-type", "event")
    )
    prelude: Final = struct.pack("!II", 12 + len(headers) + len(body) + 4, len(headers))
    message: Final = prelude + struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF) + headers + body
    return message + struct.pack("!I", zlib.crc32(message) & 0xFFFFFFFF)


def _sse_reply(frames: tuple[bytes, ...]) -> Reply:
    return Reply(chunks=frames, content_type="text/event-stream")


def _stream_reply(provider: str) -> Reply:
    match provider:
        case "openai" | "azure":
            return _sse_reply(
                tuple(
                    f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode() for chunk in OPENAI_STREAM_CHUNKS
                )
                + (b"data: [DONE]\n\n",)
            )
        case "anthropic":
            return _sse_reply(
                tuple(
                    f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
                    for event in ANTHROPIC_STREAM_EVENTS
                )
            )
        case "gemini":
            return _sse_reply(
                tuple(
                    f"data: {json.dumps(chunk, separators=(',', ':'))}\r\n\r\n".encode()
                    for chunk in GEMINI_STREAM_CHUNKS
                )
            )
        case "converse":
            return Reply(
                chunks=tuple(_aws_event_frame(event_type, payload) for event_type, payload in CONVERSE_STREAM_EVENTS),
                content_type="application/vnd.amazon.eventstream",
            )
        case "invoke":
            return Reply(
                chunks=tuple(
                    _aws_event_frame(
                        "chunk", {"bytes": base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode()}
                    )
                    for event in ANTHROPIC_STREAM_EVENTS
                ),
                content_type="application/vnd.amazon.eventstream",
            )


def _request_parameters(provider: str, wire_url: str) -> dict[str, object]:
    common: Final = {"messages": [{"role": "user", "content": "synthetic chunk control"}]}
    match provider:
        case "openai":
            return {**common, "model": "openai/gpt-4.1-mini", "api_key": "synthetic-openai-key", "api_base": wire_url}
        case "azure":
            return {
                **common,
                "model": "azure/gpt-4.1-mini",
                "api_key": "synthetic-azure-key",
                "api_base": wire_url,
                "api_version": "2025-01-01-preview",
            }
        case "anthropic":
            return {
                **common,
                "model": "anthropic/claude-sonnet-4-5",
                "api_key": "synthetic-anthropic-key",
                "api_base": wire_url,
            }
        case "gemini":
            return {
                **common,
                "model": "gemini/gemini-2.5-flash",
                "api_key": "synthetic-gemini-key",
                "api_base": wire_url,
            }
        case "converse":
            return {
                **common,
                "model": "bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0",
                "aws_access_key_id": "fake",
                "aws_secret_access_key": "fake",
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire_url,
            }
        case "invoke":
            return {
                **common,
                "model": "bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
                "aws_access_key_id": "fake",
                "aws_secret_access_key": "fake",
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire_url,
            }


def _expected_target(provider: str, streaming: bool) -> str:
    match provider:
        case "openai":
            return "/chat/completions"
        case "azure":
            return "/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview"
        case "anthropic":
            return "/v1/messages"
        case "gemini":
            return ":streamGenerateContent" if streaming else ":generateContent"
        case "converse":
            return "/converse-stream" if streaming else "/converse"
        case "invoke":
            return "/invoke-with-response-stream" if streaming else "/invoke"


def _at(value: object, *path: str) -> object:
    if not path:
        return value
    assert isinstance(value, Mapping)
    return _at(value[path[0]], *path[1:])


def _custom_key(body: Mapping[str, object], provider: str) -> object:
    match provider:
        case "anthropic":
            return _at(body, "extra_body", "custom_provider_key")
        case "converse":
            return _at(body, "additionalModelRequestFields", "extra_body", "custom_provider_key")
    return _at(body, "custom_provider_key")


def _peer(provider: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body) if request.body else {}
        streaming: Final = (
            (isinstance(body, dict) and body.get("stream") is True)
            or "streamGenerateContent" in request.target
            or request.target.endswith(("-stream",))
        )
        expected: Final = _expected_target(provider, streaming)
        assert expected in request.target, f"{provider}: expected {expected} in {request.target}"
        return _stream_reply(provider) if streaming else Reply(body=json.dumps(NON_STREAM_BODIES[provider]).encode())

    return respond


@pytest.fixture
def provider_wire_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty: Final = tmp_path / "empty-aws-config"
    empty.write_text("")
    for name in tuple(name for name in os.environ if name.startswith("AWS_")):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "AWS_CONFIG_FILE": str(empty),
        "AWS_SHARED_CREDENTIALS_FILE": str(empty),
        "AWS_EC2_METADATA_DISABLED": "true",
        "LITELLM_RUST": "false",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_internal_params_never_reach_provider_body(
    monkeypatch: pytest.MonkeyPatch,
    provider_wire_environment: None,
    provider: str,
    asynchronous: bool,
    stream: bool,
) -> None:
    recorder: Final = record_litellm_params(monkeypatch)
    with wire_server(_peer(provider)) as wire:
        parameters: Final = {
            **_request_parameters(provider, wire.url),
            "stream": stream,
            "stream_chunk_size": 64,
            "_litellm_undeclared_sentinel": "internal",
            "extra_body": {"custom_provider_key": 1},
            "max_tokens": 16,
            "timeout": 5,
            "num_retries": 0,
        }
        result: Final = (
            await litellm.acompletion(**parameters)
            if asynchronous
            else await asyncio.to_thread(litellm.completion, **parameters)
        )
        if stream:
            chunks: Final = [chunk async for chunk in result] if asynchronous else [chunk for chunk in result]
            text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
            assert text == TEXT
        else:
            assert result.choices[0].message.content == TEXT
        requests: Final = wire.drain()
        assert len(requests) == 1
        assert len(recorder.seen) == 1
        assert recorded_stream_chunk_size(recorder.seen[0]) == 64
        body: Final = json.loads(requests[0].body)
        keys: Final = keys_at_every_depth(body)
        assert "stream_chunk_size" not in keys
        assert not INTERNAL_FIELDS.intersection(keys)
        assert not frozenset(key for key in keys if key.startswith("_litellm_")), keys
        assert _custom_key(body, provider) == 1
