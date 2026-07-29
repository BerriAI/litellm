#!/usr/bin/env python3
"""
Minimal HTTP target for testing LiteLLM **Bedrock pass-through** (`/bedrock/...` on the proxy).

What it does
  - Serves a tiny Converse-shaped JSON (and optional invoke-shaped) response so the proxy can
    complete a round trip without calling AWS.
  - Does **not** verify SigV4 (Bedrock does); any Authorization header is accepted.

How to run
  uv run python scripts/mock_bedrock_passthrough_target.py --host 127.0.0.1 --port 9999

Wire LiteLLM to this host (use **one** of these patterns):

  1) model_list (recommended) — set the Bedrock runtime base to the mock:

     model_list:
       - model_name: mock-bedrock-claude
         litellm_params:
           model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
           custom_llm_provider: bedrock
           aws_region_name: us-west-2
           api_base: "http://127.0.0.1:9999"

  2) Environment (see litellm BaseAWSLLM.get_runtime_endpoint)::

       export AWS_BEDROCK_RUNTIME_ENDPOINT="http://127.0.0.1:9999"

Then call the proxy, e.g. (model_name must match config)::

  curl -sS -X POST "http://127.0.0.1:4000/bedrock/model/mock-bedrock-claude/converse" \
    -H "Authorization: Bearer $LITELLM_KEY" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":[{"text":"hi"}]}]}'

The proxy will forward to: {api_base}/model/<resolved model id>/converse (SigV4-signed).
This mock implements POST .../converse and returns a minimal valid Converse response.

Notes
  - `invoke-with-response-stream` returns a real **binary** AWS event stream
    (`application/vnd.amazon.eventstream`) with Anthropic-style JSON payloads inside each
    `PayloadPart`, matching Bedrock's InvokeModelWithResponseStream wire format. See
    https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html
    and https://docs.aws.amazon.com/awstreams/latest/devguide/message-formats.html
  - `converse-stream` is still JSON-only placeholder (different inner event shapes).
  - Use real (or any non-empty) AWS creds in the environment of the **proxy**; signing still runs.
"""
from __future__ import annotations

import argparse
import base64
import json
from binascii import crc32
from struct import pack
from typing import Any, Dict, Iterator, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

app = FastAPI(title="Mock Bedrock runtime (pass-through test target)")


# Minimal structure compatible with Converse: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Converse.html
def _converse_response_body() -> Dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "mock: ok from mock_bedrock_passthrough_target.py"}
                ],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 1,
            "outputTokens": 2,
            "totalTokens": 3,
        },
    }


# Minimal invoke (Anthropic messages on bedrock) style — adjust if you test /invoke
def _invoke_response_body() -> Dict[str, Any]:
    return {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "mock invoke response"}],
        "model": "mock",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }


def _encode_event_stream_message(headers: Dict[str, str], payload: bytes) -> bytes:
    """Single AWS binary event-stream frame (same layout botocore's ``EventStreamBuffer`` parses)."""
    header_blob = b""
    for name, value in headers.items():
        nb = name.encode("utf-8")
        vb = value.encode("utf-8")
        header_blob += bytes([len(nb)]) + nb + bytes([7]) + pack("!H", len(vb)) + vb
    headers_length = len(header_blob)
    payload_length = len(payload)
    total_length = 12 + headers_length + payload_length + 4
    prelude_wo_crc = pack("!II", total_length, headers_length)
    prelude_crc_val = crc32(prelude_wo_crc) & 0xFFFFFFFF
    prelude = prelude_wo_crc + pack("!I", prelude_crc_val)
    wo_msg_crc = prelude + header_blob + payload
    msg_crc_val = crc32(wo_msg_crc[8:], prelude_crc_val) & 0xFFFFFFFF
    return wo_msg_crc + pack("!I", msg_crc_val)


def _bedrock_payload_part(inner_event: Dict[str, Any]) -> bytes:
    """Outer JSON expected by bedrock-runtime ``ResponseStream`` / ``PayloadPart``."""
    inner_bytes = json.dumps(inner_event, separators=(",", ":")).encode("utf-8")
    outer = {
        "chunk": {
            "bytes": base64.b64encode(inner_bytes).decode("ascii"),
        }
    }
    return json.dumps(outer, separators=(",", ":")).encode("utf-8")


def _anthropic_invoke_stream_events(
    model_id: str, assistant_text: str
) -> List[Dict[str, Any]]:
    """
    Minimal Anthropic Messages stream events as returned inside Bedrock stream chunks.
    Mirrors the sequence Amazon emits for Claude on ``invoke-with-response-stream``.
    """
    msg_id = "msg_mock_bedrock_stream"
    input_tokens = 3
    output_tokens = max(1, len(assistant_text) // 4)
    events: List[Dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "model": model_id,
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ]
    # Split text into small deltas so downstream streaming behavior is visible.
    step = 24
    for i in range(0, len(assistant_text), step):
        events.append(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": assistant_text[i : i + step],
                },
            }
        )
    events.append({"type": "content_block_stop", "index": 0})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    )
    events.append(
        {
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": input_tokens,
                "outputTokenCount": output_tokens,
                "invocationLatency": 42,
                "firstByteLatency": 10,
            },
        }
    )
    return events


def _iter_invoke_with_response_stream(model_id: str) -> Iterator[bytes]:
    text = (
        "mock streaming: ok from scripts/mock_bedrock_passthrough_target.py "
        "(invoke-with-response-stream)."
    )
    headers = {
        ":event-type": "chunk",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    for ev in _anthropic_invoke_stream_events(model_id, text):
        yield _encode_event_stream_message(headers, _bedrock_payload_part(ev))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/model/{model_path:path}/converse")
async def converse(model_path: str, request: Request) -> JSONResponse:
    # Optional: log body for debugging
    _ = await request.body()
    return JSONResponse(content=_converse_response_body())


@app.post("/model/{model_path:path}/converse-stream")
async def converse_stream(model_path: str, request: Request) -> JSONResponse:
    """
    Not a real AWS event stream — returns JSON for quick smoke tests only.
    """
    _ = await request.body()
    return JSONResponse(
        content={
            "note": "This mock does not implement application/vnd.amazon.eventstream; use /converse for basic tests."
        }
    )


@app.post("/model/{model_path:path}/invoke")
async def invoke(model_path: str, request: Request) -> JSONResponse:
    _ = await request.body()
    return JSONResponse(content=_invoke_response_body())


@app.post("/model/{model_path:path}/invoke-with-response-stream")
async def invoke_with_response_stream(
    model_path: str, request: Request
) -> StreamingResponse:
    """
    Binary ``application/vnd.amazon.eventstream`` body compatible with boto3/botocore
    ``InvokeModelWithResponseStream`` / LiteLLM's Bedrock invoke streaming path.
    """
    _ = await request.body()
    return StreamingResponse(
        _iter_invoke_with_response_stream(model_id=model_path),
        media_type="application/vnd.amazon.eventstream",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
