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
  - `converse-stream` / `invoke-with-response-stream` are not real event streams here; use
    non-streaming paths first for local sanity checks.
  - Use real (or any non-empty) AWS creds in the environment of the **proxy**; signing still runs.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock Bedrock runtime (pass-through test target)")


# Minimal structure compatible with Converse: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Converse.html
def _converse_response_body() -> Dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "mock: ok from mock_bedrock_passthrough_target.py"}],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
