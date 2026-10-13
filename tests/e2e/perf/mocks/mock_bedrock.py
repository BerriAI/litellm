"""The Bedrock-shaped perf mock: the bedrock-runtime routes litellm uses for chat, Converse and
ConverseStream plus Invoke and InvokeWithResponseStream with Anthropic-native bodies. Streams are
framed as application/vnd.amazon.eventstream, which the gateway decodes with botocore. SigV4
headers are accepted without being checked; the model id in the path carries the knobs."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from mock_anthropic import anthropic_events, anthropic_message
from mock_common import Decision, RequestLog, mount_control, parse_json, sleep_for
from mock_eventstream import encode_event

EVENT_STREAM: Final = "application/vnd.amazon.eventstream"


def bedrock_error(decision: Decision) -> JSONResponse:
    kind: Final = (
        "ThrottlingException"
        if decision.status == 429
        else "ServiceUnavailableException"
        if decision.status == 503
        else "InternalServerException"
    )
    return JSONResponse(
        status_code=decision.status,
        headers={**decision.headers, "x-amzn-ErrorType": kind, "x-amzn-RequestId": uuid.uuid4().hex},
        content={"message": "perf mock flake"},
    )


def _converse_usage(decision: Decision) -> dict[str, object]:
    return {
        "inputTokens": decision.prompt_tokens,
        "outputTokens": decision.completion_tokens,
        "totalTokens": decision.prompt_tokens + decision.completion_tokens,
    }


def converse_response(decision: Decision) -> dict[str, object]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": decision.reply}]}},
        "stopReason": "end_turn",
        "usage": _converse_usage(decision),
        "metrics": {"latencyMs": round(decision.delay_s * 1000)},
    }


async def _converse_stream(decision: Decision) -> AsyncIterator[bytes]:
    await sleep_for(decision)
    yield encode_event("messageStart", {"role": "assistant"})
    for word in decision.words():
        yield encode_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": word}})
        await asyncio.sleep(decision.seconds_per_token)
    yield encode_event("contentBlockStop", {"contentBlockIndex": 0})
    yield encode_event("messageStop", {"stopReason": "end_turn"})
    yield encode_event(
        "metadata", {"usage": _converse_usage(decision), "metrics": {"latencyMs": round(decision.delay_s * 1000)}}
    )


async def _invoke_stream(decision: Decision, model: str) -> AsyncIterator[bytes]:
    await sleep_for(decision)
    for kind, payload in anthropic_events(decision, model):
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        yield encode_event("chunk", {"bytes": encoded})
        if kind == "content_block_delta":
            await asyncio.sleep(decision.seconds_per_token)


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock bedrock")
    log: Final = RequestLog()
    mount_control(app, log)

    async def converse(model_id: str, request: Request) -> Response:
        body: Final = parse_json(await request.body())
        decision: Final = log.decide(model=model_id, route="converse", stream=False, body=body)
        if decision.fail:
            return bedrock_error(decision)
        await sleep_for(decision)
        return JSONResponse(content=converse_response(decision), headers=decision.headers)

    async def converse_stream(model_id: str, request: Request) -> Response:
        body: Final = parse_json(await request.body())
        decision: Final = log.decide(model=model_id, route="converse-stream", stream=True, body=body)
        if decision.fail:
            return bedrock_error(decision)
        return StreamingResponse(_converse_stream(decision), media_type=EVENT_STREAM, headers=decision.headers)

    async def invoke(model_id: str, request: Request) -> Response:
        body: Final = parse_json(await request.body())
        decision: Final = log.decide(model=model_id, route="invoke", stream=False, body=body)
        if decision.fail:
            return bedrock_error(decision)
        await sleep_for(decision)
        return JSONResponse(content=anthropic_message(decision, model_id), headers=decision.headers)

    async def invoke_with_response_stream(model_id: str, request: Request) -> Response:
        body: Final = parse_json(await request.body())
        decision: Final = log.decide(model=model_id, route="invoke-with-response-stream", stream=True, body=body)
        if decision.fail:
            return bedrock_error(decision)
        return StreamingResponse(_invoke_stream(decision, model_id), media_type=EVENT_STREAM, headers=decision.headers)

    app.add_api_route("/model/{model_id:path}/converse", converse, methods=["POST"])
    app.add_api_route("/model/{model_id:path}/converse-stream", converse_stream, methods=["POST"])
    app.add_api_route("/model/{model_id:path}/invoke", invoke, methods=["POST"])
    app.add_api_route(
        "/model/{model_id:path}/invoke-with-response-stream", invoke_with_response_stream, methods=["POST"]
    )
    return app
