"""The Bedrock-shaped perf mock: Converse and ConverseStream under /model/{model_id}/..., replaying what bedrock-runtime answered to each request shape. Streams are the
captured application/vnd.amazon.eventstream bytes verbatim (prelude, headers, CRCs), so the gateway's
botocore-based decoder runs against the real framing."""

from __future__ import annotations

from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import Response

from mock_common import (
    RequestLog,
    json_reply,
    mount_control,
    parse_json,
    stream_reply,
)

EVENT_STREAM: Final = "application/vnd.amazon.eventstream"


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock bedrock")
    log: Final = RequestLog("bedrock")
    mount_control(app, log)

    async def handle(model_id: str, request: Request, route: str, stream: bool) -> Response:
        body: Final = parse_json(await request.body())
        decision: Final = log.decide(
            route=route,
            stream=stream,
            body=body,
        )
        return stream_reply(decision, EVENT_STREAM) if stream else await json_reply(decision)

    async def converse(model_id: str, request: Request) -> Response:
        return await handle(model_id, request, "converse", False)

    async def converse_stream(model_id: str, request: Request) -> Response:
        return await handle(model_id, request, "converse", True)

    app.add_api_route("/model/{model_id:path}/converse", converse, methods=["POST"])
    app.add_api_route("/model/{model_id:path}/converse-stream", converse_stream, methods=["POST"])
    return app
