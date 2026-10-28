"""The Anthropic-shaped perf mock: /v1/messages, JSON and SSE, answering each request with a generated reply in
api.anthropic.com's wire format (thinking blocks, tool_use, cache usage, the message_start to message_stop event
order and its chunk sizes)."""

from __future__ import annotations

from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import Response

from mock_common import (
    RequestLog,
    bool_field,
    json_reply,
    mount_control,
    parse_json,
    str_field,
    stream_reply,
)

SSE: Final = "text/event-stream"


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock anthropic")
    log: Final = RequestLog("anthropic")
    mount_control(app, log)

    async def messages(request: Request) -> Response:
        body: Final = parse_json(await request.body())
        stream: Final = bool_field(body, "stream")
        decision: Final = log.decide(
            route="messages", stream=stream, body=body, model=str_field(body, "model", "claude-sonnet-5")
        )
        return stream_reply(decision, SSE) if stream else await json_reply(decision)

    app.add_api_route("/v1/messages", messages, methods=["POST"])
    app.add_api_route("/messages", messages, methods=["POST"])
    return app
