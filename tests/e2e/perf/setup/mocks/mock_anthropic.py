"""The Anthropic-shaped perf mock: /v1/messages, JSON and SSE, replaying what api.anthropic.com answered to
each request shape (thinking blocks, tool_use, cache usage, the message_start to message_stop event order and
the chunk boundaries it arrived in)."""

from __future__ import annotations

from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from mock_common import (
    KNOBS_HEADER,
    Decision,
    RequestLog,
    bool_field,
    json_reply,
    knobs_from_header,
    mount_control,
    parse_json,
    str_field,
    stream_reply,
)

SSE: Final = "text/event-stream"


def anthropic_error(decision: Decision) -> JSONResponse:
    kind: Final = "overloaded_error" if decision.status == 529 else "api_error"
    return JSONResponse(
        status_code=decision.status,
        content={
            "type": "error",
            "error": {"type": kind, "message": f"mock failure for {decision.seed} (status {decision.status})"},
        },
        headers=decision.headers,
    )


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock anthropic")
    log: Final = RequestLog("anthropic")
    mount_control(app, log)

    async def messages(request: Request) -> Response:
        body: Final = parse_json(await request.body())
        stream: Final = bool_field(body, "stream")
        decision: Final = log.decide(
            model=str_field(body, "model", "unknown"),
            knobs=knobs_from_header(request.headers.get(KNOBS_HEADER)),
            route="messages",
            stream=stream,
            body=body,
        )
        if decision.fail:
            return anthropic_error(decision)
        return stream_reply(decision, SSE) if stream else await json_reply(decision)

    app.add_api_route("/v1/messages", messages, methods=["POST"])
    app.add_api_route("/messages", messages, methods=["POST"])
    return app
