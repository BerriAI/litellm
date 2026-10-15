"""The OpenAI-shaped perf mock: /v1/chat/completions and /v1/responses, JSON and SSE, replaying what
api.openai.com answered to each request shape, so the gateway's real OpenAI client and stream parsers run
against the provider's own bodies and chunk boundaries."""

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
    stream_reply,
)

SSE: Final = "text/event-stream"


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock openai")
    log: Final = RequestLog("openai")
    mount_control(app, log)

    async def handle(request: Request, route: str) -> Response:
        body: Final = parse_json(await request.body())
        stream: Final = bool_field(body, "stream")
        decision: Final = log.decide(
            route=route,
            stream=stream,
            body=body,
        )
        return stream_reply(decision, SSE) if stream else await json_reply(decision)

    async def chat_completions(request: Request) -> Response:
        return await handle(request, "chat")

    async def responses(request: Request) -> Response:
        return await handle(request, "responses")

    app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/v1/responses", responses, methods=["POST"])
    app.add_api_route("/responses", responses, methods=["POST"])
    return app
