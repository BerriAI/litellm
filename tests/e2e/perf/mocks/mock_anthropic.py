"""The Anthropic-shaped perf mock: /v1/messages as JSON and as the Messages SSE event sequence
(message_start, content_block_*, message_delta, message_stop), so the gateway's Anthropic client
and its stream decoder run against a real wire format."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from mock_common import Decision, RequestLog, mount_control, parse_json, sleep_for, sse
from mock_openai import bool_field, str_field


def anthropic_error(decision: Decision) -> JSONResponse:
    kind: Final = (
        "overloaded_error" if decision.status == 529 else "rate_limit_error" if decision.status == 429 else "api_error"
    )
    return JSONResponse(
        status_code=decision.status,
        headers=decision.headers,
        content={"type": "error", "error": {"type": kind, "message": "perf mock flake"}},
    )


def anthropic_message(decision: Decision, model: str) -> dict[str, object]:
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": decision.reply}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": decision.prompt_tokens, "output_tokens": decision.completion_tokens},
    }


def anthropic_events(decision: Decision, model: str) -> list[tuple[str, dict[str, object]]]:
    """The event sequence of one streamed message, in order; the text deltas are one word each."""
    opened: dict[str, object] = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": decision.prompt_tokens, "output_tokens": 1},
    }
    events: list[tuple[str, dict[str, object]]] = [
        ("message_start", {"type": "message_start", "message": opened}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        ("ping", {"type": "ping"}),
    ]
    for word in decision.words():
        events.append(
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": word}},
            )
        )
    events.extend(
        [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": decision.completion_tokens},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    return events


async def _messages_stream(decision: Decision, model: str) -> AsyncIterator[bytes]:
    await sleep_for(decision)
    for kind, payload in anthropic_events(decision, model):
        yield sse(kind, payload)
        if kind == "content_block_delta":
            await asyncio.sleep(decision.seconds_per_token)


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock anthropic")
    log: Final = RequestLog()
    mount_control(app, log)

    async def messages(request: Request) -> Response:
        body: Final = parse_json(await request.body())
        model: Final = str_field(body, "model", "unknown")
        stream: Final = bool_field(body, "stream")
        decision: Final = log.decide(model=model, route="messages", stream=stream, body=body)
        if decision.fail:
            return anthropic_error(decision)
        if stream:
            return StreamingResponse(
                _messages_stream(decision, model), media_type="text/event-stream", headers=decision.headers
            )
        await sleep_for(decision)
        return JSONResponse(content=anthropic_message(decision, model), headers=decision.headers)

    app.add_api_route("/v1/messages", messages, methods=["POST"])
    app.add_api_route("/messages", messages, methods=["POST"])
    return app
