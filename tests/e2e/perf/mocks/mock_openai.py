"""The OpenAI-shaped perf mock: /v1/chat/completions and /v1/responses, JSON and SSE, the way
api.openai.com answers them, so the gateway's real OpenAI client and stream parsers run against it."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Final, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from mock_common import Decision, RequestLog, mount_control, parse_json, sleep_for, sse


def str_field(body: object, key: str, default: str) -> str:
    if isinstance(body, dict):
        value: object = cast(dict[object, object], body).get(key)
        if isinstance(value, str):
            return value
    return default


def bool_field(body: object, key: str) -> bool:
    if isinstance(body, dict):
        value: object = cast(dict[object, object], body).get(key)
        return value is True
    return False


def openai_error(decision: Decision) -> JSONResponse:
    return JSONResponse(
        status_code=decision.status,
        headers=decision.headers,
        content={
            "error": {"message": "perf mock flake", "type": "server_error", "param": None, "code": "perf_mock_flake"}
        },
    )


def _usage(decision: Decision) -> dict[str, object]:
    return {
        "prompt_tokens": decision.prompt_tokens,
        "completion_tokens": decision.completion_tokens,
        "total_tokens": decision.prompt_tokens + decision.completion_tokens,
    }


def _chat_completion(decision: Decision, model: str) -> dict[str, object]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": decision.reply, "refusal": None},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(decision),
        "system_fingerprint": "fp_perf_mock",
    }


async def _chat_stream(decision: Decision, model: str) -> AsyncIterator[bytes]:
    await sleep_for(decision)
    completion_id: Final = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created: Final = int(time.time())

    def chunk(delta: dict[str, object], finish_reason: str | None, usage: dict[str, object] | None = None) -> bytes:
        body: dict[str, object] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
        }
        if usage is not None:
            body["usage"] = usage
        return sse(None, body)

    yield chunk({"role": "assistant", "content": ""}, None)
    for word in decision.words():
        yield chunk({"content": word}, None)
        await asyncio.sleep(decision.seconds_per_token)
    yield chunk({}, "stop", _usage(decision))
    yield b"data: [DONE]\n\n"


def _response_object(
    decision: Decision, model: str, *, response_id: str, item_id: str, completed: bool
) -> dict[str, object]:
    output: list[dict[str, object]] = (
        [
            {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": decision.reply, "annotations": []}],
            }
        ]
        if completed
        else []
    )
    body: dict[str, object] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed" if completed else "in_progress",
        "error": None,
        "incomplete_details": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "metadata": {},
    }
    if completed:
        body["usage"] = {
            "input_tokens": decision.prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": decision.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": decision.prompt_tokens + decision.completion_tokens,
        }
    return body


async def _responses_stream(decision: Decision, model: str) -> AsyncIterator[bytes]:
    await sleep_for(decision)
    response_id: Final = f"resp_{uuid.uuid4().hex}"
    item_id: Final = f"msg_{uuid.uuid4().hex}"
    sequence = 0

    def event(kind: str, **fields: object) -> bytes:
        nonlocal sequence
        sequence += 1
        return sse(kind, {"type": kind, "sequence_number": sequence, **fields})

    opened: Final = _response_object(decision, model, response_id=response_id, item_id=item_id, completed=False)
    yield event("response.created", response=opened)
    yield event("response.in_progress", response=opened)
    item: dict[str, object] = {
        "id": item_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    yield event("response.output_item.added", output_index=0, item=item)
    yield event(
        "response.content_part.added",
        item_id=item_id,
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": "", "annotations": []},
    )
    for word in decision.words():
        yield event(
            "response.output_text.delta", item_id=item_id, output_index=0, content_index=0, delta=word, logprobs=[]
        )
        await asyncio.sleep(decision.seconds_per_token)
    yield event(
        "response.output_text.done", item_id=item_id, output_index=0, content_index=0, text=decision.reply, logprobs=[]
    )
    yield event(
        "response.content_part.done",
        item_id=item_id,
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": decision.reply, "annotations": []},
    )
    done_item: dict[str, object] = {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": decision.reply, "annotations": []}],
    }
    yield event("response.output_item.done", output_index=0, item=done_item)
    done: Final = _response_object(decision, model, response_id=response_id, item_id=item_id, completed=True)
    yield event("response.completed", response=done)


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock openai")
    log: Final = RequestLog()
    mount_control(app, log)

    async def chat_completions(request: Request) -> Response:
        body: Final = parse_json(await request.body())
        model: Final = str_field(body, "model", "unknown")
        stream: Final = bool_field(body, "stream")
        decision: Final = log.decide(model=model, route="chat_completions", stream=stream, body=body)
        if decision.fail:
            return openai_error(decision)
        if stream:
            return StreamingResponse(
                _chat_stream(decision, model), media_type="text/event-stream", headers=decision.headers
            )
        await sleep_for(decision)
        return JSONResponse(content=_chat_completion(decision, model), headers=decision.headers)

    async def responses(request: Request) -> Response:
        body: Final = parse_json(await request.body())
        model: Final = str_field(body, "model", "unknown")
        stream: Final = bool_field(body, "stream")
        decision: Final = log.decide(model=model, route="responses", stream=stream, body=body)
        if decision.fail:
            return openai_error(decision)
        if stream:
            return StreamingResponse(
                _responses_stream(decision, model), media_type="text/event-stream", headers=decision.headers
            )
        await sleep_for(decision)
        content: Final = _response_object(
            decision, model, response_id=f"resp_{uuid.uuid4().hex}", item_id=f"msg_{uuid.uuid4().hex}", completed=True
        )
        return JSONResponse(content=content, headers=decision.headers)

    app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/v1/responses", responses, methods=["POST"])
    app.add_api_route("/responses", responses, methods=["POST"])
    return app
