import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

import anyio
import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.integrations.custom_logger import CustomLogger

Provider = Literal["openai", "anthropic", "gemini", "mock", "openai_text"]
Surface = Literal["chat", "text"]
Mode = Literal["sync", "async"]
Options = Literal["omitted", "none", "empty", "hidden", "visible"]
Lifecycle = Literal["success", "failure", "close", "cancel"]
MODELS: Final = {
    "openai": "gpt-6-astra",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
    "mock": "gpt-6-astra",
    "openai_text": "davinci-002",
}
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
TOOL_ARGUMENTS: Final = '{"city":"Zürich 🐈","count":2}'
TOOL_ID: Final = "call_matrix_17"


@dataclass(frozen=True, slots=True)
class Case:
    provider: Provider
    mode: Mode
    surface: Surface = "chat"
    options: Options = "visible"
    prompt_tokens: int | None = 11
    output_tokens: int = 7
    fragments: tuple[str, ...] = ("hello ", "world")
    tool: bool = False
    tool_preamble: str = ""
    mock_payload: Literal["text", "response"] = "text"
    lifecycle: Lifecycle = "success"
    admission: int | None = None
    split_bytes: int = 0
    sse_space: bool = True
    close_yield: bool = False
    admission_key: Literal["metadata", "litellm_metadata"] = "metadata"
    reservation: Literal["count", "null", "empty", "null_count"] = "count"
    model_name: str | None = None

    @property
    def response_model(self) -> str:
        return self.model_name or MODELS[self.provider]

    @property
    def model(self) -> str:
        prefix: Final = (
            "text-completion-openai"
            if self.provider == "openai_text"
            else ("openai" if self.provider == "mock" else self.provider)
        )
        return prefix + "/" + self.response_model

    @property
    def text(self) -> str:
        return "".join(self.fragments)

    @property
    def stream_options(self) -> dict[str, JsonValue]:
        match self.options:
            case "omitted":
                return {}
            case "none":
                return {"stream_options": None}
            case "empty":
                return {"stream_options": {}}
            case "hidden":
                return {"stream_options": {"include_usage": False}}
            case "visible":
                return {"stream_options": {"include_usage": True}}


def usage(case: Case) -> dict[str, JsonValue]:
    return {
        "prompt_tokens": case.prompt_tokens,
        "completion_tokens": case.output_tokens,
        "total_tokens": (case.prompt_tokens or 0) + case.output_tokens,
    }


def openai_chunk(case: Case, delta: dict[str, JsonValue], finish: str | None = None) -> dict[str, JsonValue]:
    return {
        "id": "chatcmpl-matrix",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": case.response_model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "usage": None,
    }


def openai_events(case: Case) -> tuple[dict[str, JsonValue], ...]:
    content: Final = (
        tuple(
            openai_chunk(case, {"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]})
            for fragment in case.fragments
        )
        if case.tool
        else tuple(openai_chunk(case, {"content": fragment}) for fragment in case.fragments)
    )
    initial: Final[dict[str, JsonValue]] = (
        {
            "role": "assistant",
            **({"content": case.tool_preamble} if case.tool_preamble else {}),
            "tool_calls": [
                {"index": 0, "id": TOOL_ID, "type": "function", "function": {"name": "lookup", "arguments": ""}}
            ],
        }
        if case.tool
        else {"role": "assistant", "content": ""}
    )
    trailing: Final = (
        (
            {
                "id": "chatcmpl-matrix",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": case.response_model,
                "choices": [],
                "usage": usage(case),
            },
        )
        if case.prompt_tokens is not None and case.options not in ("empty", "hidden")
        else ()
    )
    return (
        openai_chunk(case, initial),
        *content,
        openai_chunk(case, {}, "tool_calls" if case.tool else "stop"),
        *trailing,
    )


def anthropic_events(case: Case) -> tuple[dict[str, JsonValue], ...]:
    block_index: Final = 1 if case.tool_preamble else 0
    block: Final[dict[str, JsonValue]] = (
        {"type": "tool_use", "id": TOOL_ID, "name": "lookup", "input": {}}
        if case.tool
        else {"type": "text", "text": ""}
    )
    return (
        {
            "type": "message_start",
            "message": {
                "id": "msg_matrix",
                "type": "message",
                "role": "assistant",
                "model": case.response_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": case.prompt_tokens, "output_tokens": min(1, case.output_tokens)},
            },
        },
        *(
            (
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": case.tool_preamble},
                },
                {"type": "content_block_stop", "index": 0},
            )
            if case.tool_preamble
            else ()
        ),
        {"type": "content_block_start", "index": block_index, "content_block": block},
        *(
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": (
                    {"type": "input_json_delta", "partial_json": fragment}
                    if case.tool
                    else {"type": "text_delta", "text": fragment}
                ),
            }
            for fragment in case.fragments
        ),
        {"type": "content_block_stop", "index": block_index},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use" if case.tool else "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": case.output_tokens},
        },
        {"type": "message_stop"},
    )


def gemini_events(case: Case) -> tuple[dict[str, JsonValue], ...]:
    parts: Final[tuple[dict[str, JsonValue], ...]] = (
        (
            *(({"text": case.tool_preamble},) if case.tool_preamble else ()),
            {"functionCall": {"id": TOOL_ID, "name": "lookup", "args": JSON_OBJECT.validate_json(case.text)}},
        )
        if case.tool
        else tuple({"text": fragment} for fragment in case.fragments)
    )
    terminal: Final[dict[str, JsonValue]] = {
        "candidates": [{"content": {"parts": [], "role": "model"}, "index": 0, "finishReason": "STOP"}],
        "modelVersion": case.response_model,
        "responseId": "gemini_matrix",
        **(
            {
                "usageMetadata": {
                    "promptTokenCount": case.prompt_tokens,
                    "candidatesTokenCount": case.output_tokens,
                    "totalTokenCount": case.prompt_tokens + case.output_tokens,
                }
            }
            if case.prompt_tokens is not None
            else {}
        ),
    }
    return (
        *(
            {
                "candidates": [{"content": {"parts": [part], "role": "model"}, "index": 0}],
                "modelVersion": case.response_model,
                "responseId": "gemini_matrix",
            }
            for part in parts
        ),
        terminal,
    )


def text_events(case: Case) -> tuple[dict[str, JsonValue], ...]:
    chunks: Final = tuple(
        {
            "id": "cmpl-matrix",
            "object": "text_completion",
            "created": 1,
            "model": case.response_model,
            "choices": [
                {
                    "index": 0,
                    "text": fragment,
                    "logprobs": None,
                    "finish_reason": "stop" if index == len(case.fragments) else None,
                }
            ],
            "usage": None,
        }
        for index, fragment in enumerate((*case.fragments, ""))
    )
    return (
        *chunks,
        *(
            (
                {
                    "id": "cmpl-matrix",
                    "object": "text_completion",
                    "created": 1,
                    "model": case.response_model,
                    "choices": [],
                    "usage": usage(case),
                },
            )
            if case.prompt_tokens is not None and case.options == "visible"
            else ()
        ),
    )


def provider_events(case: Case) -> tuple[dict[str, JsonValue], ...]:
    match case.provider:
        case "anthropic":
            return anthropic_events(case)
        case "gemini":
            return gemini_events(case)
        case "openai_text":
            return text_events(case)
        case _:
            return openai_events(case)


def nonstream_response(case: Case) -> dict[str, JsonValue]:
    match case.provider:
        case "anthropic":
            return {
                "id": "msg_matrix",
                "type": "message",
                "role": "assistant",
                "model": case.response_model,
                "content": (
                    [
                        {
                            "type": "tool_use",
                            "id": TOOL_ID,
                            "name": "lookup",
                            "input": JSON_OBJECT.validate_json(case.text),
                        }
                    ]
                    if case.tool
                    else [{"type": "text", "text": case.text}]
                ),
                "stop_reason": "tool_use" if case.tool else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": case.prompt_tokens, "output_tokens": case.output_tokens},
            }
        case "gemini":
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": (
                                [
                                    {
                                        "functionCall": {
                                            "id": TOOL_ID,
                                            "name": "lookup",
                                            "args": JSON_OBJECT.validate_json(case.text),
                                        }
                                    }
                                ]
                                if case.tool
                                else [{"text": case.text}]
                            ),
                            "role": "model",
                        },
                        "index": 0,
                        "finishReason": "STOP",
                    }
                ],
                "modelVersion": case.response_model,
                "responseId": "gemini_matrix",
                "usageMetadata": {
                    "promptTokenCount": case.prompt_tokens,
                    "candidatesTokenCount": case.output_tokens,
                    "totalTokenCount": (case.prompt_tokens or 0) + case.output_tokens,
                },
            }
        case "openai_text":
            return {
                "id": "cmpl-matrix",
                "object": "text_completion",
                "created": 1,
                "model": case.response_model,
                "choices": [{"index": 0, "text": case.text, "finish_reason": "stop", "logprobs": None}],
                "usage": usage(case),
            }
        case _:
            return {
                "id": "chatcmpl-matrix",
                "object": "chat.completion",
                "created": 1,
                "model": case.response_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (case.tool_preamble or None) if case.tool else case.text,
                            **(
                                {
                                    "tool_calls": [
                                        {
                                            "id": TOOL_ID,
                                            "type": "function",
                                            "function": {"name": "lookup", "arguments": case.text},
                                        }
                                    ]
                                }
                                if case.tool
                                else {}
                            ),
                        },
                        "finish_reason": "tool_calls" if case.tool else "stop",
                    }
                ],
                "usage": usage(case),
            }


class Wire(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, case: Case) -> None:
        self.case: Final = case
        self.closed: bool = False
        self.waiting: Final = asyncio.Event()
        self.release: Final = asyncio.Event()
        self.requests: tuple[str, ...] = ()
        self.request_bodies: tuple[dict[str, JsonValue], ...] = ()
        self.frames: Final = tuple(
            (
                ("event: " + str(event["type"]) + "\n" if case.provider == "anthropic" else "")
                + ("data: " if case.sse_space else "data:")
                + json.dumps(event, ensure_ascii=False)
                + "\n\n"
            ).encode()
            for event in provider_events(case)
        )

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests += (str(request.url),)
        body: Final = JSON_OBJECT.validate_json(request.content)
        self.request_bodies += (body,)
        if body.get("stream") is True or "streamGenerateContent" in str(request.url):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=self)
        return httpx.Response(200, json=nonstream_response(self.case))

    def pieces(self, frame: bytes) -> tuple[bytes, ...]:
        width: Final = self.case.split_bytes
        return tuple(frame[index : index + width] for index in range(0, len(frame), width)) if width else (frame,)

    def __iter__(self) -> Iterator[bytes]:
        for index, frame in enumerate(self.frames):
            if self.case.lifecycle == "failure" and index == 3:
                raise httpx.ReadError("matrix provider disconnected after content")
            yield from self.pieces(frame)
        if self.case.provider in ("openai", "openai_text"):
            yield (b"data: " if self.case.sse_space else b"data:") + b"[DONE]\n\n"

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, frame in enumerate(self.frames):
            if self.case.lifecycle == "failure" and index == 3:
                raise httpx.ReadError("matrix provider disconnected after content")
            if self.case.lifecycle == "cancel" and index == 3:
                self.waiting.set()
                await self.release.wait()
            for piece in self.pieces(frame):
                yield piece
        if self.case.provider in ("openai", "openai_text"):
            yield (b"data: " if self.case.sse_space else b"data:") + b"[DONE]\n\n"

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        if self.case.close_yield:
            await anyio.sleep(0)
        self.closed = True


@dataclass(frozen=True, slots=True)
class Event:
    outcome: Literal["success", "failure"]
    mode: Mode
    response: dict[str, JsonValue] | str
    partial_usage: object
    exception: str | None


class Recorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: tuple[Event, ...] = ()
        self.arrived: Final = threading.Event()
        self.lock: Final = threading.Lock()

    def record(
        self, outcome: Literal["success", "failure"], mode: Mode, kwargs: Mapping[str, object], response: object
    ) -> None:
        payload: Final = (
            JSON_OBJECT.validate_json(response.model_dump_json()) if isinstance(response, BaseModel) else str(response)
        )
        with self.lock:
            self.events += (
                Event(
                    outcome,
                    mode,
                    payload,
                    kwargs.get("combined_usage_object"),
                    str(kwargs["exception"]) if "exception" in kwargs else None,
                ),
            )
            self.arrived.set()

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.record("success", "sync", kwargs, response_obj)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.record("success", "async", kwargs, response_obj)

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.record("failure", "sync", kwargs, response_obj)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.record("failure", "async", kwargs, response_obj)
