import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Final

import httpx
from integration._support.client import JSON_OBJECT, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from pydantic import JsonValue

AZURE_TARGET: Final = "/openai/v1/responses?api-version="
OPENAI_TARGET: Final = "/responses"
RATE_LIMIT_MESSAGE: Final = "Your requests to gpt-6 have exceeded token rate limit."
RATE_LIMIT_PREFIX: Final = "litellm.RateLimitError: "
SENTINEL_PREFIX: Final = "litellm.MidStreamFallbackError: "


def frame(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def response_object(identity: str, status: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "gpt-6",
        "output": [],
        "usage": None,
        **fields,
    }


def created(identity: str) -> Mapping[str, JsonValue]:
    return {"type": "response.created", "sequence_number": 0, "response": response_object(identity, "in_progress")}


def error_event(error: Mapping[str, JsonValue] | None) -> Mapping[str, JsonValue]:
    return {"type": "error", "sequence_number": 1, **({} if error is None else {"error": dict(error)})}


def openai_error_event(code: str, message: str) -> Mapping[str, JsonValue]:
    return {"type": "error", "sequence_number": 1, "code": code, "message": message, "param": None}


def openai_error_stream(identity: str, code: str, message: str) -> tuple[bytes, ...]:
    return (frame(created(identity)), frame(openai_error_event(code, message)))


def azure_rate_limit() -> Mapping[str, JsonValue]:
    return {
        "type": "too_many_requests",
        "code": "rate_limit_exceeded",
        "headers": {"x-ms-fe-error": "true"},
        "message": RATE_LIMIT_MESSAGE,
        "param": None,
    }


def failed(identity: str, code: str, message: str) -> Mapping[str, JsonValue]:
    return {
        "type": "response.failed",
        "sequence_number": 2,
        "response": response_object(identity, "failed", error={"code": code, "message": message}),
    }


def delta(identity: str, text: str) -> Mapping[str, JsonValue]:
    return {
        "type": "response.output_text.delta",
        "item_id": f"msg_{identity}",
        "output_index": 0,
        "content_index": 0,
        "delta": text,
    }


def completed(identity: str, text: str) -> Mapping[str, JsonValue]:
    message: Final = {
        "type": "message",
        "id": f"msg_{identity}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    usage: Final = {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    return {
        "type": "response.completed",
        "sequence_number": 3,
        "response": response_object(identity, "completed", output=[message], usage=usage),
    }


def rate_limited_stream(identity: str) -> tuple[bytes, ...]:
    return (
        frame(created(identity)),
        frame(error_event(azure_rate_limit())),
        frame(failed(identity, "rate_limit_exceeded", RATE_LIMIT_MESSAGE)),
    )


def healthy_stream(identity: str, text: str) -> tuple[bytes, ...]:
    return (frame(created(identity)), frame(delta(identity, text)), frame(completed(identity, text)))


def serve(stream: tuple[bytes, ...], target: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target.startswith(target), request.target
        return Reply(content_type="text/event-stream", chunks=stream)

    return respond


def function_tools() -> list[JsonValue]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]


def chat_content(frames: Sequence[Mapping[str, JsonValue]]) -> str:
    def deltas() -> Iterator[str]:
        for chunk in frames:
            for choice in chunk.get("choices") or []:
                yield string_value(object_value(object_value(choice)["delta"]).get("content") or "")

    return "".join(deltas())


def chat_body(model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": marker}],
        "tools": function_tools(),
        "num_retries": 0,
        "cache": {"no-cache": True},
        **extra,
    }


def messages_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": marker}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather for a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }
        ],
        "num_retries": 0,
        "cache": {"no-cache": True},
    }


def error_body(response: httpx.Response) -> Mapping[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def data_frames(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def sse_events(text: str) -> tuple[tuple[str, Mapping[str, JsonValue]], ...]:
    def parse(block: str) -> tuple[str, Mapping[str, JsonValue]]:
        lines: Final = block.splitlines()
        event: Final = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data: Final = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        return event, JSON_OBJECT.validate_json(data)

    return tuple(parse(block) for block in text.strip().split("\n\n") if "event: " in block)


def spend_row(call_id: str) -> Mapping[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT litellm_call_id, status, model_group, metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s',
            (call_id,),
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    assert len(rows) == 1, rows
    return rows[0]


def assert_provider_typed_rate_limit(
    error: Mapping[str, JsonValue], provider_message: str = RATE_LIMIT_MESSAGE
) -> None:
    assert error["type"] == "throttling_error", error
    assert str(error["code"]) == "429", error
    message: Final = string_value(error["message"])
    assert message.startswith(RATE_LIMIT_PREFIX) and provider_message in message, message
    assert SENTINEL_PREFIX not in message, message
