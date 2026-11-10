"""Shared helpers for the `test_dec1_native_*` contract files.

Everything here is plain data or a thin wrapper over the harness: scripted vendor replies, a way to register a
pass-through endpoint through the management API and remove it again, and readers for the `LiteLLM_SpendLogs`
rows a pass-through or native-format call leaves behind. Nothing here asserts a contract on its own.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from email.parser import BytesParser
from email.policy import HTTP
from hashlib import sha256
from itertools import accumulate, dropwhile
from typing import Final
from urllib.parse import parse_qsl, urlsplit

from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
JSON_LIST: Final = TypeAdapter(list[JsonValue])
PASS_THROUGH_CALL_TYPE: Final = "pass_through_endpoint"
SPEND_SECONDS: Final = 70

OPENAI_CHAT_USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
ANTHROPIC_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 7, "output_tokens": 3}
GEMINI_USAGE: Final[dict[str, JsonValue]] = {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18}


def digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def marker() -> str:
    return uuid.uuid4().hex


def json_reply(body: Mapping[str, JsonValue], *, status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=dict(headers or {}))


def always(reply: Reply) -> Callable[[Request], Reply]:
    return lambda _request: reply


def openai_chat_reply(text: str, model: str = "gpt-4o-mini") -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{text}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": OPENAI_CHAT_USAGE,
    }


def openai_chat_stream(text: str, model: str = "gpt-4o-mini") -> tuple[bytes, ...]:
    chunk: Final[dict[str, JsonValue]] = {
        "id": f"chatcmpl-{text}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
    }
    opening: Final = {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]}
    closing: Final = {
        **chunk,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": OPENAI_CHAT_USAGE,
    }
    return (
        f"data: {json.dumps(opening)}\n\n".encode(),
        f"data: {json.dumps(closing)}\n\n".encode(),
        b"data: [DONE]\n\n",
    )


def anthropic_message(text: str, model: str = "claude-sonnet-4-5") -> dict[str, JsonValue]:
    return {
        "id": f"msg_{text}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": ANTHROPIC_USAGE,
    }


def gemini_reply(text: str) -> dict[str, JsonValue]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": GEMINI_USAGE,
    }


def responses_object(identity: str, model: str = "gpt-5", status: str = "completed") -> dict[str, JsonValue]:
    return {"id": identity, "object": "response", "created_at": 1, "status": status, "model": model, "output": []}


def responses_text_object(identity: str, text: str, model: str = "gpt-5") -> dict[str, JsonValue]:
    return {
        **responses_object(identity, model),
        "output": [
            {
                "id": f"msg_{identity}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }


def is_model_discovery(request: Request) -> bool:
    """OpenAI-shaped deployments are probed with `GET /v1/models`; contract assertions skip that probe."""
    return request.method == "GET" and urlsplit(request.target).path.endswith("/models")


def model_discovery_reply(model: str) -> Reply:
    return json_reply({"object": "list", "data": [{"id": model, "object": "model", "created": 1, "owned_by": "x"}]})


def contract_requests(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if not is_model_discovery(request))


def only_request(wire: Wire) -> Request:
    received: Final = contract_requests(wire)
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0]


def path_of(request: Request) -> str:
    return urlsplit(request.target).path


def query_of(request: Request) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(request.target).query, keep_blank_values=True))


def json_body(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def assert_key_absent(request: Request, key: str) -> None:
    """The caller's virtual key must not reach the vendor in any header, in the URL, or in the body."""
    leaked: Final = {name: value for name, value in request.headers.items() if key in value}
    assert leaked == {}, leaked
    assert key not in request.target, request.target
    assert key.encode() not in request.body, request.body


def multipart_parts(request: Request) -> tuple[tuple[str, str | None, bytes], ...]:
    """(field name, file name or None, payload bytes) for every part of a multipart upstream request."""
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(
        (
            str(part.get_param("name", header="content-disposition")),
            part.get_filename(),
            part.get_payload(decode=True),
        )
        for part in parsed.iter_parts()
    )


def register_pass_through(gateway: Gateway, scenario: Scenario, target: str, **fields: JsonValue) -> str:
    """Create a pass-through endpoint with `POST /config/pass_through_endpoint` and return its proxy path.

    `auth` defaults to true here, as it does in the product for endpoints created through the API.
    """
    path: Final = str(fields.pop("path", f"/dec1-native-{uuid.uuid4().hex}"))
    created: Final = gateway.post("/config/pass_through_endpoint", {"path": path, "target": target, **fields})
    endpoint_id: Final = object_value(JSON_LIST.validate_python(created["endpoints"])[0])["id"]
    scenario.cleanups.callback(
        lambda: gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": str(endpoint_id)})
    )
    return path


def granted_key(scenario: Scenario, *paths: str, **fields: JsonValue) -> str:
    """A virtual key allowed to call the given auth-enforced pass-through paths."""
    return scenario.key(allowed_passthrough_routes=list(paths), **fields)


_SPEND_COLUMNS: Final = (
    "request_id, call_type, model, custom_llm_provider, spend, prompt_tokens, completion_tokens, total_tokens, "
    "status, api_base, api_key, team_id, metadata"
)


def spend_rows(gateway: Gateway, key: str, call_type: str = PASS_THROUGH_CALL_TYPE) -> list[dict[str, JsonValue]]:
    return gateway.rows(
        f'SELECT {_SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND call_type=%s',
        (digest(key), call_type),
    )


def spend_row(gateway: Gateway, key: str, call_type: str = PASS_THROUGH_CALL_TYPE) -> dict[str, JsonValue]:
    """The single spend row this key wrote for `call_type`; rows land asynchronously, so this polls."""
    rows: Final = eventually(
        lambda: spend_rows(gateway, key, call_type),
        lambda found: len(found) == 1,
        seconds=SPEND_SECONDS,
    )
    return rows[0]


def number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def first_sse_frame(raw: Iterator[bytes]) -> bytes:
    """Bytes up to and including the first blank line, read from a live stream without draining the rest."""
    return next(
        dropwhile(
            lambda buffered: b"\n\n" not in buffered,
            accumulate(raw, lambda buffered, chunk: buffered + chunk, initial=b""),
        ),
        b"",
    )


def error_message(body: bytes) -> str:
    """The human-readable message of a proxy error body, whichever of the proxy's error envelopes it uses."""
    parsed: Final = JSON_OBJECT.validate_json(body)
    error: Final = parsed.get("error", parsed.get("detail"))
    if isinstance(error, dict):
        return str(error.get("message", error.get("error", error)))
    return str(error)
