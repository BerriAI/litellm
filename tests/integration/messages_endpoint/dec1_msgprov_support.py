"""Shared rig for the December 1 `/v1/messages` contracts on providers other than Anthropic itself.

The `test_dec1_msgprov_*` files pin what `POST /v1/messages` does when the deployment behind the
model is Bedrock (InvokeModel and Converse), Vertex AI's Anthropic partner models, Azure AI Foundry
(`azure_ai/claude-*`), Azure OpenAI (`azure/`), Fireworks or Baseten. Each test registers one
deployment on the shared proxy whose `api_base` points at an owned TCP peer (`wire_server`), so the
exact upstream request (method, path, query, headers, body) can be read back, and the peer answers
in that provider's own wire format: an Anthropic message or AWS event stream for InvokeModel, a
Converse document or event stream for Converse, Anthropic JSON or SSE for Vertex and Foundry, and
OpenAI chat completions JSON or chunks for the providers reached through the chat bridge.

Harness facts that shape these helpers, learned from the first CI run of the December 1 suite:
the shared proxy has `litellm_settings.cache: true`, so every prompt carries a fresh marker (the
`Rig.body` helper from `dec1_messages_support` already does this); SpendLogs rows are read through
`SPEND_COLUMNS`, which never selects a raw timestamp column; only `POST` requests count as provider
calls because the proxy may probe a new deployment with a `GET`; and Vertex access tokens come from
the shared scripted upstream's `/_oauth/token` route (`scripted-token`), not from the owned peer.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Final
from urllib.parse import unquote, urlsplit

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.sigv4 import encoded_path, signature
from integration._support.upstream import _aws_event_frame
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint.dec1_messages_support import (
    JSON_OBJECT,
    SPEND_COLUMNS,
    Rig,
    marker,
    message_json,
    stream_chunks,
)
from pydantic import JsonValue

# Bedrock
BEDROCK_MODEL_ID: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
"""A US cross-region inference profile id, the shape most Bedrock Claude deployments use."""
BEDROCK_BASE_MODEL_ID: Final = "anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_PLAIN_MODEL_ID: Final = "us.anthropic.claude-sonnet-5"
"""A Claude id without a `:` version suffix, so the SigV4 canonical path needs no extra encoding."""
BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
ACCESS_KEY: Final = "AKIASYNTHETICMSGPROV1"
SECRET_KEY: Final = "synthetic-secret-key-for-msgprov"
SESSION_TOKEN: Final = "synthetic-session-token-for-msgprov"
BEDROCK_VERSION: Final = "bedrock-2023-05-31"
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"

# Vertex AI
VERTEX_BACKEND: Final = "claude-sonnet-4-5@20250929"
VERTEX_PROJECT: Final = "synthetic-vertex-project"
VERTEX_LOCATION: Final = "us-east5"
VERTEX_TOKEN: Final = "scripted-token"
"""What the shared scripted upstream's `/_oauth/token` route hands out (`_support/upstream.py`)."""
VERTEX_VERSION: Final = "vertex-2023-10-16"

# Azure
AZURE_AI_BACKEND: Final = "claude-sonnet-4-5"
AZURE_AI_KEY: Final = "synthetic-azure-ai-key"
AZURE_DEPLOYMENT: Final = "gpt-5.4"
AZURE_KEY: Final = "synthetic-azure-key"
AZURE_API_VERSION: Final = "2025-04-01-preview"

# OpenAI compatible providers reached through the chat bridge
FIREWORKS_BACKEND: Final = "accounts/fireworks/models/glm-5p3"
FIREWORKS_KEY: Final = "synthetic-fireworks-key"
BASETEN_BACKEND: Final = "zai-org/GLM-4.6"
BASETEN_KEY: Final = "synthetic-baseten-key"

# Deployment prices, so cost assertions do not depend on the shipped cost map
INPUT_COST: Final = 0.001
OUTPUT_COST: Final = 0.002
CACHE_READ_COST: Final = 0.0001
CACHE_CREATION_COST: Final = 0.003
PRICES: Final[Mapping[str, JsonValue]] = {
    "input_cost_per_token": INPUT_COST,
    "output_cost_per_token": OUTPUT_COST,
    "cache_read_input_token_cost": CACHE_READ_COST,
    "cache_creation_input_token_cost": CACHE_CREATION_COST,
}

TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Look up the weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
PNG_BASE64: Final = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082"
    )
).decode()


def target(request: Request) -> str:
    """The request path and query with percent escapes decoded, as a provider would route it."""
    return unquote(request.target)


def path_of(request: Request) -> str:
    return unquote(urlsplit(request.target).path)


def query_of(request: Request) -> str:
    return urlsplit(request.target).query


def body_of(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def at(value: JsonValue, *path: str | int) -> JsonValue:
    """The value at `path` inside nested JSON, asserting each step is an object key or a list index."""
    if not path:
        return value
    head, *rest = path
    if isinstance(head, int):
        assert isinstance(value, list) and len(value) > head, (value, head)
        return at(value[head], *rest)
    assert isinstance(value, dict) and head in value, (value, head)
    return at(value[head], *rest)


def message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _answering_probes(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        return respond(request)

    return answer


@contextmanager
def provider_deployment(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    model: str,
    base_suffix: str = "",
    model_info: Mapping[str, JsonValue] | None = None,
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Register `model` with `api_base` at a new owned peer, yield the rig, then delete the deployment.

    `litellm_params` override everything, including `api_key` (pass `api_key=None` for a deployment
    that must not carry the harness default `integration-provider-key`) and `api_base`.
    """
    with wire_server(_answering_probes(respond)) as wire, gateway.scenario() as scenario:
        parameters: Final[dict[str, JsonValue]] = {"api_base": wire.url + base_suffix, **litellm_params}
        name: Final = scenario.model(model=model, model_info=model_info, **parameters)
        yield Rig(gateway, scenario, wire, name)


def only_spend_row(rig: Rig, *, seconds: float = 90) -> dict[str, JsonValue]:
    """The one SpendLogs row of this rig's model group. Polls, since rows land asynchronously."""
    rows: Final = eventually(
        lambda: read_rows(SPEND_COLUMNS + "WHERE model_group=%s", (rig.model,)),
        lambda values: len(values) >= 1,
        seconds=seconds,
    )
    assert len(rows) == 1, rows
    return rows[0]


def spend_of(row: Mapping[str, JsonValue]) -> float:
    return float(str(row["spend"]))


def json_column(value: JsonValue) -> JsonValue:
    return json.loads(value) if isinstance(value, str) else value


# ---------------------------------------------------------------- Bedrock InvokeModel


def invoke_chunk(event: Mapping[str, JsonValue]) -> bytes:
    """One InvokeModel stream frame: an AWS `chunk` event whose `bytes` is the base64 Anthropic event."""
    return _aws_event_frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}, "sc", "u")


def invoke_exception(kind: str, message: str) -> bytes:
    """An AWS event stream exception frame, as Bedrock sends one mid stream (for example throttling)."""
    from integration._support.upstream import aws_event_stream_frame

    return aws_event_stream_frame(
        {":exception-type": kind, ":content-type": "application/json", ":message-type": "exception"},
        json.dumps({"message": message}).encode(),
    )


def invoke_stream_frames(
    identity: str,
    *,
    model: str = BEDROCK_BASE_MODEL_ID,
    text: str = "hello stream",
    input_tokens: int = 11,
    output_tokens: int = 7,
    metrics: Mapping[str, JsonValue] | None = None,
) -> tuple[bytes, ...]:
    """The six Anthropic events wrapped as InvokeModel `chunk` frames.

    Real Bedrock adds `amazon-bedrock-invocationMetrics` to the final `message_stop`; pass `metrics`
    to include it.
    """
    events: Final = tuple(
        JSON_OBJECT.validate_json(chunk.split(b"data: ", 1)[1])
        for chunk in stream_chunks(
            identity, model=model, text=text, input_tokens=input_tokens, output_tokens=output_tokens
        )
    )
    last: Final = events[-1] if metrics is None else {**events[-1], "amazon-bedrock-invocationMetrics": dict(metrics)}
    return tuple(invoke_chunk(event) for event in (*events[:-1], last))


def invoke_stream_reply(identity: str, **fields: object) -> Reply:
    return Reply(content_type=EVENT_STREAM, chunks=invoke_stream_frames(identity, **fields))  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def invoke_message_reply(identity: str, **fields: object) -> Reply:
    model: Final = fields.pop("model", BEDROCK_BASE_MODEL_ID)
    return Reply(body=json.dumps(message_json(identity, model=str(model), **fields)).encode())  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def invoke_echo(request: Request) -> Reply:
    """Answer an InvokeModel call: an event stream on `invoke-with-response-stream`, JSON otherwise."""
    identity: Final = message_id()
    if path_of(request).endswith("/invoke-with-response-stream"):
        return invoke_stream_reply(identity)
    return invoke_message_reply(identity)


def aws_error(status: int, error_type: str, message: str) -> Reply:
    """An AWS JSON error: `{"message": ...}` with the error class in `x-amzn-ErrorType`."""
    return Reply(
        status=status,
        body=json.dumps({"message": message}).encode(),
        headers={"x-amzn-ErrorType": f"{error_type}:http://internal.amazon.com/coral/com.amazon.bedrock/"},
    )


# ---------------------------------------------------------------- Bedrock Converse


def converse_json(
    *,
    content: Sequence[JsonValue] | None = None,
    stop_reason: str = "end_turn",
    usage: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": list(content) if content is not None else [{"text": "pong"}],
            }
        },
        "stopReason": stop_reason,
        "usage": dict(usage) if usage is not None else {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
        "metrics": {"latencyMs": 1},
    }


def converse_reply(**fields: object) -> Reply:
    return Reply(body=json.dumps(converse_json(**fields)).encode())  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def converse_stream_reply(
    pieces: Sequence[str] = ("hello ", "stream"),
    *,
    stop_reason: str = "end_turn",
    usage: Mapping[str, JsonValue] | None = None,
) -> Reply:
    events: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
        ("messageStart", {"role": "assistant"}),
        *(("contentBlockDelta", {"delta": {"text": piece}, "contentBlockIndex": 0}) for piece in pieces),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": stop_reason}),
        (
            "metadata",
            {
                "usage": dict(usage) if usage is not None else {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
                "metrics": {"latencyMs": 1},
            },
        ),
    )
    return Reply(
        content_type=EVENT_STREAM,
        chunks=tuple(_aws_event_frame(kind, payload, "sc", "u") for kind, payload in events),
    )


def converse_echo(request: Request) -> Reply:
    if path_of(request).endswith("/converse-stream"):
        return converse_stream_reply()
    return converse_reply()


# ---------------------------------------------------------------- Vertex AI and Azure AI Foundry (Anthropic wire)


def anthropic_json_reply(identity: str, *, model: str, **fields: object) -> Reply:
    return Reply(body=json.dumps(message_json(identity, model=model, **fields)).encode())  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def anthropic_sse_reply(identity: str, *, model: str, **fields: object) -> Reply:
    return Reply(content_type="text/event-stream", chunks=stream_chunks(identity, model=model, **fields))  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def vertex_echo(request: Request) -> Reply:
    identity: Final = message_id()
    if ":streamRawPredict" in path_of(request):
        return anthropic_sse_reply(identity, model="claude-sonnet-4-5-20250929")
    return anthropic_json_reply(identity, model="claude-sonnet-4-5-20250929")


def anthropic_body_echo(request: Request) -> Reply:
    """Answer an Anthropic-shaped call by its own `stream` flag (Azure AI Foundry)."""
    identity: Final = message_id()
    if body_of(request).get("stream") is True:
        return anthropic_sse_reply(identity, model=AZURE_AI_BACKEND)
    return anthropic_json_reply(identity, model=AZURE_AI_BACKEND)


def anthropic_error_reply(status: int, error_type: str, message: str) -> Reply:
    return Reply(
        status=status, body=json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()
    )


def vertex_credentials(gateway: Gateway, project: str = VERTEX_PROJECT) -> str:
    """A service account JSON whose token URI is the shared scripted upstream's `/_oauth/token`."""
    return service_account_json(project, gateway.upstream_url.rstrip("/"))


def vertex_path(*, project: str = VERTEX_PROJECT, location: str = VERTEX_LOCATION, model: str = VERTEX_BACKEND) -> str:
    return f"/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}"


# ---------------------------------------------------------------- OpenAI chat completions wire


def chat_json(
    identity: str,
    *,
    model: str,
    content: str | None = "pong",
    finish_reason: str = "stop",
    tool_calls: Sequence[JsonValue] | None = None,
    usage: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    message: Final[dict[str, JsonValue]] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = list(tool_calls)
    return {
        "id": identity,
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": dict(usage) if usage is not None else {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def chat_reply(identity: str, **fields: object) -> Reply:
    return Reply(body=json.dumps(chat_json(identity, **fields)).encode())  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def chat_stream_reply(
    identity: str,
    *,
    model: str,
    pieces: Sequence[str] = ("hello ", "stream"),
    finish_reason: str = "stop",
    usage: Mapping[str, JsonValue] | None = None,
) -> Reply:
    def chunk(delta: Mapping[str, JsonValue], finish: str | None) -> bytes:
        frame: Final = {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
        }
        return b"data: " + json.dumps(frame).encode() + b"\n\n"

    usage_frame: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [],
        "usage": dict(usage) if usage is not None else {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    return Reply(
        content_type="text/event-stream",
        chunks=(
            chunk({"role": "assistant", "content": ""}, None),
            *(chunk({"content": piece}, None) for piece in pieces),
            chunk({}, finish_reason),
            b"data: " + json.dumps(usage_frame).encode() + b"\n\n",
            b"data: [DONE]\n\n",
        ),
    )


def chat_echo(model: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
        if body_of(request).get("stream") is True:
            return chat_stream_reply(identity, model=model)
        return chat_reply(identity, model=model)

    return respond


def openai_error_reply(status: int, message: str, error_type: str = "invalid_request_error") -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": message, "type": error_type, "param": None, "code": None}}).encode(),
    )


# ---------------------------------------------------------------- SigV4


def _authorization_fields(authorization: str) -> dict[str, str]:
    return dict(part.partition("=")[::2] for part in authorization.removeprefix("AWS4-HMAC-SHA256 ").split(", "))


def sigv4_scope(request: Request) -> tuple[str, str, str]:
    """(access key, region, service) from the `Credential=` field of a SigV4 `Authorization` header."""
    authorization: Final = request.headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 "), dict(request.headers)
    access_key, date, region, service, _ = _authorization_fields(authorization)["Credential"].split("/")
    assert date == request.headers["x-amz-date"][:8], authorization
    return access_key, region, service


def signed_headers(request: Request) -> tuple[str, ...]:
    return tuple(_authorization_fields(request.headers["authorization"])["SignedHeaders"].split(";"))


def assert_sigv4_signed(request: Request, *, region: str, access_key: str = ACCESS_KEY, secret: str = SECRET_KEY) -> None:
    """Recompute the SigV4 signature with the independent equations in `_support/sigv4.py`.

    botocore signs the URI-encoded form of the path it was given, which is `encoded_path` of the
    raw request target the peer saw.
    """
    fields: Final = _authorization_fields(request.headers["authorization"])
    scope: Final = fields["Credential"].split("/", 1)[1]
    assert sigv4_scope(request) == (access_key, region, "bedrock"), fields
    raw_path, _, raw_query = request.target.partition("?")
    expected: Final = signature(
        "POST", encoded_path(raw_path), request.headers, fields["SignedHeaders"], request.body, secret, scope, raw_query
    )
    assert fields["Signature"] == expected[1], fields


def unique_text(prefix: str = "msgprov") -> str:
    return f"{prefix} {marker()}"
