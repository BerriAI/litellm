"""Shared rig for the December 1 `/v1/messages` contract tests.

Every test in the `test_dec1_messages_*` files talks to the shared integration proxy with the caller's
Anthropic request, and puts an owned TCP peer (`wire_server`) behind a freshly created
`anthropic/<model>` deployment so the exact upstream request (path, headers, JSON body) can be read
back. Nothing here patches the product: the proxy, Postgres and Redis are the real ones.

Two facts shape the helpers. The shared proxy runs with `litellm_settings.cache: true`, so every
prompt carries a fresh marker, otherwise a repeated request is answered from the response cache and
never reaches the peer. And the proxy may probe a new deployment with a `GET`, so only `POST`
requests count as provider calls.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

BACKEND: Final = "claude-sonnet-4-5-20250929"
UPSTREAM_KEY: Final = "synthetic-anthropic-key"
DEFAULT_ANTHROPIC_VERSION: Final = "2023-06-01"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
NO_CACHE: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
"""Body fields that keep a call off the response cache, which is what makes a plain Anthropic call
eligible for the Rust route (`litellm/rust_bridge/detectors.py` `response_cache`)."""

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

SPEND_COLUMNS: Final = (
    "SELECT request_id, call_type, api_key, spend, total_tokens, prompt_tokens, completion_tokens, model, "
    'model_id, model_group, custom_llm_provider, api_base, "user", team_id, end_user, session_id, status, '
    "cache_hit, request_tags, metadata, messages, response, litellm_call_id, request_duration_ms, "
    '("startTime" <= "endTime") AS times_ordered, ("completionStartTime" IS NOT NULL) AS has_completion_start '
    'FROM "LiteLLM_SpendLogs" '
)


def marker(prefix: str = "dec1") -> str:
    """A unique token for prompts, ids and tags, so the response cache and other tests cannot interfere."""
    return f"{prefix}-{uuid.uuid4().hex}"


def message_json(
    message_id: str,
    *,
    model: str = BACKEND,
    content: Sequence[JsonValue] | None = None,
    usage: Mapping[str, JsonValue] | None = None,
    stop_reason: str = "end_turn",
    extra: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": list(content) if content is not None else [{"type": "text", "text": "pong"}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": dict(usage) if usage is not None else {"input_tokens": 11, "output_tokens": 7},
        **(extra or {}),
    }


def message_reply(message_id: str, **fields: object) -> Reply:
    headers: Final = fields.pop("headers", None)
    body: Final = json.dumps(message_json(message_id, **fields)).encode()  # type: ignore[arg-type]  # test-local keyword passthrough
    if isinstance(headers, Mapping):
        return Reply(body=body, headers={str(name): str(value) for name, value in headers.items()})
    return Reply(body=body)


def sse(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def stream_chunks(
    message_id: str,
    *,
    model: str = BACKEND,
    text: str = "hello stream",
    input_tokens: int = 11,
    output_tokens: int = 7,
) -> tuple[bytes, ...]:
    """The six-event text stream Anthropic sends: start, block start, one delta, block stop, delta, stop."""
    return (
        sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 1},
                },
            },
        ),
        sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    )


def stream_reply(message_id: str, **fields: object) -> Reply:
    return Reply(content_type="text/event-stream", chunks=stream_chunks(message_id, **fields))  # type: ignore[arg-type]  # test-local keyword passthrough


def upstream_error(status: int, error_type: str, message: str, headers: Mapping[str, str] | None = None) -> Reply:
    """An error in Anthropic's own envelope, as api.anthropic.com answers it."""
    return Reply(
        status=status,
        body=json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode(),
        headers=dict(headers or {}),
    )


def sse_events(text: str) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    """Every `event:`/`data:` frame of an SSE body except keepalive pings, in arrival order."""
    frames: Final = tuple(frame for frame in text.replace("\r\n", "\n").split("\n\n") if frame.strip())
    named: Final = tuple(
        (
            next((line.removeprefix("event:").strip() for line in frame.splitlines() if line.startswith("event:")), ""),
            next((line.removeprefix("data:").strip() for line in frame.splitlines() if line.startswith("data:")), ""),
        )
        for frame in frames
    )
    return tuple((event, JSON_OBJECT.validate_json(data)) for event, data in named if event != "ping" and data)


def anthropic_error(response: httpx.Response) -> dict[str, JsonValue]:
    """The `error` object of a response that must be Anthropic's `{"type": "error", "error": {...}}` envelope."""
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert body.get("type") == "error", response.text
    error: Final = object_value(body["error"])
    assert isinstance(error.get("type"), str) and isinstance(error.get("message"), str), response.text
    return error


@dataclass(frozen=True, slots=True)
class Rig:
    """One deployment on the shared proxy whose provider is an owned TCP peer."""

    gateway: Gateway
    scenario: Scenario
    wire: Wire
    model: str

    def body(self, prompt: str | None = None, **fields: JsonValue) -> dict[str, JsonValue]:
        """A minimal valid request for this deployment; `fields` override or add top-level keys."""
        return {
            "model": self.model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt if prompt is not None else f"ping {marker()}"}],
            **fields,
        }

    def post(
        self,
        body: Mapping[str, JsonValue],
        *,
        headers: Mapping[str, str] | None = None,
        key: str | None = None,
        path: str = "/v1/messages",
    ) -> httpx.Response:
        """POST as an `Authorization: Bearer` caller (the master key unless `key` is given)."""
        return self.gateway.request("POST", path, body, key=key, headers=headers)

    def post_raw(self, headers: Mapping[str, str], *, content: bytes, path: str = "/v1/messages") -> httpx.Response:
        """POST exact bytes with exactly these headers, for auth-header and malformed-body cases."""
        return self.gateway.client.post(path, content=content, headers=dict(headers))

    def stream(
        self, body: Mapping[str, JsonValue], *, headers: Mapping[str, str] | None = None
    ) -> tuple[httpx.Response, str]:
        """POST and read the whole body through httpx's streaming API; returns the response and its text."""
        with self.gateway.client.stream(
            "POST",
            "/v1/messages",
            json=dict(body),
            headers={"Authorization": f"Bearer {self.gateway.key}", **(headers or {})},
        ) as response:
            text: Final = "".join(response.iter_text())
            return response, text

    def calls(self) -> tuple[Request, ...]:
        """Provider calls received so far, without the proxy's own `GET` probes. Drains the queue."""
        return tuple(request for request in self.wire.drain() if request.method == "POST")

    def one_call(self) -> Request:
        calls: Final = self.calls()
        assert len(calls) == 1, [(call.method, call.target) for call in calls]
        return calls[0]

    def one_body(self) -> dict[str, JsonValue]:
        return JSON_OBJECT.validate_json(self.one_call().body)


def _answering_probes(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        return respond(request)

    return answer


@contextmanager
def deployment(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    backend: str = BACKEND,
    api_key: str | None = UPSTREAM_KEY,
    model_info: Mapping[str, JsonValue] | None = None,
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Create `anthropic/<backend>` behind a new peer, yield the rig, then delete the deployment."""
    with wire_server(_answering_probes(respond)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{backend}",
            api_base=wire.url,
            model_info=model_info,
            api_key=api_key,
            **litellm_params,
        )
        yield Rig(gateway, scenario, wire, model)


def always(reply: Reply) -> Callable[[Request], Reply]:
    return lambda _request: reply


def echo_message(request: Request) -> Reply:
    """Answer any call with a small message (or the six-event stream) whose id is unique per call."""
    body: Final = JSON_OBJECT.validate_json(request.body)
    message_id: Final = f"msg_{uuid.uuid4().hex}"
    if body.get("stream") is True:
        return stream_reply(message_id)
    return message_reply(message_id)


def spend_row(request_id: str, *, seconds: float = 90) -> dict[str, JsonValue]:
    """The single SpendLogs row for a response id. The row lands asynchronously, so this polls."""
    rows: Final = eventually(
        lambda: read_rows(SPEND_COLUMNS + "WHERE request_id=%s", (request_id,)),
        lambda values: len(values) == 1,
        seconds=seconds,
    )
    return rows[0]


def spend_rows_for_model(model_group: str, expected: int, *, seconds: float = 90) -> list[dict[str, JsonValue]]:
    """Every SpendLogs row of a model group once at least `expected` have landed."""
    return eventually(
        lambda: read_rows(SPEND_COLUMNS + "WHERE model_group=%s", (model_group,)),
        lambda values: len(values) >= expected,
        seconds=seconds,
    )


def json_column(value: JsonValue) -> JsonValue:
    """A JSON column as Python data, whether the driver returned it decoded or as text."""
    return json.loads(value) if isinstance(value, str) else value


def caller_tags(row: Mapping[str, JsonValue]) -> list[JsonValue]:
    """`request_tags` without the `User-Agent: ...` tags the proxy adds on its own."""
    tags: Final = json_column(row["request_tags"])
    assert isinstance(tags, list), row
    return [tag for tag in tags if not (isinstance(tag, str) and tag.startswith("User-Agent"))]


def message_id_of(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])
