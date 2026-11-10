"""Shared building blocks for the `test_dec1_spend_*` contract files.

Every test here talks to a real proxy and a scripted provider that this module owns. The provider always answers
with the same fixed token counts, and every deployment is created with the same custom price, so each expected
spend is a literal that a reader can check by hand:

    chat (stream or not)   10 prompt + 5 completion tokens  -> 10 * 0.001 + 5 * 0.002 = 0.02
    embeddings             7 prompt tokens                  -> 7 * 0.001             = 0.007
    Anthropic messages     12 input + 3 output tokens       -> 12 * 0.001 + 3 * 0.002 = 0.018
    OpenAI responses       9 input + 4 output tokens        -> 9 * 0.001 + 4 * 0.002  = 0.017
    text completion        6 prompt + 2 completion tokens   -> 6 * 0.001 + 2 * 0.002  = 0.01

Rows land asynchronously (the request log is batched, the ledger worker then applies it to the key, team, user and
daily tables), so every read goes through `eventually` with `LANDING_SECONDS`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal

import httpx
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server

INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CHAT_PROMPT_TOKENS: Final = 10
CHAT_COMPLETION_TOKENS: Final = 5
CHAT_COST: Final = 0.02
EMBEDDING_PROMPT_TOKENS: Final = 7
EMBEDDING_COST: Final = 0.007
MESSAGES_INPUT_TOKENS: Final = 12
MESSAGES_OUTPUT_TOKENS: Final = 3
MESSAGES_COST: Final = 0.018
RESPONSES_INPUT_TOKENS: Final = 9
RESPONSES_OUTPUT_TOKENS: Final = 4
RESPONSES_COST: Final = 0.017
TEXT_PROMPT_TOKENS: Final = 6
TEXT_COMPLETION_TOKENS: Final = 2
TEXT_COST: Final = 0.01
REPLY_TEXT: Final = "scripted dec1 reply"
LANDING_SECONDS: Final = 70
TINY_BUDGET: Final = 0.0000001

Entity = Literal["key", "team", "user", "org", "end_user", "tag"]

_SPEND_QUERIES: Final[Mapping[Entity, str]] = {
    "key": 'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s',
    "team": 'SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s',
    "user": 'SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = %s',
    "org": 'SELECT spend FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s',
    "end_user": 'SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = %s',
    "tag": 'SELECT spend FROM "LiteLLM_TagTable" WHERE tag_name = %s',
}

ROW_COLUMNS: Final = (
    'request_id, call_type, api_key, spend, total_tokens, prompt_tokens, completion_tokens, "startTime"::text AS '
    'start_time, "endTime"::text AS end_time, "completionStartTime"::text AS completion_start_time, '
    '("endTime" >= "startTime") AS ends_after_start, '
    '(EXTRACT(EPOCH FROM ("endTime" - "startTime")) * 1000)::float8 AS measured_ms, '
    "to_char(\"startTime\", 'YYYY-MM-DD') AS start_day, "
    "to_char(\"startTime\" + INTERVAL '1 day', 'YYYY-MM-DD') AS next_day, "
    'request_duration_ms, model, model_id, model_group, custom_llm_provider, api_base, "user", metadata, cache_hit, '
    "cache_key, request_tags, team_id, organization_id, end_user, messages, response, session_id, status, "
    "proxy_server_request, litellm_call_id"
)


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _chat_usage() -> dict[str, JsonValue]:
    return {
        "prompt_tokens": CHAT_PROMPT_TOKENS,
        "completion_tokens": CHAT_COMPLETION_TOKENS,
        "total_tokens": CHAT_PROMPT_TOKENS + CHAT_COMPLETION_TOKENS,
    }


def _sse(value: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def _chat_stream(identity: str) -> tuple[bytes, ...]:
    base: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    return (
        _sse(
            {
                **base,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": None}],
            }
        ),
        _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _sse({**base, "choices": [], "usage": _chat_usage()}),
        b"data: [DONE]\n\n",
    )


def _chat_reply(body: Mapping[str, JsonValue]) -> Reply:
    identity: Final = f"chatcmpl-dec1-{uuid.uuid4().hex}"
    if body.get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=_chat_stream(identity))
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": "stop"}
                ],
                "usage": _chat_usage(),
            }
        ).encode()
    )


def _text_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"cmpl-dec1-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": 1,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [{"text": REPLY_TEXT, "index": 0, "logprobs": None, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": TEXT_PROMPT_TOKENS,
                    "completion_tokens": TEXT_COMPLETION_TOKENS,
                    "total_tokens": TEXT_PROMPT_TOKENS + TEXT_COMPLETION_TOKENS,
                },
            }
        ).encode()
    )


def _embedding_reply(body: Mapping[str, JsonValue]) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 0.1, 0.2]}],
                "model": body.get("model"),
                "usage": {"prompt_tokens": EMBEDDING_PROMPT_TOKENS, "total_tokens": EMBEDDING_PROMPT_TOKENS},
            }
        ).encode()
    )


def _anthropic_message(model: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": f"msg_dec1{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": REPLY_TEXT}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": MESSAGES_INPUT_TOKENS,
            "output_tokens": MESSAGES_OUTPUT_TOKENS,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _anthropic_event(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _anthropic_reply(body: Mapping[str, JsonValue]) -> Reply:
    message: Final = _anthropic_message(body.get("model"))
    if body.get("stream") is not True:
        return Reply(body=json.dumps(message).encode())
    opening: Final[dict[str, JsonValue]] = {
        **message,
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": MESSAGES_INPUT_TOKENS, "output_tokens": 0},
    }
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"type": "message_start", "message": opening},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": REPLY_TEXT}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": MESSAGES_OUTPUT_TOKENS},
        },
        {"type": "message_stop"},
    )
    return Reply(content_type="text/event-stream", chunks=tuple(_anthropic_event(event) for event in events))


def _responses_object(model: JsonValue, status: str) -> dict[str, JsonValue]:
    finished: Final = status == "completed"
    return {
        "id": f"resp_dec1{uuid.uuid4().hex}",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_dec1_output",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": REPLY_TEXT, "annotations": []}],
            }
        ]
        if finished
        else [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": RESPONSES_INPUT_TOKENS,
            "output_tokens": RESPONSES_OUTPUT_TOKENS,
            "total_tokens": RESPONSES_INPUT_TOKENS + RESPONSES_OUTPUT_TOKENS,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        if finished
        else None,
    }


def _responses_reply(body: Mapping[str, JsonValue]) -> Reply:
    completed: Final = _responses_object(body.get("model"), "completed")
    if body.get("stream") is not True:
        return Reply(body=json.dumps(completed).encode())
    created: Final[dict[str, JsonValue]] = {**completed, "status": "in_progress", "output": [], "usage": None}
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"type": "response.created", "sequence_number": 0, "response": created},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_dec1_output",
            "output_index": 0,
            "content_index": 0,
            "delta": REPLY_TEXT,
        },
        {"type": "response.completed", "sequence_number": 2, "response": completed},
    )
    return Reply(content_type="text/event-stream", chunks=tuple(_anthropic_event(event) for event in events))


def respond(request: Request) -> Reply:
    """Answer every provider call the proxy can make in these tests with fixed usage."""
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = JSON_OBJECT.validate_json(request.body) if request.body else {}
    path: Final = request.target.split("?", 1)[0]
    if path.endswith("/messages"):
        return _anthropic_reply(body)
    if path.endswith("/embeddings"):
        return _embedding_reply(body)
    if path.endswith("/chat/completions"):
        return _chat_reply(body)
    if path.endswith("/responses"):
        return _responses_reply(body)
    if path.endswith("/completions"):
        return _text_reply()
    return Reply(status=404, body=b'{"error":{"message":"dec1 scripted upstream has no such route"}}')


def failing(status: int) -> Callable[[Request], Reply]:
    """A provider that fails every inference call with `status` and a recognisable message."""

    def fail(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        return Reply(
            status=status,
            body=json.dumps(
                {
                    "error": {
                        "message": "dec1 scripted upstream failure",
                        "type": "invalid_request_error" if status == 400 else "server_error",
                        "code": str(status),
                    }
                }
            ).encode(),
        )

    return fail


@contextmanager
def scripted_upstream() -> Iterator[Wire]:
    with wire_server(respond) as wire:
        yield wire


def provider_calls(wire: Wire) -> tuple[Request, ...]:
    """The inference requests the proxy sent since the last call, model discovery excluded."""
    return tuple(request for request in wire.drain() if request.method == "POST")


def chat_model(scenario: Scenario, wire: Wire, **parameters: JsonValue) -> str:
    """An `openai/gpt-4o-mini` deployment priced at INPUT_RATE / OUTPUT_RATE that calls the scripted provider."""
    return scenario.model(
        **{
            "model": "openai/gpt-4o-mini",
            "api_base": f"{wire.url}/v1",
            "api_key": "dec1-provider-key",
            "input_cost_per_token": INPUT_RATE,
            "output_cost_per_token": OUTPUT_RATE,
            **parameters,
        }
    )


def embedding_model(scenario: Scenario, wire: Wire) -> str:
    return chat_model(scenario, wire, model="openai/text-embedding-3-small", output_cost_per_token=0.0)


def messages_model(scenario: Scenario, wire: Wire) -> str:
    return chat_model(scenario, wire, model="anthropic/claude-sonnet-4-6", api_base=wire.url)


def text_model(scenario: Scenario, wire: Wire) -> str:
    return chat_model(scenario, wire, model="text-completion-openai/gpt-3.5-turbo-instruct")


def ask(
    gateway: Gateway,
    model: str,
    key: str | None = None,
    *,
    text: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """One chat completion. The prompt is unique unless `text` is given, so the response cache never answers."""
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text or f"dec1 {uuid.uuid4().hex}"}], **extra},
        key=key,
        headers=headers,
    )


def served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def served_id(response: httpx.Response) -> str:
    return string_value(served(response)["id"])


def stream(gateway: Gateway, path: str, body: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    """Send a streaming request and return every line the proxy wrote."""
    with gateway.client.stream("POST", path, json=dict(body), headers={"Authorization": f"Bearer {key}"}) as response:
        assert response.status_code == 200, response.read().decode()
        return tuple(response.iter_lines())


def stream_chunks(lines: tuple[str, ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]"
    )


def row_by_id(gateway: Gateway, request_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: gateway.rows(f'SELECT {ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
        lambda found: len(found) == 1,
        seconds=LANDING_SECONDS,
    )
    return rows[0]


def rows_of_key(gateway: Gateway, key: str, count: int) -> list[dict[str, JsonValue]]:
    """The key's spend log rows, oldest first, once exactly `count` of them have landed."""
    return eventually(
        lambda: gateway.rows(
            f'SELECT {ROW_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime"', (hashed(key),)
        ),
        lambda found: len(found) == count,
        seconds=LANDING_SECONDS,
    )


def metadata_of(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(row["metadata"])


def number(value: JsonValue) -> float:
    assert isinstance(value, (int, float, str)) and not isinstance(value, bool), value
    return float(value)


def spend_of(gateway: Gateway, entity: Entity, identity: str) -> float | None:
    rows: Final = gateway.rows(_SPEND_QUERIES[entity], (identity,))
    return number(rows[0]["spend"]) if rows else None


def spend_reaches(gateway: Gateway, entity: Entity, identity: str, amount: float) -> float:
    """Wait until the entity's stored spend is at least `amount` and return what is stored."""
    reached: Final = eventually(
        lambda: spend_of(gateway, entity, identity),
        lambda spend: spend is not None and spend >= amount - 1e-9,
        seconds=LANDING_SECONDS,
    )
    assert reached is not None
    return reached


def refusal(response: httpx.Response) -> str:
    """Assert the default budget refusal envelope and return its message."""
    assert response.status_code == 422, f"{response.status_code} {response.text}"
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["type"] == "budget_exceeded", response.text
    assert error["code"] == "422", response.text
    return string_value(error["message"])


def daily_rows(gateway: Gateway, table: str, key: str, count: int) -> list[dict[str, JsonValue]]:
    """Rows of one `LiteLLM_Daily*Spend` table for the key, once exactly `count` of them exist."""
    assert table.startswith("LiteLLM_Daily") and table.isidentifier(), table
    return eventually(
        lambda: gateway.rows(
            f'SELECT to_jsonb(d) AS row FROM "{table}" d WHERE api_key = %s ORDER BY model, endpoint', (hashed(key),)
        ),
        lambda found: len(found) == count,
        seconds=LANDING_SECONDS,
    )


def daily_row(gateway: Gateway, table: str, key: str) -> dict[str, JsonValue]:
    return object_value(daily_rows(gateway, table, key, 1)[0]["row"])


def new_tag(gateway: Gateway, scenario: Scenario, **fields: JsonValue) -> str:
    name: Final = f"dec1-tag-{uuid.uuid4().hex}"
    gateway.post("/tag/new", {"name": name, **fields})
    scenario.cleanups.callback(gateway.post, "/tag/delete", {"name": name})
    return name


def new_customer(gateway: Gateway, scenario: Scenario, **fields: JsonValue) -> str:
    identity: Final = f"dec1-customer-{uuid.uuid4().hex}"
    gateway.post("/customer/new", {"user_id": identity, **fields})
    scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [identity]})
    return identity


def proxy_config(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    """A copy of the shared integration config with the given settings merged in, for an owned proxy."""
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {**config["general_settings"], **(general_settings or {})}
    config["litellm_settings"] = {**config.get("litellm_settings", {}), **(litellm_settings or {})}
    path: Final = directory / f"dec1-spend-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Surface:
    """One inference endpoint with the literal row it is expected to produce against the scripted provider."""

    name: str
    path: str
    call_type: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    daily_endpoint: str

    def model(self, scenario: Scenario, wire: Wire) -> str:
        if self.name == "embeddings":
            return embedding_model(scenario, wire)
        if self.name == "messages":
            return messages_model(scenario, wire)
        if self.name == "text":
            return text_model(scenario, wire)
        return chat_model(scenario, wire)

    def body(self, model: str, *, streaming: bool = False) -> dict[str, JsonValue]:
        prompt: Final = f"dec1 {self.name} {uuid.uuid4().hex}"
        flags: Final[dict[str, JsonValue]] = {"stream": True} if streaming else {}
        if self.name == "embeddings":
            return {"model": model, "input": prompt}
        if self.name == "messages":
            return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}], **flags}
        if self.name == "responses":
            return {"model": model, "input": prompt, **flags}
        if self.name == "text":
            return {"model": model, "prompt": prompt, **flags}
        return {"model": model, "messages": [{"role": "user", "content": prompt}], **flags}


CHAT: Final = Surface(
    "chat", "/v1/chat/completions", "acompletion", "openai", CHAT_PROMPT_TOKENS, CHAT_COMPLETION_TOKENS, CHAT_COST,
    "/chat/completions",
)  # fmt: skip
EMBEDDINGS: Final = Surface(
    "embeddings", "/v1/embeddings", "aembedding", "openai", EMBEDDING_PROMPT_TOKENS, 0, EMBEDDING_COST, "/embeddings"
)
MESSAGES: Final = Surface(
    "messages", "/v1/messages", "anthropic_messages", "anthropic", MESSAGES_INPUT_TOKENS, MESSAGES_OUTPUT_TOKENS,
    MESSAGES_COST, "",
)  # fmt: skip
RESPONSES: Final = Surface(
    "responses", "/v1/responses", "aresponses", "openai", RESPONSES_INPUT_TOKENS, RESPONSES_OUTPUT_TOKENS,
    RESPONSES_COST, "/responses",
)  # fmt: skip
TEXT: Final = Surface(
    "text", "/v1/completions", "atext_completion", "text-completion-openai", TEXT_PROMPT_TOKENS,
    TEXT_COMPLETION_TOKENS, TEXT_COST, "/completions",
)  # fmt: skip
SURFACES: Final = (CHAT, EMBEDDINGS, MESSAGES, RESPONSES, TEXT)
STREAMING_SURFACES: Final = (CHAT, MESSAGES, RESPONSES)
