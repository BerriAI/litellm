from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, string_value
from integration._support.database import read_rows
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

Provider = Literal["gemini", "vertex_ai"]
Endpoint = Literal["chat", "messages", "responses"]
Answer = Callable[[bool], Reply]

BACKEND: Final = "gemini-4-argon"
API_KEY: Final = "synthetic-gemini-key"
PROJECT: Final = "scripted-project"
LOCATION: Final = "us-central1"
MODEL_PATH: Final = f"/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{BACKEND}"
CONTINUED_MODEL_PATH: Final = MODEL_PATH.replace("/v1/", "/v1beta1/", 1)
SALT: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
CAP: Final = 1000
INPUT_COST: Final = 0.001
OUTPUT_COST: Final = 0.002
NO_CACHE: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
NOTHING_EXTRA: Final[Mapping[str, JsonValue]] = MappingProxyType({})
PROMPT: Final = "Write the long answer."
CHECKPOINT: Final = "checkpoint-token-never-for-clients"
HOP_TOKEN: Final = "hop-token-litellm-sends-back"
CUT_TOKEN: Final = "cut-token-for-the-caller"
REFUSAL: Final = "The scripted upstream refused this continuation request"
PROVIDERS: Final[tuple[Provider, ...]] = ("gemini", "vertex_ai")
ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
SPEND_COLUMNS: Final = (
    'SELECT request_id, call_type, status, cache_hit, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs"'
)


@dataclass(frozen=True, slots=True)
class Hop:
    pieces: tuple[str, str]
    finish: str
    token: str | None
    prompt_tokens: int
    output_tokens: int

    @property
    def text(self) -> str:
        return "".join(self.pieces)

    @property
    def usage(self) -> tuple[int, int]:
        return self.prompt_tokens, self.output_tokens


FIRST_HALF: Final = Hop(("The first ", "half, "), "CONTINUATION", HOP_TOKEN, 20, 7)
SECOND_HALF: Final = Hop(("then the ", "second half."), "STOP", None, 27, 5)
CUT: Final = Hop(("Cut at the ", "caller's cap"), "MAX_TOKENS", CUT_TOKEN, 20, 7)
PLAIN: Final = Hop(("A short ", "answer."), "STOP", None, 20, 7)
RESUMED: Final = Hop(("and the ", "rest of it."), "STOP", None, 27, 5)


def summed(*hops: Hop) -> tuple[int, int]:
    return sum(hop.prompt_tokens for hop in hops), sum(hop.output_tokens for hop in hops)


def frame(
    response_id: str, text: str, usage: tuple[int, int], finish: str | None = None, token: str | None = None
) -> dict[str, JsonValue]:
    prompt_tokens, output_tokens = usage
    return {
        "responseId": response_id,
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "index": 0,
                **({"finishReason": finish} if finish else {}),
                **({"continuationToken": token} if token else {}),
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": prompt_tokens + output_tokens,
        },
        "modelVersion": BACKEND,
    }


def stream_frames(response_id: str, hop: Hop) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    opening, closing = hop.pieces
    return (
        frame(response_id, opening, (hop.prompt_tokens, 1), token=CHECKPOINT),
        frame(response_id, closing, hop.usage, hop.finish, hop.token),
    )


def sse(frames: Sequence[Mapping[str, JsonValue]], abort_after: int | None = None) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(item)}\n\n".encode() for item in frames),
        abort_after=abort_after,
    )


def hop_reply(response_id: str, hop: Hop, stream: bool, abort_after: int | None = None) -> Reply:
    if stream:
        return sse(stream_frames(response_id, hop), abort_after)
    return Reply(body=json.dumps(frame(response_id, hop.text, hop.usage, hop.finish, hop.token)).encode())


def answers(response_id: str, hop: Hop) -> Answer:
    return lambda stream: hop_reply(response_id, hop, stream)


def refuses(status: int) -> Answer:
    error: Final = {"error": {"code": status, "message": REFUSAL, "status": "INVALID_ARGUMENT"}}
    return lambda _stream: Reply(status=status, body=json.dumps(error).encode())


def drops_before_answering(_stream: bool) -> Reply:
    return Reply(drop_connection=True)


def drops_mid_body(response_id: str, hop: Hop) -> Answer:
    return lambda stream: replace(hop_reply(response_id, hop, stream, abort_after=1), pause_between_chunks=0.5)


def streams(request: Request) -> bool:
    return ":streamGenerateContent" in request.target


def upstream(first: Answer, continued: Answer | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if "continuationToken" not in JSON_OBJECT.validate_json(request.body):
            return first(streams(request))
        assert continued is not None, f"Unscripted continuation request: {request.target}"
        return continued(streams(request))

    return respond


def register(gateway: Gateway, scenario: Scenario, provider: Provider, wire: Wire) -> str:
    if provider == "gemini":
        return scenario.model(
            model=f"gemini/{BACKEND}",
            api_base=wire.url,
            api_key=API_KEY,
            input_cost_per_token=INPUT_COST,
            output_cost_per_token=OUTPUT_COST,
        )
    return scenario.model(
        model=f"vertex_ai/{BACKEND}",
        api_base=f"{wire.url}{MODEL_PATH}",
        api_key=None,
        vertex_project=PROJECT,
        vertex_location=LOCATION,
        vertex_credentials=service_account_json(PROJECT, gateway.upstream_url.rstrip("/")),
        input_cost_per_token=INPUT_COST,
        output_cost_per_token=OUTPUT_COST,
    )


def target(provider: Provider, stream: bool, continued: bool = False) -> str:
    vertex_path: Final = CONTINUED_MODEL_PATH if continued else MODEL_PATH
    prefix: Final = f"/models/{BACKEND}" if provider == "gemini" else vertex_path
    return f"{prefix}:streamGenerateContent?alt=sse" if stream else f"{prefix}:generateContent"


def expected_auth(provider: Provider) -> tuple[str, str]:
    return ("x-goog-api-key", API_KEY) if provider == "gemini" else ("authorization", "Bearer scripted-token")


def received(
    wire: Wire, provider: Provider, stream: bool, continued: tuple[bool, ...]
) -> tuple[dict[str, JsonValue], ...]:
    requests: Final = wire.drain()
    assert [(request.method, request.target) for request in requests] == [
        ("POST", target(provider, stream, resumed)) for resumed in continued
    ], requests
    header, value = expected_auth(provider)
    assert [request.headers[header] for request in requests] == [value] * len(requests), requests
    return tuple(JSON_OBJECT.validate_json(request.body) for request in requests)


def path_of(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def request_body(
    endpoint: Endpoint,
    model: str,
    stream: bool,
    extra: Mapping[str, JsonValue] = NO_CACHE,
    *,
    prompt: str = PROMPT,
    cap: int | None = CAP,
) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                **({"max_tokens": cap} if cap is not None else {}),
                **({"stream_options": {"include_usage": True}} if stream else {}),
                **extra,
            }
        case "messages":
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                **({"max_tokens": cap} if cap is not None else {}),
                **extra,
            }
        case "responses":
            return {
                "model": model,
                "input": prompt,
                "stream": stream,
                **({"max_output_tokens": cap} if cap is not None else {}),
                **extra,
            }


def post(
    gateway: Gateway, endpoint: Endpoint, body: Mapping[str, JsonValue], *, key: str | None = None
) -> httpx.Response:
    return gateway.request("POST", path_of(endpoint), body, key=key, headers={"anthropic-version": "2023-06-01"})


def _step(value: JsonValue, key: str | int) -> JsonValue:
    if isinstance(key, str):
        return value.get(key) if isinstance(value, dict) else None
    return value[key] if isinstance(value, list) and len(value) > key else None


def at(value: JsonValue, *keys: str | int) -> JsonValue:
    return reduce(_step, keys, value)


def text_of(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def items_of(value: JsonValue) -> tuple[JsonValue, ...]:
    return tuple(value) if isinstance(value, list) else ()


def count_of(value: JsonValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def sse_payloads(text: str) -> tuple[dict[str, JsonValue], ...]:
    lines: Final = tuple(line for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]")
    return tuple(JSON_OBJECT.validate_json(line.removeprefix("data: ").encode()) for line in lines)


def payloads(text: str, stream: bool) -> tuple[dict[str, JsonValue], ...]:
    return sse_payloads(text) if stream else (JSON_OBJECT.validate_json(text.encode()),)


@dataclass(frozen=True, slots=True)
class Seen:
    ids: frozenset[str]
    answer: str
    streamed: str
    finishes: tuple[str, ...]
    usage: tuple[int | None, int | None] | None
    initial_prompt_tokens: JsonValue
    tokens: tuple[str, ...]
    token_carriers: tuple[int, ...]
    content_carriers: tuple[int, ...]
    incomplete_reason: JsonValue
    errors: tuple[dict[str, JsonValue], ...]
    raw: str


def _named(values: Sequence[JsonValue]) -> tuple[str, ...]:
    return tuple(value for value in values if isinstance(value, str))


def _errors(frames: Sequence[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        item
        for item in frames
        if item.get("error") is not None or item.get("type") in ("error", "response.failed", "response.error")
    )


def _chat_unary(payload: dict[str, JsonValue], raw: str) -> Seen:
    choices: Final = items_of(payload.get("choices"))
    answer: Final = text_of(at(payload, "choices", 0, "message", "content"))
    return Seen(
        ids=frozenset(_named((payload.get("id"),))),
        answer=answer,
        streamed=answer,
        finishes=_named(tuple(at(choice, "finish_reason") for choice in choices)),
        usage=(count_of(at(payload, "usage", "prompt_tokens")), count_of(at(payload, "usage", "completion_tokens"))),
        initial_prompt_tokens=at(payload, "usage", "prompt_tokens_details", "initial_prompt_tokens"),
        tokens=_named(
            tuple(at(choice, "message", "provider_specific_fields", "continuation_token") for choice in choices)
        ),
        token_carriers=(),
        content_carriers=(),
        incomplete_reason=None,
        errors=_errors((payload,)),
        raw=raw,
    )


def _chunk_token(chunk: dict[str, JsonValue]) -> JsonValue:
    return at(chunk, "choices", 0, "delta", "provider_specific_fields", "continuation_token")


def _chat_stream(chunks: tuple[dict[str, JsonValue], ...], raw: str) -> Seen:
    contents: Final = tuple(text_of(at(chunk, "choices", 0, "delta", "content")) for chunk in chunks)
    metered: Final = tuple(chunk for chunk in chunks if isinstance(chunk.get("usage"), dict))
    answer: Final = "".join(contents)
    return Seen(
        ids=frozenset(_named(tuple(chunk.get("id") for chunk in chunks))),
        answer=answer,
        streamed=answer,
        finishes=_named(tuple(at(chunk, "choices", 0, "finish_reason") for chunk in chunks)),
        usage=(
            (
                count_of(at(metered[-1], "usage", "prompt_tokens")),
                count_of(at(metered[-1], "usage", "completion_tokens")),
            )
            if metered
            else None
        ),
        initial_prompt_tokens=(
            at(metered[-1], "usage", "prompt_tokens_details", "initial_prompt_tokens") if metered else None
        ),
        tokens=_named(tuple(_chunk_token(chunk) for chunk in chunks)),
        token_carriers=tuple(index for index, chunk in enumerate(chunks) if _chunk_token(chunk) is not None),
        content_carriers=tuple(index for index, content in enumerate(contents) if content),
        incomplete_reason=None,
        errors=_errors(chunks),
        raw=raw,
    )


def _messages_unary(payload: dict[str, JsonValue], raw: str) -> Seen:
    blocks: Final = items_of(payload.get("content"))
    answer: Final = "".join(text_of(at(block, "text")) for block in blocks if at(block, "type") == "text")
    return Seen(
        ids=frozenset(_named((payload.get("id"),))),
        answer=answer,
        streamed=answer,
        finishes=_named((payload.get("stop_reason"),)),
        usage=(count_of(at(payload, "usage", "input_tokens")), count_of(at(payload, "usage", "output_tokens"))),
        initial_prompt_tokens=None,
        tokens=(),
        token_carriers=(),
        content_carriers=(),
        incomplete_reason=None,
        errors=_errors((payload,)),
        raw=raw,
    )


def _messages_stream(events: tuple[dict[str, JsonValue], ...], raw: str) -> Seen:
    closings: Final = tuple(event for event in events if event.get("type") == "message_delta")
    answer: Final = "".join(
        text_of(at(event, "delta", "text")) for event in events if at(event, "delta", "type") == "text_delta"
    )
    return Seen(
        ids=frozenset(
            _named(tuple(at(event, "message", "id") for event in events if event.get("type") == "message_start"))
        ),
        answer=answer,
        streamed=answer,
        finishes=_named(tuple(at(event, "delta", "stop_reason") for event in closings)),
        usage=(
            (
                count_of(at(closings[-1], "usage", "input_tokens")),
                count_of(at(closings[-1], "usage", "output_tokens")),
            )
            if closings
            else None
        ),
        initial_prompt_tokens=None,
        tokens=(),
        token_carriers=(),
        content_carriers=(),
        incomplete_reason=None,
        errors=_errors(events),
        raw=raw,
    )


def _part_texts(item: JsonValue) -> tuple[str, ...]:
    return tuple(
        text_of(at(part, "text")) for part in items_of(at(item, "content")) if at(part, "type") == "output_text"
    )


def output_text(response: JsonValue) -> str:
    messages: Final = tuple(item for item in items_of(at(response, "output")) if at(item, "type") == "message")
    return "".join("".join(_part_texts(item)) for item in messages)


def _responses(response: JsonValue, streamed: str | None, errors: tuple[dict[str, JsonValue], ...], raw: str) -> Seen:
    answer: Final = output_text(response)
    return Seen(
        ids=frozenset(_named((at(response, "id"),))),
        answer=answer,
        streamed=answer if streamed is None else streamed,
        finishes=_named((at(response, "status"),)),
        usage=(
            (count_of(at(response, "usage", "input_tokens")), count_of(at(response, "usage", "output_tokens")))
            if response is not None
            else None
        ),
        initial_prompt_tokens=None,
        tokens=_named((at(response, "provider_specific_fields", "continuation_token"),)),
        token_carriers=(),
        content_carriers=(),
        incomplete_reason=at(response, "incomplete_details", "reason"),
        errors=errors,
        raw=raw,
    )


def _responses_stream(events: tuple[dict[str, JsonValue], ...], raw: str) -> Seen:
    settled: Final = tuple(
        event.get("response") for event in events if event.get("type") in ("response.completed", "response.incomplete")
    )
    deltas: Final = "".join(
        text_of(event.get("delta")) for event in events if event.get("type") == "response.output_text.delta"
    )
    return _responses(settled[-1] if settled else None, deltas, _errors(events), raw)


def seen(endpoint: Endpoint, stream: bool, frames: tuple[dict[str, JsonValue], ...], raw: str) -> Seen:
    match (endpoint, stream):
        case ("chat", False):
            return _chat_unary(frames[0], raw)
        case ("chat", True):
            return _chat_stream(frames, raw)
        case ("messages", False):
            return _messages_unary(frames[0], raw)
        case ("messages", True):
            return _messages_stream(frames, raw)
        case ("responses", False):
            return _responses(frames[0], None, _errors(frames), raw)
        case _:
            return _responses_stream(frames, raw)


def answered(endpoint: Endpoint, stream: bool, response: httpx.Response) -> Seen:
    assert response.status_code == 200, response.text
    return seen(endpoint, stream, payloads(response.text, stream), response.text)


def ends_cleanly(endpoint: Endpoint, raw: str) -> bool:
    if endpoint == "chat":
        return raw.rstrip().endswith("data: [DONE]")
    closing: Final = sse_payloads(raw)[-1].get("type")
    if endpoint == "messages":
        return closing == "message_stop"
    return closing in ("response.completed", "response.incomplete")


def complete(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "stop"
        case "messages":
            return "end_turn"
        case "responses":
            return "completed"


def cut_off(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "length"
        case "messages":
            return "max_tokens"
        case "responses":
            return "incomplete"


def carries_token(endpoint: Endpoint, stream: bool) -> bool:
    return endpoint == "chat" or (endpoint == "responses" and stream)


def single(values: frozenset[str]) -> str:
    assert len(values) == 1, values
    return next(iter(values))


def upstream_response_id(endpoint: Endpoint, caller_id: str) -> str:
    if endpoint != "responses":
        return caller_id
    decrypted: Final = decrypt_if_encrypted_with(caller_id.removeprefix("resp_"), SALT)
    assert decrypted is not None, caller_id
    issued: Final = decrypted.split(";")[0].split("response_id:")[-1]
    return base64.b64decode(issued.removeprefix("resp_")).decode().split(";")[-1].removeprefix("response_id:")


def assert_one_id(endpoint: Endpoint, stream: bool, observed: Seen, response_id: str) -> None:
    caller_id: Final = single(observed.ids)
    if endpoint == "messages" and stream:
        assert caller_id.startswith("msg_"), caller_id
        return
    assert upstream_response_id(endpoint, caller_id) == response_id, (caller_id, response_id)


def call_type(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "acompletion"
        case "messages":
            return "anthropic_messages"
        case "responses":
            return "aresponses"


def number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def cost_of(usage: tuple[int, int]) -> float:
    prompt_tokens, completion_tokens = usage
    return prompt_tokens * INPUT_COST + completion_tokens * OUTPUT_COST


def spend_rows(model: str, expected: int = 1) -> Sequence[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: read_rows(f"{SPEND_COLUMNS} WHERE model_group=%s ORDER BY request_id", (model,)),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    assert len(rows) == expected, rows
    return rows


def assert_billed(model: str, endpoint: Endpoint, usage: tuple[int, int], identities: frozenset[str]) -> None:
    (row,) = spend_rows(model)
    assert (row["call_type"], row["status"], row["prompt_tokens"], row["completion_tokens"]) == (
        call_type(endpoint),
        "success",
        *usage,
    ), row
    assert number(row["spend"]) == pytest.approx(cost_of(usage)), row
    assert string_value(row["request_id"]) in identities, (row, identities)
