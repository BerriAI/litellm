import base64
import binascii
import json
import math
import os
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol
from urllib.parse import urlsplit

import anthropic
import httpx
import openai
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.responses_vendor import answer, error, newest_marker, sse
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

Endpoint = Literal["chat", "responses", "messages"]


JSON: Final = TypeAdapter(dict[str, JsonValue])


ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])


SIGNING_KEY: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")


TOKEN: Final = "synthetic-mantle-bearer"


GPT: Final = "bedrock_mantle/openai.gpt-5.6-sol"


GPT_REGION: Final = "bedrock_mantle/us-east-1/openai.gpt-5.6-sol"

GPT_BARE: Final = "openai.gpt-5.6-sol"


GPT_UNFLAGGED_ROW: Final = "bedrock_mantle/openai.gpt-5.4"


GPT_FLAGGED_ROW: Final = "bedrock_mantle/openai.gpt-6-luna"


GPT_ODD_STRING_FLAG: Final = "bedrock_mantle/openai.gpt-5.5"


GPT_ODD_INT_FLAG: Final = "bedrock_mantle/openai.gpt-daybreak-blue-5.6-sol"
ODD_FLAGS: Final[tuple[tuple[str, str, JsonValue], ...]] = (
    ("string-true", GPT_ODD_STRING_FLAG, "true"),
    ("int-one", GPT_ODD_INT_FLAG, 1),
    ("int-zero", GPT_ODD_INT_FLAG, 0),
    ("5kb-string", GPT_ODD_STRING_FLAG, "x" * 5120),
)


CLAUDE: Final = "bedrock_mantle/anthropic.claude-haiku-4-5"


AZURE: Final = "azure/gpt-5.6"


THIRD_PARTY: Final = "openai/gpt-5.6"


SYSTEM: Final = "Reply with the signature the user gives you."


SYSTEM_POINT: Final[list[JsonValue]] = [{"location": "message", "role": "system"}]


EXPLICIT: Final[dict[str, JsonValue]] = {"mode": "explicit"}


IMPLICIT: Final[dict[str, JsonValue]] = {"mode": "implicit"}


EPHEMERAL: Final[dict[str, JsonValue]] = {"type": "ephemeral"}


NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}


MAX_TOKENS: Final = 64


INPUT_TOKENS: Final = 2730


CACHED_TOKENS: Final = 1024


WRITTEN_TOKENS: Final = 1700


UNCACHED_TOKENS: Final = INPUT_TOKENS - CACHED_TOKENS - WRITTEN_TOKENS


OUTPUT_TOKENS: Final = 7


USAGE: Final[dict[str, JsonValue]] = {
    "input_tokens": INPUT_TOKENS,
    "output_tokens": OUTPUT_TOKENS,
    "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
    "input_tokens_details": {"cached_tokens": CACHED_TOKENS, "cache_write_tokens": WRITTEN_TOKENS},
    "output_tokens_details": {"reasoning_tokens": 0},
}


ANTHROPIC_USAGE: Final[dict[str, JsonValue]] = {
    "input_tokens": UNCACHED_TOKENS,
    "output_tokens": OUTPUT_TOKENS,
    "cache_read_input_tokens": CACHED_TOKENS,
    "cache_creation_input_tokens": WRITTEN_TOKENS,
}


CHAT_USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": INPUT_TOKENS,
    "completion_tokens": OUTPUT_TOKENS,
    "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
    "prompt_tokens_details": {"cached_tokens": CACHED_TOKENS},
}


COST_MAP: Final = JSON.validate_json(Path("model_prices_and_context_window.json").read_bytes())


RESPONSES_PATH: Final = "/openai/v1/responses"


ANTHROPIC_PATH: Final = "/anthropic/v1/messages"


BURST: Final = 30


ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "responses", "messages")


def cost_rate(model: str, field: str) -> float:
    value: Final = JSON.validate_python(COST_MAP[model])[field]
    assert isinstance(value, int | float), (model, field, value)
    return float(value)


def expected_spend(model: str) -> float:
    return (
        UNCACHED_TOKENS * cost_rate(model, "input_cost_per_token")
        + CACHED_TOKENS * cost_rate(model, "cache_read_input_token_cost")
        + WRITTEN_TOKENS * cost_rate(model, "cache_creation_input_token_cost")
        + OUTPUT_TOKENS * cost_rate(model, "output_cost_per_token")
    )


def fresh_marker() -> str:
    return uuid.uuid4().hex


def prompt_text(marker: str) -> str:
    return f"Return the signature marker-{marker}."


def valid_options(options: JsonValue) -> bool:
    if options is None:
        return True
    if not isinstance(options, dict) or not set(options) <= {"mode", "ttl"}:
        return False
    return options.get("mode", "implicit") in ("implicit", "explicit")


def message_item(identity: str, text: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{identity}",
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}] if status == "completed" else [],
    }


def responses_object(identity: str, model: str, text: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [message_item(identity, text, "completed")],
        "usage": USAGE,
    }


def responses_stream(identity: str, model: str, text: str) -> tuple[bytes, ...]:
    response: Final = responses_object(identity, model, text)
    return (
        sse({"type": "response.created", "sequence_number": 0, "response": {**response, "status": "in_progress"}}),
        sse(
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": message_item(identity, text, "in_progress"),
            }
        ),
        sse(
            {
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "item_id": f"msg_{identity}",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            }
        ),
        sse(
            {
                "type": "response.output_item.done",
                "sequence_number": 3,
                "output_index": 0,
                "item": message_item(identity, text, "completed"),
            }
        ),
        sse({"type": "response.completed", "sequence_number": 4, "response": response}),
    )


def anthropic_message(identity: str, model: str, text: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": ANTHROPIC_USAGE,
    }


def anthropic_stream(identity: str, model: str, text: str) -> tuple[bytes, ...]:
    started: Final = {**anthropic_message(identity, model, text), "content": [], "stop_reason": None}
    return (
        sse({"type": "message_start", "message": {**started, "usage": {**ANTHROPIC_USAGE, "output_tokens": 0}}}),
        sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        sse({"type": "content_block_stop", "index": 0}),
        sse(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": OUTPUT_TOKENS},
            }
        ),
        sse({"type": "message_stop"}),
    )


def chat_completion(identity: str, model: str, text: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": CHAT_USAGE,
    }


def issued_id(prefix: str, marker: str | None) -> str:
    return f"{prefix}{marker or 'unmarked'}-{uuid.uuid4().hex[:8]}"


def mantle_peer(*, pause: float = 0) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request.method
        assert request.headers.get("authorization") == f"Bearer {TOKEN}", request.headers
        body: Final = JSON.validate_json(request.body)
        model: Final = str(body["model"])
        marker: Final = newest_marker(request.body.decode())
        text: Final = answer(marker)
        streaming: Final = body.get("stream") is True
        path: Final = urlsplit(request.target).path
        if path == ANTHROPIC_PATH:
            identity: Final = issued_id("msg_", marker)
            if streaming:
                return Reply(
                    content_type="text/event-stream",
                    chunks=anthropic_stream(identity, model, text),
                    pause_between_chunks=pause,
                )
            return Reply(body=json.dumps(anthropic_message(identity, model, text)).encode())
        assert path == RESPONSES_PATH, request.target
        if not valid_options(body.get("prompt_cache_options")):
            return error(400, "Invalid prompt_cache_options", "invalid_prompt_cache_options")
        response_id: Final = issued_id("resp_", marker)
        if streaming:
            return Reply(
                content_type="text/event-stream",
                chunks=responses_stream(response_id, model, text),
                pause_between_chunks=pause,
            )
        return Reply(body=json.dumps(responses_object(response_id, model, text)).encode())

    return respond


def failing_peer(status: int, message: str, code: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert urlsplit(request.target).path == RESPONSES_PATH, request.target
        return error(status, message, code)

    return respond


def openai_shaped_peer() -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = JSON.validate_json(request.body)
        model: Final = str(body["model"])
        marker: Final = newest_marker(request.body.decode())
        text: Final = answer(marker)
        path: Final = urlsplit(request.target).path
        if path.endswith("/chat/completions"):
            return Reply(body=json.dumps(chat_completion(issued_id("chatcmpl-", marker), model, text)).encode())
        assert path.endswith("/responses"), request.target
        return Reply(body=json.dumps(responses_object(issued_id("resp_", marker), model, text)).encode())

    return respond


def system_item(*, marked: bool, endpoint: Endpoint) -> dict[str, JsonValue]:
    part: Final[dict[str, JsonValue]] = {"type": "input_text", "text": SYSTEM}
    content: Final[list[JsonValue]] = [{**part, "prompt_cache_breakpoint": EXPLICIT} if marked else part]
    if endpoint == "responses":
        return {"role": "system", "content": content}
    return {"type": "message", "role": "system", "content": content}


def user_item(text: str, endpoint: Endpoint) -> dict[str, JsonValue]:
    if endpoint == "responses":
        return {"role": "user", "content": text}
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def expected_wire(
    model: str,
    prompt: str,
    *,
    endpoint: Endpoint,
    marked: bool,
    options: JsonValue | None = IMPLICIT,
    **extra: JsonValue,
) -> dict[str, JsonValue]:
    body: Final[dict[str, JsonValue]] = {
        "model": model.rsplit("/", 1)[-1],
        "input": [system_item(marked=marked, endpoint=endpoint), user_item(prompt, endpoint)],
        "max_output_tokens": MAX_TOKENS,
        **extra,
    }
    return body if options is None else {**body, "prompt_cache_options": options}


def body_of(request: Request) -> dict[str, JsonValue]:
    return JSON.validate_json(request.body)


def without_stream(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in body.items() if key != "stream"}


def only_received(wire: Wire) -> Request:
    (request,) = wire.drain()
    return request


def assert_wire(request: Request, expected: Mapping[str, JsonValue], *, streaming: bool) -> None:
    body: Final = body_of(request)
    assert urlsplit(request.target).path == RESPONSES_PATH, request.target
    assert without_stream(body) == expected, json.dumps(body, sort_keys=True)
    assert (body.get("stream") is True) is streaming, body.get("stream")


def breakpoint_count(body: Mapping[str, JsonValue]) -> int:
    parts: Final = (
        part
        for item in ITEMS.validate_python(body["input"])
        for part in (item["content"] if isinstance(item["content"], list) else ())
    )  # comprehension-ok: nested input items
    return sum(1 for part in parts if isinstance(part, dict) and "prompt_cache_breakpoint" in part)


def endpoint_path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "responses":
            return "/v1/responses"
        case "messages":
            return "/v1/messages"


def request_body(
    endpoint: Endpoint, model: str, prompt: str, *, stream: bool = False, **extra: JsonValue
) -> dict[str, JsonValue]:
    match endpoint:
        case "chat":
            return {
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "stream": stream,
                **({"stream_options": {"include_usage": True}} if stream else {}),
                **extra,
            }
        case "responses":
            return {
                "model": model,
                "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "max_output_tokens": MAX_TOKENS,
                "stream": stream,
                **extra,
            }
        case "messages":
            return {
                "model": model,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "stream": stream,
                **extra,
            }


@dataclass(frozen=True, slots=True)
class Outcome:
    status: int
    call_id: str
    response_id: str
    text: str
    usage: dict[str, JsonValue]
    headers: Mapping[str, str]
    raw: str


def sse_payloads(lines: Iterable[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON.validate_json(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]")


def unwrapped(identity: str) -> str | None:
    try:
        decoded: Final = base64.b64decode(identity.removeprefix("resp_"), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    return decoded.rsplit("response_id:", 1)[1] if decoded.startswith("litellm:") else None


def upstream_id_of(identity: str) -> str:
    managed: Final = decrypt_if_encrypted_with(identity.removeprefix("resp_"), SIGNING_KEY)
    wrapped: Final = identity if managed is None else managed.rsplit("response_id:", 1)[1].split(";", 1)[0]
    inner: Final = unwrapped(wrapped)
    return wrapped if inner is None else inner


def row_key(request_id: str) -> str:
    return upstream_id_of(request_id.split("_cache_hit", 1)[0])


def caller_sees_upstream_id(endpoint: Endpoint, *, stream: bool) -> bool:
    return (endpoint, stream) != ("messages", True)


def usage_of(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    usage: Final = payload.get("usage")
    return JSON.validate_python(usage) if isinstance(usage, dict) else {}


def chat_text(payload: Mapping[str, JsonValue]) -> str:
    (choice,) = ITEMS.validate_python(payload["choices"])
    return str(JSON.validate_python(choice["message"])["content"])


def chat_stream_fields(payloads: Sequence[Mapping[str, JsonValue]]) -> tuple[str, str, dict[str, JsonValue]]:
    (identity,) = {str(chunk["id"]) for chunk in payloads if "id" in chunk}
    choices: Final = tuple(ITEMS.validate_python(chunk["choices"]) for chunk in payloads if chunk.get("choices"))
    deltas: Final = tuple(JSON.validate_python(choice[0]["delta"]) for choice in choices if choice)
    text: Final = "".join(str(delta["content"]) for delta in deltas if isinstance(delta.get("content"), str))
    usages: Final = tuple(usage_of(chunk) for chunk in payloads if isinstance(chunk.get("usage"), dict))
    return identity, text, usages[-1] if usages else {}


def responses_text(payload: Mapping[str, JsonValue]) -> str:
    items: Final = ITEMS.validate_python(payload["output"])
    parts: Final = tuple(ITEMS.validate_python(item["content"]) for item in items if item.get("type") == "message")
    return "".join(str(part["text"]) for part in parts[0] if part.get("type") == "output_text")


def completed_response(events: Iterable[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    (completed,) = tuple(event for event in events if event.get("type") == "response.completed")
    return JSON.validate_python(completed["response"])


def messages_text(payload: Mapping[str, JsonValue]) -> str:
    return "".join(str(block["text"]) for block in ITEMS.validate_python(payload["content"]) if "text" in block)


def messages_stream_fields(payloads: Sequence[Mapping[str, JsonValue]]) -> tuple[str, str, dict[str, JsonValue]]:
    (started,) = tuple(payload for payload in payloads if payload.get("type") == "message_start")
    message: Final = JSON.validate_python(started["message"])
    deltas: Final = tuple(JSON.validate_python(payload["delta"]) for payload in payloads if "delta" in payload)
    text: Final = "".join(str(delta["text"]) for delta in deltas if delta.get("type") == "text_delta")
    final_usages: Final = tuple(usage_of(payload) for payload in payloads if payload.get("type") == "message_delta")
    return str(message["id"]), text, {**usage_of(message), **(final_usages[-1] if final_usages else {})}


def parse_outcome(
    endpoint: Endpoint, *, stream: bool, status: int, headers: Mapping[str, str], lines: tuple[str, ...]
) -> Outcome:
    call_id: Final = headers.get("x-litellm-call-id", "")
    raw: Final = "\n".join(lines)
    if status != 200:
        return Outcome(status, call_id, "", "", {}, headers, raw)
    payloads: Final = sse_payloads(lines) if stream else (JSON.validate_json(raw),)
    match endpoint, stream:
        case "chat", True:
            identity, text, usage = chat_stream_fields(payloads)
            return Outcome(status, call_id, upstream_id_of(identity), text, usage, headers, raw)
        case "chat", False:
            return Outcome(
                status,
                call_id,
                upstream_id_of(str(payloads[0]["id"])),
                chat_text(payloads[0]),
                usage_of(payloads[0]),
                headers,
                raw,
            )
        case "responses", True:
            completed: Final = completed_response(payloads)
            return Outcome(
                status,
                call_id,
                upstream_id_of(str(completed["id"])),
                responses_text(completed),
                usage_of(completed),
                headers,
                raw,
            )
        case "responses", False:
            return Outcome(
                status,
                call_id,
                upstream_id_of(str(payloads[0]["id"])),
                responses_text(payloads[0]),
                usage_of(payloads[0]),
                headers,
                raw,
            )
        case "messages", True:
            identity, text, usage = messages_stream_fields(payloads)
            return Outcome(status, call_id, upstream_id_of(identity), text, usage, headers, raw)
        case "messages", False:
            return Outcome(
                status,
                call_id,
                upstream_id_of(str(payloads[0]["id"])),
                messages_text(payloads[0]),
                usage_of(payloads[0]),
                headers,
                raw,
            )
    raise AssertionError((endpoint, stream))


def send(gateway: Gateway, endpoint: Endpoint, body: Mapping[str, JsonValue], *, key: str | None = None) -> Outcome:
    stream: Final = body.get("stream") is True
    headers: Final = {"Authorization": f"Bearer {gateway.key if key is None else key}"}
    with gateway.client.stream("POST", endpoint_path(endpoint), json=body, headers=headers, timeout=60) as response:
        lines: Final = tuple(line for line in response.iter_lines() if line)
    return parse_outcome(endpoint, stream=stream, status=response.status_code, headers=response.headers, lines=lines)


def send_raw(gateway: Gateway, endpoint: Endpoint, content: str, *, key: str | None = None) -> Outcome:
    headers: Final = {
        "Authorization": f"Bearer {gateway.key if key is None else key}",
        "Content-Type": "application/json",
    }
    response: Final = gateway.client.post(
        endpoint_path(endpoint), content=content.encode(), headers=headers, timeout=60
    )
    lines: Final = tuple(line for line in response.text.splitlines() if line)
    return parse_outcome(endpoint, stream=False, status=response.status_code, headers=response.headers, lines=lines)


def matching_rows(name: str, markers: frozenset[str]) -> tuple[dict[str, JsonValue], ...]:
    rows: Final = read_rows(
        'SELECT request_id, status, spend, prompt_tokens, completion_tokens, cache_hit FROM "LiteLLM_SpendLogs" '
        "WHERE model_group = %s",
        (name,),
    )
    return tuple(row for row in rows if any(marker in row_key(str(row["request_id"])) for marker in markers))


def spend_rows(
    name: str, markers: frozenset[str], *, expected: int, seconds: float = 90
) -> tuple[dict[str, JsonValue], ...]:
    return eventually(lambda: matching_rows(name, markers), lambda found: len(found) >= expected, seconds=seconds)


def success_row(name: str, *needles: str) -> dict[str, JsonValue]:
    (row,) = tuple(row for row in spend_rows(name, frozenset(needles), expected=1) if row["cache_hit"] != "True")
    assert row["status"] == "success", row
    return row


def failure_row(call_id: str) -> dict[str, JsonValue]:
    assert call_id, "No call id to look the failure row up by"
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (call_id,)),
        lambda found: len(found) >= 1,
        seconds=90,
    )
    (row,) = tuple(rows)
    assert row["status"] == "failure", row
    return row


def assert_priced_row(row: Mapping[str, JsonValue], model: str) -> None:
    assert row["prompt_tokens"] == INPUT_TOKENS, row
    assert row["completion_tokens"] == OUTPUT_TOKENS, row
    assert math.isclose(float(str(row["spend"])), expected_spend(model), rel_tol=1e-9), (row, expected_spend(model))


def assert_usage(endpoint: Endpoint, usage: Mapping[str, JsonValue]) -> None:
    match endpoint:
        case "chat":
            assert usage["prompt_tokens"] == INPUT_TOKENS, usage
            assert usage["completion_tokens"] == OUTPUT_TOKENS, usage
            details: Final = JSON.validate_python(usage["prompt_tokens_details"])
            assert details["cached_tokens"] == CACHED_TOKENS, usage
            assert details["cache_write_tokens"] == WRITTEN_TOKENS, usage
            assert details["cache_creation_tokens"] == WRITTEN_TOKENS, usage
        case "responses":
            assert usage["input_tokens"] == INPUT_TOKENS, usage
            assert usage["output_tokens"] == OUTPUT_TOKENS, usage
            input_details: Final = JSON.validate_python(usage["input_tokens_details"])
            assert input_details["cached_tokens"] == CACHED_TOKENS, usage
            assert input_details["cache_write_tokens"] == WRITTEN_TOKENS, usage
        case "messages":
            assert usage["input_tokens"] == UNCACHED_TOKENS, usage
            assert usage["output_tokens"] == OUTPUT_TOKENS, usage
            assert usage["cache_read_input_tokens"] == CACHED_TOKENS, usage
            assert usage["cache_creation_input_tokens"] == WRITTEN_TOKENS, usage


def assert_answered(outcome: Outcome, marker: str) -> None:
    assert outcome.status == 200, (outcome.status, outcome.raw)
    assert outcome.text == answer(marker), outcome.raw


def deployment(
    scenario: Scenario,
    wire: Wire,
    model: str,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    points: JsonValue = SYSTEM_POINT,
    **extra: JsonValue,
) -> str:
    return scenario.model(
        model_info=model_info,
        model=model,
        api_base=wire.url,
        api_key=TOKEN,
        aws_region_name="us-east-1",
        cache_control_injection_points=points,
        **extra,
    )


def settled(gateway: Gateway, name: str, wire: Wire, *, accepted: frozenset[int] = frozenset({200})) -> None:
    eventually(
        lambda: tuple(
            gateway.request(
                "POST", "/v1/chat/completions", request_body("chat", name, prompt_text(fresh_marker()), cache=NO_CACHE)
            ).status_code
            for _ in range(12)
        ),
        lambda codes: all(code in accepted for code in codes),
        seconds=90,
    )
    wire.drain()


def mantle_deployment(
    gateway: Gateway,
    scenario: Scenario,
    wire: Wire,
    model: str = GPT,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **extra: JsonValue,
) -> str:
    name: Final = deployment(scenario, wire, model, model_info=model_info, **extra)
    settled(gateway, name, wire)
    return name


def observe(
    gateway: Gateway, wire: Wire, endpoint: Endpoint, body: Mapping[str, JsonValue], *, key: str | None = None
) -> tuple[Outcome, Request]:
    wire.drain()
    outcome: Final = send(gateway, endpoint, body, key=key)
    return outcome, only_received(wire)


def marked_cell(gateway: Gateway, endpoint: Endpoint, *, stream: bool) -> None:
    marker: Final = fresh_marker()
    with wire_server(mantle_peer()) as wire, gateway.scenario() as scenario:
        name: Final = mantle_deployment(gateway, scenario, wire)
        outcome, received = observe(
            gateway, wire, endpoint, request_body(endpoint, name, prompt_text(marker), stream=stream)
        )
        assert_answered(outcome, marker)
        assert_wire(received, expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True), streaming=stream)
        assert_usage(endpoint, outcome.usage)
        row: Final = success_row(name, marker)
        assert_priced_row(row, GPT)
        if caller_sees_upstream_id(endpoint, stream=stream):
            assert row_key(str(row["request_id"])) == outcome.response_id, (row, outcome.response_id)
        if not stream:
            assert math.isclose(float(outcome.headers["x-litellm-response-cost"]), expected_spend(GPT), rel_tol=1e-9), (
                outcome.headers
            )


def openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def async_anthropic_client(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


class Dumpable(Protocol):
    def model_dump(self, *, exclude_none: bool = ...) -> Mapping[str, object]: ...


def dump_model(value: Dumpable) -> dict[str, JsonValue]:
    return JSON.validate_python(value.model_dump(exclude_none=True))


def sdk_outcome(endpoint: Endpoint, payloads: Sequence[Mapping[str, JsonValue]], *, stream: bool) -> Outcome:
    match endpoint, stream:
        case "chat", True:
            identity, text, usage = chat_stream_fields(payloads)
            return Outcome(200, "", upstream_id_of(identity), text, usage, {}, "")
        case "chat", False:
            return Outcome(
                200, "", upstream_id_of(str(payloads[0]["id"])), chat_text(payloads[0]), usage_of(payloads[0]), {}, ""
            )
        case "responses", _:
            completed: Final = completed_response(payloads) if stream else dict(payloads[0])
            return Outcome(
                200,
                "",
                upstream_id_of(str(completed["id"])),
                responses_text(completed),
                usage_of(completed),
                {},
                "",
            )
        case "messages", True:
            identity, text, usage = messages_stream_fields(payloads)
            return Outcome(200, "", upstream_id_of(identity), text, usage, {}, "")
        case "messages", False:
            return Outcome(
                200,
                "",
                upstream_id_of(str(payloads[0]["id"])),
                messages_text(payloads[0]),
                usage_of(payloads[0]),
                {},
                "",
            )
    raise AssertionError((endpoint, stream))


def sdk_call(gateway: Gateway, endpoint: Endpoint, name: str, prompt: str, *, stream: bool) -> Outcome:
    match endpoint, stream:
        case "chat", False:
            completion: Final = openai_client(gateway).chat.completions.create(
                model=name,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
            )
            return sdk_outcome(endpoint, (dump_model(completion),), stream=False)
        case "chat", True:
            chunks: Final = openai_client(gateway).chat.completions.create(
                model=name,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
            )
            return sdk_outcome(endpoint, tuple(dump_model(chunk) for chunk in chunks), stream=True)
        case "responses", False:
            response: Final = openai_client(gateway).responses.create(
                model=name,
                input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_output_tokens=MAX_TOKENS,
            )
            return sdk_outcome(endpoint, (dump_model(response),), stream=False)
        case "responses", True:
            events: Final = openai_client(gateway).responses.create(
                model=name,
                input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_output_tokens=MAX_TOKENS,
                stream=True,
            )
            return sdk_outcome(endpoint, tuple(dump_model(event) for event in events), stream=True)
        case "messages", False:
            message: Final = anthropic_client(gateway).messages.create(
                model=name, system=SYSTEM, messages=[{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS
            )
            return sdk_outcome(endpoint, (dump_model(message),), stream=False)
        case "messages", True:
            with anthropic_client(gateway).messages.stream(
                model=name, system=SYSTEM, messages=[{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS
            ) as events_stream:
                raw_events: Final = tuple(dump_model(event) for event in events_stream)
            return sdk_outcome(endpoint, raw_events, stream=True)
    raise AssertionError((endpoint, stream))


async def async_sdk_call(gateway: Gateway, endpoint: Endpoint, name: str, prompt: str) -> Outcome:
    match endpoint:
        case "chat":
            completion: Final = await async_openai_client(gateway).chat.completions.create(
                model=name,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
            )
            return sdk_outcome(endpoint, (dump_model(completion),), stream=False)
        case "responses":
            response: Final = await async_openai_client(gateway).responses.create(
                model=name,
                input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_output_tokens=MAX_TOKENS,
            )
            return sdk_outcome(endpoint, (dump_model(response),), stream=False)
        case "messages":
            message: Final = await async_anthropic_client(gateway).messages.create(
                model=name, system=SYSTEM, messages=[{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS
            )
            return sdk_outcome(endpoint, (dump_model(message),), stream=False)


def assert_marked_sdk_cell(
    wire: Wire, endpoint: Endpoint, name: str, marker: str, outcome: Outcome, *, stream: bool
) -> None:
    assert_answered(outcome, marker)
    assert_wire(
        only_received(wire), expected_wire(GPT, prompt_text(marker), endpoint=endpoint, marked=True), streaming=stream
    )
    assert_usage(endpoint, outcome.usage)
    assert_priced_row(success_row(name, marker), GPT)


HOSTILE_OPTIONS: Final[dict[str, JsonValue]] = {
    "int": 5,
    "list": [{"mode": "explicit"}],
    "empty-string": "",
    "5kb-string": "x" * 5120,
}


MALFORMED_POINTS: Final[dict[str, JsonValue]] = {
    "null": None,
    "string": "system",
    "int": 5,
    "dict": {"location": "message", "role": "system"},
    "string-list": ["system"],
    "no-location": [{"role": "system"}],
}
MIXED_POINTS: Final[list[JsonValue]] = ["system", *SYSTEM_POINT, 3]


Step = tuple[Endpoint, bool, str]


def plan_burst(size: int) -> tuple[Step, ...]:
    return tuple((ENDPOINTS[index % 3], index % 3 == 0, fresh_marker()) for index in range(size))


def burst(gateway: Gateway, name: str, size: int) -> tuple[tuple[str, Outcome], ...]:
    def call(step: Step) -> tuple[str, Outcome]:
        endpoint, stream, marker = step
        return marker, send(gateway, endpoint, request_body(endpoint, name, prompt_text(marker), stream=stream))

    with ThreadPoolExecutor(max_workers=10) as pool:
        return tuple(pool.map(call, plan_burst(size)))


def assert_marked(request: Request, *, marked: bool) -> None:
    body: Final = body_of(request)
    assert breakpoint_count(body) == (1 if marked else 0), request.body
    assert ("prompt_cache_options" in body) is marked, request.body


def assert_burst_landed(wire: Wire, name: str, outcomes: Sequence[tuple[str, Outcome]], *, marked: bool) -> None:
    for marker, outcome in outcomes:
        assert_answered(outcome, marker)
    identities: Final = frozenset(outcome.response_id for _, outcome in outcomes)
    assert len(identities) == len(outcomes), identities
    received: Final = wire.drain()
    assert len(received) == len(outcomes), (len(received), len(outcomes))
    for request in received:
        assert_marked(request, marked=marked)
    markers: Final = frozenset(marker for marker, _ in outcomes)
    rows: Final = spend_rows(name, markers, expected=len(outcomes), seconds=120)
    assert sorted(row_key(str(row["request_id"])) for row in rows) == sorted(identities), rows
