"""Scripted provider answers, upstream observations and Redis helpers shared by the response cache cells.

The bodies carry ``$UNIQUE_ID`` so every upstream answer has its own id: an answer served from the
cache repeats the id of the request that filled it, an answer that reached the upstream again does not.
"""

from __future__ import annotations

import ast
import json
import os
import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Final

import httpx
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Scenario, eventually, object_value, string_value
from tests.integration._support.upstream import CONTROL_URL, ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse

CACHE_KEY: Final = re.compile(r"^[0-9a-f]{64}$")
PROVIDER_KEY: Final = "integration-provider-key"
CHAT_MODEL: Final = "openai/gpt-4o-mini"
TEXT_MODEL: Final = "openai/gpt-3.5-turbo-instruct"
MESSAGES_MODEL: Final = "anthropic/claude-haiku-4-5"

CHAT: Final[dict[str, JsonValue]] = {
    "id": "chatcmpl-$UNIQUE_ID",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
TEXT: Final[dict[str, JsonValue]] = {
    "id": "cmpl-$UNIQUE_ID",
    "object": "text_completion",
    "created": 1,
    "model": "gpt-3.5-turbo-instruct",
    "choices": [{"text": "scripted", "index": 0, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "id": "msg_$UNIQUE_ID",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "scripted", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}
MESSAGE: Final[dict[str, JsonValue]] = {
    "id": "msg-$UNIQUE_ID",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "scripted"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
EMBEDDING: Final[dict[str, JsonValue]] = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
    "model": "text-embedding-$UNIQUE_ID",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}
RERANK: Final[dict[str, JsonValue]] = {
    "id": "rerank-$UNIQUE_ID",
    "results": [{"index": 0, "relevance_score": 0.5}],
    "meta": {},
}
TRANSCRIPT: Final[dict[str, JsonValue]] = {"text": "scripted $UNIQUE_ID"}

CHAT_CHUNK: Final = (
    'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":'
)
CHAT_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        CHAT_CHUNK + '[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"content":"streamed "},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"content":"response"},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ),
)
CHAT_STREAM_WITHOUT_CHOICES: Final = SseResponse(
    content_type="text/event-stream",
    frames=(CHAT_CHUNK + "[]}", "data: [DONE]"),
)
TEXT_CHUNK: Final = (
    'data: {"id":"cmpl-$UNIQUE_ID","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":'
)
TEXT_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        TEXT_CHUNK + '[{"text":"streamed ","index":0,"finish_reason":null}]}',
        TEXT_CHUNK + '[{"text":"response","index":0,"finish_reason":null}]}',
        TEXT_CHUNK + '[{"text":"","index":0,"finish_reason":"stop"}]}',
        "data: [DONE]",
    ),
)
TEXT_STREAM_WITHOUT_CHOICES: Final = SseResponse(
    content_type="text/event-stream",
    frames=(TEXT_CHUNK + "[]}", "data: [DONE]"),
)
RESPONSE_CREATED: Final = (
    "event: response.created\n"
    'data: {"type":"response.created","response":{"id":"resp_$UNIQUE_ID","object":"response",'
    '"created_at":1,"status":"in_progress","model":"gpt-4o-mini","output":[],"usage":null}}'
)
RESPONSE_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        RESPONSE_CREATED,
        (
            "event: response.output_item.added\n"
            'data: {"type":"response.output_item.added","output_index":0,'
            '"item":{"type":"message","id":"msg_$UNIQUE_ID","status":"in_progress","role":"assistant","content":[]}}'
        ),
        (
            "event: response.content_part.added\n"
            'data: {"type":"response.content_part.added","item_id":"msg_$UNIQUE_ID","output_index":0,'
            '"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}'
        ),
        (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","item_id":"msg_$UNIQUE_ID","output_index":0,'
            '"content_index":0,"delta":"streamed response"}'
        ),
        (
            "event: response.output_text.done\n"
            'data: {"type":"response.output_text.done","item_id":"msg_$UNIQUE_ID","output_index":0,'
            '"content_index":0,"text":"streamed response"}'
        ),
        (
            "event: response.content_part.done\n"
            'data: {"type":"response.content_part.done","item_id":"msg_$UNIQUE_ID","output_index":0,'
            '"content_index":0,"part":{"type":"output_text","text":"streamed response","annotations":[]}}'
        ),
        (
            "event: response.output_item.done\n"
            'data: {"type":"response.output_item.done","output_index":0,'
            '"item":{"type":"message","id":"msg_$UNIQUE_ID","status":"completed","role":"assistant",'
            '"content":[{"type":"output_text","text":"streamed response","annotations":[]}]}}'
        ),
        (
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"id":"resp_$UNIQUE_ID","object":"response",'
            '"created_at":1,"status":"completed","model":"gpt-4o-mini",'
            '"output":[{"id":"msg_$UNIQUE_ID","type":"message","status":"completed","role":"assistant",'
            '"content":[{"type":"output_text","text":"streamed response","annotations":[]}]}],'
            '"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}'
        ),
    ),
)
RESPONSE_STREAM_WITHOUT_OUTPUT: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        RESPONSE_CREATED,
        (
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"id":"resp_$UNIQUE_ID","object":"response",'
            '"created_at":1,"status":"completed","model":"gpt-4o-mini","output":[],'
            '"usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}}'
        ),
    ),
)
MESSAGE_START: Final = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_$UNIQUE_ID","type":"message","role":"assistant",'
    '"model":"claude-haiku-4-5","content":[],"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":1,"output_tokens":0}}}'
)
MESSAGE_END: Final = (
    (
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
        '"usage":{"output_tokens":2}}'
    ),
    'event: message_stop\ndata: {"type":"message_stop"}',
)
MESSAGE_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        MESSAGE_START,
        (
            "event: content_block_start\n"
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'
        ),
        (
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"streamed response"}}'
        ),
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
        *MESSAGE_END,
    ),
)
MESSAGE_STREAM_WITHOUT_CONTENT_BLOCKS: Final = SseResponse(
    content_type="text/event-stream", frames=(MESSAGE_START, *MESSAGE_END)
)


class Upstream:
    """The owned upstream's observations, drained once per read and kept, filtered per deployment."""

    def __init__(self, url: str = CONTROL_URL) -> None:
        self._url: Final = url.rstrip("/")
        self._seen: tuple[dict[str, JsonValue], ...] = ()

    def received(self, identity: str) -> tuple[dict[str, JsonValue], ...]:
        with httpx.Client(timeout=10, trust_env=False) as client:
            payload: Final = JSON_OBJECT.validate_json(client.get(f"{self._url}/__observations").content)
        requests: Final = payload["requests"]
        assert isinstance(requests, list)
        self._seen = (
            *self._seen,
            *(object_value(item) for item in requests),
        )  # rebind-ok: drained observations accumulate
        return tuple(item for item in self._seen if f"/{identity}/" in string_value(item.get("path", "")))

    def calls(self, identity: str) -> int:
        """The LLM calls the deployment took; the proxy's boot-time ``GET /v1/models`` discovery of a config
        deployment is not one."""
        return sum(1 for item in self.received(identity) if item.get("method") == "POST")


def json_response(body: Mapping[str, JsonValue], status: int = 200) -> JsonResponse:
    return JsonResponse(content_type="application/json", body=dict(body), status=status)


def slowed(stream: SseResponse, frame_delay_ms: int) -> SseResponse:
    return stream.model_copy(update={"frame_delay_ms": frame_delay_ms})


def emptied(body: Mapping[str, JsonValue], field: str) -> dict[str, JsonValue]:
    return {**body, field: []}


def scenario_id() -> str:
    return f"cache-{uuid.uuid4().hex[:12]}"


def scripted(scenario: Scenario, response: JsonResponse | SseResponse) -> ScenarioHandle:
    handle: Final = register_scenario(scenario_id(), response, control_url=scenario.gateway.upstream_url)
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


@contextmanager
def scripted_in_process(response: JsonResponse | SseResponse) -> Iterator[ScenarioHandle]:
    handle: Final = register_scenario(scenario_id(), response)
    try:
        yield handle
    finally:
        delete_scenario(handle)


def rescript(handle: ScenarioHandle, response: JsonResponse | SseResponse) -> None:
    register_scenario(handle.scenario_id, response, control_url=handle.control_url)


def prompt() -> str:
    return f"cache probe {uuid.uuid4().hex}"


def redis_store() -> Redis:
    return Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))


def entry_key(store: Redis, needle: str) -> str | None:
    """The response cache key (a bare sha256) whose stored value names ``needle``, or ``None``."""
    for raw_key in store.scan_iter(count=1000):
        key: Final = raw_key.decode()
        if not CACHE_KEY.match(key) or store.type(key) != b"string":
            continue
        value: Final = store.get(key)
        if value is not None and needle.encode() in value:
            return key
    return None


def await_entry(store: Redis, needle: str) -> str:
    key: Final = eventually(lambda: entry_key(store, needle), lambda found: found is not None)
    assert key is not None
    return key


def _decode_entry(raw: bytes) -> tuple[dict[str, JsonValue], bool]:
    """The stored entry and whether it is JSON; the sync SDK path stores ``str(dict)`` instead."""
    text: Final = raw.decode()
    if text.startswith('{"'):
        return object_value(json.loads(text)), True
    return object_value(ast.literal_eval(text)), False


SHORT_TTL_SECONDS: Final = 2
SHORT_TTL_CACHE_CONTROL: Final[dict[str, JsonValue]] = {"ttl": SHORT_TTL_SECONDS}
STALE_ENTRY_TTL_SECONDS: Final = 60
STALE_ID_SUFFIX: Final = "-stale"


def flip_entry_to_empty(store: Redis, key: str, field: str) -> str:
    """Rewrite the stored response so ``field`` is ``[]`` and its id carries ``-stale``, keeping the encoding.

    Returns the stale id. The proxy worker that wrote the entry keeps a copy in its own memory for the entry's ttl,
    so a cell fills the entry with ``SHORT_TTL_CACHE_CONTROL`` and reads the rewritten entry once that copy lapsed.
    """
    raw: Final = store.get(key)
    assert raw is not None, key
    entry, is_json = _decode_entry(raw)
    response: Final = entry["response"]
    stored: Final = object_value(json.loads(response) if isinstance(response, str) else response)
    assert field in stored, f"{field!r} missing from the cached response {sorted(stored)}"
    stale_id: Final = f"{string_value(stored['id'])}{STALE_ID_SUFFIX}"
    flipped: Final[dict[str, JsonValue]] = {**stored, field: [], "id": stale_id}
    rewritten: Final[dict[str, JsonValue]] = {
        **entry,
        "response": json.dumps(flipped) if isinstance(response, str) else flipped,
    }
    store.set(key, json.dumps(rewritten) if is_json else str(rewritten), ex=STALE_ENTRY_TTL_SECONDS)
    return stale_id


def chat_body(model: str, text: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}


def message_body(model: str, text: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}], **extra}


def response_id(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def choices(response: httpx.Response) -> list[JsonValue]:
    assert response.status_code == 200, response.text
    found: Final = JSON_OBJECT.validate_json(response.content)["choices"]
    assert isinstance(found, list), response.text
    return found


def content_of_first_choice(response: httpx.Response) -> JsonValue:
    return object_value(object_value(choices(response)[0])["message"])["content"]


def text_of_first_choice(response: httpx.Response) -> JsonValue:
    return object_value(choices(response)[0])["text"]
