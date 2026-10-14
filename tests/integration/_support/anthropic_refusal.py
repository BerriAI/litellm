import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

import yaml
from integration._support.anthropic_thinking import (
    JSON_LIST,
    JSON_OBJECT,
    Event,
    marker_of,
    stream_reply,
    streams,
)
from integration._support.client import eventually
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue, TypeAdapter
from redis import Redis

DOCS_REFUSAL_URL: Final = "https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#refusal-response"
DOCS_STREAMING_REFUSAL_URL: Final = (
    "https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals"
)

# Copied verbatim from DOCS_REFUSAL_URL and DOCS_STREAMING_REFUSAL_URL on 2026-10-12.
DOCS_REFUSAL: Final[Event] = {
    "id": "msg_01XFUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "model": "claude-fable-5",
    "content": [],
    "stop_reason": "refusal",
    "stop_details": {
        "type": "refusal",
        "category": "cyber",
        "explanation": "This request was declined because it could enable cyber harm.",
    },
    "usage": {"input_tokens": 412, "output_tokens": 0},
}
DOCS_PARTIAL_REFUSAL: Final[Event] = {
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello.."}],
    "stop_reason": "refusal",
    "stop_details": {
        "type": "refusal",
        "category": "cyber",
        "explanation": "This request was declined because it could enable cyber harm.",
    },
}

# Captured from api.anthropic.com (claude-sonnet-5-5, prompt "Reply with the single word: ready") on 2026-10-13.
CAPTURED_TURN: Final[Event] = {
    "model": "claude-sonnet-5-5",
    "id": "msg_011CfyXFGze1cDkk8o7gDRRS",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "ready"}],
    "container": None,
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "stop_details": None,
    "usage": {
        "input_tokens": 18,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"thinking_tokens": 0},
        "service_tier": "standard",
        "inference_geo": "global",
    },
    "diagnostics": None,
}
CAPTURED_EVENTS: Final[tuple[Event, ...]] = (
    {
        "type": "message_start",
        "message": {
            "model": "claude-sonnet-5-5",
            "id": "msg_011CfyXFLwXUxsp7nNAm3gcu",
            "type": "message",
            "role": "assistant",
            "content": [],
            "container": None,
            "stop_reason": None,
            "stop_sequence": None,
            "stop_details": None,
            "usage": {
                "input_tokens": 18,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                "output_tokens": 4,
                "service_tier": "standard",
                "inference_geo": "global",
            },
            "diagnostics": None,
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ready"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None, "stop_details": None, "container": None},
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 4,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    {"type": "message_stop"},
)

DOCS_STOP_DETAILS: Final = DOCS_REFUSAL["stop_details"]
REFUSAL_FIELDS: Final[Event] = {"stop_reason": "refusal", "stop_details": DOCS_STOP_DETAILS}
STOP_FIELDS: Final = ("stop_reason", "stop_details")
LONG_STOP_DETAILS: Final = "d" * 5120
MALFORMED_STOP_DETAILS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"int": 7, "list": ["refusal"], "empty": "", "long": LONG_STOP_DETAILS}
)
_FIXTURE: Final = re.compile(r"refusal-fixture-([a-z]+)")

Location: TypeAlias = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class Fixture:
    status: int
    body: bytes
    events: tuple[Event, ...] = ()


def _text_block(index: int, block: JsonValue) -> tuple[Event, ...]:
    text: Final = JSON_OBJECT.validate_python(block)["text"]
    return (
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": index},
    )


def refusal_events(message: Mapping[str, JsonValue]) -> tuple[Event, ...]:
    usage: Final = JSON_OBJECT.validate_python(message["usage"])
    start: Final[Event] = {
        "type": "message_start",
        "message": {
            "id": message["id"],
            "type": "message",
            "role": "assistant",
            "model": message["model"],
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "stop_details": None,
            "usage": usage,
        },
    }
    blocks: Final = chain.from_iterable(
        _text_block(index, block) for index, block in enumerate(JSON_LIST.validate_python(message["content"]))
    )
    delta: Final[Event] = {
        "type": "message_delta",
        "delta": {
            "stop_reason": message["stop_reason"],
            "stop_sequence": None,
            "stop_details": message["stop_details"],
        },
        "usage": {name: usage[name] for name in ("output_tokens", "output_tokens_details") if name in usage},
    }
    return (start, *blocks, delta, {"type": "message_stop"})


def _without_stop_details(value: Mapping[str, JsonValue]) -> Event:
    return {name: item for name, item in value.items() if name != "stop_details"}


def _captured_event(event: Event, stop_details: JsonValue, *, keep: bool) -> Event:
    if event["type"] == "message_start":
        message: Final = JSON_OBJECT.validate_python(event["message"])
        return {**event, "message": message if keep else _without_stop_details(message)}
    if event["type"] != "message_delta":
        return event
    delta: Final = JSON_OBJECT.validate_python(event["delta"])
    replaced: Final = {**delta, "stop_details": stop_details} if keep else _without_stop_details(delta)
    return {**event, "delta": replaced}


def _captured(stop_details: JsonValue, *, keep: bool = True) -> Fixture:
    body: Final = {**CAPTURED_TURN, "stop_details": stop_details} if keep else _without_stop_details(CAPTURED_TURN)
    return Fixture(
        200,
        json.dumps(body).encode(),
        tuple(_captured_event(event, stop_details, keep=keep) for event in CAPTURED_EVENTS),
    )


def _duplicated_stop_details() -> bytes:
    pair: Final = f'"stop_details": {json.dumps(DOCS_STOP_DETAILS)}'
    return json.dumps(DOCS_REFUSAL).replace(pair, f"{pair}, {pair}").encode()


def _anthropic_error(kind: str, message: str) -> bytes:
    return json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()


_PARTIAL: Final[Event] = {**DOCS_REFUSAL, **DOCS_PARTIAL_REFUSAL}
_REASONED: Final[Event] = {
    **_PARTIAL,
    "usage": {"input_tokens": 412, "output_tokens": 10, "output_tokens_details": {"thinking_tokens": 4}},
}

FIXTURES: Final[Mapping[str, Fixture]] = MappingProxyType(
    {
        "refusal": Fixture(200, json.dumps(DOCS_REFUSAL).encode(), refusal_events(DOCS_REFUSAL)),
        "partial": Fixture(200, json.dumps(_PARTIAL).encode(), refusal_events(_PARTIAL)),
        "reasoned": Fixture(200, json.dumps(_REASONED).encode(), refusal_events(_REASONED)),
        "duplicate": Fixture(200, _duplicated_stop_details()),
        "null": _captured(None),
        "missing": _captured(None, keep=False),
        **{name: _captured(value) for name, value in MALFORMED_STOP_DETAILS.items()},
        "nondict": Fixture(200, b"[]"),
        "unauthorized": Fixture(401, _anthropic_error("authentication_error", "invalid x-api-key")),
        "invalid": Fixture(400, _anthropic_error("invalid_request_error", "max_tokens: Field required")),
    }
)


def refusal_prompt(fixture: str, marker: str) -> str:
    return f"refusal-fixture-{fixture} marker-{marker}"


CITED_FIELDS: Final[Event] = {"citations": ["https://example.com/source"]}
_PLAIN_DELTA: Final[Event] = {"role": "assistant", "content": "ready"}
_CITED_DELTA: Final[Event] = {**_PLAIN_DELTA, "provider_specific_fields": CITED_FIELDS}
_OPENAI_USAGE: Final[Event] = {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12,
    "completion_tokens_details": {"reasoning_tokens": 4, "text_tokens": 3},
}


@dataclass(frozen=True, slots=True)
class OpenAIFixture:
    first_delta: Event
    usage: tuple[Event, ...] = ()


_OPENAI_FIXTURES: Final[Mapping[str, OpenAIFixture]] = MappingProxyType(
    {
        "plain": OpenAIFixture(_PLAIN_DELTA),
        "cited": OpenAIFixture(_CITED_DELTA),
        "plainusage": OpenAIFixture(_PLAIN_DELTA, (_OPENAI_USAGE,)),
        "citedusage": OpenAIFixture(_CITED_DELTA, (_OPENAI_USAGE,)),
    }
)


def openai_id(marker: str) -> str:
    return f"chatcmpl-{marker}"


def _openai_frame(marker: str, body: Event) -> bytes:
    chunk: Final[Event] = {
        "id": openai_id(marker),
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "gpt-4o-mini",
        **body,
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def _openai_choice(delta: Event, finish_reason: str | None) -> Event:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def _openai_stream(name: str, request: Request) -> Reply:
    assert streams(request), request.body
    marker: Final = marker_of(request)
    fixture: Final = _OPENAI_FIXTURES[name]
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _openai_frame(marker, _openai_choice(fixture.first_delta, None)),
            _openai_frame(marker, _openai_choice({"content": " now"}, "stop")),
            *(_openai_frame(marker, {"choices": [], "usage": usage}) for usage in fixture.usage),
            b"data: [DONE]\n\n",
        ),
    )


def respond(request: Request) -> Reply:
    found: Final[Sequence[str]] = _FIXTURE.findall(request.body.decode())
    assert found, request.body
    if request.target.endswith("/chat/completions"):
        return _openai_stream(found[-1], request)
    fixture: Final = FIXTURES[found[-1]]
    if fixture.status != 200 or not fixture.events or not streams(request):
        return Reply(status=fixture.status, body=fixture.body)
    return stream_reply(request, fixture.events)


def _stop_fields(fields: JsonValue) -> Event:
    if not isinstance(fields, dict):
        return {}
    return {name: fields[name] for name in STOP_FIELDS if name in fields}


def relayed(value: JsonValue, location: Location = ()) -> tuple[tuple[Location, Event], ...]:
    if isinstance(value, list):
        return tuple(chain.from_iterable(relayed(item, (*location, index)) for index, item in enumerate(value)))
    if not isinstance(value, dict):
        return ()
    found: Final = _stop_fields(value.get("provider_specific_fields"))
    here: Final = (((*location, "provider_specific_fields"), found),) if found else ()
    return (*here, *chain.from_iterable(relayed(item, (*location, name)) for name, item in value.items()))


_CACHE_KEY: Final = re.compile(r"[0-9a-f]{64}")
_STORED: Final = TypeAdapter(bytes | None)
STALE_ID_SUFFIX: Final = "-stale"


def _stored(store: Redis, key: str) -> bytes:
    if not _CACHE_KEY.fullmatch(key) or store.type(key) != b"string":
        return b""
    return _STORED.validate_python(store.get(key)) or b""


def _entry_key(store: Redis, needle: str) -> str | None:
    keys: Final = (raw.decode() for raw in store.scan_iter(count=1000))
    return next((key for key in keys if needle.encode() in _stored(store, key)), None)


def _with_message_fields(response: Event, fields: JsonValue) -> Event:
    choice: Final = JSON_OBJECT.validate_python(JSON_LIST.validate_python(response["choices"])[0])
    message: Final = JSON_OBJECT.validate_python(choice["message"])
    return {
        **response,
        "id": f"{response['id']}{STALE_ID_SUFFIX}",
        "choices": [{**choice, "message": {**message, "provider_specific_fields": fields}}],
    }


def rewrite_cached_message_fields(needle: str, fields: JsonValue) -> str:
    store: Final = Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    key: Final = eventually(lambda: _entry_key(store, needle), lambda found: found is not None)
    assert key is not None, needle
    entry: Final = JSON_OBJECT.validate_json(_stored(store, key))
    stored: Final = entry["response"]
    encoded: Final = isinstance(stored, str)
    response: Final = (
        JSON_OBJECT.validate_json(stored) if isinstance(stored, str) else JSON_OBJECT.validate_python(stored)
    )
    rewritten: Final = _with_message_fields(response, fields)
    store.set(key, json.dumps({**entry, "response": json.dumps(rewritten) if encoded else rewritten}), ex=60)
    return str(rewritten["id"])


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _key_hash_of(event: Mapping[str, JsonValue]) -> JsonValue:
    return JSON_OBJECT.validate_python(event.get("metadata") or {}).get("user_api_key_hash")


class Ledger:
    def __init__(self, wire: Wire) -> None:
        self.wire: Final = wire
        self.batches: tuple[Request, ...] = ()

    def events(self) -> tuple[Event, ...]:
        self.batches = (*self.batches, *self.wire.drain())
        decoded: Final = chain.from_iterable(JSON_LIST.validate_json(batch.body) for batch in self.batches)
        return tuple(JSON_OBJECT.validate_python(event) for event in decoded)

    def for_key(self, key: str) -> tuple[Event, ...]:
        hashed: Final = key_hash(key)
        return tuple(event for event in self.events() if _key_hash_of(event) == hashed)

    def await_key(self, key: str, count: int = 1, seconds: float = 30) -> tuple[Event, ...]:
        return eventually(lambda: self.for_key(key), lambda found: len(found) >= count, seconds=seconds)


_HIDDEN_PARAMS_PROBE: Final = """
import httpx
from litellm.integrations.custom_logger import CustomLogger


class HiddenParamsProbe(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        payload = kwargs.get("standard_logging_object") or {{}}
        hidden = getattr(response_obj, "_hidden_params", None) or {{}}
        event = {{
            "metadata": {{"user_api_key_hash": (payload.get("metadata") or {{}}).get("user_api_key_hash")}},
            "response_id": getattr(response_obj, "id", None),
            "received_model_id": hidden.get("received_model_id"),
        }}
        async with httpx.AsyncClient(trust_env=False) as client:
            await client.post({url!r}, json=[event])


instance = HiddenParamsProbe()
"""


def hidden_params_probe(directory: Path, url: str) -> str:
    (directory / "hidden_params_probe.py").write_text(_HIDDEN_PARAMS_PROBE.format(url=url))
    return "hidden_params_probe.instance"


def sink_config(
    directory: Path,
    sinks: Mapping[str, str],
    *,
    retries: int = 0,
    extra_callbacks: Sequence[str] = (),
    store_prompts: bool = False,
) -> Path:
    shared: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = {
        **JSON_OBJECT.validate_python(shared["litellm_settings"]),
        "callbacks": [*sinks, *extra_callbacks],
        "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
    }
    callback_settings: Final = {
        name: {
            "callback_type": "generic_api",
            "endpoint": url,
            "headers": {"x-refusal-sink": name},
            "max_retries": retries,
            "retry_delay": 0.5,
        }
        for name, url in sinks.items()
    }
    general_settings: Final = {
        **JSON_OBJECT.validate_python(shared["general_settings"]),
        "store_prompts_in_spend_logs": store_prompts,
    }
    path: Final = directory / "anthropic-refusal-sinks.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **shared,
                "general_settings": general_settings,
                "litellm_settings": settings,
                "callback_settings": callback_settings,
            }
        )
    )
    return path


def frames(text: str) -> tuple[Event, ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    )


def frame_types(events: Sequence[Mapping[str, JsonValue]]) -> tuple[JsonValue, ...]:
    return tuple(event.get("type") for event in events)
