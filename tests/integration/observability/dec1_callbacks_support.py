"""Scripted provider, proxy config and readers shared by the dec1 callback contract tests.

One rig is one owned proxy with three owned wire servers: the provider every deployment points at, the sink the
`generic_api` callback posts StandardLoggingPayload batches to, and the probe sink the custom callback module
`dec1_callbacks_probe` reports its hook calls to. Tests match events by the `x-litellm-call-id` they send.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

PROBE_MODULE: Final = "integration.observability.dec1_callbacks_probe"
RECORDER: Final = f"{PROBE_MODULE}.recorder"
PROVIDER_KEY: Final = "synthetic-dec1-provider-key"
SINK_SECRET: Final = "synthetic-dec1-sink-secret"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CHAT_USAGE: Final = (11, 4)
EMBEDDING_TOKENS: Final = 5
MESSAGES_USAGE: Final = (9, 3)
RESPONSES_USAGE: Final = (7, 2)
CHAT_COST: Final = 11 * INPUT_RATE + 4 * OUTPUT_RATE
EMBEDDING_COST: Final = 5 * INPUT_RATE
MESSAGES_COST: Final = 9 * INPUT_RATE + 3 * OUTPUT_RATE
RESPONSES_COST: Final = 7 * INPUT_RATE + 2 * OUTPUT_RATE
REPLY_TEXT: Final = "dec1 scripted reply"
PROVIDER_REJECTION: Final = "dec1 scripted provider rejection"
PROVIDER_OUTAGE: Final = "dec1 scripted provider outage"
PROVIDER_LIMIT: Final = "dec1 scripted provider rate limit"
PROVIDER_DENIAL: Final = "dec1 scripted provider denial"

CHAT: Final = "dec1-chat"
CHAT_REJECTED: Final = "dec1-chat-rejected"
CHAT_OUTAGE: Final = "dec1-chat-outage"
CHAT_LIMITED: Final = "dec1-chat-limited"
CHAT_DENIED: Final = "dec1-chat-denied"
CHAT_FLAKY: Final = "dec1-chat-flaky"
CHAT_FALLBACK_PRIMARY: Final = "dec1-chat-fallback-primary"
EMBEDDING: Final = "dec1-embedding"
EMBEDDING_REJECTED: Final = "dec1-embedding-rejected"
MESSAGES: Final = "dec1-messages"
MESSAGES_REJECTED: Final = "dec1-messages-rejected"
RESPONSES: Final = "dec1-responses"
RESPONSES_REJECTED: Final = "dec1-responses-rejected"

STANDARD_PAYLOAD_FIELDS: Final = frozenset(
    {
        "id",
        "trace_id",
        "session_id",
        "litellm_call_id",
        "call_type",
        "stream",
        "response_cost",
        "cost_breakdown",
        "response_cost_failure_debug_info",
        "status",
        "status_fields",
        "custom_llm_provider",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "startTime",
        "endTime",
        "completionStartTime",
        "response_time",
        "model_map_information",
        "model",
        "model_id",
        "model_group",
        "api_base",
        "metadata",
        "cache_hit",
        "cache_key",
        "saved_cache_cost",
        "request_tags",
        "end_user",
        "requester_ip_address",
        "user_agent",
        "messages",
        "response",
        "error_str",
        "error_information",
        "model_parameters",
        "hidden_params",
        "guardrail_information",
        "standard_built_in_tools_params",
    }
)
STANDARD_METADATA_FIELDS: Final = frozenset(
    {
        "user_api_key_hash",
        "user_api_key_alias",
        "user_api_key_spend",
        "user_api_key_max_budget",
        "user_api_key_budget_reset_at",
        "user_api_key_org_id",
        "user_api_key_team_id",
        "user_api_key_user_id",
        "user_api_key_user_email",
        "user_api_key_team_alias",
        "user_api_key_end_user_id",
        "user_api_key_request_route",
        "spend_logs_metadata",
        "requester_ip_address",
        "user_agent",
        "requester_metadata",
        "requester_custom_headers",
        "applied_guardrails",
        "usage_object",
    }
)


def _chat_reply(identity: str, model: str, text: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": CHAT_USAGE[0],
                "completion_tokens": CHAT_USAGE[1],
                "total_tokens": sum(CHAT_USAGE),
            },
        }
    ).encode()


def _chat_frames(identity: str, model: str, text: str) -> tuple[bytes, ...]:
    first, second = text[: len(text) // 2], text[len(text) // 2 :]
    chunks: Final = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": first}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": second}, "finish_reason": None}]},
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": CHAT_USAGE[0],
                "completion_tokens": CHAT_USAGE[1],
                "total_tokens": sum(CHAT_USAGE),
            },
        },
    )
    head: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": model}
    return (*(f"data: {json.dumps({**head, **chunk})}\n\n".encode() for chunk in chunks), b"data: [DONE]\n\n")


def _embedding_reply(model: str) -> bytes:
    return json.dumps(
        {
            "object": "list",
            "model": model,
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
            "usage": {"prompt_tokens": EMBEDDING_TOKENS, "total_tokens": EMBEDDING_TOKENS},
        }
    ).encode()


def _message_reply(identity: str, model: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": REPLY_TEXT}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": MESSAGES_USAGE[0], "output_tokens": MESSAGES_USAGE[1]},
        }
    ).encode()


def _message_frames(identity: str, model: str) -> tuple[bytes, ...]:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "message_start",
            "message": {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": MESSAGES_USAGE[0], "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": REPLY_TEXT}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": MESSAGES_USAGE[1]},
        },
        {"type": "message_stop"},
    )
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _response_object(identity: str, model: str, status: str) -> dict[str, JsonValue]:
    done: Final = status == "completed"
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": model,
        "output": [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": REPLY_TEXT, "annotations": []}],
            }
        ]
        if done
        else [],
        "usage": {
            "input_tokens": RESPONSES_USAGE[0],
            "output_tokens": RESPONSES_USAGE[1],
            "total_tokens": sum(RESPONSES_USAGE),
        }
        if done
        else None,
    }


def _response_frames(identity: str, model: str) -> tuple[bytes, ...]:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": _response_object(identity, model, "in_progress"),
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "delta": REPLY_TEXT,
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": _response_object(identity, model, "completed"),
        },
    )
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _error(status: int, message: str, kind: str) -> Reply:
    return Reply(status=status, body=json.dumps({"error": {"message": message, "type": kind, "code": None}}).encode())


@dataclass
class ScriptedProvider:
    """Answers chat, embeddings, messages and responses. `/rejected/` is a 400, `/outage/` a 500, `/limited/` a
    429, `/denied/` a 401, and `/flaky/` a 500 for the first request carrying a given user text and a success for
    every later one."""

    attempts: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        if request.target.startswith("/rejected/"):
            return _error(400, PROVIDER_REJECTION, "invalid_request_error")
        if request.target.startswith("/outage/"):
            return _error(500, PROVIDER_OUTAGE, "server_error")
        if request.target.startswith("/limited/"):
            return _error(429, PROVIDER_LIMIT, "rate_limit_error")
        if request.target.startswith("/denied/"):
            return _error(401, PROVIDER_DENIAL, "authentication_error")
        body: Final = json.loads(request.body or b"{}")
        if request.target.startswith("/flaky/") and self._first_attempt(json.dumps(body.get("messages"))):
            return _error(500, PROVIDER_OUTAGE, "server_error")
        identity: Final = "dec1-" + uuid.uuid4().hex
        model: Final = str(body.get("model", "gpt-4o-mini"))
        streams: Final = body.get("stream") is True
        target: Final = request.target.removeprefix("/flaky")
        if target == "/v1/chat/completions":
            if streams:
                return Reply(
                    chunks=_chat_frames("chatcmpl-" + identity, model, REPLY_TEXT), content_type="text/event-stream"
                )
            return Reply(body=_chat_reply("chatcmpl-" + identity, model, REPLY_TEXT))
        if target == "/v1/embeddings":
            return Reply(body=_embedding_reply(model))
        if target == "/v1/messages":
            if streams:
                return Reply(chunks=_message_frames("msg_" + identity, model), content_type="text/event-stream")
            return Reply(body=_message_reply("msg_" + identity, model))
        if target == "/v1/responses":
            if streams:
                return Reply(chunks=_response_frames("resp_" + identity, model), content_type="text/event-stream")
            return Reply(body=json.dumps(_response_object("resp_" + identity, model, "completed")).encode())
        return _error(404, f"dec1 provider has no route {request.target}", "invalid_request_error")

    def _first_attempt(self, text: str) -> bool:
        with self.lock:
            self.attempts[text] = self.attempts.get(text, 0) + 1
            return self.attempts[text] == 1


def _deployment(name: str, model: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": model,
            "api_base": api_base,
            "api_key": PROVIDER_KEY,
            "input_cost_per_token": INPUT_RATE,
            "output_cost_per_token": OUTPUT_RATE,
            **params,
        },
    }


def write_config(
    path: Path,
    provider_url: str,
    litellm_settings: Mapping[str, JsonValue],
    *,
    guardrails: Sequence[Mapping[str, JsonValue]] = (),
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    sections: Mapping[str, JsonValue] | None = None,
) -> Path:
    chat: Final = "openai/gpt-4o-mini"
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    _deployment(CHAT, chat, provider_url + "/v1"),
                    _deployment(CHAT_REJECTED, chat, provider_url + "/rejected/v1"),
                    _deployment(CHAT_OUTAGE, chat, provider_url + "/outage/v1"),
                    _deployment(CHAT_LIMITED, chat, provider_url + "/limited/v1"),
                    _deployment(CHAT_DENIED, chat, provider_url + "/denied/v1"),
                    _deployment(CHAT_FLAKY, chat, provider_url + "/flaky/v1"),
                    _deployment(CHAT_FALLBACK_PRIMARY, chat, provider_url + "/outage/v1"),
                    _deployment(EMBEDDING, "openai/text-embedding-3-small", provider_url + "/v1"),
                    _deployment(EMBEDDING_REJECTED, "openai/text-embedding-3-small", provider_url + "/rejected/v1"),
                    _deployment(MESSAGES, "anthropic/claude-haiku-4-5", provider_url),
                    _deployment(MESSAGES_REJECTED, "anthropic/claude-haiku-4-5", provider_url + "/rejected"),
                    _deployment(RESPONSES, chat, provider_url + "/v1"),
                    _deployment(RESPONSES_REJECTED, chat, provider_url + "/rejected/v1"),
                ],
                "litellm_settings": dict(litellm_settings),
                **({"guardrails": [dict(guardrail) for guardrail in guardrails]} if guardrails else {}),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_model_info_refresh": True,
                    "proxy_batch_write_at": 1,
                    **(general_settings or {}),
                },
                "router_settings": {
                    "num_retries": 0,
                    "disable_cooldowns": True,
                    "fallbacks": [{CHAT_FALLBACK_PRIMARY: [CHAT]}],
                    **(router_settings or {}),
                },
                **(sections or {}),
            }
        )
    )
    return path


@dataclass
class Journal:
    """Every JSON record a wire server received, kept across drains because draining consumes the queue."""

    wire: Wire
    batched: bool
    seen: list[dict[str, JsonValue]] = field(default_factory=list)
    raw: list[bytes] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def records(self) -> tuple[dict[str, JsonValue], ...]:
        with self.lock:
            for request in self.wire.drain():
                self.raw.append(request.body)
                decoded = json.loads(request.body)
                self.seen.extend(decoded if self.batched else [decoded])
            return tuple(self.seen)

    def bodies(self) -> bytes:
        self.records()
        with self.lock:
            return b"\n".join(self.raw)


@dataclass(frozen=True)
class CallbackRig:
    proxy: Gateway
    provider: Wire
    sink: Journal
    probe: Journal

    def events(self, call_id: str) -> tuple[dict[str, JsonValue], ...]:
        """StandardLoggingPayloads the generic_api callback delivered for one x-litellm-call-id."""
        return tuple(event for event in self.sink.records() if event.get("litellm_call_id") == call_id)

    def event(self, call_id: str, status: str | None = None, seconds: float = 20) -> dict[str, JsonValue]:
        """The single payload delivered for the call, optionally the single one with the given status."""

        def matching() -> tuple[dict[str, JsonValue], ...]:
            return tuple(event for event in self.events(call_id) if status is None or event["status"] == status)

        found: Final = eventually(matching, lambda events: len(events) >= 1, seconds=seconds)
        assert len(found) == 1, [(event["id"], event["status"]) for event in found]
        return found[0]

    def events_mentioning(self, text: str) -> tuple[dict[str, JsonValue], ...]:
        """StandardLoggingPayloads whose logged request `messages` contain `text`.

        A request rejected during key auth is logged before the endpoint reads `x-litellm-call-id`, so its payload
        carries a call id the proxy generated (`_handle_logging_proxy_only_error`, litellm/proxy/utils.py). Tests
        of such failures send a unique text and find the payload by it instead.
        """
        return tuple(event for event in self.sink.records() if text in json.dumps(event.get("messages")))

    def event_mentioning(self, text: str, status: str, seconds: float = 20) -> dict[str, JsonValue]:
        """The single payload with the given status whose logged request carried `text`."""
        found: Final = eventually(
            lambda: tuple(event for event in self.events_mentioning(text) if event["status"] == status),
            lambda events: len(events) >= 1,
            seconds=seconds,
        )
        assert len(found) == 1, [(event["id"], event["status"]) for event in found]
        return found[0]

    def hook_names_mentioning(self, text: str, logger: str = "recorder") -> tuple[str, ...]:
        """Hook names, in call order, of the probe records from one logger whose request text contains `text`."""
        found: Final = tuple(
            record
            for record in self.probe.records()
            if record.get("logger") == logger and text in str(record.get("text"))
        )
        return tuple(str(record["hook"]) for record in sorted(found, key=_sequence))

    def settled_events(self, call_id: str, count: int, *, holding: float = 3) -> tuple[dict[str, JsonValue], ...]:
        """Exactly `count` payloads for the call, still `count` after `holding` more seconds."""
        return held(lambda: self.events(call_id), lambda events: len(events) == count, holding=holding, seconds=30)

    def hooks(self, call_id: str, logger: str = "recorder") -> tuple[dict[str, JsonValue], ...]:
        """Probe records for one call from one logger instance, in the order the proxy process made the calls."""
        found: Final = tuple(
            record
            for record in self.probe.records()
            if record.get("litellm_call_id") == call_id and record.get("logger") == logger
        )
        return tuple(sorted(found, key=_sequence))

    def hook_names(self, call_id: str, logger: str = "recorder") -> tuple[str, ...]:
        return tuple(str(record["hook"]) for record in self.hooks(call_id, logger))

    def settled_hook_names(
        self, call_id: str, expected: Sequence[str], logger: str = "recorder", *, holding: float = 3
    ) -> tuple[str, ...]:
        """Hook names for the call once they equal `expected` as a multiset and stay so for `holding` seconds."""
        return held(
            lambda: self.hook_names(call_id, logger),
            lambda names: sorted(names) == sorted(expected),
            holding=holding,
            seconds=30,
        )

    def hook(self, call_id: str, name: str, logger: str = "recorder") -> dict[str, JsonValue]:
        found: Final = eventually(
            lambda: tuple(record for record in self.hooks(call_id, logger) if record["hook"] == name),
            lambda records: len(records) >= 1,
            seconds=20,
        )
        return found[0]

    def stream_hook_texts(self, key_alias: str) -> tuple[str, ...]:
        """Texts the per-chunk streaming hook received, in order, for streams made with the key of this alias.

        The per-chunk hook is handed no request data, so the key alias is the only way to tie a call to a caller.
        """
        return tuple(
            str(record["stream_text"])
            for record in sorted(self.probe.records(), key=_sequence)
            if record.get("hook") == "async_post_call_streaming_hook" and record.get("key_alias") == key_alias
        )


def _sequence(record: Mapping[str, JsonValue]) -> int:
    sequence: Final = record.get("seq")
    assert isinstance(sequence, int), record
    return sequence


@contextmanager
def callback_rig(
    directory: Path,
    litellm_settings: Mapping[str, JsonValue],
    *,
    environment: Mapping[str, str] | None = None,
    guardrails: Sequence[Mapping[str, JsonValue]] = (),
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    sections: Mapping[str, JsonValue] | None = None,
) -> Iterator[CallbackRig]:
    """Boot one single-worker proxy on the given litellm_settings, wired to an owned provider and two owned sinks.

    The generic_api destination comes from GENERIC_LOGGER_ENDPOINT and its bearer from GENERIC_LOGGER_HEADERS, and
    DEFAULT_FLUSH_INTERVAL_SECONDS=1 keeps the batch logger from holding events for its default five seconds.
    """
    with (
        wire_server(ScriptedProvider()) as provider,
        wire_server(lambda request: Reply()) as sink,
        wire_server(lambda request: Reply()) as probe,
        gateway_from_environment() as shared,
    ):
        config: Final = write_config(
            directory / f"dec1-callbacks-{uuid.uuid4().hex}.yaml",
            provider.url,
            litellm_settings,
            guardrails=guardrails,
            general_settings=general_settings,
            router_settings=router_settings,
            sections=sections,
        )
        overrides: Final = {
            "GENERIC_LOGGER_ENDPOINT": sink.url,
            "GENERIC_LOGGER_HEADERS": f"Authorization=Bearer {SINK_SECRET}",
            "DEC1_CALLBACKS_PROBE_URL": probe.url,
            "DEFAULT_FLUSH_INTERVAL_SECONDS": "1",
            "STORE_MODEL_IN_DB": "False",
            **(environment or {}),
        }
        with owned_proxy(shared, directory, overrides, config=config) as proxy:
            yield CallbackRig(proxy, provider, Journal(sink, batched=True), Journal(probe, batched=False))


def new_call_id() -> str:
    return str(uuid.uuid4())


def marker() -> str:
    return "dec1" + uuid.uuid4().hex


def key_hash(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def post(
    rig: CallbackRig,
    path: str,
    body: Mapping[str, JsonValue],
    call_id: str,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    return rig.proxy.request("POST", path, body, key=key, headers={"x-litellm-call-id": call_id, **(headers or {})})


def chat(
    rig: CallbackRig,
    call_id: str,
    text: str = "dec1 callback contract",
    *,
    model: str = CHAT,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": text}], **extra}
    return post(rig, "/v1/chat/completions", body, call_id, key=key, headers=headers)


def stream_chat(
    rig: CallbackRig, call_id: str, text: str = "dec1 callback contract", *, model: str = CHAT, key: str | None = None
) -> tuple[int, Mapping[str, str], str]:
    """Send a streamed chat request and read it to the end: status, response headers, raw SSE text."""
    with rig.proxy.client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": text}], "stream": True},
        headers={"Authorization": f"Bearer {key or rig.proxy.key}", "x-litellm-call-id": call_id},
    ) as response:
        return response.status_code, dict(response.headers), response.read().decode()


def stream(rig: CallbackRig, path: str, body: Mapping[str, JsonValue], call_id: str) -> tuple[int, str]:
    with rig.proxy.client.stream(
        "POST",
        path,
        json={**body, "stream": True},
        headers={"Authorization": f"Bearer {rig.proxy.key}", "x-litellm-call-id": call_id},
    ) as response:
        return response.status_code, response.read().decode()


def streamed_text(sse: str) -> str:
    """Concatenated delta content of a chat completions SSE body."""
    payloads: Final = (line[6:] for line in sse.splitlines() if line.startswith("data: ") and line != "data: [DONE]")
    deltas: Final = (json.loads(payload)["choices"] for payload in payloads)
    return "".join(choices[0]["delta"].get("content") or "" for choices in deltas if choices)


def provider_bodies(rig: CallbackRig) -> tuple[dict[str, JsonValue], ...]:
    """JSON bodies of the POST requests the provider received since the last read."""
    return tuple(json.loads(request.body) for request in rig.provider.drain() if request.method == "POST")


def error_message(response: httpx.Response) -> str:
    return str(response.json()["error"]["message"])
