"""Shared pieces for the ``test_dec1_limits_*`` contracts.

Every test in that family owns its provider (a ``wire_server``) so the usage the provider reports and the
requests it receives are exact. The proxy caches responses by default in this suite, so every prompt carries
a fresh marker. A probe is always 40 characters, which the limiter estimates as 10 input tokens
(``len(text) // 4``, ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` ``DEFAULT_CHARS_PER_TOKEN``).
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value, string_value
from tests.integration._support.process import OwnedState, owned_proxy
from tests.integration._support.wire import Reply, Request, Wire

WINDOW_SECONDS: Final = 60
PROBE_CHARACTERS: Final = 40
PROBE_TOKENS: Final = 10
PROVIDER_MODEL: Final = "gpt-4o-mini"
ANTHROPIC_MODEL: Final = "claude-sonnet-4-5-20250929"
HOLD_SECONDS: Final = 30
RESET_AT: Final = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$")
REDIS_ENVIRONMENT: Final = ("REDIS_HOST", "REDIS_PORT", "REDIS_URL", "REDIS_PASSWORD")


@dataclass(frozen=True, slots=True)
class Usage:
    prompt: int = 11
    completion: int = 7
    cached: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion


DEFAULT_USAGE: Final = Usage()


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def probe() -> str:
    text: Final = "limits::" + uuid.uuid4().hex
    assert len(text) == PROBE_CHARACTERS
    return text


def _openai_usage(usage: Usage) -> dict[str, JsonValue]:
    details: Final[dict[str, JsonValue]] = (
        {"prompt_tokens_details": {"cached_tokens": usage.cached}} if usage.cached else {}
    )
    return {
        "prompt_tokens": usage.prompt,
        "completion_tokens": usage.completion,
        "total_tokens": usage.total,
        **details,
    }


def _frame(payload: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chat_stream(usage: Usage) -> tuple[bytes, ...]:
    chunk: Final[dict[str, JsonValue]] = {
        "id": "chatcmpl-limits",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": PROVIDER_MODEL,
    }
    return (
        _frame({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "limits"}}]}),
        _frame({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _frame({**chunk, "choices": [], "usage": _openai_usage(usage)}),
        b"data: [DONE]\n\n",
    )


def _chat_body(usage: Usage) -> dict[str, JsonValue]:
    return {
        "id": "chatcmpl-limits",
        "object": "chat.completion",
        "created": 1,
        "model": PROVIDER_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "limits"}, "finish_reason": "stop"}],
        "usage": _openai_usage(usage),
    }


def _embedding_body(usage: Usage) -> dict[str, JsonValue]:
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}],
        "usage": {"prompt_tokens": usage.prompt, "total_tokens": usage.prompt},
    }


def _responses_body(usage: Usage) -> dict[str, JsonValue]:
    return {
        "id": "resp_limits",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": PROVIDER_MODEL,
        "output": [
            {
                "id": "msg_limits",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "limits", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": usage.prompt, "output_tokens": usage.completion, "total_tokens": usage.total},
    }


def _text_completion_body(usage: Usage) -> dict[str, JsonValue]:
    return {
        "id": "cmpl-limits",
        "object": "text_completion",
        "created": 1,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{"index": 0, "text": "limits", "finish_reason": "stop", "logprobs": None}],
        "usage": _openai_usage(usage),
    }


def _anthropic_body(usage: Usage) -> dict[str, JsonValue]:
    return {
        "id": "msg_limits",
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": "limits"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage.prompt, "output_tokens": usage.completion},
    }


def _success(request: Request, usage: Usage) -> Reply:
    path: Final = request.target.split("?", 1)[0]
    if path.endswith("/chat/completions"):
        if JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_chat_stream(usage))
        return Reply(body=json.dumps(_chat_body(usage)).encode())
    if path.endswith("/embeddings"):
        return Reply(body=json.dumps(_embedding_body(usage)).encode())
    if path.endswith("/responses"):
        return Reply(body=json.dumps(_responses_body(usage)).encode())
    if path.endswith("/messages"):
        return Reply(body=json.dumps(_anthropic_body(usage)).encode())
    if path.endswith("/completions"):
        return Reply(body=json.dumps(_text_completion_body(usage)).encode())
    raise AssertionError(f"The limits provider has no reply for POST {request.target}")


def provider(
    usage: Usage = DEFAULT_USAGE,
    *,
    status: int = 200,
    hold: threading.Event | None = None,
    gate_stream: threading.Event | None = None,
) -> Callable[[Request], Reply]:
    """A provider that answers every generation route with ``usage``.

    ``status`` other than 200 answers every POST with that OpenAI-shaped error. ``hold`` keeps every POST
    unanswered until it is set. ``gate_stream`` sends the first chat stream frame and then waits for the event
    before the rest, which keeps a stream open on the proxy.
    """

    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        if hold is not None:
            assert hold.wait(timeout=HOLD_SECONDS), "The held provider reply was never released"
        if status != 200:
            return Reply(
                status=status,
                body=json.dumps(
                    {"error": {"message": f"scripted {status}", "type": "invalid_request_error", "code": None}}
                ).encode(),
            )
        reply: Final = _success(request, usage)
        if gate_stream is not None and reply.chunks is not None:
            return Reply(content_type=reply.content_type, chunks=reply.chunks, gate_after_first=gate_stream)
        return reply

    return respond


def posts(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    """The JSON bodies of every POST the provider has received since the last drain."""
    return tuple(JSON_OBJECT.validate_json(request.body) for request in wire.drain() if request.method == "POST")


def chat(
    gateway: Gateway,
    model: str,
    key: str,
    *,
    text: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text or probe()}], **extra},
        key=key,
        headers=headers,
    )


def embed(gateway: Gateway, model: str, key: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/embeddings", {"model": model, "input": probe(), **extra}, key=key)


def respond(gateway: Gateway, model: str, key: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/responses", {"model": model, "input": probe(), "store": False, **extra}, key=key
    )


def anthropic_messages(gateway: Gateway, model: str, key: str, *, max_tokens: int = 16) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/messages",
        {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": probe()}]},
        key=key,
        headers={"anthropic-version": "2023-06-01"},
    )


def complete(gateway: Gateway, model: str, key: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/completions", {"model": model, "prompt": probe(), **extra}, key=key)


def statuses(responses: tuple[httpx.Response, ...] | list[httpx.Response]) -> list[int]:
    return [response.status_code for response in responses]


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def limit_message(scope: str, value: str, kind: str, limit: int, remaining: int = 0) -> str:
    """The v3 limiter's refusal text up to the reset time, which is the only part that moves."""
    return (
        f"Rate limit exceeded for {scope}: {value}. Limit type: {kind}. "
        f"Current limit: {limit}, Remaining: {remaining}. Limit resets at: "
    )


def assert_limited(
    response: httpx.Response,
    scope: str,
    value: str,
    kind: str,
    limit: int,
    remaining: int = 0,
    *,
    message_prefix: str = "",
) -> None:
    """The response is the v3 limiter's OpenAI-shaped 429 for exactly this scope, limit kind and limit.

    ``message_prefix`` is text an endpoint puts in front of the limiter's message (see the text completions
    test, which pins the ``litellm.RateLimitError: `` prefix that route adds).
    """
    assert response.status_code == 429, f"{response.status_code} {response.text}"
    error: Final = error_of(response)
    message: Final = string_value(error["message"])
    prefix: Final = message_prefix + limit_message(scope, value, kind, limit, remaining)
    assert message.startswith(prefix), message
    assert RESET_AT.match(message.removeprefix(prefix)), message
    assert {name: error[name] for name in ("type", "param", "code")} == {
        "type": "throttling_error",
        "param": None,
        "code": "429",
    }, response.text


def scope_headers(response: httpx.Response, scope: str) -> dict[str, str]:
    """Every ``x-ratelimit-{scope}-*`` header on the response, keyed by the part after the scope."""
    prefix: Final = f"x-ratelimit-{scope}-"
    return {name.removeprefix(prefix): value for name, value in response.headers.items() if name.startswith(prefix)}


def ratelimit_header_names(response: httpx.Response) -> frozenset[str]:
    return frozenset(name for name in response.headers if name.startswith("x-ratelimit-"))


def counter_key(scope: str, value: str, suffix: str) -> str:
    """The Redis key of one limiter value: ``{scope:value}:requests|tokens|window|max_parallel_requests``
    (``litellm-rust/crates/redis-store/src/rate_limits/keys.rs``)."""
    return f"{{{scope}:{value}}}:{suffix}"


@contextmanager
def shared_redis() -> Iterator[Redis]:
    """The Redis the shared gateway and its peer keep their rate-limit counters in."""
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as client:
        yield client


def redis_integer(client: Redis, name: str) -> int | None:
    value: Final = client.get(name)
    return None if value is None else int(float(str(value)))


def limits_config(
    directory: Path,
    *,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    without_redis: bool = False,
) -> Path:
    """A copy of the suite's proxy config with the given settings merged in.

    ``without_redis`` drops the Redis cache block, which leaves the limiter with only its in-memory counters.
    """
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    inherited_litellm: Final = {
        name: value
        for name, value in object_value(base["litellm_settings"]).items()
        if not (without_redis and name in ("cache", "cache_params", "enable_redis_auth_cache"))
    }
    config: Final = {
        **base,
        "general_settings": {**object_value(base["general_settings"]), **(general_settings or {})},
        "litellm_settings": {**inherited_litellm, **(litellm_settings or {})},
        "router_settings": {**object_value(base["router_settings"]), "num_retries": 0, **(router_settings or {})},
    }
    path: Final = directory / f"dec1-limits-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def limits_proxy(
    gateway: Gateway,
    directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    without_redis: bool = False,
    shared: bool = False,
) -> Iterator[Gateway]:
    """A proxy of our own, started with the given settings and environment.

    By default ``owned_proxy`` boots it on a private scratch database and a private Redis, so it sees nothing
    the shared gateway created and its counters never reach ``shared_redis()``. ``shared=True`` boots it on the
    shared gateway's database and Redis instead (``OwnedState.SHARED_WITH_WORKER``), for tests that create
    state through the shared gateway or read the candidate's counters from ``shared_redis()``. With
    ``without_redis`` the Redis variables are removed either way.
    """
    config: Final = limits_config(
        directory,
        litellm_settings=litellm_settings,
        general_settings=general_settings,
        router_settings=router_settings,
        without_redis=without_redis,
    )
    with owned_proxy(
        gateway,
        directory,
        dict(environment or {}),
        config=config,
        remove_environment=REDIS_ENVIRONMENT if without_redis else (),
        state=OwnedState.SHARED_WITH_WORKER if shared else None,
    ) as candidate:
        yield candidate
