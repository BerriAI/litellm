"""Shared pieces for the ``test_dec1_ratelim_*`` contracts.

These files extend the ``test_dec1_limits_*`` family and reuse its provider, probes and assertions from
``dec1_limits_support``. What is added here: the ``rate_limited_by`` header check, a raw-socket probe of the
refusal that the proxy answers before the request body arrives (#51428), helpers for tagged, agent and session
requests, and small Redis readers.

The proxy's limiter is ``litellm/proxy/hooks/parallel_request_limiter_v3.py``. Its Rust port (draft PR #51537,
crate ``callbacks-parallel-request-limiter``) rebuilds every descriptor and settlement path these tests touch, so
every expectation here is a literal read from the Python code.
"""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Final

import httpx
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import Gateway, Scenario, string_value
from tests.integration.authorization.dec1_limits_support import (
    PROVIDER_MODEL,
    assert_limited,
    counter_key,
    probe,
    redis_integer,
)

WINDOW_SECONDS: Final = 60
DECLARED_BODY_BYTES: Final = 8 * 1024 * 1024
SENT_BODY_BYTES: Final = 1024
NO_ANSWER: Final = 0


def assert_refused_by(
    response: httpx.Response,
    scope: str,
    value: str,
    kind: str,
    limit: int,
    remaining: int = 0,
) -> None:
    """The v3 limiter's 429 for exactly this scope, plus the headers ``_handle_rate_limit_error`` sets:
    ``rate_limited_by`` equal to the scope word, ``rate_limit_type`` equal to the limit kind and
    ``retry-after`` equal to the window (``parallel_request_limiter_v3.py`` ``_handle_rate_limit_error``)."""
    assert_limited(response, scope, value, kind, limit, remaining)
    assert response.headers["rate_limited_by"] == scope, dict(response.headers)
    assert response.headers["rate_limit_type"] == kind, dict(response.headers)
    assert response.headers["retry-after"] == str(WINDOW_SECONDS), dict(response.headers)


def tagged_chat(
    gateway: Gateway,
    model: str,
    key: str,
    *,
    metadata_tags: tuple[str, ...] = (),
    body_tags: tuple[str, ...] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """A chat request carrying ``metadata.tags`` and, when given, the top-level ``tags`` field too."""
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": probe()}],
        "metadata": {"tags": list(metadata_tags)},
        **({"tags": list(body_tags)} if body_tags is not None else {}),
        **extra,
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def chat_with(
    gateway: Gateway,
    model: str,
    key: str,
    *,
    metadata: Mapping[str, JsonValue] | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """A chat request with an arbitrary ``metadata`` object and top-level fields."""
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": probe()}],
        **({"metadata": dict(metadata)} if metadata is not None else {}),
        **extra,
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key, headers=headers)


def tag(gateway: Gateway, scenario: Scenario, **limits: JsonValue) -> str:
    """A tag created through ``POST /tag/new`` with the given limits, deleted when the scenario ends."""
    name: Final = f"ratelim-tag-{uuid.uuid4().hex[:12]}"
    gateway.post("/tag/new", {"name": name, **limits})
    scenario.cleanups.callback(gateway.post, "/tag/delete", {"name": name})
    return name


def _agent_card(name: str, url: str) -> dict[str, JsonValue]:
    return {
        "protocolVersion": "0.3",
        "name": name,
        "description": "Synthetic rate limit agent",
        "version": "1.0.0",
        "url": url + "/",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def agent(gateway: Gateway, scenario: Scenario, url: str, **limits: JsonValue) -> str:
    """An agent registered through ``POST /v1/agents`` with ``rpm_limit``, ``tpm_limit``, ``session_rpm_limit``
    or ``session_tpm_limit`` (``litellm/types/agents.py`` ``AgentConfig``), deleted when the scenario ends."""
    name: Final = f"ratelim-agent-{uuid.uuid4().hex[:12]}"
    created: Final = gateway.request(
        "POST", "/v1/agents", {"agent_name": name, "agent_card_params": _agent_card(name, url), **limits}
    )
    assert created.status_code == 200, created.text
    identity: Final = string_value(created.json()["agent_id"])
    scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/agents/{identity}")
    return identity


def end_user(gateway: Gateway, scenario: Scenario, **budget: JsonValue) -> tuple[str, str]:
    """An end user whose budget carries the given limits; returns ``(end_user_id, budget_id)``."""
    budget_id: Final = scenario.budget(**budget)
    identity: Final = f"ratelim-customer-{uuid.uuid4().hex}"
    gateway.post("/end_user/new", {"user_id": identity, "budget_id": budget_id})
    scenario.cleanups.callback(gateway.post, "/end_user/delete", {"user_ids": [identity]})
    return identity, budget_id


def status_before_body_completes(
    gateway: Gateway,
    key: str,
    *,
    path: str = "/v1/chat/completions",
    model: str = PROVIDER_MODEL,
    timeout: float = 5.0,
) -> int:
    """POST ``path`` with a declared 8 MB body, send only its first kilobyte, and return the status line's code.

    Returns ``NO_ANSWER`` (0) when the proxy keeps waiting for the rest of the body, which means the request
    could not be refused from its headers alone. Same technique as ``test_reject_before_body.py``.
    """
    url: Final = gateway.client.base_url
    head: Final = (
        f"POST {path} HTTP/1.1\r\nHost: {url.host}\r\nAuthorization: Bearer {key}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {DECLARED_BODY_BYTES}\r\n\r\n"
    ).encode()
    partial_body: Final = (b'{"model": "' + model.encode() + b'", "messages": [{"role": "user", "content": "').ljust(
        SENT_BODY_BYTES, b"A"
    )
    with socket.create_connection((url.host, url.port or 80), timeout=timeout) as connection:
        connection.sendall(head + partial_body)
        try:
            status_line: Final = connection.recv(4096).split(b"\r\n", 1)[0]
        except TimeoutError:
            return NO_ANSWER
    return int(status_line.split()[1]) if status_line else NO_ANSWER


def raw_answer_before_body_completes(gateway: Gateway, key: str, *, timeout: float = 5.0) -> tuple[int, str]:
    """Like ``status_before_body_completes`` but returns ``(status, raw head and body text)`` for header checks."""
    url: Final = gateway.client.base_url
    head: Final = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {url.host}\r\nAuthorization: Bearer {key}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {DECLARED_BODY_BYTES}\r\n\r\n"
    ).encode()
    with socket.create_connection((url.host, url.port or 80), timeout=timeout) as connection:
        connection.sendall(head + b'{"model": "'.ljust(SENT_BODY_BYTES, b"A"))
        try:
            answer: Final = connection.recv(65536).decode(errors="replace")
        except TimeoutError:
            return NO_ANSWER, ""
    status_line: Final = answer.split("\r\n", 1)[0]
    return (int(status_line.split()[1]) if status_line else NO_ANSWER), answer


def lower_headers(raw_answer: str) -> dict[str, str]:
    """The header block of a raw HTTP answer as a lower-cased name to value map."""
    head: Final = raw_answer.split("\r\n\r\n", 1)[0].split("\r\n")[1:]
    return {name.strip().lower(): value.strip() for name, _, value in (line.partition(":") for line in head)}


@contextmanager
def redis_counters() -> Iterator[Redis]:
    """The shared Redis the gateway keeps its limiter counters in (same as ``dec1_limits_support.shared_redis``)."""
    import os

    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as client:
        yield client


def requests_counter(client: Redis, scope: str, value: str) -> int | None:
    return redis_integer(client, counter_key(scope, value, "requests"))


def tokens_counter(client: Redis, scope: str, value: str) -> int | None:
    return redis_integer(client, counter_key(scope, value, "tokens"))


def in_flight(client: Redis, scope: str, value: str) -> int:
    """Members of the ``{scope:value}:max_parallel_requests`` sorted set, one per request holding a slot."""
    return int(client.zcard(counter_key(scope, value, "max_parallel_requests")))


def sse_lines(response: httpx.Response) -> tuple[str, ...]:
    return tuple(line for line in response.text.splitlines() if line.startswith("data: "))


def json_line(line: str) -> dict[str, JsonValue]:
    return json.loads(line.removeprefix("data: "))
