"""Shared building blocks for the `test_dec1_spendlogs_*` contract files (Spend Logs V2 parity).

These files pin the `LiteLLM_SpendLogs` row and what is built from it, so the Rust `callbacks-spend-log` port
(plan: spend logs on CustomLoggerV2, PRs 1 to 4) can be checked column by column against today's Python writer.
They build on `dec1_spend_support`, whose scripted provider answers with fixed usage and whose deployments are
priced at 0.001 per input token and 0.002 per output token, so a plain chat call costs 0.02.

What this module adds: providers that report prompt caching usage, providers that hang (to force a timeout),
fail once and then succeed (to force a retry), an owned proxy with extra settings, and small readers for the
spend log list routes.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.spend.dec1_spend_support import (
    LANDING_SECONDS,
    REPLY_TEXT,
    ROW_COLUMNS,
    hashed,
    proxy_config,
    respond,
)

CACHED_PROMPT_TOKENS: Final = 4
"""`prompt_tokens_details.cached_tokens` reported by `caching_upstream` on a chat call (10 prompt tokens in all)."""
CACHE_WRITE_TOKENS: Final = 2
CACHE_READ_TOKENS: Final = 3
"""Anthropic `cache_creation_input_tokens` / `cache_read_input_tokens` reported by `caching_upstream`."""

ERROR_STATUS_HEADER: Final = "x-dec1-fail-first"
"""A request header that, when present, makes `flaky_upstream` fail the first call it sees with that status."""


def caching_reply(request: Request) -> Reply:
    """Like `respond`, but chat usage carries `prompt_tokens_details.cached_tokens` 4 and Anthropic usage carries
    `cache_creation_input_tokens` 2 and `cache_read_input_tokens` 3 (input 12, output 3 as usual)."""
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    path: Final = request.target.split("?", 1)[0]
    body: Final = JSON_OBJECT.validate_json(request.body) if request.body else {}
    if path.endswith("/messages"):
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg_dec1sl{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model"),
                    "content": [{"type": "text", "text": REPLY_TEXT}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": CACHE_WRITE_TOKENS,
                        "cache_read_input_tokens": CACHE_READ_TOKENS,
                    },
                }
            ).encode()
        )
    if path.endswith("/chat/completions") and body.get("stream") is not True:
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-dec1sl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "prompt_tokens_details": {"cached_tokens": CACHED_PROMPT_TOKENS},
                    },
                }
            ).encode()
        )
    return respond(request)


@contextmanager
def caching_upstream() -> Iterator[Wire]:
    with wire_server(caching_reply) as wire:
        yield wire


@contextmanager
def hanging_upstream() -> Iterator[Wire]:
    """A provider that sends response headers for every inference call and then never sends the body, so a
    deployment with a short `timeout` fails every attempt with `litellm.Timeout`. The gate is released at
    teardown so the server can stop."""
    gate: Final = threading.Event()

    def hang(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        return Reply(chunks=(b"{}",), gate_before_first=gate)

    with wire_server(hang) as wire:
        try:
            yield wire
        finally:
            gate.set()


@contextmanager
def flaky_upstream(status: int = 500) -> Iterator[Wire]:
    """A provider whose first inference call fails with `status`; every later call is answered normally."""
    lock: Final = threading.Lock()
    calls: Final = [0]

    def answer(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        with lock:
            calls[0] += 1
            first: Final = calls[0] == 1
        if first:
            return Reply(
                status=status,
                body=json.dumps(
                    {"error": {"message": "dec1 spendlogs flaky failure", "type": "server_error", "code": str(status)}}
                ).encode(),
            )
        return respond(request)

    with wire_server(answer) as wire:
        yield wire


@contextmanager
def failing_upstream(status: int, message: str = "dec1 spendlogs upstream failure") -> Iterator[Wire]:
    def fail(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        return Reply(
            status=status,
            body=json.dumps({"error": {"message": message, "type": "server_error", "code": str(status)}}).encode(),
        )

    with wire_server(fail) as wire:
        yield wire


@contextmanager
def owned_spend_proxy(
    gateway: Gateway,
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
    upstream: Callable[[], AbstractContextManager[Wire]] | None = None,
) -> Iterator[tuple[Gateway, Scenario, Wire]]:
    """An owned proxy on the worker's database with settings merged into the shared integration config, a
    scenario on it, and a scripted provider (`respond` unless another upstream factory is given)."""
    config: Final = proxy_config(directory, general_settings=general_settings, litellm_settings=litellm_settings)
    provider: Final = upstream() if upstream is not None else wire_server(respond)
    with (
        provider as wire,
        owned_proxy(gateway, directory, dict(environment or {}), config=config) as owned,
        owned.scenario() as scenario,
    ):
        yield owned, scenario, wire


def rows_where(gateway: Gateway, condition: str, parameters: tuple[str, ...], count: int) -> list[dict[str, JsonValue]]:
    """Spend log rows matching a fixed SQL condition, oldest first, once exactly `count` of them exist."""
    return eventually(
        lambda: gateway.rows(
            f'SELECT {ROW_COLUMNS}, requester_ip_address, mcp_namespaced_tool_name, agent_id FROM "LiteLLM_SpendLogs" '
            f'WHERE {condition} ORDER BY "startTime", request_id',
            parameters,
        ),
        lambda found: len(found) == count,
        seconds=LANDING_SECONDS,
    )


def key_rows(gateway: Gateway, key: str, count: int) -> list[dict[str, JsonValue]]:
    return rows_where(gateway, "api_key = %s", (hashed(key),), count)


def deliveries(gateway: Gateway, request_id: str, count: int) -> list[dict[str, JsonValue]]:
    """`LiteLLM_SpendLogDelivery` rows for one request id, once exactly `count` exist."""
    return eventually(
        lambda: gateway.rows(
            "SELECT delivery_id, request_id, state::text AS state, accounting_payload, is_batch, attempts "
            'FROM "LiteLLM_SpendLogDelivery" WHERE request_id = %s',
            (request_id,),
        ),
        lambda found: len(found) == count,
        seconds=LANDING_SECONDS,
    )


def today_window() -> dict[str, str]:
    """A `start_date`/`end_date` pair in the `YYYY-MM-DD HH:MM:SS` form both list routes accept, covering
    yesterday to tomorrow in UTC so a row written now is inside it whatever the runner's clock zone."""
    now: Final = datetime.now(timezone.utc)
    return {
        "start_date": (now - timedelta(days=1)).strftime("%Y-%m-%d 00:00:00"),
        "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%d 23:59:59"),
    }


def list_logs(gateway: Gateway, path: str, params: Mapping[str, str], *, key: str | None = None) -> httpx.Response:
    return gateway.request("GET", path, params=dict(params), key=key)


def list_body(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def error_text(response: httpx.Response) -> str:
    """The human readable error of a proxy error body, whether it is `{"error": {"message"}}` or `{"detail"}`."""
    body: Final = JSON_OBJECT.validate_json(response.content)
    error: Final = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message"))
    return json.dumps(body.get("detail"))


SLOW_STREAM_CHUNKS: Final = 40
SLOW_STREAM_PAUSE: Final = 0.25


def _slow_chat_stream(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    identity: Final = f"chatcmpl-dec1sl-slow-{uuid.uuid4().hex}"
    piece: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "slow "}, "finish_reason": None}],
    }
    usage: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": SLOW_STREAM_CHUNKS,
            "total_tokens": 10 + SLOW_STREAM_CHUNKS,
        },
    }
    return Reply(
        content_type="text/event-stream",
        chunks=(
            *(b"data: " + json.dumps(piece).encode() + b"\n\n" for _ in range(SLOW_STREAM_CHUNKS)),
            b"data: " + json.dumps(usage).encode() + b"\n\n",
            b"data: [DONE]\n\n",
        ),
        pause_between_chunks=SLOW_STREAM_PAUSE,
    )


@contextmanager
def slow_stream_upstream() -> Iterator[Wire]:
    """A chat provider that streams 40 content chunks 0.25 s apart, then a usage chunk (10 prompt and 40
    completion tokens) and `[DONE]`: slow enough for a caller to hang up in the middle, and for the first token
    to arrive well before the last."""
    with wire_server(_slow_chat_stream) as wire:
        yield wire
