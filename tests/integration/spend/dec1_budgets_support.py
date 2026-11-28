"""Shared building blocks for the `test_dec1_budgets_*` contract files.

Budgets are being moved inline into Rust at auth: the admission hold ("reservation") is taken by the
`callbacks-budgets` crate (`litellm/rust/crates/callbacks-budgets/src/engine.rs`) and the 17 auth budget checks
decide through `litellm/rust/crates/callbacks-budgets/src/checks.rs`. These helpers let a test watch that hold
from the outside: the Redis spend counters it lives in, a provider that can keep a call in flight, and a price
that makes every estimate a literal.

Pricing used by the reservation tests (`hold_model`): input tokens are free and every output token costs 0.004.
The scripted provider answers with 5 completion tokens, so one served chat call costs exactly 0.02, and the
admission estimate is exactly `max_tokens * 0.004` (input is free, so the token count of the prompt never matters):

    max_tokens 5   -> hold 0.02
    max_tokens 10  -> hold 0.04
    no max_tokens  -> hold 16384 * 0.004 = 65.536 (`DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK`, or the model ceiling,
                      which is also 16384 for gpt-4o-mini), shrunk to whatever budget is left

The plain price of the other spend contract files (`chat_model` from `dec1_spend_support`, 0.001 in and 0.002
out, also 0.02 per call) is reused where the estimate does not matter.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import httpx
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    TINY_BUDGET,
    chat_model,
    hashed,
    respond,
)

HOLD_OUTPUT_RATE: Final = 0.004
NO_MAX_TOKENS_HOLD: Final = 16384 * HOLD_OUTPUT_RATE
HOLD_MARKER: Final = "dec1-budgets-hold"
ARRIVAL_SECONDS: Final = 30

__all__ = [
    "ARRIVAL_SECONDS",
    "CHAT_COST",
    "HOLD_MARKER",
    "HOLD_OUTPUT_RATE",
    "LANDING_SECONDS",
    "NO_MAX_TOKENS_HOLD",
    "TINY_BUDGET",
    "HeldProvider",
    "background",
    "budget_message",
    "call",
    "counter",
    "counter_settles",
    "hashed",
    "held_provider",
    "hold_model",
    "key_counter",
    "masked",
    "parse_costs",
    "redis_client",
    "refused",
]


@dataclass
class HeldProvider:
    """A scripted provider that keeps every call whose prompt contains `HOLD_MARKER` waiting until `release()`.

    Calls without the marker are answered at once with the fixed usage of `dec1_spend_support.respond`.
    `failing_status` makes every inference call fail with that status instead.
    """

    wire: Wire | None = None
    release_gate: threading.Event = field(default_factory=threading.Event)
    arrived_lock: threading.Condition = field(default_factory=threading.Condition)
    arrivals: int = 0
    failing_status: int | None = None

    def answer(self, request: Request) -> Reply:
        if request.method == "POST" and HOLD_MARKER.encode() in request.body:
            with self.arrived_lock:
                self.arrivals += 1
                self.arrived_lock.notify_all()
            assert self.release_gate.wait(timeout=60), "the held call was never released"
        if request.method == "POST" and self.failing_status is not None:
            return Reply(
                status=self.failing_status,
                body=b'{"error":{"message":"dec1 budgets scripted failure","type":"server_error","code":"500"}}',
            )
        return respond(request)

    def wait_for(self, count: int, seconds: float = ARRIVAL_SECONDS) -> None:
        with self.arrived_lock:
            assert self.arrived_lock.wait_for(lambda: self.arrivals >= count, timeout=seconds), (
                f"only {self.arrivals} of {count} held calls reached the provider"
            )

    def release(self) -> None:
        self.release_gate.set()

    @property
    def url(self) -> str:
        assert self.wire is not None
        return self.wire.url


@contextmanager
def held_provider(failing_status: int | None = None) -> Iterator[HeldProvider]:
    provider: Final = HeldProvider(failing_status=failing_status)
    with wire_server(provider.answer) as wire:
        provider.wire = wire
        try:
            yield provider
        finally:
            provider.release()


def hold_model(scenario: Scenario, provider: HeldProvider | Wire, **parameters: JsonValue) -> str:
    """A gpt-4o-mini deployment where input is free and output costs 0.004 a token (see the module docstring)."""
    wire: Final = provider.wire if isinstance(provider, HeldProvider) else provider
    assert wire is not None
    return chat_model(scenario, wire, input_cost_per_token=0.0, output_cost_per_token=HOLD_OUTPUT_RATE, **parameters)


def call(
    gateway: Gateway,
    model: str,
    key: str,
    *,
    hold: bool = False,
    max_tokens: int | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """One chat completion with a unique prompt; `hold=True` makes the `HeldProvider` keep it in flight."""
    prompt: Final = f"{HOLD_MARKER if hold else 'dec1-budgets'} {uuid.uuid4().hex}"
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": prompt}], **extra}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


@contextmanager
def background() -> Iterator[Callable[[Callable[[], httpx.Response]], Future[httpx.Response]]]:
    with ThreadPoolExecutor(max_workers=16) as pool:
        yield pool.submit


@contextmanager
def redis_client() -> Iterator[Redis]:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
        yield cache


def counter(cache: Redis, name: str) -> float | None:
    """The value of one spend counter, for example `spend:key:<sha256 of the key>` or `spend:team:<team_id>`."""
    raw: Final = cache.get(name)
    return None if raw is None else float(raw)


def counter_settles(cache: Redis, name: str, amount: float, seconds: float = LANDING_SECONDS) -> float | None:
    """Wait until the counter reads `amount` (a missing counter reads as 0)."""
    return eventually(
        lambda: counter(cache, name),
        lambda value: abs((value or 0.0) - amount) < 1e-9,
        seconds=seconds,
    )


def refused(response: httpx.Response, status: int = 422) -> str:
    """Assert a budget refusal envelope (`error.type` `budget_exceeded`) and return its message."""
    assert response.status_code == status, f"{response.status_code} {response.text}"
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["type"] == "budget_exceeded", response.text
    assert error["code"] == str(status), response.text
    return string_value(error["message"])


def masked(key: str) -> str:
    """The `key_name` the proxy stores for a generated key (`abbreviate_api_key`, auth_utils.py:2944)."""
    return f"sk-...{key[-4:]}"


_COSTS: Final = re.compile(
    r"^Budget has been exceeded! (?P<label>\S+) Current cost: (?P<current>[0-9.e+-]+), "
    r"(?:Estimated request cost: (?P<estimate>[0-9.e+-]+), )?Max budget: (?P<max>[0-9.e+-]+)$"
)


def parse_costs(message: str) -> Mapping[str, str | None]:
    """Split a `Budget has been exceeded! <Entity>=<id> Current cost: ...` message into its parts."""
    matched: Final = _COSTS.match(message)
    assert matched is not None, message
    return matched.groupdict()


def budget_message(label: str, current: str, maximum: str, estimate: str | None = None) -> str:
    """The wording shared by the Rust auth checks (`checks.rs::current`) and the admission engine
    (`engine.rs::exceeded`)."""
    estimated: Final = "" if estimate is None else f"Estimated request cost: {estimate}, "
    return f"Budget has been exceeded! {label} Current cost: {current}, {estimated}Max budget: {maximum}"


def key_counter(key: str) -> str:
    return f"spend:key:{hashed(key)}"
