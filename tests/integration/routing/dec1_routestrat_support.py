"""Shared rig for the `test_dec1_routestrat_*` files: one peer that speaks every big provider, and owned proxies.

These files pin which deployment each routing strategy of the Rust router core (`litellm/rust/crates/router-core`)
picks, with deployments on OpenAI, Azure OpenAI, Bedrock, Vertex AI and Anthropic in the same model group. The
router core runs simple-shuffle natively (`select.rs`, `simple_shuffle`) and asks the Python host for every other
strategy (`Host::select` with `SelectStage::Strategy`, answered by `litellm/router_strategy/*.py`), so a rewrite of
either side can change a pick.

One in-process peer answers every deployment. Each deployment's `api_base` carries a label as its first path
segment (`<peer>/<label>/...`), so the peer knows which deployment a request reached whatever the provider's URL
shape is, and every 200 it sends carries the text `served <label>`. A test therefore reads the serving deployment
from the response content, and cross-checks it with the `x-litellm-model-id` header, whose value is the
deployment id `dec1-routestrat-<label>` given in `model_info.id`.

The peer can delay a label (`Peer.delay`), hold a label until released (`Peer.hold` / `Peer.release`), queue
error replies (`Peer.plan`), and change the completion token count of an OpenAI-shaped answer
(`Peer.completion_tokens`), which latency routing divides the response time by.

The Rust core draws its shuffle from the thread RNG (`ThreadRandomness` in `select.rs`); the seeded randomness
there is only reachable from the crate's own unit tests, so a proxy cannot be given a fixed seed. Distribution
tests therefore send many requests and assert generous bounds (about five standard deviations either side of
the expected share), which keeps the false failure rate far below one in a million runs.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.providers.dec1_vertex_support import TOKEN_PATH
from tests.integration.routing.dec1_reliab_support import (
    BUILDERS,
    OPENAI_MODEL,
    success_for,
)
from tests.integration.routing.dec1_routing_support import routing_proxy

ID_PREFIX: Final = "dec1-routestrat-"
HELD_SECONDS: Final = 30
NO_DEPLOYMENTS: Final = "No deployments available for selected model"
CROSSED_BUDGET: Final = "No deployments available - crossed budget"
TAGS_REFUSED: Final = "Not allowed to access model due to tags configuration"
PROVIDERS: Final = ("openai", "azure", "bedrock", "vertex", "anthropic")
LONG_PROMPT: Final = "routing strategy contract " * 300


@dataclass(frozen=True, slots=True)
class Hit:
    label: str
    path: str
    body: Mapping[str, JsonValue]
    at: float
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Peer:
    wire: Wire
    log: list[Hit] = field(default_factory=list)  # mutable-ok: appended by the peer's handler threads
    plans: dict[str, deque[Reply]] = field(default_factory=dict)  # mutable-ok: queued by tests
    fixed: dict[str, Reply] = field(default_factory=dict)  # mutable-ok: set by tests
    delays: dict[str, float] = field(default_factory=dict)  # mutable-ok: seconds a label waits before answering
    gates: dict[str, threading.Event] = field(default_factory=dict)  # mutable-ok: one release gate per held label
    tokens: dict[str, int] = field(default_factory=dict)  # mutable-ok: completion tokens of a label's OpenAI answer

    @property
    def url(self) -> str:
        return self.wire.url

    def plan(self, label: str, *replies: Reply) -> None:
        self.plans[label] = deque(replies)

    def always(self, label: str, reply: Reply) -> None:
        self.fixed[label] = reply

    def delay(self, label: str, seconds: float) -> None:
        self.delays[label] = seconds

    def completion_tokens(self, label: str, tokens: int) -> None:
        self.tokens[label] = tokens

    def hold(self, label: str) -> None:
        self.gates.setdefault(label, threading.Event()).clear()

    def release(self, label: str) -> None:
        self.gates.setdefault(label, threading.Event()).set()

    def hits(self, label: str) -> tuple[Hit, ...]:
        return tuple(hit for hit in tuple(self.log) if hit.label == label)

    def count(self, label: str) -> int:
        return len(self.hits(label))

    def order(self, *labels: str) -> tuple[str, ...]:
        return tuple(hit.label for hit in tuple(self.log) if hit.label in labels)


def _label_of(path: str) -> str:
    return path.lstrip("/").split("/", 1)[0].split(":", 1)[0]


def _body(request: Request) -> Mapping[str, JsonValue]:
    try:
        return JSON_OBJECT.validate_json(request.body) if request.body else {}
    except ValueError:
        return {}


def openai_reply_with_tokens(text: str, completion_tokens: int) -> Reply:
    """An OpenAI chat completion whose `usage.completion_tokens` is `completion_tokens`."""
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": OPENAI_MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 10 + completion_tokens,
                },
            }
        ).encode()
    )


def _token_reply() -> Reply:
    return Reply(
        body=json.dumps({"access_token": "dec1-routestrat-token", "expires_in": 3600, "token_type": "Bearer"}).encode()
    )


def _answer(peer: Peer, request: Request) -> Reply:
    path: Final = request.target.split("?", 1)[0]
    if path == TOKEN_PATH:
        return _token_reply()
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    label: Final = _label_of(path)
    body: Final = _body(request)
    peer.log.append(Hit(label, path, body, time.monotonic(), request.headers))
    if label in peer.gates:
        assert peer.gates[label].wait(timeout=HELD_SECONDS), f"{label} was never released"
    if label in peer.delays:
        time.sleep(peer.delays[label])
    queued: Final = peer.plans.get(label)
    if queued:
        return queued.popleft()
    if label in peer.fixed:
        return peer.fixed[label]
    if label in peer.tokens and path.endswith("/chat/completions") and body.get("stream") is not True:
        return openai_reply_with_tokens(f"served {label}", peer.tokens[label])
    return success_for(path, body, f"served {label}")


@contextmanager
def strat_peer() -> Iterator[Peer]:
    holder: Final[list[Peer]] = []  # mutable-ok: the handler needs the Peer built from its own wire

    def respond(request: Request) -> Reply:
        return _answer(holder[0], request)

    with wire_server(respond) as wire:
        holder.append(Peer(wire))
        try:
            yield holder[0]
        finally:
            for gate in tuple(holder[0].gates.values()):
                gate.set()


def dep(
    provider: str,
    group: str,
    label: str,
    base: str,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    """One deployment of `provider` in `group`, served by the peer under `label`, with id `dec1-routestrat-<label>`.

    Builders come from `dec1_reliab_support`: `openai/gpt-4o-mini`, `azure/<label>`, `bedrock/converse/amazon.nova-pro-v1:0`
    with a bearer key, `vertex_ai/gemini-2.5-flash` with a service account whose token URI is the peer, and
    `anthropic/claude-sonnet-4-5-20250929`. Extra `params` land in `litellm_params`.
    """
    return BUILDERS[provider](
        group, label, base, model_info={"id": f"{ID_PREFIX}{label}", **(model_info or {})}, **params
    )


def ident(label: str) -> str:
    return f"{ID_PREFIX}{label}"


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    peer: Peer


@contextmanager
def strat_rig(
    directory: Path,
    models: Callable[[str], Sequence[Mapping[str, JsonValue]]],
    router_settings: Mapping[str, JsonValue],
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Rig]:
    """The peer plus one owned proxy whose `model_list` is `models(peer.url)` and whose router settings are given."""
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        peer: Final = stack.enter_context(strat_peer())
        gateway: Final = stack.enter_context(
            routing_proxy(
                shared,
                directory,
                model_list=models(peer.url),
                router_settings=router_settings,
                general_settings=general_settings,
                environment=environment,
            )
        )
        yield Rig(gateway, peer)


def marker() -> str:
    return f"dec1-routestrat-{uuid.uuid4().hex}"


def chat(
    gateway: Gateway,
    model: str,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": marker()}], **extra}
    return gateway.request("POST", "/v1/chat/completions", body, key=key, headers=headers)


def stream_chat(gateway: Gateway, model: str, **extra: JsonValue) -> tuple[httpx.Headers, str]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": marker()}],
        **extra,
    }
    with gateway.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return response.headers, response.read().decode()


def content_of(response: httpx.Response) -> str:
    choices: Final = JSON_OBJECT.validate_json(response.content)["choices"]
    assert isinstance(choices, list) and choices, response.text
    return string_value(object_value(object_value(choices[0])["message"])["content"])


def served(gateway: Gateway, model: str, *, key: str | None = None, **extra: JsonValue) -> str:
    """The label of the deployment that served one successful request."""
    response: Final = chat(gateway, model, key=key, **extra)
    assert response.status_code == 200, response.text
    label: Final = content_of(response).removeprefix("served ")
    assert response.headers["x-litellm-model-id"] == ident(label), response.headers
    return label


def served_many(
    gateway: Gateway, model: str, requests: int, *, key: str | None = None, **extra: JsonValue
) -> Counter[str]:
    return Counter(served(gateway, model, key=key, **extra) for _ in range(requests))


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def reported_strategy(gateway: Gateway) -> JsonValue:
    return object_value(gateway.get("/router/settings")["current_values"])["routing_strategy"]


def early_in_a_minute(latest_second: int = 40) -> None:
    """Wait, by polling the clock, until the UTC second is below `latest_second`.

    Per-minute counters (`%H-%M` buckets in `lowest_tpm_rpm_v2.py`, `%Y-%m-%d-%H-%M` in `lowest_cost.py` and
    `lowest_latency.py`) reset at the minute boundary, so a test that counts requests within one minute starts
    early in a minute instead of sleeping."""
    eventually(lambda: datetime.now(UTC).second, lambda second: second < latest_second, seconds=70)
