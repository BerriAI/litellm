"""Shared rig for the `test_dec1_routing_*` files: one scripted OpenAI-shaped upstream and owned proxies.

Every deployment in these files points at one in-process upstream. The upstream decides what to answer
from the upstream model name the proxy sends (the part after `openai/` in `litellm_params.model`), so a
config can describe a failing, slow or healthy deployment without any per-test server:

`ok-<label>` answers 200 with the text `served ok-<label>`. `status-<code>-<label>` always answers that
HTTP status. `script-<label>` answers the statuses queued with `Upstream.script` and 200 after that.
`retryafter-<label>` is a script whose 429 answers carry `retry-after: 2`. `slow-<label>` answers 200 after
`SLOW_SECONDS`, `mid-<label>` after `MID_SECONDS` and `lag-<label>` after `LAG_SECONDS`. `held-<label>` answers 200 once
`Upstream.release(label)` is called. `context-<label>` and `policy-<label>` answer the 400 bodies OpenAI
uses for a context window overflow and a content policy refusal. `drop-<label>` closes the connection without
answering.

The upstream keeps every request it saw, so a test counts attempts per upstream model name. That count is
the observable these contracts are written against: how many times and in which order the proxy called
each deployment.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.openai_wire import chat_reply, responses_reply
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

SLOW_SECONDS: Final = 4
MID_SECONDS: Final = 1.5
LAG_SECONDS: Final = 0.4
HELD_SECONDS: Final = 30
RETRY_AFTER_HEADER_SECONDS: Final = 2
UNREACHABLE_API_BASE: Final = "http://127.0.0.1:9/v1"
PROVIDER_KEY: Final = "dec1-routing-provider-key"
CONTEXT_MESSAGE: Final = "This model's maximum context length is 4096 tokens"
POLICY_MESSAGE: Final = "scripted content policy rejection"
NO_DEPLOYMENTS: Final = "No deployments available for selected model"
INVALID_MODEL: Final = "Invalid model name passed in model="
MESSAGE_PREFIX: Final[Mapping[int, str]] = {
    400: "litellm.BadRequestError: OpenAIException - ",
    401: "litellm.AuthenticationError: AuthenticationError: OpenAIException - ",
    404: "litellm.NotFoundError: NotFoundError: OpenAIException - ",
    408: "litellm.Timeout: Timeout Error: OpenAIException - ",
    429: "litellm.RateLimitError: RateLimitError: OpenAIException - ",
    500: "litellm.InternalServerError: InternalServerError: OpenAIException - ",
    503: "litellm.ServiceUnavailableError: ServiceUnavailableError: OpenAIException - ",
}


@dataclass(frozen=True, slots=True)
class Seen:
    model: str
    target: str
    body: Mapping[str, JsonValue]
    at: float
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Upstream:
    wire: Wire
    log: list[Seen] = field(default_factory=list)  # mutable-ok: appended by the upstream's handler threads
    scripts: dict[str, deque[int]] = field(default_factory=dict)  # mutable-ok: queued by tests, popped by the handler
    gates: dict[str, threading.Event] = field(default_factory=dict)  # mutable-ok: one release gate per held label

    @property
    def api_base(self) -> str:
        return f"{self.wire.url}/v1"

    def calls(self, model: str) -> tuple[Seen, ...]:
        return tuple(seen for seen in tuple(self.log) if seen.model == model)

    def count(self, model: str) -> int:
        return len(self.calls(model))

    def order(self, *models: str) -> tuple[str, ...]:
        return tuple(seen.model for seen in tuple(self.log) if seen.model in models)

    def script(self, model: str, *statuses: int) -> None:
        self.scripts[model] = deque(statuses)

    def gate(self, model: str) -> threading.Event:
        return self.gates.setdefault(model, threading.Event())

    def release(self, model: str) -> None:
        self.gate(model).set()

    def hold(self, model: str) -> None:
        self.gate(model).clear()


def error_reply(
    status: int,
    message: str,
    *,
    kind: str = "scripted_error",
    code: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": message, "type": kind, "code": code}}).encode(),
        headers=headers or {},
    )


def _success(request: Request, model: str, body: Mapping[str, JsonValue]) -> Reply:
    stream: Final = body.get("stream") is True
    text: Final = f"served {model}"
    if request.target.endswith("/embeddings"):
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "model": model,
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}],
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                }
            ).encode()
        )
    if request.target.endswith("/responses"):
        return responses_reply(f"resp_{uuid.uuid4().hex}", model, text, stream=stream)
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", model, text, stream=stream)


def _scripted_status(upstream: Upstream, model: str) -> int:
    queued: Final = upstream.scripts.get(model)
    return queued.popleft() if queued else 200


def _answer(upstream: Upstream, request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = JSON_OBJECT.validate_json(request.body)
    model: Final = str(body.get("model", ""))
    upstream.log.append(Seen(model, request.target, body, time.monotonic(), request.headers))
    kind: Final = model.split("-", 1)[0]
    if kind == "status":
        status: Final = int(model.split("-")[1])
        return error_reply(status, f"scripted {status} from {model}")
    if kind in ("script", "retryafter"):
        scripted: Final = _scripted_status(upstream, model)
        if scripted != 200:
            retry_after: Final = {"retry-after": str(RETRY_AFTER_HEADER_SECONDS)} if kind == "retryafter" else {}
            return error_reply(scripted, f"scripted {scripted} from {model}", headers=retry_after)
    if kind == "context":
        return error_reply(400, CONTEXT_MESSAGE, kind="invalid_request_error", code="context_length_exceeded")
    if kind == "policy":
        return error_reply(400, POLICY_MESSAGE, kind="invalid_request_error", code="content_policy_violation")
    if kind == "drop":
        return Reply(drop_connection=True)
    if kind == "slow":
        time.sleep(SLOW_SECONDS)
    if kind == "mid":
        time.sleep(MID_SECONDS)
    if kind == "lag":
        time.sleep(LAG_SECONDS)
    if kind == "held":
        assert upstream.gate(model).wait(timeout=HELD_SECONDS), f"{model} was never released"
    return _success(request, model, body)


@contextmanager
def scripted_upstream() -> Iterator[Upstream]:
    holder: Final[list[Upstream]] = []  # mutable-ok: the handler needs the Upstream built from its own wire

    def respond(request: Request) -> Reply:
        return _answer(holder[0], request)

    with wire_server(respond) as wire:
        upstream: Final = Upstream(wire)
        holder.append(upstream)
        try:
            yield upstream
        finally:
            for gate in tuple(upstream.gates.values()):
                gate.set()


def deployment(
    group: str,
    upstream_model: str,
    api_base: str,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    """One `model_list` entry whose deployment id is `dec1-<upstream_model>` unless `model_info` names one."""
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"openai/{upstream_model}",
            "api_key": PROVIDER_KEY,
            "api_base": api_base,
            **params,
        },
        "model_info": {"id": f"dec1-{upstream_model}", **(model_info or {})},
    }


def routing_config(
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]],
    router_settings: Mapping[str, JsonValue],
    general_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    assert isinstance(base, dict)
    config: Final = {
        **base,
        "model_list": [dict(entry) for entry in model_list],
        "router_settings": dict(router_settings),
        "general_settings": {**base.get("general_settings", {}), **(general_settings or {})},
    }
    path: Final = directory / f"dec1-routing-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def routing_proxy(
    shared: Gateway,
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]],
    router_settings: Mapping[str, JsonValue],
    general_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Gateway]:
    """An owned proxy whose `model_list` and `router_settings` are exactly the ones given."""
    config: Final = routing_config(
        directory, model_list=model_list, router_settings=router_settings, general_settings=general_settings
    )
    with owned_proxy(shared, directory, dict(environment or {}), config=config) as candidate:
        yield candidate


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: Upstream


@contextmanager
def routing_rig(
    directory: Path,
    models: Callable[[str], Sequence[Mapping[str, JsonValue]]],
    router_settings: Mapping[str, JsonValue],
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[Rig]:
    """The scripted upstream plus one owned proxy whose `model_list` is `models(upstream.api_base)`."""
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        upstream: Final = stack.enter_context(scripted_upstream())
        gateway: Final = stack.enter_context(
            routing_proxy(
                shared,
                directory,
                model_list=models(upstream.api_base),
                router_settings=router_settings,
                general_settings=general_settings,
            )
        )
        yield Rig(gateway, upstream)


def pair(group: str, upstream_model: str, api_base: str, **params: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    """Two deployments of one group that call the same upstream model, with ids `dec1-<group>-a` and `-b`.

    A group of two never sleeps between retries, so retry counts can be observed without waiting for backoff."""
    return tuple(
        deployment(group, upstream_model, api_base, model_info={"id": f"dec1-{group}-{suffix}"}, **params)
        for suffix in ("a", "b")
    )


def marker() -> str:
    return f"dec1-routing-{uuid.uuid4().hex}"


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


def embed(gateway: Gateway, model: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/embeddings", {"model": model, "input": marker(), **extra})


def anthropic_message(gateway: Gateway, model: str, **extra: JsonValue) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": marker()}],
        **extra,
    }
    return gateway.request("POST", "/v1/messages", body, headers={"anthropic-version": "2023-06-01"})


def respond(gateway: Gateway, model: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/responses", {"model": model, "input": marker(), **extra})


def stream_chat(gateway: Gateway, model: str, **extra: JsonValue) -> tuple[int, str]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": marker()}],
        **extra,
    }
    with gateway.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        return response.status_code, response.read().decode()


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def content_of(response: httpx.Response) -> str:
    choices: Final = JSON_OBJECT.validate_json(response.content)["choices"]
    assert isinstance(choices, list) and choices, response.text
    return string_value(object_value(object_value(choices[0])["message"])["content"])


def statuses(responses: Sequence[httpx.Response]) -> tuple[int, ...]:
    return tuple(response.status_code for response in responses)
