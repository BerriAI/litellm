"""Shared pieces for the ``test_dec1_limcap4_*`` contracts (rate limits and model capacity, round 4).

Round 4 works from the "Not covered" list of ``test_dec1_limcap3_``: batch rows charged per model (end user daily
tokens, project input and output windows, rows rewritten from the proxy alias to the deployment model by #51843),
MCP rate limits per team and their interaction with the server window, extraction requests in the token estimate,
image routes under model capacity, a provider failure returning a borrowing team's held tokens, router retries and
fallbacks in key and capacity windows, and limit changes reaching a second instance.

The limits provider, probes, refusal assertions and Redis readers come from ``dec1_limits_support``; the batch
provider from ``dec1_limcap2_support``. This module adds a provider that fails its first answers, an image provider,
batch files whose rows each name their own model, and a few readers.

Every literal is read from the Python product on main (49b2058c): the v3 limiter
(``litellm/proxy/hooks/parallel_request_limiter_v3.py``, Rust port in draft PR #51537), the batch limiter
(``litellm/proxy/hooks/batch_rate_limiter.py``) and the dynamic rate limiter v3
(``litellm/proxy/hooks/dynamic_rate_limiter_v3.py``, Rust port in draft PR #51540).
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire
from tests.integration.authorization.dec1_budlim2_support import PNG_BYTES
from tests.integration.authorization.dec1_limits_support import (
    DEFAULT_USAGE,
    Usage,
    chat,
    counter_key,
    provider,
    redis_integer,
    shared_redis,
)

PROVIDER_TOTAL: Final = DEFAULT_USAGE.total
DEPLOYMENT_MODEL: Final = "gpt-4o-mini"
PRETOKENIZED_ROW_TOKENS: Final = 30
TPD_WINDOW_SECONDS: Final = 86400
PEER_PICKUP_SECONDS: Final = 30
IMAGE_MODEL: Final = "openai/gpt-image-1"


def unique(label: str) -> str:
    return f"limcap4-{label}-{uuid.uuid4().hex[:10]}"


def _error_reply(status: int) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": f"scripted {status}", "type": "server_error", "code": None}}).encode(),
    )


@dataclass
class FlakyProvider:
    """The limits provider, except that the first ``failures`` POSTs answer ``status`` (500 by default).

    ``failures=-1`` fails every POST. ``attempts`` counts the generation POSTs it received."""

    failures: int = 1
    status: int = 500
    usage: Usage = DEFAULT_USAGE
    attempts: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, request: Request) -> Reply:
        if request.method != "POST":
            return provider(self.usage)(request)
        with self._lock:
            self.attempts += 1
            failing = self.failures < 0 or self.attempts <= self.failures
        return _error_reply(self.status) if failing else provider(self.usage)(request)


def _image_reply() -> Reply:
    return Reply(body=json.dumps({"created": 1, "data": [{"b64_json": "aW1hZ2U=", "revised_prompt": None}]}).encode())


def image_provider() -> Callable[[Request], Reply]:
    """Answers ``POST .../images/generations`` and ``POST .../images/edits`` with one base64 image and no usage;
    every other route is the limits provider."""
    base: Final = provider()

    def respond(request: Request) -> Reply:
        path: Final = request.target.split("?", 1)[0]
        if request.method == "POST" and path.endswith(("/images/generations", "/images/edits")):
            return _image_reply()
        return base(request)

    return respond


def image_posts(wire: Wire) -> int:
    return sum(
        1 for request in wire.drain() if request.method == "POST" and "/images/" in request.target.split("?", 1)[0]
    )


def generate_image(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/images/generations", {"model": model, "prompt": f"limcap4 {uuid.uuid4().hex}"}, key=key
    )


def edit_image(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request_multipart(
        "/v1/images/edits",
        {"model": model, "prompt": f"limcap4 {uuid.uuid4().hex}"},
        {"image": ("probe.png", PNG_BYTES, "image/png")},
        key=key,
    )


@dataclass(frozen=True, slots=True)
class Row:
    """One batch input row. ``tokens`` builds a pretokenized ``/v1/embeddings`` row whose ``input`` is that many token
    ids (each counts one token, ``litellm/batches/batch_utils.py:777``); otherwise a chat row with ``body`` extras."""

    model: str
    url: str = "/v1/chat/completions"
    tokens: int | None = None
    body: Mapping[str, JsonValue] = field(default_factory=dict)


def embedding_row(model: str, tokens: int = PRETOKENIZED_ROW_TOKENS) -> Row:
    return Row(model=model, url="/v1/embeddings", tokens=tokens)


def chat_row(model: str, **body: JsonValue) -> Row:
    return Row(model=model, body=body)


def responses_row(model: str, **body: JsonValue) -> Row:
    return Row(model=model, url="/v1/responses", body=body)


def batch_file(rows: Sequence[Row]) -> bytes:
    """Compact JSONL, one line per row, each line starting with ``{"custom_id"`` as ``BatchProvider`` expects."""

    def line(index: int, row: Row) -> str:
        body: dict[str, JsonValue] = (
            {"model": row.model, "input": list(range(1, row.tokens + 1))}
            if row.tokens is not None
            else {"model": row.model, "messages": [{"role": "user", "content": "limcap4 row"}], **dict(row.body)}
            if row.url == "/v1/chat/completions"
            else {"model": row.model, "input": "limcap4 row", **dict(row.body)}
        )
        return json.dumps(
            {"custom_id": f"r{index}", "method": "POST", "url": row.url, "body": body}, separators=(",", ":")
        )

    return "".join(line(index, row) + "\n" for index, row in enumerate(rows)).encode()


def upload(gateway: Gateway, routing_model: str, key: str, rows: Sequence[Row]) -> str:
    """Upload ``rows`` through ``/v1/files`` routed by ``routing_model`` (the model-routed upload of #51843, which
    rewrites rows naming that alias to the deployment model) and return the encoded file id."""
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "model": routing_model},
        {"file": ("in.jsonl", batch_file(rows), "application/jsonl")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    return string_value(JSON_OBJECT.validate_json(uploaded.content)["id"])


def create_batch(
    gateway: Gateway,
    routing_model: str,
    key: str,
    rows: Sequence[Row],
    *,
    endpoint: str = "/v1/chat/completions",
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """Upload ``rows`` and create a batch from the file with the given extra request headers."""
    file_id: Final = upload(gateway, routing_model, key, rows)
    return gateway.request(
        "POST",
        "/v1/batches",
        {"input_file_id": file_id, "endpoint": endpoint, "completion_window": "24h", "model": routing_model},
        key=key,
        headers=headers,
    )


def batch_creates(wire: Wire) -> int:
    """How many ``POST /batches`` the provider received since the last drain."""
    return sum(
        1
        for request in wire.drain()
        if request.method == "POST" and request.target.split("?", 1)[0].endswith("/batches")
    )


def uploaded_rows(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    """The JSONL rows the provider received in ``POST /files`` uploads since the last drain."""
    found: list[dict[str, JsonValue]] = []  # mutable-ok: collected across every upload body
    for request in wire.drain():
        if request.method != "POST" or not request.target.split("?", 1)[0].endswith("/files"):
            continue
        for raw in request.body.split(b"\n"):
            start = raw.find(b'{"custom_id"')
            if start >= 0:
                found.append(JSON_OBJECT.validate_json(raw[start:].strip()))
    return tuple(found)


def row_model(row: Mapping[str, JsonValue]) -> JsonValue:
    return object_value(row["body"]).get("model")


def counter(scope: str, value: str, kind: str) -> int | None:
    with shared_redis() as redis:
        return redis_integer(redis, counter_key(scope, value, kind))


def counter_settles(scope: str, value: str, kind: str, expected: int | None, seconds: float = 30) -> int | None:
    """Poll ``{scope:value}:kind`` until it holds ``expected``; return what it holds."""
    return eventually(
        lambda: counter(scope, value, kind),
        lambda held: held == expected,
        seconds=seconds,
        return_last_on_timeout=True,
    )


def counter_ttl(scope: str, value: str, kind: str) -> int:
    with shared_redis() as redis:
        return int(redis.ttl(counter_key(scope, value, kind)))


def until_served(instance: Gateway, model: str, key: str, seconds: float = 60) -> httpx.Response:
    """Wait until ``instance`` loaded ``model`` (it may come from another instance's database write)."""
    return eventually(
        lambda: chat(instance, model, key),
        lambda response: response.status_code == 200,
        seconds=seconds,
    )


def end_user_with(gateway: Gateway, scenario: Scenario, **budget: JsonValue) -> str:
    """An end user whose budget carries ``budget`` (``rpm_limit``, ``tpm_limit``, ``tpd_limit``); returns its id."""
    budget_id: Final = scenario.budget(**budget)
    identity: Final = unique("customer")
    gateway.post("/end_user/new", {"user_id": identity, "budget_id": budget_id})
    scenario.cleanups.callback(gateway.post, "/end_user/delete", {"user_ids": [identity]})
    return identity


def model_with_id(scenario: Scenario, api_base: str, **parameters: JsonValue) -> tuple[str, str]:
    """A DB model created through ``POST /model/new`` on ``scenario``'s instance; returns ``(name, model id)``."""
    name: Final = unique("group")
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": f"openai/{DEPLOYMENT_MODEL}",
                "api_key": "limcap4-provider-key",
                "api_base": api_base,
                **parameters,
            },
            "model_info": {},
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return name, identity


def update_model(gateway: Gateway, identity: str, **litellm_params: JsonValue) -> None:
    response: Final = gateway.request("PATCH", f"/model/{identity}/update", {"litellm_params": litellm_params})
    assert response.status_code == 200, response.text


__all__ = (
    "DEPLOYMENT_MODEL",
    "IMAGE_MODEL",
    "PEER_PICKUP_SECONDS",
    "PRETOKENIZED_ROW_TOKENS",
    "PROVIDER_TOTAL",
    "TPD_WINDOW_SECONDS",
    "FlakyProvider",
    "Row",
    "batch_creates",
    "batch_file",
    "chat_row",
    "counter",
    "counter_settles",
    "counter_ttl",
    "create_batch",
    "edit_image",
    "embedding_row",
    "end_user_with",
    "generate_image",
    "image_posts",
    "image_provider",
    "model_with_id",
    "responses_row",
    "row_model",
    "unique",
    "until_served",
    "update_model",
    "upload",
    "uploaded_rows",
)
