"""Shared helpers for the ``test_dec1_cache_*`` response cache contracts.

A ``Probe`` is one proxy model pointed at one scripted upstream scenario: every ``send`` returns the
caller's response together with whether that request reached the upstream, which is the only
black-box proof of a cache hit (a hit never reaches the provider). ``cache_rig`` boots an owned
proxy with another ``litellm_settings.cache`` / ``cache_params`` block than the shared rig, on the
worker's database and either the worker's Redis or an owned one.
"""

from __future__ import annotations

import json
import os
import uuid
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    list_value,
    object_value,
    string_value,
)
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration._support.redis_process import OwnedRedis
from tests.integration._support.upstream import ScenarioHandle
from tests.integration._support.wire import Reply, Request
from tests.integration.caching.response_cache_case import (
    CHAT,
    CHAT_MODEL,
    EMBEDDING,
    PROVIDER_KEY,
    Upstream,
    await_entry,
    chat_body,
    json_response,
    message_body,
    scripted,
)
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse

CACHE_KEY_HEADER: Final = "x-litellm-cache-key"
COST_HEADER: Final = "x-litellm-response-cost"
CALL_ID_HEADER: Final = "x-litellm-call-id"
SPEND_SECONDS: Final = 70
SETTLE_SECONDS: Final = 15
RIG_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
REMOVE_FROM_ENVIRONMENT: Final = ("DATABASE_URL_READ_REPLICA",)
SPEND_SQL: Final = (
    "SELECT request_id, cache_hit, cache_key, spend, call_type, model, prompt_tokens, completion_tokens "
    'FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime", request_id'
)
EMBEDDING_TOKENS_PER_INPUT: Final = 3


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def spend_rows(gateway: Gateway, key: str) -> list[dict[str, JsonValue]]:
    return gateway.rows(SPEND_SQL, (hashed(key),))


def await_spend_rows(gateway: Gateway, key: str, count: int) -> list[dict[str, JsonValue]]:
    """Exactly ``count`` spend rows for ``key``; the writer is asynchronous, so this polls."""
    rows: Final = eventually(lambda: spend_rows(gateway, key), lambda found: len(found) >= count, SPEND_SECONDS)
    assert len(rows) == count, rows
    return rows


def error_message(response: httpx.Response) -> str:
    """The ``error.message`` of a proxy error body, or the raw text when the body has another shape."""
    try:
        payload: Final = JSON_OBJECT.validate_json(response.content)
    except ValueError:
        return response.text
    error: Final = payload.get("error", payload.get("detail", response.text))
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error)


@dataclass(frozen=True, slots=True)
class Exchange:
    response: httpx.Response
    reached_upstream: bool

    def json(self) -> dict[str, JsonValue]:
        assert self.response.status_code == 200, self.response.text
        return JSON_OBJECT.validate_json(self.response.content)

    def id(self) -> str:
        return string_value(self.json()["id"])

    def cache_key(self) -> str | None:
        return self.response.headers.get(CACHE_KEY_HEADER)


@dataclass(frozen=True, slots=True)
class Probe:
    """One proxy model on one scripted upstream scenario, asked through ``gateway``."""

    gateway: Gateway
    handle: ScenarioHandle
    model: str

    def calls(self) -> int:
        return Upstream(self.gateway.upstream_url).calls(self.handle.scenario_id)

    def upstream_bodies(self) -> tuple[dict[str, JsonValue], ...]:
        received: Final = Upstream(self.gateway.upstream_url).received(self.handle.scenario_id)
        return tuple(object_value(item["body"]) for item in received if item.get("method") == "POST")

    def via(self, gateway: Gateway) -> Probe:
        """The same model and scenario asked through another proxy instance."""
        return Probe(gateway, self.handle, self.model)

    def send(
        self,
        path: str,
        body: Mapping[str, JsonValue],
        *,
        key: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Exchange:
        before: Final = self.calls()
        response: Final = self.gateway.request("POST", path, body, key=key, headers=headers)
        return Exchange(response, self.calls() > before)

    def chat(
        self, text: str, *, key: str | None = None, headers: Mapping[str, str] | None = None, **extra: JsonValue
    ) -> Exchange:
        return self.send("/v1/chat/completions", chat_body(self.model, text, **extra), key=key, headers=headers)

    def messages(self, text: str, *, key: str | None = None, **extra: JsonValue) -> Exchange:
        return self.send("/v1/messages", message_body(self.model, text, **extra), key=key)

    def responses(self, text: str, *, key: str | None = None, **extra: JsonValue) -> Exchange:
        return self.send("/v1/responses", {"model": self.model, "input": text, **extra}, key=key)

    def completion(self, text: str, *, key: str | None = None, **extra: JsonValue) -> Exchange:
        return self.send("/v1/completions", {"model": self.model, "prompt": text, **extra}, key=key)


def probe(
    scenario: Scenario,
    response: JsonResponse | SseResponse | None = None,
    *,
    name: str | None = None,
    model_info: Mapping[str, JsonValue] | None = None,
    **parameters: JsonValue,
) -> Probe:
    """Register a scripted scenario and a proxy model on it through ``scenario.gateway``."""
    handle: Final = scripted(scenario, response if response is not None else json_response(CHAT))
    model: Final = scenario.model(name=name, model_info=model_info, api_base=handle.api_base(), **parameters)
    return Probe(scenario.gateway, handle, model)


def filled(store: Redis[bytes], exchange: Exchange) -> str:
    """The Redis key of the entry ``exchange`` wrote; the write is a background task, so this polls."""
    assert exchange.reached_upstream, f"the filling request was itself served from the cache: {exchange.response.text}"
    return await_entry(store, exchange.id())


def hit_after_fill(store: Redis[bytes], ask: Callable[[], Exchange]) -> tuple[Exchange, Exchange]:
    """Ask twice: the first answer fills the entry (awaited in Redis), the second is returned alongside it."""
    first: Final = ask()
    filled(store, first)
    return first, ask()


def assert_hit(first: Exchange, second: Exchange) -> None:
    assert not second.reached_upstream, f"an identical request reached the upstream again: {second.response.text}"
    assert second.id() == first.id(), (first.response.text, second.response.text)
    assert second.cache_key(), dict(second.response.headers)


def assert_miss(first: Exchange, second: Exchange) -> None:
    assert second.reached_upstream, f"the request was served the stored answer: {second.response.text}"
    assert second.id() != first.id(), (first.response.text, second.response.text)


def sse_payloads(response: httpx.Response) -> tuple[dict[str, JsonValue], ...]:
    assert response.status_code == 200, response.text
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )


def sse_event_names(response: httpx.Response) -> tuple[str, ...]:
    return tuple(line.removeprefix("event: ") for line in response.text.splitlines() if line.startswith("event: "))


def chat_stream_text(payloads: Sequence[Mapping[str, JsonValue]]) -> str:
    def pieces() -> Iterator[str]:
        for payload in payloads:
            for choice in list_value(payload.get("choices") or []):
                content = object_value(object_value(choice).get("delta") or {}).get("content")
                if isinstance(content, str):
                    yield content

    return "".join(pieces())


def embedding_vector(text: str) -> list[float]:
    return [zlib.crc32(text.encode()) / 2**32, 0.5]


def embedding_inputs(request: Request) -> list[str]:
    raw: Final = object_value(json.loads(request.body))["input"]
    return [str(item) for item in raw] if isinstance(raw, list) else [str(raw)]


def openai_embedding_peer(request: Request) -> Reply:
    """An OpenAI ``/embeddings`` peer for ``wire_server``: one deterministic vector per input, in input order,
    and ``EMBEDDING_TOKENS_PER_INPUT`` prompt tokens for each."""
    if request.method != "POST" or not request.target.split("?")[0].endswith("/embeddings"):
        return Reply(status=404, body=b'{"error":{"message":"not an embeddings call"}}')
    inputs: Final = embedding_inputs(request)
    tokens: Final = EMBEDDING_TOKENS_PER_INPUT * len(inputs)
    body: Final = {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": index, "embedding": embedding_vector(text)}
            for index, text in enumerate(inputs)
        ],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }
    return Reply(body=json.dumps(body).encode())


def embedding_model(scenario: Scenario, wire_url: str) -> str:
    return scenario.model(model="openai/text-embedding-3-small", api_base=f"{wire_url}/v1")


def unique_embedding() -> tuple[JsonResponse, str]:
    """A scripted ``EMBEDDING`` answer whose vector carries a float no other answer has, and that float as text.

    The proxy restamps the ``model`` of an embedding answer to the public model name the caller asked for,
    while the stored entry keeps the provider's model, so the answer's ``model`` cannot find the entry. The
    stored vector is written back with Python's shortest float repr, which is the marker text unchanged.
    """
    marker: Final = f"0.5{uuid.uuid4().int % 10**12:012d}7"
    body: Final[dict[str, JsonValue]] = {
        **EMBEDDING,
        "data": [{"object": "embedding", "embedding": [0.1, float(marker)], "index": 0}],
    }
    return json_response(body), marker


def vectors(answer: Mapping[str, JsonValue]) -> list[JsonValue]:
    data: Final = [object_value(item) for item in list_value(answer["data"])]
    assert [item["index"] for item in data] == list(range(len(data))), answer
    return [item["embedding"] for item in data]


def redis_params(**extra: JsonValue) -> dict[str, JsonValue]:
    """The shared rig's ``cache_params`` (Redis from ``REDIS_HOST`` / ``REDIS_PORT``) with ``extra`` on top."""
    return {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT", **extra}


def deployment_entry(name: str, handle: ScenarioHandle, litellm_model: str = CHAT_MODEL) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": litellm_model, "api_base": handle.api_base(), "api_key": PROVIDER_KEY},
    }


def cache_config(
    directory: Path,
    *,
    cache: bool | None = True,
    cache_params: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    model_list: Sequence[Mapping[str, JsonValue]] = (),
    extra: Mapping[str, JsonValue] | None = None,
) -> Path:
    """The shared rig's proxy config with its cache block replaced.

    ``cache=None`` leaves ``litellm_settings.cache`` out altogether, ``cache_params=None`` leaves that block out.
    A placeholder deployment keeps the router alive when a test adds its models through ``/model/new``.
    """
    rig: Final = JSON_OBJECT.validate_python(yaml.safe_load(RIG_CONFIG.read_text()))
    kept: Final = {
        name: value
        for name, value in object_value(rig["litellm_settings"]).items()
        if name not in ("cache", "cache_params")
    }
    cache_block: Final[dict[str, JsonValue]] = {
        **({} if cache is None else {"cache": cache}),
        **({} if cache_params is None else {"cache_params": dict(cache_params)}),
    }
    placeholder: Final[dict[str, JsonValue]] = {
        "model_name": f"dec1-cache-placeholder-{uuid.uuid4().hex[:8]}",
        "litellm_params": {"model": CHAT_MODEL, "api_base": "http://127.0.0.1:9/v1", "api_key": PROVIDER_KEY},
    }
    config: Final = {
        **rig,
        **(extra or {}),
        "model_list": [dict(entry) for entry in model_list] or [placeholder],
        "litellm_settings": {**kept, **cache_block, **(litellm_settings or {})},
        "router_settings": {**object_value(rig.get("router_settings") or {}), **(router_settings or {})},
        "general_settings": {**object_value(rig.get("general_settings") or {}), **(general_settings or {})},
    }
    path: Final = directory / f"dec1-cache-config-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@dataclass(frozen=True, slots=True)
class CacheRig:
    gateway: Gateway
    owned: OwnedProxy
    redis: OwnedRedis | None

    def store(self) -> Redis[bytes]:
        """A client for the Redis this proxy's response cache was pointed at."""
        if self.redis is None:
            return Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
        return Redis(host=self.redis.host, port=self.redis.port)

    def get(self, path: str, *, key: str | None = None) -> httpx.Response:
        return self.gateway.request("GET", path, key=key)


@contextmanager
def cache_rig(
    gateway: Gateway,
    directory: Path,
    config: Path,
    *,
    redis: OwnedRedis | None = None,
    workers: int = 1,
    environment: Mapping[str, str] | None = None,
) -> Iterator[CacheRig]:
    """An owned proxy on ``config`` sharing the worker's database; ``redis`` points ``REDIS_HOST`` / ``REDIS_PORT``
    at an owned Redis, otherwise the worker's Redis is used."""
    overrides: Final = {
        "DATABASE_URL": os.environ["DATABASE_URL"],
        **({} if redis is None else {"REDIS_HOST": redis.host, "REDIS_PORT": str(redis.port)}),
        **(environment or {}),
    }
    with owned_proxy_process(
        gateway, directory, overrides, config=config, remove_environment=REMOVE_FROM_ENVIRONMENT, workers=workers
    ) as owned:
        yield CacheRig(owned.gateway, owned, redis)


def cache_keys(store: Redis[bytes], pattern: str = "*") -> frozenset[str]:
    return frozenset(raw.decode() for raw in store.scan_iter(match=pattern, count=1000))


def entry_keys_naming(store: Redis[bytes], needle: str) -> frozenset[str]:
    """Every string key, namespaced or not, whose stored value names ``needle``."""

    def found() -> Iterator[str]:
        for raw_key in store.scan_iter(count=1000):
            if store.type(raw_key) != b"string":
                continue
            value = store.get(raw_key)
            if value is not None and needle.encode() in value:
                yield raw_key.decode()

    return frozenset(found())


def await_named_entry(store: Redis[bytes], needle: str) -> str:
    keys: Final = eventually(lambda: entry_keys_naming(store, needle), lambda found: len(found) > 0, SETTLE_SECONDS)
    assert len(keys) == 1, keys
    return next(iter(keys))


def stored_entry(store: Redis[bytes], key: str) -> dict[str, JsonValue]:
    raw: Final = store.get(key)
    assert raw is not None, key
    return JSON_OBJECT.validate_json(raw)
