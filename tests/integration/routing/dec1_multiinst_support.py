"""Shared rig for the `test_dec1_multiinst_*` files: two proxy instances on one database and one Redis.

A `Fleet` is what a production deployment with several pods looks like from outside: two owned proxies booted
from the same generated config, on one scratch Postgres database and one owned Redis that the test can read,
write, stop, restart and pause. Every deployment points at the in-process scripted upstream of
`dec1_routing_support`, which answers `served <upstream model>` and counts the requests each upstream model
received, so "which instance served it" and "did it reach the provider" are both observable.

The generated config starts from `tests/integration/proxy_config.yaml` (so `store_model_in_db: true` and the
Redis response cache are on, with `REDIS_HOST` / `REDIS_PORT` pointing at the owned Redis) and replaces
`model_list`, `router_settings` and, when given, parts of `litellm_settings` and `general_settings`.
`redis=False` drops the response cache block and every Redis variable, which gives two instances that share
only the database.

`PROXY_CONFIG_RELOAD_INTERVAL_SECONDS` defaults to 3600 here, so a peer that picks up a database change within
seconds can only have been told through Redis pub/sub, never by the periodic reload.
"""

from __future__ import annotations

import ast
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx
import psycopg
import yaml
from psycopg.rows import dict_row
from pydantic import JsonValue, TypeAdapter
from redis import Redis

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from tests.integration._support.database import scratch_database
from tests.integration._support.process import owned_proxy
from tests.integration._support.redis_process import OwnedRedis, owned_redis
from tests.integration.routing.dec1_routing_support import PROVIDER_KEY, Upstream, scripted_upstream

BASE_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
CONFIG_SYNC_CHANNEL: Final = "litellm_proxy.config_change"
MODEL_TABLE: Final = "litellm_proxymodeltable"
HEALTH_KEY: Final = "litellm:health_check:deployment_health_state"
CACHE_KEY_HEADER: Final = "x-litellm-cache-key"
LONG_RELOAD_SECONDS: Final = 3600
PEER_PICKUP_SECONDS: Final = 20
REMOVED_ENVIRONMENT: Final = ("DATABASE_URL_READ_REPLICA",)
SUBSCRIBER_COUNTS: Final = TypeAdapter(tuple[tuple[str, int], ...])


@dataclass(frozen=True, slots=True)
class Fleet:
    first: Gateway
    second: Gateway
    upstream: Upstream
    redis: OwnedRedis | None
    database_url: str

    @property
    def instances(self) -> tuple[Gateway, Gateway]:
        return (self.first, self.second)

    def store(self) -> Redis:
        """A fresh client on the fleet's Redis; the caller closes it with `with`."""
        assert self.redis is not None, "this fleet runs without Redis"
        return Redis(host=self.redis.host, port=self.redis.port, socket_timeout=2, socket_connect_timeout=2)

    def raw(self, key: str) -> bytes | None:
        with self.store() as client:
            value: Final = client.get(key)
        return None if value is None else bytes(value)

    def ttl(self, key: str) -> int:
        with self.store() as client:
            return int(client.ttl(key))

    def keys(self, pattern: str) -> frozenset[str]:
        with self.store() as client:
            return frozenset(raw.decode() for raw in client.scan_iter(match=pattern, count=1000))

    def subscribers(self, channel: str = CONFIG_SYNC_CHANNEL) -> int:
        with self.store() as client:
            return SUBSCRIBER_COUNTS.validate_python(client.pubsub_numsub(channel))[0][1]

    def rows(self, query: str, parameters: tuple[object, ...] = ()) -> list[dict[str, object]]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            return list(connection.execute(query, parameters).fetchall())


def decoded(raw: bytes | None) -> JsonValue:
    """A value the Python stack wrote either as `json.dumps(value)` or as `str(value)` (a Python literal)."""
    assert raw is not None
    text: Final = raw.decode()
    try:
        return JSON_OBJECT.validate_json(text) if text.startswith("{") and '"' in text else json.loads(text)
    except ValueError:
        return TypeAdapter(JsonValue).validate_python(ast.literal_eval(text))


def unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def utc_minute() -> str:
    """The `HH-MM` window usage counters are keyed by (`get_utc_datetime().strftime("%H-%M")`)."""
    return datetime.now(UTC).strftime("%H-%M")


def fleet_config(
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]],
    router_settings: Mapping[str, JsonValue],
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    cache_params: Mapping[str, JsonValue] | None = None,
    redis: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    base: Final = yaml.safe_load(BASE_CONFIG.read_text())
    assert isinstance(base, dict)
    settings: Final = dict(base.get("litellm_settings", {}))
    if redis:
        settings["cache_params"] = {**settings.get("cache_params", {}), **(cache_params or {})}
    else:
        for name in ("cache", "cache_params", "enable_redis_auth_cache"):
            settings.pop(name, None)
    config: Final = {
        **base,
        "model_list": [dict(entry) for entry in model_list],
        "router_settings": dict(router_settings),
        "general_settings": {**base.get("general_settings", {}), **(general_settings or {})},
        "litellm_settings": {**settings, **(litellm_settings or {})},
    }
    path: Final = directory / f"dec1-multiinst-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def fleet(
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]] | None = None,
    models: Callable[[str], Sequence[Mapping[str, JsonValue]]] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    second_general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    cache_params: Mapping[str, JsonValue] | None = None,
    second_cache_params: Mapping[str, JsonValue] | None = None,
    redis: bool = True,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Fleet]:
    """Two instances of one config on one scratch database and, unless `redis=False`, one owned Redis.

    `models`, when given, is called with the upstream's `api_base` to build the model list. The `second_*`
    arguments give the second instance a different value for that block, for contracts about two pods that
    disagree (one runs background health checks, or uses another cache namespace)."""
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        upstream: Final = stack.enter_context(scripted_upstream())
        database: Final = stack.enter_context(scratch_database("dec1_multiinst"))
        cache: Final = stack.enter_context(owned_redis(directory / "redis")) if redis else None
        entries: Final = tuple(models(upstream.api_base)) if models is not None else tuple(model_list or ())
        overrides: Final[dict[str, str]] = {
            "DATABASE_URL": database,
            "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": str(LONG_RELOAD_SECONDS),
            **({"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)} if cache is not None else {}),
            **(environment or {}),
        }
        removed: Final = REMOVED_ENVIRONMENT + (() if cache is not None else ("REDIS_HOST", "REDIS_PORT"))
        gateways: list[Gateway] = []  # mutable-ok: filled once per instance below
        for name, general, cache_block in (
            ("first", general_settings, cache_params),
            (
                "second",
                second_general_settings if second_general_settings is not None else general_settings,
                second_cache_params if second_cache_params is not None else cache_params,
            ),
        ):
            config = fleet_config(
                directory / name,
                model_list=entries,
                router_settings=router_settings or {"num_retries": 0},
                general_settings=general,
                litellm_settings=litellm_settings,
                cache_params=cache_block,
                redis=cache is not None,
            )
            gateways.append(
                stack.enter_context(
                    owned_proxy(shared, directory / name, overrides, config=config, remove_environment=removed)
                )
            )
        built: Final = Fleet(gateways[0], gateways[1], upstream, cache, database)
        if cache is not None:
            channels: Final = frozenset(
                sync_channel(block)
                for block in (cache_params, second_cache_params if second_cache_params is not None else cache_params)
            )
            eventually(
                lambda: sum(built.subscribers(channel) for channel in channels), lambda count: count >= 2, seconds=30
            )
        yield built


def sync_channel(cache_params: Mapping[str, JsonValue] | None) -> str:
    """The config sync channel an instance subscribes to: prefixed `<namespace>:` when its cache has a namespace.

    The subscriber listens on the coordination Redis, which is the response cache's Redis with its namespace
    (`config_sync_channel`, litellm/proxy/common_utils/config_sync_pubsub.py:93)."""
    namespace: Final = (cache_params or {}).get("namespace")
    return CONFIG_SYNC_CHANNEL if not isinstance(namespace, str) else f"{namespace}:{CONFIG_SYNC_CHANNEL}"


def ask(
    gateway: Gateway,
    model: str,
    text: str,
    *,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """A chat request with a caller-chosen text, so two calls can be identical (a response cache hit)."""
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": text}], **extra}
    return gateway.request("POST", "/v1/chat/completions", body, headers=headers)


def served(response: httpx.Response) -> str:
    """The upstream model that produced a 200, from the scripted upstream's `served <model>` answer."""
    assert response.status_code == 200, response.text
    choices: Final = list_value(JSON_OBJECT.validate_json(response.content)["choices"])
    content: Final = string_value(object_value(object_value(choices[0])["message"])["content"])
    return content.removeprefix("served ")


def add_model(gateway: Gateway, group: str, upstream_model: str, api_base: str, **params: JsonValue) -> str:
    """`POST /model/new` with a chosen `model_info.id`, which is the deployment id it is stored and routed under."""
    model_id: Final = unique("dec1-multiinst-model")
    gateway.post(
        "/model/new",
        {
            "model_name": group,
            "litellm_params": {
                "model": f"openai/{upstream_model}",
                "api_key": PROVIDER_KEY,
                "api_base": api_base,
                **params,
            },
            "model_info": {"id": model_id},
        },
    )
    return model_id


def listed_models(gateway: Gateway) -> frozenset[str]:
    return frozenset(string_value(object_value(entry)["id"]) for entry in list_value(gateway.get("/v1/models")["data"]))


def router_values(gateway: Gateway) -> dict[str, JsonValue]:
    return object_value(gateway.get("/router/settings")["current_values"])


@contextmanager
def redis_stopped(fleet_value: Fleet) -> Iterator[None]:
    """Stops the owned Redis and starts it again on the same port when the block ends."""
    assert fleet_value.redis is not None
    fleet_value.redis.stop()
    try:
        yield
    finally:
        fleet_value.redis.start()
