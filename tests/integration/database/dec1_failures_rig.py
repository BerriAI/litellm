"""Shared rig for the `test_dec1_failures_*` files: one owned proxy whose Postgres and Redis sit behind
switchable TCP relays, plus one healthy scripted provider.

The relay (`FaultRelay`) is a plain byte forwarder with four switches, so a test can take a dependency away
without signalling any process and without losing the dependency's own state:

`refuse()` drops every live connection with a TCP reset and resets every new one: the dependency is down.
`stall()` keeps connections open but forwards no bytes until `open()`: the dependency hangs.
`slow(seconds)` forwards every chunk after that many seconds: the dependency is slow but answers.
`restart()` resets every live connection once and keeps accepting new ones: the dependency restarted.

The proxy under test only ever sees the relay's port. Tests read Postgres and Redis directly, around the
relay, to observe rows and keys while the proxy cannot.

The provider answers 200 on chat completions, embeddings, the Responses API and Anthropic messages, records
every request body, and can hold its answers behind a gate so a request stays in flight.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.async_server import LoopbackServer, close_writer
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
from tests.integration._support.database_relay import plaintext_database_url
from tests.integration._support.openai_wire import chat_reply, responses_reply
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import OwnedProxy, WorkerHealthcheck, owned_proxy_process
from tests.integration._support.redis_process import OwnedRedis, owned_redis
from tests.integration._support.wire import Reply, Request, Wire, wire_server

Mode = Literal["open", "refuse", "stall"]

CHAT: Final = "dec1-failures-chat"
EMBED: Final = "dec1-failures-embed"
MESSAGES: Final = "dec1-failures-messages"
PROVIDER_KEY: Final = "dec1-failures-provider-key"
REPLY_TEXT: Final = "dec1 failures reply"
HELD_SECONDS: Final = 60
LANDING_SECONDS: Final = 90
RECOVERY_SECONDS: Final = 90
DB_UNAVAILABLE_MESSAGE: Final = (
    "Service Unavailable, the authentication database is temporarily unreachable. Please retry shortly."
)
DB_UNAVAILABLE_ERROR: Final[Mapping[str, JsonValue]] = {
    "message": DB_UNAVAILABLE_MESSAGE,
    "type": "no_db_connection",
    "param": None,
    "code": "503",
}
READY_BODY: Final[Mapping[str, JsonValue]] = {"status": "healthy", "db": "connected"}
DISCONNECTED_BODY: Final[Mapping[str, JsonValue]] = {"status": "healthy", "db": "disconnected"}
_POLL_SECONDS: Final = 0.02
_SETTLE_PROBE_SECONDS: Final = 1.0
_SETTLE_STREAK: Final = 5


class FaultRelay:
    """A TCP forwarder in front of one dependency whose behaviour a test switches while the proxy runs."""

    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        self.port: Final = allocate_port()
        self._upstream: Final = (upstream_host, upstream_port)
        self._server: Final = LoopbackServer(self.port, self._serve)
        self._mode: Mode = "open"
        self._delay = 0.0
        self._writers: set[asyncio.StreamWriter] = set()  # mutable-ok: touched on the relay loop only
        self.accepted = 0
        self.refused = 0

    def start(self) -> None:
        self._server.start()

    def stop(self) -> None:
        self._mode = "open"
        self._server.stop()

    def open(self) -> None:
        """Forward normally again. Bytes held by `stall` are released in order."""
        self._delay = 0.0
        self._mode = "open"

    def refuse(self) -> None:
        """The dependency is down: live connections are reset and new ones are reset on accept."""
        self._mode = "refuse"
        self._reset_live_connections()

    def stall(self) -> None:
        """The dependency hangs: connections stay open and accept writes, nothing is forwarded either way."""
        self._mode = "stall"

    def slow(self, seconds: float) -> None:
        """The dependency is slow: every forwarded chunk waits `seconds` first."""
        self._delay = seconds
        self._mode = "open"

    def restart(self) -> None:
        """The dependency restarted: every live connection is reset once, new connections work at once."""
        self._reset_live_connections()

    def live_connections(self) -> int:
        return len(self._writers) // 2

    def _reset_live_connections(self) -> None:
        asyncio.run_coroutine_threadsafe(self._reset(), self._server.loop).result(timeout=10)

    async def _reset(self) -> None:
        for writer in tuple(self._writers):
            transport: Final = writer.transport
            if transport is not None:
                transport.abort()
        self._writers.clear()

    async def _serve(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        if self._mode == "refuse":
            self.refused += 1
            client_writer.transport.abort()
            return
        try:
            server_reader, server_writer = await asyncio.open_connection(*self._upstream)
        except OSError:
            client_writer.transport.abort()
            return
        self.accepted += 1
        pair: Final = (client_writer, server_writer)
        self._writers.update(pair)

        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                while chunk := await reader.read(65536):
                    while self._mode == "stall":
                        await asyncio.sleep(_POLL_SECONDS)
                    if self._delay:
                        await asyncio.sleep(self._delay)
                    writer.write(chunk)
                    await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError, OSError):
                return
            finally:
                await close_writer(writer)

        try:
            await asyncio.gather(forward(client_reader, server_writer), forward(server_reader, client_writer))
        finally:
            self._writers.difference_update(pair)


@contextmanager
def fault_relay(host: str, port: int) -> Iterator[FaultRelay]:
    relay: Final = FaultRelay(host, port)
    try:
        relay.start()
        yield relay
    finally:
        relay.stop()


def relayed_database_url(database_url: str, port: int) -> str:
    parts: Final = urlsplit(database_url)
    credentials: Final = f"{parts.username}:{parts.password}@" if parts.username else ""
    return plaintext_database_url(urlunsplit(parts._replace(netloc=f"{credentials}127.0.0.1:{port}")))


def unreachable_database_url(database_url: str) -> str:
    """The same database URL on a port nothing listens on, so every connection attempt is refused."""
    return relayed_database_url(database_url, allocate_port())


@dataclass(frozen=True, slots=True)
class Provider:
    """The healthy scripted provider: answers 200 and remembers every POST body it received."""

    wire: Wire
    gate: threading.Event
    log: list[Mapping[str, JsonValue]] = field(default_factory=list)  # mutable-ok: appended by handler threads

    @property
    def url(self) -> str:
        return self.wire.url

    def hold(self) -> None:
        """Keep every following answer back until `release`, so requests stay in flight."""
        self.gate.clear()

    def release(self) -> None:
        self.gate.set()

    def count(self, marker: str) -> int:
        """How many requests the provider received whose body mentions `marker`."""
        return sum(1 for body in tuple(self.log) if marker in json.dumps(body))


def _embedding(model: str) -> Reply:
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


def _anthropic_message(model: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": REPLY_TEXT}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }
        ).encode()
    )


@contextmanager
def healthy_provider() -> Iterator[Provider]:
    gate: Final = threading.Event()
    gate.set()
    log: Final[list[Mapping[str, JsonValue]]] = []

    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        body: Final = JSON_OBJECT.validate_json(request.body)
        log.append(body)
        model: Final = str(body.get("model", ""))
        assert gate.wait(timeout=HELD_SECONDS), "the provider gate was never released"
        if request.target.endswith("/embeddings"):
            return _embedding(model)
        if request.target.endswith("/messages"):
            return _anthropic_message(model)
        if request.target.endswith("/responses"):
            return responses_reply(f"resp_{uuid.uuid4().hex}", model, REPLY_TEXT, stream=body.get("stream") is True)
        return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", model, REPLY_TEXT, stream=body.get("stream") is True)

    with wire_server(respond, teardown_grace=HELD_SECONDS) as wire:
        try:
            yield Provider(wire, gate, log)
        finally:
            gate.set()


def deployment(name: str, model: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_key": PROVIDER_KEY, "api_base": api_base, **params},
        "model_info": {"id": f"dec1-failures-{name}"},
    }


def failures_config(
    directory: Path,
    provider_url: str,
    *,
    with_redis: bool,
    model_list: Sequence[Mapping[str, JsonValue]] = (),
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    """A complete proxy config: three healthy deployments, no retries, no cooldowns, one second spend flush.

    With `with_redis` the response cache and the Redis auth cache are on, as in the shared harness config."""
    cache: Final[dict[str, JsonValue]] = (
        {
            "enable_redis_auth_cache": True,
            "cache": True,
            "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
        }
        if with_redis
        else {}
    )
    config: Final = {
        "model_list": [
            deployment(CHAT, "openai/gpt-4o-mini", f"{provider_url}/v1"),
            deployment(EMBED, "openai/text-embedding-3-small", f"{provider_url}/v1"),
            deployment(MESSAGES, "anthropic/claude-haiku-4-5", provider_url),
            *(dict(entry) for entry in model_list),
        ],
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "database_url": "os.environ/DATABASE_URL",
            "disable_model_info_refresh": True,
            "proxy_batch_write_at": 1,
            "proxy_batch_polling_interval": 1,
            **(general_settings or {}),
        },
        "litellm_settings": {**cache, **(litellm_settings or {})},
        "router_settings": {"num_retries": 0, "disable_cooldowns": True, **(router_settings or {})},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / f"dec1-failures-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    owned: OwnedProxy
    provider: Provider
    database: FaultRelay
    redis: FaultRelay | None
    cache: OwnedRedis | None

    def redis_client(self) -> Redis:
        """A client on the real Redis, around the relay, so a test can read keys while the proxy cannot."""
        assert self.cache is not None, "this rig runs without Redis"
        return Redis(host=self.cache.host, port=self.cache.port, socket_timeout=2, decode_responses=True)


ModelList = Callable[[str], Sequence[Mapping[str, JsonValue]]]


@contextmanager
def failures_rig(
    directory: Path,
    *,
    with_redis: bool = True,
    model_list: ModelList | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
    healthcheck: WorkerHealthcheck = WorkerHealthcheck.STRICT,
    client_timeout: float = 30,
    extra_arguments: tuple[str, ...] = (),
    before_boot: Callable[[FaultRelay, FaultRelay | None], None] | None = None,
) -> Iterator[Rig]:
    """One owned single-worker proxy on a scratch database and its own Redis, each reached through a relay.

    `model_list` receives the provider URL and returns extra deployments. `before_boot` receives the database
    and Redis relays before the proxy starts, so a dependency can already be down at boot. The relays are
    reopened before the proxy stops, so teardown never runs against a dependency a test left down."""
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        provider: Final = stack.enter_context(healthy_provider())
        database_url: Final = stack.enter_context(scratch_database("dec1_failures"))
        parts: Final = urlsplit(database_url)
        assert parts.hostname is not None and parts.port is not None, database_url
        database: Final = stack.enter_context(fault_relay(parts.hostname, parts.port))
        cache: Final = stack.enter_context(owned_redis(directory / "redis")) if with_redis else None
        redis: Final = stack.enter_context(fault_relay(cache.host, cache.port)) if cache is not None else None
        config: Final = failures_config(
            directory,
            provider.url,
            with_redis=with_redis,
            model_list=model_list(provider.url) if model_list is not None else (),
            general_settings=general_settings,
            litellm_settings=litellm_settings,
            router_settings=router_settings,
        )
        overrides: Final = {
            "DATABASE_URL": relayed_database_url(database_url, database.port),
            **({"REDIS_HOST": "127.0.0.1", "REDIS_PORT": str(redis.port)} if redis is not None else {}),
            **(environment or {}),
        }
        if before_boot is not None:
            before_boot(database, redis)
        removed: Final = (
            "DATABASE_URL_READ_REPLICA",
            *(() if with_redis else ("REDIS_HOST", "REDIS_PORT", "REDIS_URL")),
        )
        owned: Final = stack.enter_context(
            owned_proxy_process(
                shared,
                directory / "proxy",
                overrides,
                config=config,
                remove_environment=removed,
                healthcheck=healthcheck,
                client_timeout=client_timeout,
                extra_arguments=extra_arguments,
            )
        )
        stack.callback(provider.release)
        stack.callback(database.open)
        if redis is not None:
            stack.callback(redis.open)
        proxy: Final = dataclasses.replace(owned.gateway, database_url=database_url)
        yield Rig(proxy, owned, provider, database, redis, cache)


def marker() -> str:
    return f"dec1-failures-{uuid.uuid4().hex}"


def digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def chat(
    proxy: Gateway, model: str = CHAT, *, key: str | None = None, text: str | None = None, **extra: JsonValue
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": text or marker()}],
        **extra,
    }
    return proxy.request("POST", "/v1/chat/completions", body, key=key)


def embed(proxy: Gateway, model: str = EMBED, *, key: str | None = None, text: str | None = None) -> httpx.Response:
    return proxy.request("POST", "/v1/embeddings", {"model": model, "input": text or marker()}, key=key)


def anthropic_message(
    proxy: Gateway, model: str = MESSAGES, *, key: str | None = None, text: str | None = None
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": text or marker()}],
    }
    return proxy.request("POST", "/v1/messages", body, key=key, headers={"anthropic-version": "2023-06-01"})


def stream_chat(
    proxy: Gateway, model: str = CHAT, *, key: str | None = None, text: str | None = None
) -> tuple[int, str]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": text or marker()}],
    }
    with proxy.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key or proxy.key}"}
    ) as response:
        return response.status_code, response.read().decode()


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def content_of(response: httpx.Response) -> str:
    choices: Final = list_value(JSON_OBJECT.validate_json(response.content)["choices"])
    return string_value(object_value(object_value(choices[0])["message"])["content"])


def readiness(proxy: Gateway) -> tuple[int, dict[str, JsonValue]]:
    """`GET /health/readiness` with no credential, as an orchestrator probe sends it."""
    response: Final = proxy.client.get("/health/readiness")
    return response.status_code, JSON_OBJECT.validate_json(response.content)


def await_connected(proxy: Gateway, seconds: float = RECOVERY_SECONDS) -> None:
    """Wait until the proxy itself reports its database connected again, with no proxy restart."""
    eventually(lambda: readiness(proxy), lambda seen: seen == (200, dict(READY_BODY)), seconds=seconds)


def await_serving(proxy: Gateway, seconds: float = RECOVERY_SECONDS) -> None:
    """Wait until database-backed admin reads answer 200 several times in a row.

    Readiness can report `connected` while a reconnect cycle the outage started is still replacing the Prisma
    client, and queries in that window fail with `Client is not connected to the query engine`. A test that
    creates or deletes keys right after an outage needs the client settled, not just one good probe."""
    streak: Final[list[int]] = [0]  # mutable-ok: the run of good probes, read and reset by probe()

    def probe() -> int:
        response: Final = proxy.request("GET", "/key/list", params={"page": "1", "size": "1"})
        streak[0] = streak[0] + 1 if response.status_code == 200 else 0
        time.sleep(_SETTLE_PROBE_SECONDS)
        return streak[0]

    eventually(probe, lambda seen: seen >= _SETTLE_STREAK, seconds=seconds)


@contextmanager
def database_down(rig: Rig) -> Iterator[None]:
    """Postgres is unreachable inside the block. Afterwards the relay is open, the proxy reports itself
    connected, and database-backed admin calls are answered again."""
    rig.database.refuse()
    try:
        yield
    finally:
        rig.database.open()
        await_connected(rig.proxy)
        await_serving(rig.proxy)


@contextmanager
def redis_down(rig: Rig) -> Iterator[None]:
    """Redis is unreachable inside the block and reachable again, with all its keys, afterwards."""
    assert rig.redis is not None, "this rig runs without Redis"
    rig.redis.refuse()
    try:
        yield
    finally:
        rig.redis.open()


def spend_rows(proxy: Gateway, key: str, count: int) -> list[dict[str, JsonValue]]:
    """The key's `LiteLLM_SpendLogs` rows once exactly `count` have landed, read around the relay."""
    return eventually(
        lambda: proxy.rows(
            'SELECT request_id, spend, status, model_group, call_type, metadata FROM "LiteLLM_SpendLogs" '
            'WHERE api_key = %s ORDER BY "startTime"',
            (digest(key),),
        ),
        lambda rows: len(rows) >= count,
        seconds=LANDING_SECONDS,
    )


def key_spend(proxy: Gateway, key: str) -> float:
    rows: Final = proxy.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest(key),))
    assert len(rows) == 1, rows
    spend: Final = rows[0]["spend"]
    assert isinstance(spend, (int, float, str)) and not isinstance(spend, bool), spend
    return float(spend)


def number(value: JsonValue) -> float:
    assert isinstance(value, (int, float, str)) and not isinstance(value, bool), value
    return float(value)


def served_ids(responses: Sequence[httpx.Response]) -> tuple[str, ...]:
    """The `id` of every response, after asserting each one is a 200."""
    for response in responses:
        assert response.status_code == 200, f"{response.status_code} {response.text}"
    return tuple(string_value(JSON_OBJECT.validate_json(response.content)["id"]) for response in responses)


def statuses(responses: Sequence[httpx.Response]) -> tuple[int, ...]:
    return tuple(response.status_code for response in responses)
