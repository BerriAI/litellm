import itertools
import json
import os
import signal
import threading
import uuid
from bisect import bisect_left
from collections.abc import Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import httpcore
import httpx
import jwt
import psutil
import psycopg
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows, scratch_database, write_rows
from integration._support.database_relay import database_relay
from integration._support.process import OwnedProxy, group_members, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from jwt.algorithms import RSAAlgorithm
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.constants import SPEND_LOG_KEY_METADATA_MISS_CACHE_TTL

MODEL: Final = "key-metadata-recovery-audit"
LOOKUP_MARKER: Final = "first_alias"
FAILED_LOOKUP_FLOOR: Final = timedelta(seconds=4)
MISS_TTL_BOUND: Final = 60
MISS_WINDOW: Final = timedelta(seconds=SPEND_LOG_KEY_METADATA_MISS_CACHE_TTL)
WORKERS: Final = 2
USAGE: Final = MappingProxyType({"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40})
REPLY_TEXT: Final = "You spent a little this week."
WAITING_LOOKUPS: Final = (
    "SELECT pid, query_start::text AS started FROM pg_stat_activity "
    "WHERE datname = current_database() AND pid <> pg_backend_pid() AND state = 'active' "
    "AND wait_event_type = 'Lock' AND position(%s in query) > 0"
)
JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
WAITING_ROWS: Final[TypeAdapter[tuple[tuple[int, str], ...]]] = TypeAdapter(tuple[tuple[int, str], ...])
SOCKET_ADDRESS: Final[TypeAdapter[tuple[str, int]]] = TypeAdapter(tuple[str, int])
NO_SETTINGS: Final[Mapping[str, JsonValue]] = MappingProxyType({})
NO_ENVIRONMENT: Final[Mapping[str, str]] = MappingProxyType({})
JWT_SETTINGS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"enable_jwt_auth": True, "litellm_jwtauth": {"user_id_jwt_field": "sub", "user_id_upsert": True}}
)
JWKS_KEY_ID: Final = "key-metadata-recovery-audit"
SPEND_TABLES: Final = (
    "LiteLLM_SpendLogs",
    "LiteLLM_DailyUserSpend",
    "LiteLLM_DailyTeamSpend",
    "LiteLLM_DailyOrganizationSpend",
    "LiteLLM_DailyEndUserSpend",
    "LiteLLM_DailyTagSpend",
    "LiteLLM_DailyAgentSpend",
)
LANDED_KEYS: Final = " UNION ".join(
    f"SELECT DISTINCT '{table}' AS source, api_key FROM \"{table}\"" for table in SPEND_TABLES
)
TENANT_KEYS: Final = ("chat", "stream", "messages", "responses", "live", "deleted")
KEYS_ONLY_IN_SPEND_LOGS: Final = frozenset(("chat", "stream", "messages", "responses"))
KEYS_A_SEARCH_FINDS_BY_ALIAS: Final = frozenset(("live", "deleted"))
AGGREGATED: Final = "/user/daily/activity/aggregated"
BURST: Final = 16


class _KeyMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    key_alias: str | None = None
    user_id: str | None = None
    user_email: str | None = None


class _KeyBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)
    metadata: _KeyMetadata


class _Breakdown(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_keys: Mapping[str, _KeyBreakdown]


class _Day(BaseModel):
    model_config = ConfigDict(frozen=True)
    breakdown: _Breakdown


class _Activity(BaseModel):
    model_config = ConfigDict(frozen=True)
    results: tuple[_Day, ...]


class _UpstreamMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: str | None = None


class _UpstreamRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    stream: bool | None = None
    tools: tuple[JsonValue, ...] | None = None
    messages: tuple[_UpstreamMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class Spender:
    alias: str
    user_id: str
    user_email: str
    digest: str


@dataclass(frozen=True, slots=True)
class Pinned:
    client: httpx.Client
    port: int

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, JsonValue] | None = None,
    ) -> httpx.Response:
        response: Final = self.client.request(method, path, params=params, json=body)
        assert _local_port(response) == self.port, "Pinned connection moved to another worker socket"
        return response


@dataclass(frozen=True, slots=True)
class Probe:
    ran_lookup: bool
    aliases: tuple[str | None, ...]


def _day(offset: int) -> str:
    return (datetime.now(UTC) + timedelta(days=offset)).date().isoformat()


def _completion(message: Mapping[str, JsonValue], finish_reason: str) -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": dict(message), "finish_reason": finish_reason}],
            "usage": dict(USAGE),
        }
    ).encode()


def _stream_frames() -> tuple[bytes, ...]:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    frames: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": dict(USAGE)},
    )
    envelope: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return (*(f"data: {json.dumps({**envelope, **frame})}\n\n".encode() for frame in frames), b"data: [DONE]\n\n")


def _usage_tool_call() -> Mapping[str, JsonValue]:
    arguments: Final = json.dumps({"start_date": _day(-1), "end_date": _day(1)})
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_usage", "type": "function", "function": {"name": "get_usage_data", "arguments": arguments}}
        ],
    }


def _response_object() -> bytes:
    return json.dumps(
        {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "status": "completed",
            "created_at": 1,
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": REPLY_TEXT, "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": USAGE["prompt_tokens"],
                "output_tokens": USAGE["completion_tokens"],
                "total_tokens": USAGE["total_tokens"],
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    ).encode()


def _respond(request: Request) -> Reply:
    if request.target.endswith("/responses"):
        return Reply(body=_response_object())
    body: Final = _UpstreamRequest.model_validate_json(request.body or b"{}")
    if body.stream:
        return Reply(content_type="text/event-stream", chunks=_stream_frames())
    roles: Final = frozenset(message.role for message in body.messages)
    if body.tools and "tool" not in roles:
        return Reply(body=_completion(_usage_tool_call(), "tool_calls"))
    return Reply(body=_completion({"role": "assistant", "content": REPLY_TEXT}, "stop"))


def _config(wire_url: str, general_settings: Mapping[str, JsonValue]) -> str:
    return json.dumps(
        {
            "model_list": [
                {
                    "model_name": MODEL,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": f"{wire_url}/v1",
                        "api_key": "sk-upstream",
                        "input_cost_per_token": 0.001,
                        "output_cost_per_token": 0.002,
                    },
                }
            ],
            "general_settings": {
                "master_key": "os.environ/LITELLM_MASTER_KEY",
                "database_url": "os.environ/DATABASE_URL",
                "store_model_in_db": True,
                "disable_spend_logs": False,
                "proxy_batch_write_at": 1,
                "proxy_batch_polling_interval": 1,
                **general_settings,
            },
            "router_settings": {"disable_cooldowns": True},
        }
    )


@contextmanager
def _owned_proxy(
    gateway: Gateway,
    directory: Path,
    database_url: str,
    wire_url: str,
    *,
    general_settings: Mapping[str, JsonValue] = NO_SETTINGS,
    environment: Mapping[str, str] = NO_ENVIRONMENT,
) -> Generator[OwnedProxy]:
    config: Final = directory / "key_metadata_recovery.yaml"
    config.write_text(_config(wire_url, general_settings))
    with owned_proxy_process(
        gateway,
        directory,
        {
            "DATABASE_URL": database_url,
            "KEEPALIVE_TIMEOUT": "600",
            "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
            "OPENAI_API_KEY": "sk-upstream",
            "OPENAI_BASE_URL": f"{wire_url}/v1",
            "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
            **environment,
        },
        config=config,
        remove_environment=("DATABASE_URL_READ_REPLICA",),
        workers=WORKERS,
    ) as owned:
        yield owned


@contextmanager
def _proxy(
    gateway: Gateway,
    directory: Path,
    database_url: str,
    wire_url: str,
    *,
    general_settings: Mapping[str, JsonValue] = NO_SETTINGS,
    environment: Mapping[str, str] = NO_ENVIRONMENT,
) -> Generator[Gateway]:
    with _owned_proxy(
        gateway, directory, database_url, wire_url, general_settings=general_settings, environment=environment
    ) as owned:
        yield owned.gateway


def _local_port(response: httpx.Response) -> int:
    match response.extensions:
        case {"network_stream": httpcore.NetworkStream() as stream}:
            return SOCKET_ADDRESS.validate_python(stream.get_extra_info("client_addr"))[1]
        case _:
            raise AssertionError(f"No network stream on {response.request.url}")


@contextmanager
def _pinned(proxy: Gateway) -> Generator[Pinned]:
    with httpx.Client(
        base_url=proxy.client.base_url,
        headers={"Authorization": f"Bearer {proxy.key}"},
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=600),
        timeout=60,
        trust_env=False,
    ) as client:
        opened: Final = client.get("/health/liveliness")
        assert opened.status_code == 200, opened.text
        yield Pinned(client, _local_port(opened))


def _landed(database_url: str, digest: str) -> bool:
    daily: Final = read_rows(
        'SELECT 1 FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s', (digest,), database_url=database_url
    )
    logged: Final = read_rows(
        'SELECT 1 FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,), database_url=database_url
    )
    return bool(daily) and bool(logged)


def _spender(proxy: Gateway, database_url: str, label: str) -> Spender:
    user_id: Final = f"{label}-{uuid.uuid4().hex[:8]}"
    user_email: Final = f"{user_id}@example.com"
    alias: Final = f"{label}-laptop-key"
    proxy.post("/user/new", {"user_id": user_id, "user_email": user_email, "auto_create_key": False})
    key: Final = string_value(
        proxy.post("/key/generate", {"user_id": user_id, "key_alias": alias, "models": [MODEL]})["key"]
    )
    digest: Final = sha256(key.encode()).hexdigest()
    usage: Final = object_value(proxy.chat(MODEL, key=key, text=f"spend {uuid.uuid4().hex}")["usage"])
    assert {name: usage.get(name) for name in USAGE} == dict(USAGE), usage
    eventually(lambda: _landed(database_url, digest), bool, seconds=70)
    write_rows('DELETE FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,), database_url=database_url)
    return Spender(alias, user_id, user_email, digest)


def _fetch(pinned: Pinned, digest: str) -> httpx.Response:
    return pinned.request(
        "GET",
        "/user/daily/activity/aggregated",
        params={"start_date": _day(-1), "end_date": _day(1), "api_key": digest},
    )


def _read(pinned: Pinned, digest: str) -> httpx.Response:
    response: Final = _fetch(pinned, digest)
    assert response.status_code == 200, response.text
    return response


def _metadata(response: httpx.Response, digest: str) -> tuple[_KeyMetadata, ...]:
    days: Final = _Activity.model_validate_json(response.content).results
    return tuple(day.breakdown.api_keys[digest].metadata for day in days)


def _aliases(response: httpx.Response, digest: str) -> tuple[str | None, ...]:
    return tuple(meta.key_alias for meta in _metadata(response, digest))


def _named(pinned: Pinned, spender: Spender) -> httpx.Response:
    return eventually(
        lambda: _read(pinned, spender.digest),
        lambda response: _aliases(response, spender.digest) == (spender.alias,),
        seconds=MISS_TTL_BOUND,
    )


def _named_once_reconnected(pinned: Pinned, spender: Spender) -> httpx.Response:
    return eventually(
        lambda: _fetch(pinned, spender.digest),
        lambda response: response.status_code == 200 and _aliases(response, spender.digest) == (spender.alias,),
        seconds=MISS_TTL_BOUND,
    )


@contextmanager
def _locked_spend_logs(database_url: str) -> Generator[None]:
    with psycopg.connect(database_url) as connection:
        connection.execute('LOCK TABLE "LiteLLM_SpendLogs" IN ACCESS EXCLUSIVE MODE')
        try:
            yield
        finally:
            connection.rollback()


def _poll_waiting_lookups(database_url: str, stop: threading.Event, seen: SimpleQueue[tuple[int, str]]) -> None:
    with psycopg.connect(database_url, autocommit=True) as connection:
        while not stop.wait(0.05):
            for lookup in WAITING_ROWS.validate_python(
                connection.execute(WAITING_LOOKUPS, (LOOKUP_MARKER,)).fetchall()
            ):
                seen.put(lookup)


@contextmanager
def _recording(database_url: str) -> Generator[SimpleQueue[tuple[int, str]]]:
    seen: Final[SimpleQueue[tuple[int, str]]] = SimpleQueue()
    stop: Final = threading.Event()
    poller: Final = threading.Thread(target=_poll_waiting_lookups, args=(database_url, stop, seen))
    poller.start()
    try:
        yield seen
    finally:
        stop.set()
        poller.join(timeout=5)
        assert not poller.is_alive(), "Lookup recorder survived its recording window"


def _lookups(seen: SimpleQueue[tuple[int, str]]) -> frozenset[tuple[int, str]]:
    return frozenset(seen.get_nowait() for _ in range(seen.qsize()))


def _busiest_miss_window(lookups: frozenset[tuple[int, str]]) -> int:
    starts: Final = sorted(datetime.fromisoformat(started) for _, started in lookups)
    return max((bisect_left(starts, start + MISS_WINDOW) - index for index, start in enumerate(starts)), default=0)


def _waiting_lookup_count(database_url: str) -> int:
    return len(read_rows(WAITING_LOOKUPS, (LOOKUP_MARKER,), database_url=database_url))


def _probe(pinned: Pinned, database_url: str, digest: str) -> Probe:
    with ThreadPoolExecutor(max_workers=1) as pool:
        with _locked_spend_logs(database_url):
            pending: Final = pool.submit(_read, pinned, digest)
            ran_lookup: Final = eventually(
                lambda: (pending.done(), _waiting_lookup_count(database_url) > 0),
                lambda state: state[0] or state[1],
                seconds=15,
            )[1]
        return Probe(ran_lookup, _aliases(pending.result(), digest))


def _park(database_url: str, digest: str) -> None:
    write_rows(
        """UPDATE "LiteLLM_SpendLogs" SET api_key = 'parked-' || api_key WHERE api_key = %s""",
        (digest,),
        database_url=database_url,
    )


def _restore(database_url: str, digest: str) -> None:
    write_rows(
        """UPDATE "LiteLLM_SpendLogs" SET api_key = substr(api_key, 8) WHERE api_key = 'parked-' || %s""",
        (digest,),
        database_url=database_url,
    )


def _events(response: httpx.Response) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")
    )


@dataclass(frozen=True, slots=True)
class Tenant:
    label: str
    team_id: str
    organization_id: str

    @property
    def owner(self) -> str:
        return f"{self.label}-owner"

    @property
    def email(self) -> str:
        return f"{self.owner}@example.com"

    @property
    def customer(self) -> str:
        return f"{self.label}-customer"

    @property
    def tag(self) -> str:
        return f"{self.label}-tag"

    @property
    def agent(self) -> str:
        return f"{self.label}-agent"

    @property
    def headers(self) -> Mapping[str, str]:
        return MappingProxyType(
            {"x-litellm-end-user-id": self.customer, "x-litellm-tags": self.tag, "x-litellm-agent-id": self.agent}
        )


@dataclass(frozen=True, slots=True)
class TenantKey:
    name: str
    key: str
    digest: str
    alias: str


@dataclass(frozen=True, slots=True)
class KeyRow:
    digest: str
    key_alias: str | None
    user_id: str | None
    user_email: str | None


def _tenant(proxy: Gateway) -> Tenant:
    label: Final = f"audit-{uuid.uuid4().hex[:6]}"
    organization: Final = proxy.post("/organization/new", {"organization_alias": f"{label}-org", "models": [MODEL]})
    organization_id: Final = string_value(organization["organization_id"])
    owner: Final = f"{label}-owner"
    proxy.post("/user/new", {"user_id": owner, "user_email": f"{owner}@example.com", "auto_create_key": False})
    team: Final = proxy.post(
        "/team/new",
        {
            "team_alias": f"{label}-team",
            "organization_id": organization_id,
            "models": [MODEL],
            "members_with_roles": [{"role": "user", "user_id": owner}],
        },
    )
    return Tenant(label, string_value(team["team_id"]), organization_id)


def _tenant_key(proxy: Gateway, tenant: Tenant, name: str) -> TenantKey:
    alias: Final = f"{tenant.label}-{name}"
    generated: Final = proxy.post(
        "/key/generate", {"user_id": tenant.owner, "team_id": tenant.team_id, "key_alias": alias, "models": [MODEL]}
    )
    key: Final = string_value(generated["key"])
    return TenantKey(name, key, sha256(key.encode()).hexdigest(), alias)


def _spend_request(name: str) -> tuple[str, Mapping[str, JsonValue]]:
    prompt: Final = f"spend {uuid.uuid4().hex}"
    match name:
        case "stream":
            return "/v1/chat/completions", {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        case "messages":
            return "/v1/messages", {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]}
        case "responses":
            return "/v1/responses", {"model": MODEL, "input": prompt}
        case _:
            return "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}


def _chunk_text(frame: JsonValue) -> str:
    match frame:
        case {"choices": [{"delta": {"content": str() as text}}]}:
            return text
        case _:
            return ""


def _spent_text(response: httpx.Response) -> str:
    if response.headers["content-type"].startswith("text/event-stream"):
        return "".join(
            _chunk_text(JSON_VALUE.validate_json(line.removeprefix("data: ")))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
    match JSON_VALUE.validate_json(response.content):
        case {"choices": [{"message": {"content": str() as text}}]}:
            return text
        case {"content": [{"text": str() as text}]}:
            return text
        case {"output": [{"content": [{"text": str() as text}]}]}:
            return text
        case _:
            return response.text


def _spend(proxy: Gateway, tenant: Tenant, key: TenantKey) -> None:
    path, body = _spend_request(key.name)
    response: Final = proxy.request("POST", path, body, key=key.key, headers=tenant.headers)
    assert response.status_code == 200, f"{key.name}: {response.status_code} {response.text}"
    assert _spent_text(response) == REPLY_TEXT, f"{key.name}: {response.text}"


def _landed_rows(database_url: str) -> frozenset[tuple[str, str]]:
    return frozenset(
        (str(row["source"]), str(row["api_key"])) for row in read_rows(LANDED_KEYS, (), database_url=database_url)
    )


def _named_key_row(name: str, child: JsonValue, digests: frozenset[str]) -> tuple[KeyRow, ...]:
    match child:
        case {"metadata": dict() as metadata} if name in digests:
            meta: Final = _KeyMetadata.model_validate(metadata)
            return (KeyRow(name, meta.key_alias, meta.user_id, meta.user_email),)
        case _:
            return ()


def _key_rows(value: JsonValue, digests: frozenset[str]) -> Iterator[KeyRow]:
    match value:
        case list():
            for item in value:
                yield from _key_rows(item, digests)
        case {"api_key": str() as digest, "metadata": dict() as metadata} if digest in digests:
            meta: Final = _KeyMetadata.model_validate(metadata)
            yield KeyRow(digest, meta.key_alias, meta.user_id, meta.user_email)
        case dict():
            for name, child in value.items():
                yield from _named_key_row(name, child, digests)
                yield from _key_rows(child, digests)
        case _:
            return


def _walked(response: httpx.Response, digests: frozenset[str]) -> frozenset[KeyRow]:
    return frozenset(_key_rows(JSON_VALUE.validate_json(response.content), digests))


def _routes(tenant: Tenant) -> Mapping[str, tuple[str, Mapping[str, str]]]:
    return MappingProxyType(
        {
            "user": ("/user/daily/activity", {}),
            "user aggregated": (AGGREGATED, {}),
            "user search": ("/user/daily/activity/aggregated/search", {"search": tenant.label}),
            "team": ("/team/daily/activity", {"team_ids": tenant.team_id}),
            "team aggregated": ("/team/daily/activity/aggregated", {"team_ids": tenant.team_id}),
            "team search": (
                "/team/daily/activity/aggregated/search",
                {"search": tenant.label, "team_ids": tenant.team_id},
            ),
            "organization": ("/organization/daily/activity", {"organization_ids": tenant.organization_id}),
            "customer": ("/customer/daily/activity", {"end_user_ids": tenant.customer}),
            "end user": ("/end_user/daily/activity", {"end_user_ids": tenant.customer}),
            "tag": ("/tag/daily/activity", {"tags": tenant.tag}),
            "agent": ("/agent/daily/activity", {"agent_ids": tenant.agent}),
        }
    )


def _get(pinned: Pinned, path: str, params: Mapping[str, str]) -> httpx.Response:
    response: Final = pinned.request(
        "GET", path, params={"start_date": _day(-1), "end_date": _day(1), "page_size": "100", **params}
    )
    assert response.status_code == 200, f"GET {path}: {response.status_code} {response.text}"
    return response


def _sweep(pinned: Pinned, tenant: Tenant, digests: frozenset[str]) -> Mapping[str, frozenset[KeyRow]]:
    return MappingProxyType(
        {name: _walked(_get(pinned, path, params), digests) for name, (path, params) in _routes(tenant).items()}
    )


def _named_row(tenant: Tenant, key: TenantKey) -> KeyRow:
    return KeyRow(key.digest, key.alias, tenant.owner, tenant.email)


def _outage_row(tenant: Tenant, key: TenantKey) -> KeyRow:
    if key.name in KEYS_ONLY_IN_SPEND_LOGS:
        return KeyRow(key.digest, None, tenant.owner, tenant.email)
    return _named_row(tenant, key)


def _expected(tenant: Tenant, keys: tuple[TenantKey, ...], *, outage: bool) -> Mapping[str, frozenset[KeyRow]]:
    every: Final = frozenset(_outage_row(tenant, key) if outage else _named_row(tenant, key) for key in keys)
    found: Final = frozenset(_named_row(tenant, key) for key in keys if key.name in KEYS_A_SEARCH_FINDS_BY_ALIAS)
    searched: Final = frozenset(("user search", "team search"))
    return MappingProxyType({name: found if name in searched else every for name in _routes(tenant)})


def _burst(proxy: Gateway, digests: frozenset[str]) -> tuple[frozenset[KeyRow], ...]:
    params: Final = {"start_date": _day(-1), "end_date": _day(1)}
    with (
        httpx.Client(
            base_url=proxy.client.base_url,
            headers={"Authorization": f"Bearer {proxy.key}"},
            timeout=60,
            trust_env=False,
        ) as client,
        ThreadPoolExecutor(max_workers=BURST) as pool,
    ):

        def read_aggregated(_: int) -> httpx.Response:
            return client.get(AGGREGATED, params=params)

        responses: Final = tuple(pool.map(read_aggregated, range(BURST)))
    assert all(response.status_code == 200 for response in responses), tuple(r.text for r in responses)
    return tuple(_walked(response, digests) for response in responses)


def _jwks_reply(public_jwk: str) -> Reply:
    return Reply(body=json.dumps({"keys": [{**json.loads(public_jwk), "kid": JWKS_KEY_ID}]}).encode())


@contextmanager
def _rig(gateway: Gateway, directory: Path) -> Generator[tuple[Gateway, str]]:
    with (
        scratch_database() as database_url,
        wire_server(_respond) as wire,
        _proxy(gateway, directory, database_url, wire.url) as proxy,
    ):
        yield proxy, database_url


@pytest.mark.timeout(360)
def test_usage_ai_chat_timeout_then_second_timeout_still_recovers_key_alias_after_database_frees(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path) as (proxy, database_url):
        spender: Final = _spender(proxy, database_url, "ai-chat")
        with _pinned(proxy) as pinned:
            with _locked_spend_logs(database_url):
                with _recording(database_url) as during_chat:
                    chat: Final = pinned.request(
                        "POST",
                        "/usage/ai/chat",
                        body={
                            "messages": [{"role": "user", "content": "What did we spend?"}],
                            "model": "openai/gpt-4o-mini",
                        },
                    )
                assert chat.status_code == 200, chat.text
                assert chat.elapsed >= FAILED_LOOKUP_FLOOR, chat.elapsed
                tool_call: Final = {
                    "type": "tool_call",
                    "tool_name": "get_usage_data",
                    "tool_label": "global usage data",
                    "arguments": {"start_date": _day(-1), "end_date": _day(1)},
                }
                assert _events(chat) == (
                    {"type": "status", "message": "Thinking..."},
                    {**tool_call, "status": "running"},
                    {**tool_call, "status": "complete"},
                    {"type": "status", "message": "Analyzing results..."},
                    {"type": "chunk", "content": REPLY_TEXT},
                    {"type": "done"},
                ), chat.text
                assert len(_lookups(during_chat)) == 1
                cached_miss: Final = _read(pinned, spender.digest)
                assert cached_miss.elapsed < FAILED_LOOKUP_FLOOR, cached_miss.elapsed
                assert _aliases(cached_miss, spender.digest) == (None,), cached_miss.text
                with _recording(database_url) as during_retry:
                    retried: Final = eventually(
                        lambda: _read(pinned, spender.digest),
                        lambda response: response.elapsed >= FAILED_LOOKUP_FLOOR,
                        seconds=MISS_TTL_BOUND,
                    )
                assert _aliases(retried, spender.digest) == (None,), retried.text
                assert len(_lookups(during_retry)) == 1
            recovered: Final = _named(pinned, spender)
            assert _metadata(recovered, spender.digest) == (
                _KeyMetadata(key_alias=spender.alias, user_id=spender.user_id, user_email=spender.user_email),
            ), recovered.text


@pytest.mark.timeout(300)
def test_usage_page_timeout_then_genuine_miss_still_recovers_key_alias_once_spend_logs_return(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path) as (proxy, database_url):
        spender: Final = _spender(proxy, database_url, "miss-after-timeout")
        with _pinned(proxy) as pinned:
            with _locked_spend_logs(database_url), _recording(database_url) as during_timeout:
                timed_out: Final = _read(pinned, spender.digest)
            assert timed_out.elapsed >= FAILED_LOOKUP_FLOOR, timed_out.elapsed
            assert _aliases(timed_out, spender.digest) == (None,), timed_out.text
            assert len(_lookups(during_timeout)) == 1
            _park(database_url, spender.digest)
            missed: Final = eventually(
                lambda: _probe(pinned, database_url, spender.digest),
                lambda probe: probe.ran_lookup,
                seconds=MISS_TTL_BOUND,
            )
            assert missed.aliases == (None,), missed
            _restore(database_url, spender.digest)
            recovered: Final = _named(pinned, spender)
            assert _metadata(recovered, spender.digest) == (
                _KeyMetadata(key_alias=spender.alias, user_id=spender.user_id, user_email=spender.user_email),
            ), recovered.text


@pytest.mark.timeout(360)
def test_usage_page_pins_a_key_blank_only_after_two_genuine_misses(gateway: Gateway, tmp_path: Path) -> None:
    with _rig(gateway, tmp_path) as (proxy, database_url):
        missed_twice: Final = _spender(proxy, database_url, "missed-twice")
        missed_once: Final = _spender(proxy, database_url, "missed-once")
        with _pinned(proxy) as pinned:
            _park(database_url, missed_twice.digest)
            _park(database_url, missed_once.digest)
            first_misses: Final = (
                _probe(pinned, database_url, missed_twice.digest),
                _probe(pinned, database_url, missed_once.digest),
            )
            assert first_misses == (Probe(True, (None,)), Probe(True, (None,))), first_misses
            _restore(database_url, missed_once.digest)
            second_miss: Final = eventually(
                lambda: _probe(pinned, database_url, missed_twice.digest),
                lambda probe: probe.ran_lookup,
                seconds=MISS_TTL_BOUND,
            )
            assert second_miss.aliases == (None,), second_miss
            _restore(database_url, missed_twice.digest)
            _named(pinned, missed_once)
            pinned_blank: Final = eventually(
                lambda: _probe(pinned, database_url, missed_twice.digest),
                lambda probe: probe.ran_lookup,
                seconds=45,
                return_last_on_timeout=True,
            )
            assert pinned_blank == Probe(False, (None,)), pinned_blank


@pytest.mark.timeout(300)
def test_usage_page_survives_a_dropped_database_connection_during_alias_recovery(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        scratch_database() as database_url,
        database_relay(database_url, b"AS " + LOOKUP_MARKER.encode()) as (relay, relayed_url),
        wire_server(_respond) as wire,
        _proxy(gateway, tmp_path, relayed_url, wire.url) as proxy,
    ):
        spender: Final = _spender(proxy, database_url, "dropped-connection")
        with _pinned(proxy) as pinned:
            relay.arm()
            dropped: Final = _read(pinned, spender.digest)
            assert relay.tripped.is_set(), dropped.text
            assert _aliases(dropped, spender.digest) == (None,), dropped.text
            recovered: Final = _named_once_reconnected(pinned, spender)
            assert _metadata(recovered, spender.digest) == (
                _KeyMetadata(key_alias=spender.alias, user_id=spender.user_id, user_email=spender.user_email),
            ), recovered.text


def _drop_token_rows(database_url: str, keys: tuple[TenantKey, ...]) -> None:
    write_rows(
        """DELETE FROM "LiteLLM_VerificationToken" WHERE token = ANY(string_to_array(%s, ','))""",
        (",".join(key.digest for key in keys if key.name in KEYS_ONLY_IN_SPEND_LOGS),),
        database_url=database_url,
    )


@pytest.mark.timeout(420)
def test_every_usage_route_names_spend_log_only_keys_again_once_an_outage_ends(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path) as (proxy, database_url):
        tenant: Final = _tenant(proxy)
        keys: Final = tuple(_tenant_key(proxy, tenant, name) for name in TENANT_KEYS)
        for key in keys:
            _spend(proxy, tenant, key)
        digests: Final = frozenset(key.digest for key in keys)
        every_table: Final = frozenset(itertools.product(SPEND_TABLES, digests))
        eventually(lambda: every_table - _landed_rows(database_url), lambda missing: not missing, seconds=90)
        _drop_token_rows(database_url, keys)
        deleted: Final = next(key for key in keys if key.name == "deleted")
        proxy.post("/key/delete", {"keys": [deleted.key]})
        outage: Final = _expected(tenant, keys, outage=True)
        healthy: Final = _expected(tenant, keys, outage=False)
        with _pinned(proxy) as pinned:
            with _locked_spend_logs(database_url):
                with _recording(database_url) as during_outage:
                    burst: Final = _burst(proxy, digests)
                    blank: Final = _sweep(pinned, tenant, digests)
                with _recording(database_url) as during_retry:
                    retried: Final = eventually(
                        lambda: _get(pinned, AGGREGATED, {}),
                        lambda response: response.elapsed >= FAILED_LOOKUP_FLOOR,
                        seconds=MISS_TTL_BOUND,
                    )
            assert frozenset(burst) == {outage["user aggregated"]}, burst
            assert dict(blank) == dict(outage)
            outage_lookups: Final = _lookups(during_outage)
            assert outage_lookups, "No alias lookup reached the locked spend logs"
            assert _busiest_miss_window(outage_lookups) <= WORKERS, sorted(outage_lookups)
            assert _walked(retried, digests) == outage["user aggregated"], retried.text
            assert len(_lookups(during_retry)) == 1
            eventually(
                lambda: _walked(_get(pinned, AGGREGATED, {}), digests),
                lambda rows: rows == healthy["user aggregated"],
                seconds=MISS_TTL_BOUND,
            )
            named: Final = _sweep(pinned, tenant, digests)
            assert dict(named) == dict(healthy)
            assert frozenset(_burst(proxy, digests)) == {healthy["user aggregated"]}


@pytest.mark.timeout(300)
def test_usage_page_retries_the_spend_log_lookup_for_a_rejected_jwt_caller_once_spend_logs_free_up(
    gateway: Gateway, tmp_path: Path
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = RSAAlgorithm.to_jwk(private_key.public_key())
    subject: Final = f"jwt-{uuid.uuid4().hex[:12]}"
    token: Final = jwt.encode(
        {"sub": subject, "exp": int((datetime.now(UTC) + timedelta(minutes=10)).timestamp())},
        private_key,
        algorithm="RS256",
        headers={"kid": JWKS_KEY_ID},
    )
    digest: Final = f"hashed-jwt-{sha256(token.encode()).hexdigest()}"
    with (
        scratch_database() as database_url,
        wire_server(lambda _: _jwks_reply(public_jwk)) as jwks,
        wire_server(_respond) as wire,
        _proxy(
            gateway,
            tmp_path,
            database_url,
            wire.url,
            general_settings=JWT_SETTINGS,
            environment={"JWT_PUBLIC_KEY_URL": jwks.url},
        ) as proxy,
    ):
        broke: Final = proxy.request("POST", "/user/new", {"user_id": subject, "max_budget": 0})
        assert broke.status_code == 200, broke.text
        rejected: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": MODEL, "messages": [{"role": "user", "content": f"spend {uuid.uuid4().hex}"}]},
            key=token,
        )
        assert rejected.status_code == 422, rejected.text
        assert f"User={subject} over budget" in rejected.text, rejected.text
        eventually(lambda: _landed(database_url, digest), bool, seconds=70)
        with _pinned(proxy) as pinned:
            with _locked_spend_logs(database_url), _recording(database_url) as during_outage:
                blank: Final = _read(pinned, digest)
            assert blank.elapsed >= FAILED_LOOKUP_FLOOR, blank.elapsed
            assert _metadata(blank, digest) == (_KeyMetadata(user_id=subject),), blank.text
            assert len(_lookups(during_outage)) == 1
            retried: Final = eventually(
                lambda: _probe(pinned, database_url, digest),
                lambda probe: probe.ran_lookup,
                seconds=MISS_TTL_BOUND,
            )
            assert retried.aliases == (None,), retried
            named: Final = _read(pinned, digest)
            assert _metadata(named, digest) == (_KeyMetadata(user_id=subject),), named.text


def _full_metadata(spender: Spender) -> tuple[_KeyMetadata, ...]:
    return (_KeyMetadata(key_alias=spender.alias, user_id=spender.user_id, user_email=spender.user_email),)


@pytest.mark.timeout(600)
def test_second_proxy_instance_names_the_key_while_the_first_recovers_from_its_own_misses(
    gateway: Gateway, tmp_path: Path
) -> None:
    first_home: Final = tmp_path / "first"
    second_home: Final = tmp_path / "second"
    first_home.mkdir()
    second_home.mkdir()
    with (
        scratch_database() as database_url,
        wire_server(_respond) as wire,
        _proxy(gateway, first_home, database_url, wire.url) as first,
        _proxy(gateway, second_home, database_url, wire.url) as second,
    ):
        spender: Final = _spender(first, database_url, "second-instance")
        with _pinned(first) as pinned_first, _pinned(second) as pinned_second:
            with _locked_spend_logs(database_url):
                timed_out: Final = _read(pinned_first, spender.digest)
                assert timed_out.elapsed >= FAILED_LOOKUP_FLOOR, timed_out.elapsed
                assert _aliases(timed_out, spender.digest) == (None,), timed_out.text
                retried: Final = eventually(
                    lambda: _read(pinned_first, spender.digest),
                    lambda response: response.elapsed >= FAILED_LOOKUP_FLOOR,
                    seconds=MISS_TTL_BOUND,
                )
                assert _aliases(retried, spender.digest) == (None,), retried.text
            fresh: Final = _read(pinned_second, spender.digest)
            assert fresh.elapsed < FAILED_LOOKUP_FLOOR, fresh.elapsed
            assert _metadata(fresh, spender.digest) == _full_metadata(spender), fresh.text
            recovered: Final = _named(pinned_first, spender)
            assert _metadata(recovered, spender.digest) == _full_metadata(spender), recovered.text


@pytest.mark.timeout(600)
def test_usage_page_names_the_key_at_once_after_a_restart_ends_the_outage(gateway: Gateway, tmp_path: Path) -> None:
    with scratch_database() as database_url, wire_server(_respond) as wire:
        with _proxy(gateway, tmp_path, database_url, wire.url) as first:
            spender: Final = _spender(first, database_url, "restart")
            with _pinned(first) as pinned:
                with _locked_spend_logs(database_url):
                    timed_out: Final = _read(pinned, spender.digest)
                assert timed_out.elapsed >= FAILED_LOOKUP_FLOOR, timed_out.elapsed
                assert _aliases(timed_out, spender.digest) == (None,), timed_out.text
                cached_miss: Final = _read(pinned, spender.digest)
                assert cached_miss.elapsed < FAILED_LOOKUP_FLOOR, cached_miss.elapsed
                assert _aliases(cached_miss, spender.digest) == (None,), cached_miss.text
        with _proxy(gateway, tmp_path, database_url, wire.url) as restarted, _pinned(restarted) as pinned_again:
            named: Final = _read(pinned_again, spender.digest)
            assert named.elapsed < FAILED_LOOKUP_FLOOR, named.elapsed
            assert _metadata(named, spender.digest) == _full_metadata(spender), named.text


def _fresh_read(owned: OwnedProxy, digest: str) -> httpx.Response:
    with httpx.Client(base_url=owned.gateway.client.base_url, timeout=60, trust_env=False) as client:
        return client.get(
            AGGREGATED,
            params={"start_date": _day(-1), "end_date": _day(1), "api_key": digest},
            headers={"Authorization": f"Bearer {owned.gateway.key}", "Connection": "close"},
        )


def _running_children(owned: OwnedProxy) -> tuple[int, ...]:
    return tuple(
        member.pid
        for member in group_members(owned.process.pid)
        if member.pid != owned.process.pid and member.is_running() and member.status() != psutil.STATUS_ZOMBIE
    )


def _worker_pids(owned: OwnedProxy) -> tuple[int, ...]:
    return tuple(
        member.pid
        for member in group_members(owned.process.pid)
        if member.pid != owned.process.pid and any("spawn_main" in part for part in member.cmdline())
    )


@pytest.mark.timeout(600)
def test_usage_page_keeps_serving_when_a_worker_dies_mid_outage(gateway: Gateway, tmp_path: Path) -> None:
    with (
        scratch_database() as database_url,
        wire_server(_respond) as wire,
        _owned_proxy(gateway, tmp_path, database_url, wire.url) as owned,
    ):
        spender: Final = _spender(owned.gateway, database_url, "worker-death")
        children: Final = _running_children(owned)
        workers: Final = _worker_pids(owned)
        assert len(workers) == WORKERS, workers
        with _locked_spend_logs(database_url):
            timed_out: Final = _fresh_read(owned, spender.digest)
            assert timed_out.status_code == 200, timed_out.text
            assert timed_out.elapsed >= FAILED_LOOKUP_FLOOR, timed_out.elapsed
            assert _aliases(timed_out, spender.digest) == (None,), timed_out.text
            os.kill(workers[0], signal.SIGKILL)
            after_kill: Final = _fresh_read(owned, spender.digest)
            assert after_kill.status_code == 200, after_kill.text
            assert _aliases(after_kill, spender.digest) == (None,), after_kill.text
            respawned: Final = eventually(
                lambda: _running_children(owned),
                lambda pids: len(pids) >= len(children) and any(pid not in children for pid in pids),
                seconds=30,
            )
            assert workers[0] not in respawned, respawned
        recovered: Final = eventually(
            lambda: _fresh_read(owned, spender.digest),
            lambda response: response.status_code == 200 and _aliases(response, spender.digest) == (spender.alias,),
            seconds=MISS_TTL_BOUND,
        )
        assert _metadata(recovered, spender.digest) == _full_metadata(spender), recovered.text


def _status(pinned: Pinned, path: str, params: Mapping[str, str]) -> int:
    return pinned.request("GET", path, params={"start_date": _day(-1), "end_date": _day(1), **params}).status_code


@pytest.mark.timeout(360)
def test_usage_page_rejects_bad_key_filters_and_unrelated_routes_ignore_a_locked_spend_log_table(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path) as (proxy, database_url):
        spender: Final = _spender(proxy, database_url, "bad-filters")
        with _pinned(proxy) as pinned, _locked_spend_logs(database_url):
            odd_keys: Final = ("k" * 5000, "", "123", json.dumps([spender.digest]))
            odd_statuses: Final = tuple(_status(pinned, AGGREGATED, {"api_key": api_key}) for api_key in odd_keys)
            assert odd_statuses == (200, 200, 200, 200), odd_statuses
            repeated: Final = _status(pinned, f"{AGGREGATED}?api_key={spender.digest}&api_key={spender.digest}", {})
            assert repeated == 200, repeated
            page_sizes: Final = tuple(
                _status(pinned, "/user/daily/activity", {"page_size": page_size}) for page_size in ("0", "abc")
            )
            assert page_sizes == (422, 422), page_sizes
            liveliness: Final = pinned.request("GET", "/health/liveliness")
            assert liveliness.status_code == 200, liveliness.text
            readiness: Final = pinned.request("GET", "/health/readiness")
            assert readiness.status_code == 200, readiness.text
            gateway_activity: Final = _status(pinned, "/gateway/daily/activity", {})
            assert gateway_activity == 200, gateway_activity
            chat: Final = proxy.chat(MODEL, text=f"locked {uuid.uuid4().hex}")
            assert object_value(chat["usage"]) == dict(USAGE), chat


@pytest.mark.timeout(300)
def test_usage_ai_chat_survives_a_dropped_database_connection_during_alias_recovery(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        scratch_database() as database_url,
        database_relay(database_url, b"AS " + LOOKUP_MARKER.encode()) as (relay, relayed_url),
        wire_server(_respond) as wire,
        _proxy(gateway, tmp_path, relayed_url, wire.url) as proxy,
    ):
        spender: Final = _spender(proxy, database_url, "dropped-ai-chat")
        with _pinned(proxy) as pinned:
            relay.arm()
            chat: Final = pinned.request(
                "POST",
                "/usage/ai/chat",
                body={
                    "messages": [{"role": "user", "content": "What did we spend?"}],
                    "model": "openai/gpt-4o-mini",
                },
            )
            assert chat.status_code == 200, chat.text
            assert relay.tripped.is_set(), chat.text
            tool_call: Final = {
                "type": "tool_call",
                "tool_name": "get_usage_data",
                "tool_label": "global usage data",
                "arguments": {"start_date": _day(-1), "end_date": _day(1)},
            }
            assert _events(chat) == (
                {"type": "status", "message": "Thinking..."},
                {**tool_call, "status": "running"},
                {**tool_call, "status": "complete"},
                {"type": "status", "message": "Analyzing results..."},
                {"type": "chunk", "content": REPLY_TEXT},
                {"type": "done"},
            ), chat.text
            recovered: Final = _named_once_reconnected(pinned, spender)
            assert _metadata(recovered, spender.digest) == _full_metadata(spender), recovered.text
