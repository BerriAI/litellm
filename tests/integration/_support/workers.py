from __future__ import annotations

import os
import socket
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
import psycopg
from psycopg import sql
from pydantic import TypeAdapter
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from tests.integration._support.client import Gateway
from tests.integration._support.database import scratch_database
from tests.integration._support.redis_process import owned_redis

READINESS_OBJECT: Final = TypeAdapter(dict[str, object])
SUBSCRIBER_COUNTS: Final = TypeAdapter(tuple[tuple[str, int], ...])


class ServiceSubscribers(Protocol):
    def pubsub_numsub(self, *channels: str) -> object: ...


def readiness_failure(
    client: httpx.Client, upstream: str, proxies: tuple[Gateway, ...], cache: ServiceSubscribers
) -> str | None:
    try:
        if client.get(upstream + "/health", timeout=2).status_code != 200:
            return "scripted upstream health"
        for index, proxy in enumerate(proxies):
            if proxy.client.get("/health/readiness", timeout=2).status_code != 200:
                return f"proxy {index} readiness"
            response: Final = proxy.client.get(
                "/cache/ping", headers={"Authorization": f"Bearer {proxy.key}"}, timeout=2
            )
            if response.status_code != 200:
                return f"proxy {index} Redis health HTTP {response.status_code}"
            result: Final = READINESS_OBJECT.validate_json(response.content)
            if (
                result.get("status") != "healthy"
                or result.get("cache_type") != "redis"
                or result.get("ping_response") is not True
                or result.get("set_cache_response") != "success"
            ):
                return f"proxy {index} Redis read/write health"
        subscribers: Final = SUBSCRIBER_COUNTS.validate_python(
            cache.pubsub_numsub("litellm_proxy.auth_cache_invalidation")
        )
        if subscribers[0][1] < len(proxies):
            return f"auth-cache subscribers {subscribers[0][1]}/{len(proxies)}"
    except (httpx.TransportError, RedisConnectionError, RedisTimeoutError) as error:
        return f"service transport {type(error).__name__}"
    return None


def wait_for_services(upstream: str, proxies: tuple[Gateway, ...], cache: Redis) -> None:
    with httpx.Client(trust_env=False, timeout=2) as client:
        deadline: Final = time.monotonic() + 90
        while True:
            failure: Final = readiness_failure(client, upstream, proxies, cache)
            if failure is None:
                return
            assert time.monotonic() < deadline, f"Integration services did not become ready: {failure}"
            time.sleep(0.2)


def worker_endpoints() -> Mapping[str, str]:
    with socket.socket() as upstream, socket.socket() as provider:
        upstream.bind(("127.0.0.1", 0))
        provider.bind(("127.0.0.1", 0))
        return MappingProxyType(
            {
                "INTEGRATION_UPSTREAM_URL": f"http://127.0.0.1:{upstream.getsockname()[1]}",
                "INTEGRATION_PROVIDER_PORT": str(provider.getsockname()[1]),
            }
        )


def _replica_urls(database: str) -> Mapping[str, str]:
    writer: Final = urlsplit(os.environ["INTEGRATION_PROXY_DATABASE_URL"])
    reader: Final = urlsplit(os.environ["INTEGRATION_PROXY_READ_REPLICA_URL"])
    database_path: Final = urlsplit(database).path
    assert writer.username is not None and reader.username is not None
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute(
            sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(database_path.removeprefix("/")), sql.Identifier(writer.username)
            )
        )
        admin.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(database_path.removeprefix("/")), sql.Identifier(reader.username)
            )
        )
        admin.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(reader.username)))
        admin.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(
                sql.Identifier(writer.username), sql.Identifier(reader.username)
            )
        )
    return MappingProxyType(
        {
            "INTEGRATION_PROXY_DATABASE_URL": urlunsplit(writer._replace(path=database_path)),
            "INTEGRATION_PROXY_READ_REPLICA_URL": urlunsplit(reader._replace(path=database_path)),
        }
    )


@contextmanager
def worker_services(directory: Path, *, replica: bool = False) -> Iterator[Mapping[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    with (
        scratch_database() as database,
        owned_redis(directory) as cache,
    ):
        yield MappingProxyType(
            {
                "DATABASE_URL": database,
                "INTEGRATION_PROXY_DATABASE_URL": database,
                "INTEGRATION_PROXY_READ_REPLICA_URL": "",
                "REDIS_HOST": cache.host,
                "REDIS_PORT": str(cache.port),
                **(_replica_urls(database) if replica else {}),
            }
        )


@contextmanager
def worker_gateway(directory: Path, *, peer: bool, replica: bool) -> Iterator[Mapping[str, str]]:
    from tests.integration._support.asgi import asgi_server
    from tests.integration._support.process import owned_proxy
    from tests.integration._support.upstream import Provider

    port: Final = urlsplit(os.environ["INTEGRATION_UPSTREAM_URL"]).port
    assert port is not None
    key: Final = f"sk-{uuid.uuid4().hex}"
    with worker_services(directory, replica=replica) as services:
        overrides: Final = {
            **services,
            "DATABASE_URL": services["INTEGRATION_PROXY_DATABASE_URL"],
            **({"DATABASE_URL_READ_REPLICA": services["INTEGRATION_PROXY_READ_REPLICA_URL"]} if replica else {}),
        }
        with (
            asgi_server(Provider().app(), port=port, keepalive_seconds=125) as upstream,
            httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client,
            owned_proxy(Gateway(client, key, upstream), directory, overrides) as primary,
            owned_proxy(primary, directory, overrides) if peer else nullcontext(None) as secondary,
        ):
            with Redis(host=services["REDIS_HOST"], port=int(services["REDIS_PORT"]), socket_timeout=2) as cache:
                wait_for_services(upstream, (primary, secondary) if secondary is not None else (primary,), cache)
            primary.post("/config/update", {"router_settings": {"num_retries": 0}})
            yield MappingProxyType(
                {
                    **services,
                    "INTEGRATION_PROXY_URL": str(primary.client.base_url),
                    "INTEGRATION_PEER_URL": str(secondary.client.base_url) if secondary is not None else "",
                    "INTEGRATION_MASTER_KEY": key,
                    "LITELLM_MASTER_KEY": key,
                    "INTEGRATION_UPSTREAM_URL": upstream,
                }
            )
