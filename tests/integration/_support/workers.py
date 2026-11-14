from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
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
from tests.integration._support.ports import allocate_port
from tests.integration._support.redis_process import owned_redis
from tests.integration._support.routing import WORKER_DATABASE_PREFIX

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
    return MappingProxyType(
        {
            "INTEGRATION_UPSTREAM_URL": f"http://127.0.0.1:{allocate_port()}",
            "INTEGRATION_PROVIDER_PORT": str(allocate_port()),
        }
    )


def replica_urls(database: str, inherited: Mapping[str, str] = os.environ) -> Mapping[str, str]:
    writer: Final = urlsplit(inherited["INTEGRATION_PROXY_DATABASE_URL"])
    reader: Final = urlsplit(inherited["INTEGRATION_PROXY_READ_REPLICA_URL"])
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


@dataclass(frozen=True, slots=True)
class GatewayProfile:
    peer: bool
    replica: bool
    cost_map: bool
    coverage_file: Path | None


def gateway_profile(environment: Mapping[str, str]) -> GatewayProfile:
    coverage: Final = environment.get("INTEGRATION_COVERAGE_FILE", "")
    return GatewayProfile(
        peer=environment.get("INTEGRATION_WORKER_PEER") == "1",
        replica=bool(environment.get("INTEGRATION_PROXY_READ_REPLICA_URL")),
        cost_map=environment.get("INTEGRATION_WORKER_COST_MAP") == "1",
        coverage_file=Path(coverage) if coverage else None,
    )


def cost_map_environment(upstream: str) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "LITELLM_MODEL_COST_MAP_URL": f"{upstream}/_cost_map",
            "MODEL_COST_MAP_MIN_MODEL_COUNT": "1",
            "MODEL_COST_MAP_MAX_SHRINK_RATIO": "0",
            "GEMINI_API_BASE": upstream,
            "ANTHROPIC_API_BASE": upstream,
            "GEMINI_API_KEY": "sk-scripted-provider",
            "ANTHROPIC_API_KEY": "sk-scripted-provider",
        }
    )


@contextmanager
def worker_services(
    directory: Path, *, replica: bool = False, inherited: Mapping[str, str] = os.environ
) -> Iterator[Mapping[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    with (
        scratch_database(WORKER_DATABASE_PREFIX) as database,
        owned_redis(directory, inherited=inherited) as cache,
    ):
        yield MappingProxyType(
            {
                "DATABASE_URL": database,
                "INTEGRATION_PROXY_DATABASE_URL": database,
                "INTEGRATION_PROXY_READ_REPLICA_URL": "",
                "REDIS_HOST": cache.host,
                "REDIS_PORT": str(cache.port),
                **(replica_urls(database, inherited) if replica else {}),
            }
        )


@contextmanager
def worker_gateway(
    directory: Path, profile: GatewayProfile, inherited: Mapping[str, str]
) -> Iterator[Mapping[str, str]]:
    from tests.integration._support.process import owned_proxy, owned_upstream

    upstream: Final = inherited["INTEGRATION_UPSTREAM_URL"]
    port: Final = urlsplit(upstream).port
    assert port is not None
    key: Final = f"sk-{uuid.uuid4().hex}"
    removed: Final = ("LITELLM_LOCAL_MODEL_COST_MAP",) if profile.cost_map else ()
    with worker_services(directory, replica=profile.replica, inherited=inherited) as services:
        overrides: Final = {
            **services,
            "DATABASE_URL": services["INTEGRATION_PROXY_DATABASE_URL"],
            **(
                {"DATABASE_URL_READ_REPLICA": services["INTEGRATION_PROXY_READ_REPLICA_URL"]} if profile.replica else {}
            ),
            "LITELLM_ENABLE_MCP_STDIO": "true",
            **(cost_map_environment(upstream) if profile.cost_map else {}),
        }
        with (
            owned_upstream(directory, port=port, inherited=inherited),
            httpx.Client(base_url="http://127.0.0.1:1", trust_env=False) as client,
            owned_proxy(
                Gateway(client, key, upstream),
                directory,
                overrides,
                remove_environment=removed,
                coverage_file=profile.coverage_file,
                inherited=inherited,
            ) as primary,
            owned_proxy(
                primary,
                directory,
                overrides,
                remove_environment=removed,
                coverage_file=profile.coverage_file,
                inherited=inherited,
            )
            if profile.peer
            else nullcontext(None) as secondary,
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


@dataclass(frozen=True, slots=True)
class WorkerStack:
    resources: ExitStack
    environment: Mapping[str, str]


def worker_stack(results: Path, profile: GatewayProfile) -> WorkerStack:
    endpoints: Final = worker_endpoints()
    inherited: Final = MappingProxyType({**os.environ, **endpoints, "INTEGRATION_RESULTS_DIR": str(results)})
    resources: Final = ExitStack()
    try:
        services: Final = resources.enter_context(worker_gateway(results / "runtime", profile, inherited))
    except BaseException:
        resources.close()
        raise
    return WorkerStack(resources, MappingProxyType({**endpoints, **services, "INTEGRATION_RESULTS_DIR": str(results)}))


def start_worker_stacks(
    directories: tuple[Path, ...], build: Callable[[Path], WorkerStack], resources: ExitStack, concurrency: int
) -> tuple[Mapping[str, str], ...]:
    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="integration-worker-stack") as pool:
        futures: Final = tuple(pool.submit(build, directory) for directory in directories)
    for future in futures:
        if future.exception() is None:
            resources.enter_context(future.result().resources)
    failure: Final = next((error for future in futures if (error := future.exception()) is not None), None)
    if failure is not None:
        resources.close()
        raise failure
    return tuple(future.result().environment for future in futures)
