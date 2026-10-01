import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import httpx
import psycopg
import pytest
import yaml
from psycopg.rows import dict_row
from pydantic import JsonValue, TypeAdapter
from redis import Redis

from tests.integration._support.client import Gateway, Scenario, eventually
from tests.integration._support.customer_model_allowlist_wire import customer_model_allowlist_reply
from tests.integration._support.database import scratch_database
from tests.integration._support.process import owned_proxy, owned_proxy_process
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_AUDIT_HEAD_ROOT: Final = Path(os.environ.get("INTEGRATION_AUDIT_HEAD_ROOT", str(_REPO_ROOT))).resolve()
_PRISMA_DIR: Final = _AUDIT_HEAD_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
_MIGRATIONS_DIR: Final = _PRISMA_DIR / "migrations"
_MODELS_MIGRATION: Final = "20260930000000_add_end_user_models"
_MIGRATIONS: Final = tuple(sorted(path.name for path in _MIGRATIONS_DIR.iterdir() if path.is_dir()))


@dataclass(frozen=True, slots=True)
class BurstRequest:
    endpoint: Literal["chat", "messages", "responses"]
    customer: str
    marker: str
    stream: bool


_BURST_ENDPOINTS: Final[tuple[Literal["chat"], Literal["messages"], Literal["responses"]]] = (
    "chat",
    "messages",
    "responses",
)


def _customer(scenario: Scenario, models: tuple[str, ...]) -> str:
    identity: Final = f"integration-customer-{uuid.uuid4().hex}"
    scenario.gateway.post("/customer/new", {"user_id": identity, "models": list(models)})
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [identity]})
    return identity


def _customer_chat(
    gateway: Gateway,
    key: str,
    model: str,
    customer: str,
    marker: str,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker}], "user": customer},
        key=key,
    )


def _assert_upstream_request(wire: Wire, marker: str) -> None:
    requests: Final = wire.drain()
    matching: Final = tuple(request for request in requests if marker.encode() in request.body)
    assert len(matching) == 1, matching


def _assert_no_upstream_request(wire: Wire, marker: str) -> None:
    requests: Final = wire.drain()
    matching: Final = tuple(request for request in requests if marker.encode() in request.body)
    assert matching == (), matching


def test_customer_cache_update_reaches_the_peer_proxy(gateway: Gateway, peer: Gateway) -> None:
    with wire_server(customer_model_allowlist_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        disallowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, disallowed])
        customer: Final = _customer(scenario, (allowed, disallowed))
        for proxy in (gateway, peer):
            marker: Final = uuid.uuid4().hex
            response: Final = _customer_chat(proxy, key, disallowed, customer, marker)
            assert response.status_code == 200, response.text
            _assert_upstream_request(wire, marker)

        gateway.post("/customer/update", {"user_id": customer, "models": [allowed]})
        eventual_marker: Final = uuid.uuid4().hex
        denied: Final = eventually(
            lambda: _customer_chat(peer, key, disallowed, customer, eventual_marker),
            lambda response: response.status_code == 403,
            seconds=10,
        )
        assert denied.status_code == 403, denied.text
        assert "customer_model_access_denied" in denied.text, denied.text
        marker: Final = uuid.uuid4().hex
        final_denial: Final = _customer_chat(peer, key, disallowed, customer, marker)
        assert final_denial.status_code == 403, final_denial.text
        assert "customer_model_access_denied" in final_denial.text, final_denial.text
        _assert_no_upstream_request(wire, marker)


def test_customer_access_does_not_leak_between_alternating_customers(gateway: Gateway) -> None:
    with wire_server(customer_model_allowlist_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        restricted_model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, restricted_model])
        restricted: Final = _customer(scenario, (allowed,))
        unrestricted: Final = _customer(scenario, ())

        for _ in range(20):
            restricted_marker: Final = uuid.uuid4().hex
            denied: Final = _customer_chat(gateway, key, restricted_model, restricted, restricted_marker)
            assert denied.status_code == 403, denied.text
            assert "customer_model_access_denied" in denied.text, denied.text
            _assert_no_upstream_request(wire, restricted_marker)

            unrestricted_marker: Final = uuid.uuid4().hex
            served: Final = _customer_chat(gateway, key, restricted_model, unrestricted, unrestricted_marker)
            assert served.status_code == 200, served.text
            _assert_upstream_request(wire, unrestricted_marker)


def _burst_request(gateway: Gateway, key: str, model: str, request: BurstRequest) -> httpx.Response:
    if request.endpoint == "chat":
        return gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": request.marker}],
                "user": request.customer,
                "stream": request.stream,
            },
            key=key,
        )
    if request.endpoint == "messages":
        return gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": request.marker}],
                "metadata": {"user_id": request.customer},
                "stream": request.stream,
            },
            key=key,
        )
    return gateway.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": request.marker,
            "user": request.customer,
            "stream": request.stream,
        },
        key=key,
    )


def test_customer_allowlist_flip_is_consistent_during_concurrent_mixed_streams(gateway: Gateway) -> None:
    release_streams: Final = threading.Event()
    first_stream_started: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if request.method == "POST" and request.body:
            body: Final = _JSON_OBJECT.validate_json(request.body)
            if body.get("stream") is True and not first_stream_started.is_set():
                first_stream_started.set()
        return customer_model_allowlist_reply(request, release_streams)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        restricted_model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, restricted_model])
        restricted: Final = _customer(scenario, (allowed,))
        unrestricted: Final = _customer(scenario, ())
        initial_marker: Final = uuid.uuid4().hex
        initial_denial: Final = _customer_chat(gateway, key, restricted_model, restricted, initial_marker)
        assert initial_denial.status_code == 403, initial_denial.text
        assert "customer_model_access_denied" in initial_denial.text, initial_denial.text
        _assert_no_upstream_request(wire, initial_marker)
        requests: Final = tuple(
            BurstRequest(
                endpoint=_BURST_ENDPOINTS[index % len(_BURST_ENDPOINTS)],
                customer=restricted if index % 2 else unrestricted,
                marker=uuid.uuid4().hex,
                stream=index % 3 != 0,
            )
            for index in range(30)
        )
        barrier: Final = threading.Barrier(len(requests) + 1)

        def send(request: BurstRequest) -> tuple[BurstRequest, httpx.Response]:
            barrier.wait(timeout=20)
            return request, _burst_request(gateway, key, restricted_model, request)

        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures: Final = tuple(executor.submit(send, request) for request in requests)
            barrier.wait(timeout=20)
            try:
                eventually(first_stream_started.is_set, lambda started: started, seconds=20)
                gateway.post("/customer/update", {"user_id": restricted, "models": []})
            finally:
                release_streams.set()
            results: Final = tuple(future.result(timeout=30) for future in futures)

        for request, response in results:
            assert response.status_code in (200, 403), response.text
            if response.status_code == 403:
                assert request.customer == restricted, response.text
                assert "customer_model_access_denied" in response.text, response.text

        upstream: Final = wire.drain()
        for request, response in results:
            matching: Final = tuple(item for item in upstream if request.marker.encode() in item.body)
            if response.status_code == 200:
                assert len(matching) == 1, matching
            else:
                assert matching == (), matching

        final_marker: Final = uuid.uuid4().hex
        after_flip: Final = eventually(
            lambda: _customer_chat(gateway, key, restricted_model, restricted, final_marker),
            lambda response: response.status_code == 200,
            seconds=10,
        )
        assert after_flip.status_code == 200, after_flip.text
        _assert_upstream_request(wire, final_marker)


def _send_mixed_burst(
    gateway: Gateway,
    key: str,
    model: str,
    requests: tuple[BurstRequest, ...],
) -> tuple[tuple[BurstRequest, httpx.Response], ...]:
    barrier: Final = threading.Barrier(len(requests) + 1)

    def send(request: BurstRequest) -> tuple[BurstRequest, httpx.Response]:
        barrier.wait(timeout=20)
        return request, _burst_request(gateway, key, model, request)

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures: Final = tuple(executor.submit(send, request) for request in requests)
        barrier.wait(timeout=20)
        return tuple(future.result(timeout=30) for future in futures)


def _mixed_burst(customer: str, count: int) -> tuple[BurstRequest, ...]:
    return tuple(
        BurstRequest(
            endpoint=_BURST_ENDPOINTS[index % len(_BURST_ENDPOINTS)],
            customer=customer,
            marker=uuid.uuid4().hex,
            stream=index % 2 == 0,
        )
        for index in range(count)
    )


def test_customer_allowlist_is_enforced_after_owned_proxy_restarts_mid_burst(gateway: Gateway, tmp_path: Path) -> None:
    release_streams: Final = threading.Event()
    first_stream_started: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if request.method == "POST" and request.body:
            body: Final = _JSON_OBJECT.validate_json(request.body)
            if body.get("stream") is True and not first_stream_started.is_set():
                first_stream_started.set()
        return customer_model_allowlist_reply(request, release_streams)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        disallowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer: Final = _customer(scenario, (allowed,))
        initial_info: Final = gateway.get("/customer/info", {"end_user_id": customer})
        assert initial_info["models"] == [allowed], initial_info
        config: Final = {
            "model_list": [
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "integration-provider-key",
                        "api_base": f"{wire.url}/v1",
                    },
                }
                for model in (allowed, disallowed)
            ],
            "general_settings": {
                "master_key": "os.environ/LITELLM_MASTER_KEY",
                "database_url": "os.environ/DATABASE_URL",
                "store_model_in_db": False,
            },
            "litellm_settings": {"cache": False},
        }
        config_path: Final = tmp_path / f"customer-restart-{uuid.uuid4().hex}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        requests: Final = _mixed_burst(customer, 30)

        with owned_proxy_process(gateway, tmp_path, {}, config=config_path) as owned:
            candidate: Final = owned.gateway
            boot_info: Final = candidate.get("/customer/info", {"end_user_id": customer})
            assert boot_info["models"] == [allowed], f"{customer}: {boot_info}"
            barrier: Final = threading.Barrier(len(requests) + 1)

            def send(request: BurstRequest) -> tuple[BurstRequest, httpx.Response | None]:
                barrier.wait(timeout=20)
                try:
                    return request, _burst_request(candidate, gateway.key, allowed, request)
                except httpx.TransportError:
                    return request, None

            with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                futures: Final = tuple(executor.submit(send, request) for request in requests)
                barrier.wait(timeout=20)
                try:
                    eventually(first_stream_started.is_set, lambda started: started, seconds=20)
                    assert owned.process.poll() is None
                    owned.process.terminate()
                    release_streams.set()
                    owned.process.wait(timeout=35)
                finally:
                    release_streams.set()
                active_results: Final = tuple(future.result(timeout=30) for future in futures)

        for _, response in active_results:
            if response is not None:
                assert response.status_code == 200, response.text

        pre_restart_upstream: Final = wire.drain()
        for request in requests:
            matching: Final = tuple(item for item in pre_restart_upstream if request.marker.encode() in item.body)
            assert len(matching) <= 1, matching
        between_processes_info: Final = gateway.get("/customer/info", {"end_user_id": customer})
        assert between_processes_info["models"] == [allowed], f"{customer}: {between_processes_info}"

        with owned_proxy(gateway, tmp_path, {}, config=config_path) as candidate:
            info: Final = candidate.get("/customer/info", {"end_user_id": customer})
            assert info["models"] == [allowed], f"{customer}: {info}"
            denied_requests: Final = _mixed_burst(customer, 30)
            denied_results: Final = _send_mixed_burst(candidate, gateway.key, disallowed, denied_requests)
            for request, response in denied_results:
                assert response.status_code == 403, response.text
                assert "customer_model_access_denied" in response.text, response.text

            post_restart_upstream: Final = wire.drain()
            for request in denied_requests:
                matching: Final = tuple(item for item in post_restart_upstream if request.marker.encode() in item.body)
                assert matching == (), matching


def _custom_auth_config(directory: Path, wire: Wire, run_common_checks: bool) -> tuple[Path, str, str]:
    first: Final = f"custom-m1-{uuid.uuid4().hex}"
    second: Final = f"custom-m2-{uuid.uuid4().hex}"
    config: Final = {
        "model_list": [
            {
                "model_name": model,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": f"{wire.url}/v1",
                },
            }
            for model in (first, second)
        ],
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "database_url": "os.environ/DATABASE_URL",
            "custom_auth": "integration._support.customer_model_allowlist_auth.user_api_key_auth",
            "custom_auth_run_common_checks": run_common_checks,
            "store_model_in_db": False,
        },
        "litellm_settings": {"cache": False},
    }
    path: Final = directory / f"customer-custom-auth-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, first, second


def _custom_auth_request(
    gateway: Gateway,
    model: str,
    customer: str,
    marker: str,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker}]},
        key="integration-custom-auth-token",
        headers={"x-litellm-customer-id": customer},
    )


def test_custom_auth_common_checks_enforce_customer_model_allowlist(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(customer_model_allowlist_reply) as wire:
        config, allowed, disallowed = _custom_auth_config(tmp_path, wire, run_common_checks=True)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            with candidate.scenario() as scenario:
                customer: Final = _customer(scenario, (allowed,))
                marker: Final = uuid.uuid4().hex
                response: Final = _custom_auth_request(candidate, disallowed, customer, marker)
                assert response.status_code == 403, response.text
                assert "customer_model_access_denied" in response.text, response.text
                _assert_no_upstream_request(wire, marker)


def test_custom_auth_without_common_checks_serves_disallowed_customer_model(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(customer_model_allowlist_reply) as wire:
        config, allowed, disallowed = _custom_auth_config(tmp_path, wire, run_common_checks=False)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            with candidate.scenario() as scenario:
                customer: Final = _customer(scenario, (allowed,))
                marker: Final = uuid.uuid4().hex
                response: Final = _custom_auth_request(candidate, disallowed, customer, marker)
                assert response.status_code == 200, response.text
                _assert_upstream_request(wire, marker)


def _deploy_base_shaped_schema(database_url: str, directory: Path) -> None:
    base: Final = directory / "base-shaped"
    migrations: Final = base / "migrations"
    migrations.mkdir(parents=True)
    shutil.copy(_PRISMA_DIR / "schema.prisma", base / "schema.prisma")
    shutil.copy(_MIGRATIONS_DIR / "migration_lock.toml", migrations / "migration_lock.toml")
    base_migrations: Final = tuple(name for name in _MIGRATIONS if name != _MODELS_MIGRATION)
    for migration in base_migrations:
        shutil.copytree(_MIGRATIONS_DIR / migration, migrations / migration)
    result: Final = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "prisma",
            "migrate",
            "deploy",
            "--schema",
            str(base / "schema.prisma"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=80,
        env={**os.environ, "DATABASE_URL": database_url},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _applied_migrations(database_url: str) -> tuple[str, ...]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        rows: Final = connection.execute(
            'SELECT migration_name FROM "_prisma_migrations" '
            "WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL ORDER BY migration_name"
        ).fetchall()
    return tuple(str(row["migration_name"]) for row in rows)


def _models_column_exists(database_url: str) -> bool:
    with psycopg.connect(database_url) as connection:
        row: Final = connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'LiteLLM_EndUserTable' AND column_name = 'models'"
        ).fetchone()
    return row is not None


def test_head_migration_adds_empty_allowlist_without_rewriting_customers(
    gateway: Gateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        _deploy_base_shaped_schema(database_url, tmp_path)
        assert not _models_column_exists(database_url)
        assert _applied_migrations(database_url) == tuple(name for name in _MIGRATIONS if name != _MODELS_MIGRATION)
        customer: Final = f"integration-upgrade-{uuid.uuid4().hex}"
        with psycopg.connect(database_url) as connection:
            connection.execute(
                'INSERT INTO "LiteLLM_EndUserTable" ("user_id", "alias", "spend", "blocked") '
                "VALUES (%s, NULL, 0.0, false)",
                (customer,),
            )

        entrypoint: Final = subprocess.run(
            [sys.executable, "-m", "litellm.proxy.prisma_migration"],
            check=False,
            capture_output=True,
            text=True,
            timeout=80,
            cwd=_AUDIT_HEAD_ROOT,
            env={
                **os.environ,
                "DATABASE_URL": database_url,
                "PYTHONPATH": os.pathsep.join((str(_AUDIT_HEAD_ROOT / "tests"), str(_AUDIT_HEAD_ROOT))),
            },
        )
        assert entrypoint.returncode == 0, entrypoint.stdout + entrypoint.stderr
        assert _MODELS_MIGRATION in _applied_migrations(database_url), entrypoint.stdout
        assert _models_column_exists(database_url)
        migration: Final = (_MIGRATIONS_DIR / _MODELS_MIGRATION / "migration.sql").read_text()
        assert re.search(r"\b(?:UPDATE|DELETE)\b", migration, re.IGNORECASE) is None, migration

        with wire_server(customer_model_allowlist_reply) as wire:
            monkeypatch.setenv("INTEGRATION_PROXY_ROOT", str(_AUDIT_HEAD_ROOT))
            monkeypatch.setenv(
                "PYTHONPATH",
                os.pathsep.join((str(_AUDIT_HEAD_ROOT / "tests"), str(_AUDIT_HEAD_ROOT))),
            )
            with owned_proxy(
                gateway,
                tmp_path,
                {"DATABASE_URL": database_url, "DISABLE_SCHEMA_UPDATE": "true"},
                remove_environment=("DATABASE_URL_READ_REPLICA",),
            ) as upgraded:
                info: Final = upgraded.get("/customer/info", {"end_user_id": customer})
                assert info["models"] == [], info
                with upgraded.scenario() as scenario:
                    model: Final = scenario.model(api_base=f"{wire.url}/v1")
                    key: Final = scenario.key(models=[model])
                    marker: Final = uuid.uuid4().hex
                    response: Final = _customer_chat(upgraded, key, model, customer, marker)
                    assert response.status_code == 200, response.text
                    _assert_upstream_request(wire, marker)


def test_cached_end_user_without_models_decodes_as_unrestricted(gateway: Gateway) -> None:
    with wire_server(customer_model_allowlist_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        requested: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, requested])
        customer: Final = _customer(scenario, ())
        info: Final = gateway.get("/customer/info", {"end_user_id": customer})
        if "models" in info:
            gateway.post("/customer/update", {"user_id": customer, "models": [allowed]})

        cache_key: Final = f"end_user_id:{customer}"
        legacy: Final = {
            "user_id": customer,
            "blocked": False,
            "alias": None,
            "spend": 0.0,
            "allowed_model_region": None,
            "default_model": None,
            "budget_id": None,
        }
        with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
            cache.set(cache_key, json.dumps(legacy))
            marker: Final = uuid.uuid4().hex
            response: Final = _customer_chat(gateway, key, requested, customer, marker)
            assert response.status_code == 200, response.text
            _assert_upstream_request(wire, marker)
            cache.delete(cache_key)
