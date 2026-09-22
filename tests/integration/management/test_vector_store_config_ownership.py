import os
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import httpx
import psycopg
import pytest
from psycopg import sql
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.redis_process import owned_redis

CONFIG_STORE_ID: Final = "vs_integration_config_store"
CONFIG_STORE_NAME: Final = "integration-config-store"
SEARCH_PATH: Final = f"/vector_stores/{CONFIG_STORE_ID}/search"
PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"


def listed_rows(response: httpx.Response) -> tuple[dict[str, JsonValue], ...]:
    rows: Final = object_value(response.json()).get("data")
    assert isinstance(rows, list), response.text
    return tuple(object_value(row) for row in rows)


def listed_store(gateway: Gateway, vector_store_id: str, *, key: str | None = None) -> dict[str, JsonValue]:
    listed: Final = gateway.request("GET", "/vector_store/list", key=key)
    assert listed.status_code == 200, listed.text
    matches: Final = tuple(row for row in listed_rows(listed) if row["vector_store_id"] == vector_store_id)
    assert len(matches) == 1, f"{vector_store_id} appears {len(matches)} times in {listed.text}"
    return matches[0]


def listed_ids(gateway: Gateway) -> tuple[str, ...]:
    rows: Final = gateway.get("/vector_store/list")["data"]
    assert isinstance(rows, list)
    return tuple(str(object_value(row)["vector_store_id"]) for row in rows)


def config_store_info(gateway: Gateway) -> dict[str, JsonValue]:
    return object_value(gateway.post("/vector_store/info", {"vector_store_id": CONFIG_STORE_ID})["vector_store"])


def store_rows(vector_store_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT vector_store_id, vector_store_name FROM "LiteLLM_ManagedVectorStoresTable" WHERE vector_store_id = %s',
        (vector_store_id,),
    )


def assert_config_write_refused(gateway: Gateway) -> None:
    for path, body in (
        ("/vector_store/update", {"vector_store_id": CONFIG_STORE_ID, "vector_store_name": "renamed"}),
        ("/vector_store/delete", {"vector_store_id": CONFIG_STORE_ID}),
        ("/vector_store/new", {"vector_store_id": CONFIG_STORE_ID, "custom_llm_provider": "openai"}),
    ):
        refused = gateway.request("POST", path, body)
        assert refused.status_code == 400, f"{path}: {refused.status_code} {refused.text}"
        error = object_value(object_value(refused.json())["detail"])
        assert error["vector_store_id"] == CONFIG_STORE_ID, refused.text
        assert "config file" in str(error["error"]), refused.text


def burst_list(gateway: Gateway) -> tuple[int, str]:
    response: Final = gateway.request("GET", "/vector_store/list")
    if response.status_code != 200:
        return response.status_code, response.text
    ids: Final = tuple(str(row["vector_store_id"]) for row in listed_rows(response))
    return response.status_code, "config" if CONFIG_STORE_ID in ids else response.text


def burst_post(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[int, str]:
    response: Final = gateway.request("POST", path, body)
    return response.status_code, response.text


def upstream_requests(upstream: httpx.Client, marker: str) -> list[dict[str, JsonValue]]:
    observed: Final = upstream.get("/__observations")
    observed.raise_for_status()
    requests: Final = object_value(observed.json())["requests"]
    assert isinstance(requests, list), observed.text
    return [object_value(value) for value in requests if marker in str(object_value(value)["body"])]


@pytest.mark.covers("mgmt.vector_store.list.keeps_config_store_beside_db_stores")
def test_config_store_is_listed_beside_db_store_and_survives_listing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        db_store_id: Final = f"vs_db_{uuid.uuid4().hex}"
        gateway.post("/vector_store/new", {"vector_store_id": db_store_id, "custom_llm_provider": "openai"})
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": db_store_id})
        before: Final = config_store_info(gateway)
        assert before["vector_store_id"] == CONFIG_STORE_ID, before

        config_row: Final = listed_store(gateway, CONFIG_STORE_ID)
        assert config_row["is_config"] is True, config_row
        assert config_row["vector_store_name"] == CONFIG_STORE_NAME, config_row
        assert object_value(config_row["litellm_params"])["api_key"] != "integration-provider-key", config_row
        db_row: Final = listed_store(gateway, db_store_id)
        assert db_row["is_config"] is False, db_row

        after: Final = config_store_info(gateway)
        assert after["vector_store_id"] == CONFIG_STORE_ID, after
        assert after["is_config"] is True, after
        assert after["vector_store_description"] == "declared in tests/integration/proxy_config.yaml", after
        assert store_rows(CONFIG_STORE_ID) == [], "config store must not need a database row"
        assert listed_store(gateway, CONFIG_STORE_ID)["is_config"] is True


@pytest.mark.covers("mgmt.vector_store.write.config_store_is_read_only")
def test_config_store_refuses_new_update_and_delete(gateway: Gateway) -> None:
    assert_config_write_refused(gateway)
    row: Final = listed_store(gateway, CONFIG_STORE_ID)
    assert row["vector_store_name"] == CONFIG_STORE_NAME, row
    assert row["is_config"] is True, row
    assert config_store_info(gateway)["vector_store_name"] == CONFIG_STORE_NAME


@pytest.mark.covers("mgmt.vector_store.write.db_store_lifecycle_unchanged_beside_config_store")
def test_db_store_lifecycle_is_unchanged_beside_config_store(gateway: Gateway) -> None:
    incomplete: Final = gateway.request("POST", "/vector_store/new", {"custom_llm_provider": "openai"})
    assert incomplete.status_code == 400, incomplete.text
    db_store_id: Final = f"vs_db_{uuid.uuid4().hex}"
    created: Final = gateway.request(
        "POST",
        "/vector_store/new",
        {"vector_store_id": db_store_id, "custom_llm_provider": "openai", "vector_store_name": "first"},
    )
    assert created.status_code == 200, created.text
    assert store_rows(db_store_id) == [{"vector_store_id": db_store_id, "vector_store_name": "first"}]
    updated: Final = gateway.post(
        "/vector_store/update", {"vector_store_id": db_store_id, "vector_store_name": "second"}
    )
    assert object_value(updated["vector_store"])["vector_store_name"] == "second", updated
    assert store_rows(db_store_id) == [{"vector_store_id": db_store_id, "vector_store_name": "second"}]
    row: Final = listed_store(gateway, db_store_id)
    assert row["vector_store_name"] == "second" and row["is_config"] is False, row
    info: Final = object_value(gateway.post("/vector_store/info", {"vector_store_id": db_store_id})["vector_store"])
    assert info["vector_store_name"] == "second" and info["is_config"] is False, info
    gateway.post("/vector_store/delete", {"vector_store_id": db_store_id})
    assert store_rows(db_store_id) == []
    assert db_store_id not in listed_ids(gateway)
    assert CONFIG_STORE_ID in listed_ids(gateway)
    missing: Final = gateway.request("POST", "/vector_store/info", {"vector_store_id": db_store_id})
    assert missing.status_code == 404, missing.text


@pytest.mark.covers("other.vector_store.chat.config_store_search_reaches_upstream_after_listing")
def test_chat_with_config_store_searches_upstream_and_injects_context_after_listing(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        marker: Final = f"lit6337 {uuid.uuid4().hex}"
        assert CONFIG_STORE_ID in listed_ids(gateway)
        upstream.get("/__observations").raise_for_status()
        completion: Final = gateway.post(
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "vector_store_ids": [CONFIG_STORE_ID]},
        )
        assert object_value(completion["usage"])["total_tokens"] == 40, completion
        requests: Final = upstream_requests(upstream, marker)
        searches: Final = [value for value in requests if value["path"] == SEARCH_PATH]
        assert len(searches) == 1, requests
        assert object_value(searches[0]["body"])["query"] == marker, searches
        assert searches[0]["authorization"] == "Bearer integration-provider-key", searches
        chats: Final = [value for value in requests if value["path"] == "/v1/chat/completions"]
        assert len(chats) == 1, requests
        messages: Final = object_value(chats[0]["body"])["messages"]
        assert isinstance(messages, list), chats
        contents: Final = tuple(str(object_value(message)["content"]) for message in messages)
        assert contents == (f"Context:\n\nscripted context for {marker}\n\n", marker), contents


@pytest.mark.covers("other.vector_store.search.config_store_passthrough_uses_yaml_credentials_after_listing")
def test_passthrough_search_on_config_store_uses_yaml_credentials_after_listing(gateway: Gateway) -> None:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        marker: Final = f"lit6337 passthrough {uuid.uuid4().hex}"
        assert CONFIG_STORE_ID in listed_ids(gateway)
        upstream.get("/__observations").raise_for_status()
        searched: Final = gateway.request("POST", f"/v1/vector_stores/{CONFIG_STORE_ID}/search", {"query": marker})
        assert searched.status_code == 200, searched.text
        data: Final = listed_rows(searched)
        assert len(data) == 1, searched.text
        content: Final = data[0]["content"]
        assert isinstance(content, list), searched.text
        assert object_value(content[0])["text"] == f"scripted context for {marker}", searched.text
        requests: Final = upstream_requests(upstream, marker)
        assert [value["path"] for value in requests] == [SEARCH_PATH], requests
        assert requests[0]["authorization"] == "Bearer integration-provider-key", requests


@pytest.mark.covers("authz.vector_store.list.non_admin_key_access_to_config_store_follows_grants")
def test_non_admin_key_access_to_config_store_follows_grants_after_admin_listing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        granted: Final = scenario.key(object_permission={"vector_stores": [CONFIG_STORE_ID]})
        plain: Final = scenario.key()
        assert CONFIG_STORE_ID in listed_ids(gateway)
        row: Final = listed_store(gateway, CONFIG_STORE_ID, key=granted)
        assert row["is_config"] is True and row["vector_store_name"] == CONFIG_STORE_NAME, row
        unlisted: Final = gateway.request("GET", "/vector_store/list", key=plain)
        assert unlisted.status_code == 200, unlisted.text
        assert CONFIG_STORE_ID not in {value["vector_store_id"] for value in listed_rows(unlisted)}, unlisted.text
        for key in (granted, plain):
            info = gateway.request("POST", "/vector_store/info", {"vector_store_id": CONFIG_STORE_ID}, key=key)
            assert info.status_code == 200, info.text
            assert object_value(object_value(info.json())["vector_store"])["is_config"] is True, info.text
        forbidden: Final = gateway.request(
            "POST", "/vector_store/delete", {"vector_store_id": CONFIG_STORE_ID}, key=granted
        )
        assert forbidden.status_code in {400, 401, 403}, forbidden.text
        assert CONFIG_STORE_ID in listed_ids(gateway)


@pytest.mark.covers("mgmt.vector_store.list.peer_process_keeps_config_store_and_sees_db_store")
def test_peer_process_keeps_config_store_and_sees_db_store_created_elsewhere(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        db_store_id: Final = f"vs_db_{uuid.uuid4().hex}"
        gateway.post("/vector_store/new", {"vector_store_id": db_store_id, "custom_llm_provider": "openai"})
        scenario.cleanups.callback(gateway.request, "POST", "/vector_store/delete", {"vector_store_id": db_store_id})
        for side in (gateway, peer, gateway, peer):
            assert listed_store(side, CONFIG_STORE_ID)["is_config"] is True
            assert listed_store(side, db_store_id)["is_config"] is False
            assert config_store_info(side)["is_config"] is True
            assert_config_write_refused(side)
        gateway.post("/vector_store/delete", {"vector_store_id": db_store_id})
        assert db_store_id not in listed_ids(peer)
        assert CONFIG_STORE_ID in listed_ids(peer)


@pytest.mark.covers("mgmt.vector_store.chaos.concurrent_burst_keeps_config_store_across_workers")
def test_concurrent_burst_keeps_config_store_and_refuses_every_config_write(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        db_store_ids: Final = tuple(f"vs_db_{uuid.uuid4().hex}" for _ in range(6))
        for db_store_id in db_store_ids:
            scenario.cleanups.callback(
                gateway.request, "POST", "/vector_store/delete", {"vector_store_id": db_store_id}
            )

        def act(index: int) -> tuple[str, int, str]:
            match index % 5:
                case 0:
                    return ("list", *burst_list(gateway))
                case 1:
                    return ("info", *burst_post(gateway, "/vector_store/info", {"vector_store_id": CONFIG_STORE_ID}))
                case 2:
                    return (
                        "config-update",
                        *burst_post(
                            gateway,
                            "/vector_store/update",
                            {"vector_store_id": CONFIG_STORE_ID, "vector_store_name": str(index)},
                        ),
                    )
                case 3:
                    return (
                        "db-new",
                        *burst_post(
                            gateway,
                            "/vector_store/new",
                            {
                                "vector_store_id": db_store_ids[index % len(db_store_ids)],
                                "custom_llm_provider": "openai",
                            },
                        ),
                    )
                case _:
                    return (
                        "chat",
                        *burst_post(
                            gateway,
                            "/v1/chat/completions",
                            {
                                "model": model,
                                "messages": [{"role": "user", "content": f"burst {index}"}],
                                "vector_store_ids": [CONFIG_STORE_ID],
                            },
                        ),
                    )

        with ThreadPoolExecutor(max_workers=10) as pool:
            outcomes: Final = tuple(pool.map(act, range(30)))
        expected: Final = {"list": 200, "info": 200, "config-update": 400, "db-new": 200, "chat": 200}
        assert [(kind, status) for kind, status, _ in outcomes] == [
            (kind, expected[kind]) for kind, _, _ in outcomes
        ], outcomes
        assert all(detail == "config" for kind, _, detail in outcomes if kind == "list"), outcomes
        assert listed_store(gateway, CONFIG_STORE_ID)["vector_store_name"] == CONFIG_STORE_NAME
        assert config_store_info(gateway)["vector_store_name"] == CONFIG_STORE_NAME
        assert store_rows(CONFIG_STORE_ID) == []
        assert all(len(store_rows(db_store_id)) == 1 for db_store_id in db_store_ids), "each DB store exactly once"


@pytest.mark.timeout(180)
@pytest.mark.covers("mgmt.vector_store.chaos.redis_outage_keeps_config_store_and_recovers")
def test_redis_outage_keeps_config_store_served_and_recovers(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original: Final = os.environ["DATABASE_URL"]
    identity: Final = "integration_vs_outage_" + uuid.uuid4().hex
    parsed: Final = urlsplit(original)
    database_url: Final = urlunsplit((parsed.scheme, parsed.netloc, "/" + identity, "", ""))
    with psycopg.connect(original, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(identity)))
        try:
            with owned_redis(tmp_path) as cache, monkeypatch.context() as environment:
                environment.setenv("DATABASE_URL", database_url)
                overrides: Final = {
                    "DATABASE_URL": database_url,
                    "REDIS_HOST": cache.host,
                    "REDIS_PORT": str(cache.port),
                    "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "1",
                }
                with owned_proxy(gateway, tmp_path, overrides, config=PROXY_CONFIG, workers=2) as candidate:
                    db_store_id: Final = f"vs_db_{uuid.uuid4().hex}"
                    for phase in ("before", "during", "after"):
                        if phase == "during":
                            cache.stop()
                        if phase == "after":
                            cache.start()
                        for _ in range(4):
                            assert listed_store(candidate, CONFIG_STORE_ID)["is_config"] is True, phase
                            assert config_store_info(candidate)["vector_store_name"] == CONFIG_STORE_NAME, phase
                        assert_config_write_refused(candidate)
                        created = candidate.request(
                            "POST",
                            "/vector_store/new",
                            {"vector_store_id": f"{db_store_id}_{phase}", "custom_llm_provider": "openai"},
                        )
                        assert created.status_code == 200, (phase, created.text)
                        assert eventually(
                            lambda phase=phase: store_rows(f"{db_store_id}_{phase}"), lambda rows: len(rows) == 1
                        ), phase
                        assert f"{db_store_id}_{phase}" in listed_ids(candidate), phase
                    assert store_rows(CONFIG_STORE_ID) == []
                    with psycopg.connect(database_url) as fresh:
                        counted: Final = fresh.execute(
                            'SELECT count(*) FROM "LiteLLM_ManagedVectorStoresTable" WHERE vector_store_id LIKE %s',
                            (f"{db_store_id}%",),
                        ).fetchone()
                        assert counted is not None and counted[0] == 3, counted
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(identity)))
        assert admin.execute("SELECT datname FROM pg_database WHERE datname=%s", (identity,)).fetchall() == []
