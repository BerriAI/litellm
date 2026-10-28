import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import scratch_database
from integration._support.mcp import McpCaller, McpPeer, mcp_peer, paginated_mcp_peer, register_mcp, tool_calls
from integration._support.process import owned_proxy
from integration._support.redis_process import OwnedRedis, owned_redis
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ListToolsResult, PaginatedRequestParams

REMOVE_DATABASE: Final = ("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "LITELLM_LICENSE", "LITELLM_LICENSE_PATH")


@asynccontextmanager
async def _catalog_session(gateway: Gateway) -> AsyncIterator[ClientSession]:
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=15,
        trust_env=False,
    ) as client:
        async with streamable_http_client(
            f"{str(gateway.client.base_url).rstrip('/')}/mcp/",
            http_client=client,
        ) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session


async def _list_tools(gateway: Gateway, cursor: str | None = None) -> ListToolsResult | MCPError:
    async with _catalog_session(gateway) as session:
        try:
            if cursor is None:
                return await session.list_tools()
            return await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
        except MCPError as error:
            return error


def _tool_items(result: ListToolsResult) -> tuple[dict[str, object], ...]:
    return tuple(tool.model_dump(mode="json") for tool in result.tools)


def _has_method(calls: tuple[dict[str, object], ...], method: str) -> bool:
    return any(
        isinstance(call.get("body"), dict) and isinstance(call["body"], dict) and call["body"].get("method") == method
        for call in calls
    )


def _config_file(
    directory: Path,
    master_key: str,
    redis: OwnedRedis,
    *,
    upstream: str | None = None,
    rpm: int | None = None,
    allowed_tools: tuple[str, ...] = (),
    store_model_in_db: bool,
) -> Path:
    config: dict[str, object] = {
        "model_list": [],
        "general_settings": {
            "master_key": master_key,
            "store_model_in_db": store_model_in_db,
            "coordination_redis": {"host": redis.host, "port": redis.port},
        },
    }
    if upstream is not None:
        server: dict[str, object] = {"url": upstream, "transport": "http", "rpm": rpm}
        if allowed_tools:
            server["allowed_tools"] = list(allowed_tools)
        config["mcp_servers"] = {"rpm": server}
    path: Final = directory / "proxy.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _call_tool(gateway: Gateway, key: str, server_id: str, name: str) -> httpx.Response:
    return gateway.client.post(
        "/mcp-rest/tools/call",
        headers={"x-litellm-api-key": key},
        json={"name": name, "arguments": {"a": 1, "b": 2}, "server_id": server_id},
    )


def _assert_rate_limit(response: httpx.Response, descriptor: str) -> None:
    assert response.status_code == 429, response.text
    assert descriptor in response.text


def test_shared_redis_enforces_paginated_tools_and_rest_listings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_directory: Final = tmp_path / "results"
    results_directory.mkdir()
    monkeypatch.setenv("INTEGRATION_RESULTS_DIR", str(results_directory))

    async def exercise(
        first_replica: Gateway, second_replica: Gateway, peer: McpPeer
    ) -> tuple[str, tuple[dict[str, object], ...]]:
        first_page: Final = await _list_tools(first_replica)
        assert isinstance(first_page, ListToolsResult)
        assert first_page.next_cursor is not None
        first_cursor: Final = first_page.next_cursor

        continued_page: Final = await _list_tools(second_replica, first_cursor)
        assert isinstance(continued_page, ListToolsResult)
        continued_items: Final = _tool_items(continued_page)
        assert continued_items

        peer.drain()
        rejected: Final = await _list_tools(first_replica, first_cursor)
        assert isinstance(rejected, MCPError)
        assert "mcp_server" in str(rejected)
        assert not _has_method(peer.drain(), "tools/list")

        peer.drain()
        rest_rejected: Final = second_replica.client.get(
            "/mcp-rest/tools/list",
            headers={"x-litellm-api-key": second_replica.key},
            params={"server_id": "rpm"},
        )
        assert rest_rejected.status_code == 429, rest_rejected.text
        assert not _has_method(peer.drain(), "tools/list")

        return first_cursor, continued_items

    with paginated_mcp_peer(page_size=1) as peer, owned_redis(tmp_path) as redis, httpx.Client() as client:
        seed: Final = Gateway(client, "sk-mcp-pagination-rate-limit", peer.url)
        config: Final = _config_file(
            tmp_path,
            seed.key,
            redis,
            upstream=peer.url,
            rpm=2,
            store_model_in_db=False,
        )
        environment: Final = {
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": "true",
            "LITELLM_SALT_KEY": "shared-mcp-pagination-rate-limit",
            "LITELLM_RATE_LIMIT_WINDOW_SIZE": "10",
        }
        options: Final = {
            "config": config,
            "database_setup": (),
            "remove_environment": REMOVE_DATABASE,
        }
        with (
            owned_proxy(seed, tmp_path / "first", environment, **options) as first_replica,
            owned_proxy(seed, tmp_path / "second", environment, **options) as second_replica,
        ):
            first_cursor, continued_items = asyncio.run(exercise(first_replica, second_replica, peer))

            def retry() -> tuple[dict[str, object], ...] | None:
                result: Final = asyncio.run(_list_tools(second_replica, first_cursor))
                return _tool_items(result) if isinstance(result, ListToolsResult) else None

            retried_items: Final = eventually(retry, lambda items: items is not None, seconds=30)
            assert retried_items == continued_items


def test_mcp_key_team_and_server_rpm_limits_share_redis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_directory: Final = tmp_path / "results"
    results_directory.mkdir()
    monkeypatch.setenv("INTEGRATION_RESULTS_DIR", str(results_directory))
    assert os.environ.get("DATABASE_URL"), "This integration case requires disposable-database access"

    with (
        owned_redis(tmp_path) as redis,
        scratch_database() as database_url,
        mcp_peer() as peer,
        httpx.Client() as client,
    ):
        monkeypatch.setenv("DATABASE_URL", database_url)
        seed: Final = Gateway(client, "sk-mcp-key-team-server-rate-limit", peer.url)
        config: Final = _config_file(
            tmp_path,
            seed.key,
            redis,
            store_model_in_db=True,
        )
        environment: Final = {
            "DATABASE_URL": database_url,
            "LITELLM_SALT_KEY": "shared-mcp-key-team-server-rate-limit",
            "LITELLM_RATE_LIMIT_WINDOW_SIZE": "30",
        }
        second_environment: Final = {**environment, "DISABLE_SCHEMA_UPDATE": "true"}
        options: Final = {
            "config": config,
            "remove_environment": ("DATABASE_URL_READ_REPLICA", "LITELLM_LICENSE", "LITELLM_LICENSE_PATH"),
        }
        with (
            owned_proxy(seed, tmp_path / "first", environment, **options) as first_replica,
            first_replica.scenario() as scenario,
        ):
            server_id: Final = register_mcp(
                scenario,
                peer,
                "rpm",
                rpm=5,
                allowed_tools=["add"],
            )
            permission: Final = {"mcp_servers": [server_id]}
            team_id: Final = scenario.team(
                mcp_rpm_limit={"rpm": 3},
                object_permission=permission,
            )
            key_one: Final = scenario.key(
                team_id=team_id,
                mcp_rpm_limit={"rpm": 1},
                object_permission=permission,
            )
            key_two: Final = scenario.key(team_id=team_id, object_permission=permission)
            key_three: Final = scenario.key(object_permission=permission)
            key_four: Final = scenario.key(rpm_limit=1, object_permission=permission)

            with owned_proxy(
                seed, tmp_path / "second", second_environment, database_setup=(), **options
            ) as second_replica:
                first_call: Final = _call_tool(first_replica, key_one, server_id, "rpm-add")
                assert first_call.status_code == 200, first_call.text

                peer.drain()
                key_one_rejected: Final = _call_tool(second_replica, key_one, server_id, "rpm-add")
                _assert_rate_limit(key_one_rejected, "mcp_per_key")
                assert tool_calls(peer.drain()) == ()

                second_call: Final = _call_tool(first_replica, key_two, server_id, "rpm-add")
                assert second_call.status_code == 200, second_call.text
                third_call: Final = _call_tool(second_replica, key_two, server_id, "rpm-add")
                assert third_call.status_code == 200, third_call.text

                peer.drain()
                key_two_rejected: Final = _call_tool(first_replica, key_two, server_id, "rpm-add")
                _assert_rate_limit(key_two_rejected, "mcp_per_team")
                assert tool_calls(peer.drain()) == ()

                key_four_second_replica: Final = McpCaller(second_replica, key_four, "mcp")
                key_four_first_replica: Final = McpCaller(first_replica, key_four, "mcp")
                key_four_first_call: Final = key_four_second_replica.call("rpm-add", {"a": 1, "b": 2})
                assert key_four_first_call.ok, key_four_first_call.raw

                peer.drain()
                key_four_rejected: Final = key_four_first_replica.call("rpm-add", {"a": 1, "b": 2})
                assert not key_four_rejected.ok, key_four_rejected.raw
                assert "api_key" in (key_four_rejected.error or "")
                assert tool_calls(peer.drain()) == ()

                peer.drain()
                forbidden: Final = _call_tool(second_replica, key_three, server_id, "rpm-multiply")
                assert forbidden.status_code == 403, forbidden.text
                assert tool_calls(peer.drain()) == ()

                key_three_first_call: Final = _call_tool(first_replica, key_three, server_id, "rpm-add")
                assert key_three_first_call.status_code == 200, key_three_first_call.text

                peer.drain()
                server_rejected: Final = _call_tool(second_replica, key_three, server_id, "rpm-add")
                _assert_rate_limit(server_rejected, "mcp_server")
                assert tool_calls(peer.drain()) == ()
