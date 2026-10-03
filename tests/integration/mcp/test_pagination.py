import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import yaml
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, PaginatedRequestParams

from integration._support.client import Gateway
from integration._support.mcp import paginated_mcp_peer
from integration._support.process import owned_proxy
from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPTransport


@asynccontextmanager
async def catalog_session(gateway):
    async with httpx.AsyncClient(headers={"Authorization": "Bearer " + gateway.key}) as client:
        async with streamable_http_client(
            str(gateway.client.base_url).rstrip("/") + "/mcp/", http_client=client
        ) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session


def config_file(directory: Path, upstream: str) -> Path:
    config = directory / "proxy.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [],
                "mcp_servers": {"pages": {"url": upstream, "transport": "http"}},
                "general_settings": {"master_key": "sk-pagination-test", "store_model_in_db": False},
            }
        )
    )
    return config


REMOVE_DATABASE = ("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "LITELLM_LICENSE", "LITELLM_LICENSE_PATH")


def test_catalog_pages_are_portable_repeatable_and_complete(tmp_path: Path):
    async def exercise(first_replica, second_replica):
        for method, field in (
            ("list_tools", "tools"),
            ("list_prompts", "prompts"),
            ("list_resources", "resources"),
            ("list_resource_templates", "resource_templates"),
        ):
            async with catalog_session(first_replica) as first_session:
                first = await getattr(first_session, method)()
            assert first.next_cursor
            async with catalog_session(second_replica) as second_session:
                second = await getattr(second_session, method)(params=PaginatedRequestParams(cursor=first.next_cursor))
                replay = await getattr(second_session, method)(params=PaginatedRequestParams(cursor=first.next_cursor))
                assert getattr(second, field) == getattr(replay, field)
                assert second.next_cursor
                third = await getattr(second_session, method)(params=PaginatedRequestParams(cursor=second.next_cursor))
            names = [item.name for page in (first, second, third) for item in getattr(page, field)]
            assert len(names) == len(set(names)) == 3
            assert third.next_cursor is None
        async with catalog_session(second_replica) as session:
            called = await session.call_tool("pages-add2", {"a": 3, "b": 4})
            assert not called.is_error
            assert called.content[0].text == "7"

    with paginated_mcp_peer() as peer, httpx.Client() as client:
        seed = Gateway(client, "sk-pagination-test", peer.url)
        config = config_file(tmp_path, peer.url)
        options = dict(config=config, database_setup=(), remove_environment=REMOVE_DATABASE)
        environment = {
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": "true",
            "LITELLM_SALT_KEY": "shared-pagination-test",
        }
        with (
            owned_proxy(seed, tmp_path / "a", environment, **options) as a,
            owned_proxy(seed, tmp_path / "b", environment, **options) as b,
        ):
            asyncio.run(exercise(a, b))


@pytest.mark.parametrize("page_size", [1, 3])
def test_no_salt_preserves_complete_lists_and_calls_but_rejects_continuations(tmp_path: Path, page_size: int):
    async def exercise(gateway):
        async with catalog_session(gateway) as session:
            for method, field in (
                ("list_tools", "tools"),
                ("list_prompts", "prompts"),
                ("list_resources", "resources"),
                ("list_resource_templates", "resource_templates"),
            ):
                if page_size == 1:
                    with pytest.raises(MCPError, match="LITELLM_SALT_KEY"):
                        await getattr(session, method)()
                else:
                    result = await getattr(session, method)()
                    assert len(getattr(result, field)) == 3
                    assert result.next_cursor is None
            called = await session.send_request(
                CallToolRequest(params=CallToolRequestParams(name="pages-add2", arguments={"a": 3, "b": 4})),
                CallToolResult,
            )
            assert not called.is_error
            assert called.content[0].text == "7"

    with paginated_mcp_peer(page_size=page_size) as peer, httpx.Client() as client:
        seed = Gateway(client, "sk-pagination-test", peer.url)
        config = config_file(tmp_path, peer.url)
        environment = {"STORE_MODEL_IN_DB": "False", "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_SALT_KEY": ""}
        with owned_proxy(
            seed, tmp_path / "proxy", environment, config=config, database_setup=(), remove_environment=REMOVE_DATABASE
        ) as gateway:
            asyncio.run(exercise(gateway))


def test_continuations_reauthorize_and_reject_registry_changes(tmp_path: Path, monkeypatch):
    import os
    import subprocess
    import sys

    from integration._support.database import scratch_database
    from integration._support.mcp import register_mcp

    assert os.environ.get("DATABASE_URL"), "This integration case requires disposable-database access"

    async def exercise(a, b, peer, identity, owner, stranger):
        owner_a = Gateway(a.client, owner, peer.url)
        owner_b = Gateway(b.client, owner, peer.url)
        stranger_b = Gateway(b.client, stranger, peer.url)
        first_pages = {}
        for method in ("list_tools", "list_prompts", "list_resources", "list_resource_templates"):
            async with catalog_session(owner_a) as session:
                first_pages[method] = await getattr(session, method)()
                assert first_pages[method].next_cursor
            async with catalog_session(owner_b) as session:
                continued = await getattr(session, method)(
                    params=PaginatedRequestParams(cursor=first_pages[method].next_cursor)
                )
                assert continued.next_cursor
            async with catalog_session(stranger_b) as session:
                peer.drain()
                with pytest.raises(MCPError, match="fresh listing"):
                    await getattr(session, method)(
                        params=PaginatedRequestParams(cursor=first_pages[method].next_cursor)
                    )
                assert not any(call["body"].get("method", "").endswith("/list") for call in peer.drain())

        a.post("/key/update", {"key": owner, "object_permission": {"mcp_servers": ["no-mcp-servers"]}})
        for method, first in first_pages.items():
            async with catalog_session(owner_b) as session:
                peer.drain()
                with pytest.raises(MCPError, match="fresh listing"):
                    await getattr(session, method)(params=PaginatedRequestParams(cursor=first.next_cursor))
                assert not any(call["body"].get("method", "").endswith("/list") for call in peer.drain())
        a.post("/key/update", {"key": owner, "object_permission": {"mcp_servers": [identity]}})
        changed = a.request("PUT", "/v1/mcp/server", {"server_id": identity, "description": "new catalog generation"})
        assert changed.status_code == 202, changed.text
        for method, first in first_pages.items():
            async with catalog_session(owner_b) as session:
                peer.drain()
                with pytest.raises(MCPError, match="fresh listing"):
                    await getattr(session, method)(params=PaginatedRequestParams(cursor=first.next_cursor))
                assert not any(call["body"].get("method", "").endswith("/list") for call in peer.drain())
                fresh = await getattr(session, method)()
                assert fresh.next_cursor

    with scratch_database() as database_url:
        monkeypatch.setenv("DATABASE_URL", database_url)
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "prisma",
                "db",
                "push",
                "--schema",
                "litellm/proxy/schema.prisma",
                "--skip-generate",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with paginated_mcp_peer() as peer, httpx.Client() as client:
            seed = Gateway(client, "sk-pagination-test", peer.url)
            config = tmp_path / "database-proxy.yaml"
            config.write_text(
                yaml.safe_dump(
                    {"model_list": [], "general_settings": {"master_key": seed.key, "store_model_in_db": True}}
                )
            )
            environment = {
                "DATABASE_URL": database_url,
                "DISABLE_SCHEMA_UPDATE": "true",
                "LITELLM_SALT_KEY": "shared-pagination-test",
            }
            options = dict(
                config=config,
                database_setup=(),
                remove_environment=("DATABASE_URL_READ_REPLICA", "LITELLM_LICENSE", "LITELLM_LICENSE_PATH"),
            )
            with (
                owned_proxy(seed, tmp_path / "a", environment, **options) as a,
                owned_proxy(seed, tmp_path / "b", environment, **options) as b,
                a.scenario() as scenario,
            ):
                identity = register_mcp(scenario, peer, "pages")
                owner = scenario.key(object_permission={"mcp_servers": [identity]})
                stranger = scenario.key(object_permission={"mcp_servers": [identity]})
                assert owner != stranger
                asyncio.run(exercise(a, b, peer, identity, owner, stranger))


@pytest.mark.parametrize("changed", ["key", "snapshot"])
def test_cursor_rejects_changed_replica_configuration_before_dispatch(tmp_path: Path, changed: str):
    async def exercise(a, b, peer):
        for method in ("list_tools", "list_prompts", "list_resources", "list_resource_templates"):
            async with catalog_session(a) as session:
                first = await getattr(session, method)()
                assert first.next_cursor
            async with catalog_session(b) as session:
                peer.drain()
                with pytest.raises(MCPError, match="fresh listing"):
                    await getattr(session, method)(params=PaginatedRequestParams(cursor=first.next_cursor))
                assert not any(call["body"].get("method", "").endswith("/list") for call in peer.drain())
                fresh = await getattr(session, method)()
                assert fresh.next_cursor

    with paginated_mcp_peer() as peer, httpx.Client() as client:
        seed = Gateway(client, "sk-pagination-test", peer.url)
        config = config_file(tmp_path, peer.url)
        second_config = tmp_path / "second-proxy.yaml"
        second_values = yaml.safe_load(config.read_text())
        if changed == "snapshot":
            second_values["mcp_servers"]["pages"]["description"] = "changed registry definition"
        second_config.write_text(yaml.safe_dump(second_values))
        environment = {
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": "true",
            "LITELLM_SALT_KEY": "original-key",
        }
        second_environment = {**environment, "LITELLM_SALT_KEY": "rotated-key" if changed == "key" else "original-key"}
        options = dict(database_setup=(), remove_environment=REMOVE_DATABASE)
        with (
            owned_proxy(seed, tmp_path / "a", environment, config=config, **options) as a,
            owned_proxy(seed, tmp_path / "b", second_environment, config=second_config, **options) as b,
        ):
            asyncio.run(exercise(a, b, peer))


def test_pages_preserve_supported_protocol_versions(tmp_path: Path):
    from mcp.types import ListPromptsRequest, ListResourcesRequest, ListResourceTemplatesRequest, ListToolsRequest
    from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS

    async def exercise(gateway):
        for revision in HANDSHAKE_PROTOCOL_VERSIONS:
            client = MCPClient(
                server_url=str(gateway.client.base_url).rstrip("/") + "/mcp",
                transport_type=MCPTransport.http,
                protocol_version=revision,
                extra_headers={"Authorization": "Bearer " + gateway.key},
                timeout=15,
            )
            for request, field in (
                (ListToolsRequest, "tools"),
                (ListPromptsRequest, "prompts"),
                (ListResourcesRequest, "resources"),
                (ListResourceTemplatesRequest, "resource_templates"),
            ):
                first = await client.list_page(request())
                assert first.next_cursor, revision
                second = await client.list_page(request(params=PaginatedRequestParams(cursor=first.next_cursor)))
                assert second.next_cursor, revision
                third = await client.list_page(request(params=PaginatedRequestParams(cursor=second.next_cursor)))
                assert third.next_cursor is None, revision
                names = [item.name for page in (first, second, third) for item in getattr(page, field)]
                assert len(names) == len(set(names)) == 3, revision

    with paginated_mcp_peer() as peer, httpx.Client() as client:
        seed = Gateway(client, "sk-pagination-test", peer.url)
        config = config_file(tmp_path, peer.url)
        environment = {"STORE_MODEL_IN_DB": "False", "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_SALT_KEY": "shared-key"}
        with owned_proxy(
            seed, tmp_path / "proxy", environment, config=config, database_setup=(), remove_environment=REMOVE_DATABASE
        ) as gateway:
            asyncio.run(exercise(gateway))


def test_incomplete_discovery_never_establishes_a_bare_tool_route(tmp_path: Path):
    from integration._support.mcp import tool_calls

    async def exercise(gateway, peer):
        async with catalog_session(gateway) as session:
            first = await session.list_tools()
            assert [tool.name for tool in first.tools] == ["pages-add0"]
            assert first.next_cursor
            with pytest.raises(MCPError, match="repeated"):
                await session.list_tools(params=PaginatedRequestParams(cursor=first.next_cursor))
            peer.drain()
            result = await session.send_request(
                CallToolRequest(params=CallToolRequestParams(name="add0", arguments={"a": 3, "b": 4})),
                CallToolResult,
            )
            assert result.is_error
            assert tool_calls(peer.drain()) == ()

    with paginated_mcp_peer(repeat_cursor=True) as peer, httpx.Client() as client:
        seed = Gateway(client, "sk-pagination-test", peer.url)
        config = config_file(tmp_path, peer.url)
        environment = {"STORE_MODEL_IN_DB": "False", "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_SALT_KEY": "shared-key"}
        with owned_proxy(
            seed, tmp_path / "proxy", environment, config=config, database_setup=(), remove_environment=REMOVE_DATABASE
        ) as gateway:
            asyncio.run(exercise(gateway, peer))


def test_partial_catalog_keeps_sanitized_failure_metadata_on_following_pages(tmp_path: Path):
    async def exercise(gateway):
        for method, field in (
            ("list_tools", "tools"),
            ("list_prompts", "prompts"),
            ("list_resources", "resources"),
            ("list_resource_templates", "resource_templates"),
        ):
            async with catalog_session(gateway) as session:
                first = await getattr(session, method)()
                assert first.next_cursor
                assert len(getattr(first, field)) == 1
                fault = first.meta["litellm.ai/server_outcomes"]["broken"]
                assert fault["status"] != "ok"
                assert "tag" not in fault
                assert "untrusted upstream message" not in first.model_dump_json()
                second = await getattr(session, method)(params=PaginatedRequestParams(cursor=first.next_cursor))
                assert len(getattr(second, field)) == 1
                assert second.meta["litellm.ai/server_outcomes"]["broken"] == fault

    with paginated_mcp_peer() as healthy, paginated_mcp_peer(fail_listing=True) as broken, httpx.Client() as client:
        seed = Gateway(client, "sk-pagination-test", healthy.url)
        config = config_file(tmp_path, healthy.url)
        values = yaml.safe_load(config.read_text())
        values["mcp_servers"]["broken"] = {"url": broken.url, "transport": "http"}
        config.write_text(yaml.safe_dump(values))
        environment = {"STORE_MODEL_IN_DB": "False", "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_SALT_KEY": "shared-key"}
        with owned_proxy(
            seed, tmp_path / "proxy", environment, config=config, database_setup=(), remove_environment=REMOVE_DATABASE
        ) as gateway:
            asyncio.run(exercise(gateway))


def test_failed_upstream_continuation_requires_restart_for_every_catalog(tmp_path: Path):
    async def exercise(gateway):
        async with catalog_session(gateway) as session:
            for method in ("list_tools", "list_prompts", "list_resources", "list_resource_templates"):
                first = await getattr(session, method)()
                assert first.next_cursor
                with pytest.raises(MCPError, match="fresh listing") as caught:
                    await getattr(session, method)(params=PaginatedRequestParams(cursor=first.next_cursor))
                assert "untrusted upstream message" not in str(caught.value)

    with paginated_mcp_peer(fail_continuation=True) as peer, httpx.Client() as client:
        seed = Gateway(client, "sk-pagination-test", peer.url)
        config = config_file(tmp_path, peer.url)
        environment = {"STORE_MODEL_IN_DB": "False", "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_SALT_KEY": "shared-key"}
        with owned_proxy(
            seed, tmp_path / "proxy", environment, config=config, database_setup=(), remove_environment=REMOVE_DATABASE
        ) as gateway:
            asyncio.run(exercise(gateway))
