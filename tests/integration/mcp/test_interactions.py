import json
import os
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue

from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


_KEY: Final = "sk-interaction-test"
_PROXY_PYTHONPATH: Final = os.pathsep.join(
    (str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[2]))
)
_META: Final = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "continuation-test", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {"elicitation": {"form": {}, "url": {}}},
}


def interaction_peer(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(status=405)
    body: Final = json.loads(request.body)
    method: Final = body["method"]
    params: Final = body.get("params", {})
    if method == "server/discover":
        result = {
            "supportedVersions": ["2026-07-28"],
            "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
            "cacheScope": "private",
            "ttlMs": 0,
        }
    elif method == "tools/list":
        result = {
            "tools": [{"name": "confirm", "inputSchema": {"type": "object"}}],
            "cacheScope": "private",
            "ttlMs": 0,
        }
    elif method == "prompts/list":
        result = {"prompts": [{"name": "confirm"}], "cacheScope": "private", "ttlMs": 0}
    elif method == "resources/list":
        result = {"resources": [{"name": "confirm", "uri": "test://confirm"}], "cacheScope": "private", "ttlMs": 0}
    elif method == "resources/templates/list":
        result = {"resourceTemplates": [], "cacheScope": "private", "ttlMs": 0}
    elif not params.get("requestState"):
        return Reply(
            body=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "resultType": "input_required",
                        "requestState": "opaque:" + method,
                        "inputRequests": {
                            "consent": {
                                "method": "elicitation/create",
                                "params": {
                                    "mode": "form",
                                    "message": "Confirm",
                                    "requestedSchema": {"type": "object", "properties": {}},
                                },
                            }
                        },
                    },
                }
            ).encode()
        )
    else:
        assert params["requestState"] == "opaque:" + method
        assert params["inputResponses"] == {"consent": {"action": "accept"}}
        if method == "tools/call":
            result = {"content": [{"type": "text", "text": "confirmed"}], "isError": False}
        elif method == "prompts/get":
            result = {"messages": [{"role": "user", "content": {"type": "text", "text": "confirmed"}}]}
        else:
            assert method == "resources/read"
            result = {"contents": [{"uri": "test://confirm", "text": "confirmed"}]}
    return Reply(
        body=json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"resultType": "complete", **result}}).encode()
    )


def rpc(gateway: Gateway, method: str, params: dict[str, JsonValue], *, status: int = 200) -> dict:
    response: Final = gateway.client.post(
        "/mcp/",
        headers={
            "Authorization": "Bearer " + gateway.key,
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": method,
            "Mcp-Name": str(params.get("name", params.get("uri", ""))),
            "Accept": "application/json, text/event-stream",
        },
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": {**params, "_meta": _META}},
    )
    assert response.status_code == status, response.text
    return response.json()


@pytest.mark.parametrize("changed_target", [False, True])
def test_continuations_resume_on_another_replica_and_reject_changed_operations(
    tmp_path: Path, changed_target: bool
) -> None:
    with wire_server(interaction_peer) as peer, httpx.Client() as client:
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {
                        "mrtr": {
                            "url": peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_elicitation": True,
                        }
                    },
                    "general_settings": {
                        "master_key": _KEY,
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2025-11-25", "2026-07-28"],
                    },
                }
            )
        )
        other_config: Final = tmp_path / "other-proxy.yaml"
        other_config.write_text(
            config.read_text().replace(peer.url, peer.url + "/changed-target") if changed_target else config.read_text()
        )
        seed: Final = Gateway(client, _KEY, peer.url)
        environment: Final = {
            "PYTHONPATH": _PROXY_PYTHONPATH,
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": "true",
            "LITELLM_SALT_KEY": "shared-interaction-test",
        }
        options: Final = {
            "database_setup": (),
            "remove_environment": (
                "DATABASE_URL",
                "DATABASE_URL_READ_REPLICA",
                "LITELLM_LICENSE",
                "LITELLM_LICENSE_PATH",
                "REDIS_URL",
                "REDIS_HOST",
            ),
        }
        with (
            owned_proxy(seed, tmp_path / "a", environment, config=config, **options) as first,
            owned_proxy(seed, tmp_path / "b", environment, config=other_config, **options) as second,
        ):
            for method, params, terminal_field in (
                ("tools/call", {"name": "mrtr-confirm", "arguments": {}}, "content"),
                ("prompts/get", {"name": "mrtr-confirm", "arguments": {}}, "messages"),
                ("resources/read", {"uri": "test://confirm"}, "contents"),
            ):
                initial: Final = rpc(first, method, params)
                assert initial["result"]["resultType"] == "input_required", initial
                state: Final = initial["result"]["requestState"]
                assert state.startswith("mcp_state_v1."), initial
                retry: Final = {**params, "requestState": state, "inputResponses": {"consent": {"action": "accept"}}}
                if changed_target:
                    peer.drain()
                    refused: Final = rpc(second, method, retry, status=400)
                    assert refused["error"]["code"] == -32602, refused
                    assert peer.drain() == (), "Changed target must reject before upstream dispatch"
                    continue
                completed: Final = rpc(second, method, retry)
                assert "confirmed" in json.dumps(completed["result"][terminal_field]), completed
                assert rpc(first, method, retry) == completed
                peer.drain()
                changed: Final = {
                    **retry,
                    **({"uri": "test://other"} if method == "resources/read" else {"name": "mrtr-other"}),
                }
                rejected: Final = rpc(second, method, changed, status=400)
                assert rejected["error"]["code"] == -32602, rejected
                assert peer.drain() == (), "Rejected continuation must not contact the upstream"


def test_continuation_reauthenticates_caller_and_rechecks_revoked_permissions(tmp_path: Path) -> None:
    auth_module: Final = tmp_path / "interaction_auth.py"
    auth_module.write_text(
        "from pathlib import Path\n"
        "from fastapi import HTTPException, Request\n"
        "from litellm.proxy._types import UserAPIKeyAuth\n"
        "async def authenticate(request: Request, api_key: str) -> UserAPIKeyAuth:\n"
        "    if api_key != 'sk-interaction-test':\n"
        "        raise HTTPException(status_code=401, detail='Unknown test caller')\n"
        "    return UserAPIKeyAuth.model_validate_json(Path(__file__).with_suffix('.json').read_text())\n"
    )
    identity: Final = {
        "user_id": "alice",
        "team_id": "team-a",
        "user_role": "internal_user",
        "object_permission": {"object_permission_id": "test-permission", "mcp_servers": ["interaction-server"]},
    }
    auth_state: Final = auth_module.with_suffix(".json")
    auth_state.write_text(json.dumps(identity))
    with wire_server(interaction_peer) as peer, wire_server(interaction_peer) as other_peer, httpx.Client() as client:
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {
                        "mrtr": {
                            "server_id": "interaction-server",
                            "url": peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_elicitation": True,
                        },
                        "other": {
                            "server_id": "other-server",
                            "url": other_peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_elicitation": True,
                        },
                    },
                    "general_settings": {
                        "master_key": _KEY,
                        "custom_auth": "interaction_auth.authenticate",
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2025-11-25", "2026-07-28"],
                    },
                }
            )
        )
        with owned_proxy(
            Gateway(client, _KEY, peer.url),
            tmp_path / "proxy",
            {
                "PYTHONPATH": _PROXY_PYTHONPATH,
                "STORE_MODEL_IN_DB": "False",
                "DISABLE_SCHEMA_UPDATE": "true",
                "LITELLM_SALT_KEY": "caller-test-salt",
            },
            config=config,
            database_setup=(),
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "REDIS_URL", "REDIS_HOST"),
        ) as gateway:
            params: Final = {"name": "mrtr-confirm", "arguments": {}}
            initial: Final = rpc(gateway, "tools/call", params)
            assert initial["result"]["resultType"] == "input_required", initial
            retry: Final = {
                **params,
                "requestState": initial["result"]["requestState"],
                "inputResponses": {"consent": {"action": "accept"}},
            }
            for changed in ({"user_id": "bob"}, {"team_id": "team-b"}):
                auth_state.write_text(json.dumps({**identity, **changed}))
                peer.drain()
                rejected: Final = rpc(gateway, "tools/call", retry, status=400)
                assert rejected["error"]["code"] == -32602, rejected
                assert peer.drain() == (), "Caller-bound state must reject before contacting upstream"
            auth_state.write_text(json.dumps(identity))
            resumed: Final = rpc(gateway, "tools/call", retry)
            assert resumed["result"]["content"] == [{"type": "text", "text": "confirmed"}], resumed
            auth_state.write_text(
                json.dumps(
                    {
                        **identity,
                        "object_permission": {
                            "object_permission_id": "test-permission",
                            "mcp_servers": ["denied-server"],
                        },
                    }
                )
            )
            peer.drain()
            revoked: Final = rpc(gateway, "tools/call", retry, status=403)
            assert revoked["detail"] == "MCP continuation target is no longer authorized", revoked
            assert peer.drain() == (), "Revoked access must reject a valid continuation before upstream dispatch"

            auth_state.write_text(json.dumps(identity))
            resource: Final = rpc(gateway, "resources/read", {"uri": "test://confirm"})
            assert resource["result"]["resultType"] == "input_required", resource
            auth_state.write_text(
                json.dumps(
                    {
                        **identity,
                        "object_permission": {
                            "object_permission_id": "test-permission",
                            "mcp_servers": ["other-server"],
                        },
                    }
                )
            )
            peer.drain()
            other_peer.drain()
            response: Final = gateway.client.post(
                "/mcp/",
                headers={
                    "Authorization": "Bearer " + gateway.key,
                    "MCP-Protocol-Version": "2026-07-28",
                    "Mcp-Method": "resources/read",
                    "Mcp-Name": "test://confirm",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "resources/read",
                    "params": {
                        "uri": "test://confirm",
                        "_meta": _META,
                        "requestState": resource["result"]["requestState"],
                        "inputResponses": {"consent": {"action": "accept"}},
                    },
                },
            )
            other_requests: Final = tuple(
                (json.loads(request.body)["method"], json.loads(request.body).get("params", {}).get("requestState"))
                for request in other_peer.drain()
            )
            assert other_requests == (), "Continuation must never send upstream state to another authorized server"
            assert peer.drain() == (), "Revoked original target must not receive a retry"
            assert response.status_code >= 400 or "error" in response.json(), response.text

            auth_state.write_text(
                json.dumps(
                    {
                        **identity,
                        "object_permission": {
                            "object_permission_id": "test-permission",
                            "mcp_servers": ["interaction-server", "other-server"],
                        },
                    }
                )
            )
            completed: Final = rpc(
                gateway,
                "resources/read",
                {
                    "uri": "test://confirm",
                    "requestState": resource["result"]["requestState"],
                    "inputResponses": {"consent": {"action": "accept"}},
                },
            )
            assert completed["result"]["contents"] == [{"uri": "test://confirm", "text": "confirmed"}], completed
            assert other_peer.drain() == (), "Expanded access must keep the continuation on its original target"


def test_missing_continuation_key_reports_configuration_for_each_carrier(tmp_path: Path) -> None:
    with wire_server(interaction_peer) as peer, httpx.Client() as client:
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {
                        "mrtr": {
                            "url": peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_elicitation": True,
                        }
                    },
                    "general_settings": {
                        "master_key": _KEY,
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2025-11-25", "2026-07-28"],
                    },
                }
            )
        )
        with owned_proxy(
            Gateway(client, _KEY, peer.url),
            tmp_path / "proxy",
            {
                "PYTHONPATH": _PROXY_PYTHONPATH,
                "STORE_MODEL_IN_DB": "False",
                "DISABLE_SCHEMA_UPDATE": "true",
                "LITELLM_SALT_KEY": "",
            },
            config=config,
            database_setup=(),
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "REDIS_URL", "REDIS_HOST"),
        ) as gateway:
            for method, params in (
                ("tools/call", {"name": "mrtr-confirm", "arguments": {}}),
                ("prompts/get", {"name": "mrtr-confirm", "arguments": {}}),
                ("resources/read", {"uri": "test://confirm"}),
            ):
                response: Final = rpc(gateway, method, params, status=400)
                assert response["error"]["code"] == -32602, response
                assert "LITELLM_SALT_KEY" in response["error"]["message"], response


@pytest.mark.parametrize(
    "event_kind,access",
    [
        ("tools", "normal"),
        ("prompts", "normal"),
        ("resources", "normal"),
        ("updated", "normal"),
        ("tools", "revoked"),
        ("tools", "identity"),
        ("tools", "idle"),
        ("tools", "hidden"),
        ("tools", "upstream-ended"),
        ("tools", "legacy"),
        ("tools", "pages"),
        ("tools", "fanout"),
    ],
)
def test_modern_subscription_delivers_changes_between_calls_and_refreshes_tools(
    tmp_path: Path, event_kind: str, access: str
) -> None:
    import asyncio
    from contextlib import asynccontextmanager

    import httpx2
    from mcp import ClientSession, MCPError
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.subscriptions import listen, SubscriptionLost
    from mcp.server.mcpserver import MCPServer
    from mcp.server.subscriptions import (
        InMemorySubscriptionBus,
        ListenHandler,
        ToolsListChanged,
        PromptsListChanged,
        ResourcesListChanged,
        ResourceUpdated,
    )
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import DiscoverResult, ListToolsResult, PaginatedRequestParams, SubscriptionsListenRequestParams
    from sse_starlette.sse import AppStatus
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from integration._support.asgi import asgi_server
    from integration._support.mcp import _drain_sse_streams

    from threading import Event
    from collections.abc import Callable
    from mcp.shared.subscriptions import ServerEvent

    disconnected: Final = Event()
    connected: Final = Event()
    listener_credentials = []

    class ObservedBus(InMemorySubscriptionBus):
        def subscribe(self, listener: Callable[[ServerEvent], None]) -> Callable[[], None]:
            unsubscribe = super().subscribe(listener)
            connected.set()

            def close() -> None:
                unsubscribe()
                disconnected.set()

            return close

    auth_module: Final = tmp_path / "subscription_auth.py"
    auth_module.write_text(
        "from pathlib import Path\n"
        "from fastapi import HTTPException, Request\n"
        "from litellm.proxy._types import UserAPIKeyAuth\n"
        "async def authenticate(request: Request, api_key: str) -> UserAPIKeyAuth:\n"
        "    if api_key != 'sk-subscriptions-test':\n"
        "        raise HTTPException(status_code=401, detail='Unknown caller')\n"
        "    return UserAPIKeyAuth.model_validate_json(Path(__file__).with_suffix('.json').read_text())\n"
    )
    identity: Final = {
        "user_id": "alice",
        "user_role": "internal_user",
        "object_permission": {
            "object_permission_id": "subscription-permission",
            "mcp_servers": ["subscription-server", "second-server"] if access == "fanout" else ["subscription-server"],
        },
    }
    auth_state: Final = auth_module.with_suffix(".json")
    auth_state.write_text(json.dumps(identity))
    bus: Final = ObservedBus()
    upstream_listener: Final = ListenHandler(bus)

    async def control_listen(ctx, call_next):
        if ctx.method == "subscriptions/listen":
            listener_credentials.append(ctx.request.headers.get("authorization"))
            return await upstream_listener(ctx, SubscriptionsListenRequestParams.model_validate(ctx.params))
        result = await call_next(ctx)
        if access == "pages" and ctx.method == "tools/list":
            result = ListToolsResult.model_validate(result)
            offset = int((ctx.params or {}).get("cursor") or "0")
            return result.model_copy(
                update={
                    "tools": result.tools[offset : offset + 1],
                    "next_cursor": str(offset + 1) if offset + 1 < len(result.tools) else None,
                }
            )
        return result

    upstream: Final = MCPServer("subscription-peer", subscriptions=bus, middleware=[control_listen])

    @upstream.resource("test://value")
    def value() -> str:
        return "value"

    @upstream.tool()
    async def change_tools() -> str:
        async def added() -> str:
            return "new tool"

        if access == "upstream-ended":
            upstream_listener.close()
            return "ended"
        if event_kind == "tools":
            upstream.add_tool(added)
            await bus.publish(ToolsListChanged())
        elif event_kind == "prompts":
            upstream.prompt()(added)
            await bus.publish(PromptsListChanged())
        elif event_kind == "resources":
            upstream.resource("test://added")(added)
            await bus.publish(ResourcesListChanged())
        else:
            await bus.publish(ResourceUpdated(uri="test://value"))
        return "changed"

    async def exercise(gateway: Gateway) -> None:
        headers = {"Authorization": "Bearer " + gateway.key}
        if access == "fanout":
            headers = {"x-litellm-api-key": gateway.key, "Authorization": "Bearer upstream-only-test-token"}
        async with httpx2.AsyncClient(headers=headers, timeout=30) as client:
            async with streamable_http_client(
                str(gateway.client.base_url).rstrip("/") + "/mcp/", http_client=client
            ) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    session.adopt(DiscoverResult.model_validate(await session.send_discover("2026-07-28")))
                    before = await session.list_tools()
                    assert "peer-change_tools" in {tool.name for tool in before.tools}
                    assert "peer-added" not in {tool.name for tool in before.tools}
                    if access == "legacy":
                        with pytest.raises(MCPError, match="Select modern upstream"):
                            async with listen(session, tools_list_changed=True):
                                pytest.fail("Unsupported upstream was acknowledged")
                        assert not connected.is_set()
                        return
                    async with listen(
                        session,
                        tools_list_changed=event_kind == "tools",
                        prompts_list_changed=event_kind == "prompts",
                        resources_list_changed=event_kind == "resources",
                        resource_subscriptions=["test://value"] if event_kind == "updated" else (),
                    ) as subscription:
                        if access == "fanout":
                            assert listener_credentials == [None, None], (
                                "Unscoped caller credential reached multiple upstreams"
                            )
                        if access in ("revoked", "identity", "idle"):
                            changed_identity = (
                                {**identity, "user_id": "bob"}
                                if access == "identity"
                                else {
                                    **identity,
                                    "object_permission": {
                                        "object_permission_id": "subscription-permission",
                                        "mcp_servers": ["denied"],
                                    },
                                }
                            )
                            auth_state.write_text(json.dumps(changed_identity))
                        if access == "idle":
                            with pytest.raises((SubscriptionLost, StopAsyncIteration)):
                                await asyncio.wait_for(anext(subscription), 20)
                            assert await asyncio.to_thread(disconnected.wait, 3)
                            return
                        async with streamable_http_client(upstream_url + "/mcp") as direct_streams:
                            async with ClientSession(direct_streams[0], direct_streams[1]) as direct:
                                direct.adopt(DiscoverResult.model_validate(await direct.send_discover("2026-07-28")))
                                changed = await direct.call_tool("change_tools", {})
                                assert not changed.is_error
                        if access in ("revoked", "identity", "idle", "upstream-ended"):
                            with pytest.raises((SubscriptionLost, StopAsyncIteration)):
                                await asyncio.wait_for(anext(subscription), 5)
                            assert await asyncio.to_thread(disconnected.wait, 3)
                            return
                        if access == "hidden":
                            after = await session.list_tools()
                            assert {tool.name for tool in after.tools} == {"peer-change_tools"}
                            with pytest.raises(TimeoutError):
                                await asyncio.wait_for(anext(subscription), 0.5)
                            return
                        event = await asyncio.wait_for(anext(subscription), 5)
                        expected = {
                            "tools": ToolsListChanged(),
                            "prompts": PromptsListChanged(),
                            "resources": ResourcesListChanged(),
                            "updated": ResourceUpdated(uri="test://value"),
                        }
                        assert event == expected[event_kind]
                        if event_kind == "tools":
                            assert subscription.honored.tools_list_changed
                            after = await session.list_tools()
                            if access == "pages":
                                assert after.next_cursor
                                assert "peer-added" not in {tool.name for tool in after.tools}
                                after = await session.list_tools(
                                    params=PaginatedRequestParams(cursor=after.next_cursor)
                                )
                            assert "peer-added" in {tool.name for tool in after.tools}
                        elif event_kind == "prompts":
                            assert subscription.honored.prompts_list_changed
                            prompts = await session.list_prompts()
                            assert "peer-added" in {prompt.name for prompt in prompts.prompts}
                        elif event_kind == "resources":
                            assert subscription.honored.resources_list_changed
                            resources = await session.list_resources()
                            assert "test://added" in {str(resource.uri) for resource in resources.resources}
                        else:
                            assert subscription.honored.resource_subscriptions == ["test://value"]
                    assert await asyncio.to_thread(disconnected.wait, 3), "Upstream listener survived disconnect"

    app: Final = upstream.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
    )

    @asynccontextmanager
    async def lifespan(_: Starlette):
        AppStatus.should_exit = False
        async with app.router.lifespan_context(app):
            yield
        _drain_sse_streams()
        watchers = tuple(task for task in asyncio.all_tasks() if "_shutdown_watcher" in repr(task.get_coro()))
        await asyncio.wait_for(asyncio.gather(*watchers), 2)

    hosted: Final = Starlette(routes=[Mount("/", app=app)], lifespan=lifespan)
    with asgi_server(hosted) as upstream_url, httpx.Client() as client:
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {
                        "peer": {
                            "server_id": "subscription-server",
                            "url": upstream_url + "/mcp",
                            "transport": "http",
                            "protocol_version": "2025-11-25" if access == "legacy" else "2026-07-28",
                            **({"allowed_tools": ["change_tools"]} if access == "hidden" else {}),
                            **(
                                {"auth_type": "oauth_delegate", "extra_headers": ["Authorization"]}
                                if access == "fanout"
                                else {}
                            ),
                        },
                        **(
                            {
                                "second": {
                                    "server_id": "second-server",
                                    "url": upstream_url + "/mcp",
                                    "transport": "http",
                                    "protocol_version": "2026-07-28",
                                    "auth_type": "oauth_delegate",
                                    "extra_headers": ["Authorization"],
                                }
                            }
                            if access == "fanout"
                            else {}
                        ),
                    },
                    "general_settings": {
                        "master_key": "sk-subscriptions-test",
                        "custom_auth": "subscription_auth.authenticate",
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2025-11-25", "2026-07-28"],
                    },
                }
            )
        )
        source: Final = Path(__file__).resolve().parents[3]
        environment: Final = {
            "PYTHONPATH": os.pathsep.join((str(source), str(source / "tests"))),
            "LITELLM_SALT_KEY": "subscription-pagination-key",
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": "true",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY": "true",
        }
        seed: Final = Gateway(client, "sk-subscriptions-test", upstream_url)
        with owned_proxy(
            seed,
            tmp_path / "gateway",
            environment,
            config=config,
            database_setup=(),
            remove_environment=(
                "DATABASE_URL",
                "DATABASE_URL_READ_REPLICA",
                "DIRECT_URL",
                "INTEGRATION_PROXY_DATABASE_URL",
                "INTEGRATION_PROXY_READ_REPLICA_URL",
            ),
        ) as gateway:
            asyncio.run(exercise(gateway))


def test_subscription_partial_setup_releases_upstream_capacity() -> None:
    import asyncio
    from contextlib import asynccontextmanager
    from threading import Event

    import httpx2
    from mcp import ClientSession, MCPError
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.subscriptions import listen
    from mcp.server.mcpserver import MCPServer
    from mcp.server.subscriptions import InMemorySubscriptionBus
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import DiscoverResult, SubscriptionsListenRequestParams
    from sse_starlette.sse import AppStatus
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from integration._support.asgi import asgi_server
    from integration._support.mcp import _drain_sse_streams
    from litellm.experimental_mcp_client.client import MCPClient
    from litellm.proxy._experimental.mcp_server.contracts import OperationContext
    from litellm.proxy._experimental.mcp_server.subscriptions import GatewaySubscriptions, SubscriptionTarget
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer as RegisteredServer

    disconnected = Event()

    class ObservedBus(InMemorySubscriptionBus):
        def subscribe(self, listener):
            unsubscribe = super().subscribe(listener)

            def close():
                unsubscribe()
                disconnected.set()

            return close

    selected = []

    async def resolve(context):
        return tuple(selected)

    async def authorize():
        return OperationContext(_caller=None)

    service = GatewaySubscriptions(max_listeners=1, resolve_targets=resolve)

    async def dispatch(ctx, call_next):
        if ctx.method == "subscriptions/listen":
            return await service.listen(
                ctx, SubscriptionsListenRequestParams.model_validate(ctx.params), authorize, "alice"
            )
        return await call_next(ctx)

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    upstream = MCPServer("capacity-upstream", subscriptions=ObservedBus()).streamable_http_app(
        transport_security=security
    )
    gateway = MCPServer("capacity-gateway", middleware=[dispatch]).streamable_http_app(transport_security=security)

    @asynccontextmanager
    async def lifespan(app):
        AppStatus.should_exit = False
        async with upstream.router.lifespan_context(upstream), gateway.router.lifespan_context(gateway):
            yield
        _drain_sse_streams()
        watchers = tuple(task for task in asyncio.all_tasks() if "_shutdown_watcher" in repr(task.get_coro()))
        await asyncio.wait_for(asyncio.gather(*watchers), 2)

    hosted = Starlette(routes=[Mount("/upstream", app=upstream), Mount("/gateway", app=gateway)], lifespan=lifespan)

    async def exercise(url):
        client = MCPClient(
            server_url=url + "/upstream/mcp", transport_type=MCPTransport.http, protocol_version="2026-07-28"
        )
        target = SubscriptionTarget(
            RegisteredServer(
                server_id="capacity-peer", name="capacity-peer", transport="http", protocol_version="2026-07-28"
            ),
            client,
            "uncredentialed",
        )
        selected.extend((target, target))
        async with httpx2.AsyncClient(timeout=30) as http:
            async with streamable_http_client(url + "/gateway/mcp", http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    session.adopt(DiscoverResult.model_validate(await session.send_discover("2026-07-28")))
                    with pytest.raises(MCPError):
                        async with listen(session):
                            pytest.fail("Partial upstream setup was acknowledged")
                    assert await asyncio.to_thread(disconnected.wait, 3), "Partial listener leaked its upstream"
                    selected.pop()
                    disconnected.clear()
                    async with listen(session) as subscription:
                        assert not subscription.honored.tools_list_changed
                    assert await asyncio.to_thread(disconnected.wait, 3), "Retry leaked its upstream"

    with asgi_server(hosted) as url:
        asyncio.run(exercise(url))
