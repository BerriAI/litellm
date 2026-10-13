import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.mcp import McpCaller
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

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


@contextmanager
def sdk_interaction_peer(*, sampling: bool = False):
    import asyncio
    import queue
    from contextlib import asynccontextmanager

    from mcp.server import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import (
        CallToolResult,
        CreateMessageRequest,
        CreateMessageRequestParams,
        SamplingMessage,
        ElicitRequest,
        ElicitRequestFormParams,
        GetPromptResult,
        InputRequiredResult,
        ListToolsResult,
        ListPromptsResult,
        ListResourcesResult,
        Prompt,
        PromptMessage,
        ReadResourceResult,
        Resource,
        TextContent,
        TextResourceContents,
        Tool,
    )
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from integration._support.asgi import asgi_server
    from integration._support.mcp import McpPeer, _capturing, _drain_sse_streams
    from sse_starlette.sse import AppStatus

    async def tools(ctx, params):
        return ListToolsResult(tools=[Tool(name="confirm", input_schema={"type": "object"})])

    async def prompts(ctx, params):
        return ListPromptsResult(prompts=[Prompt(name="confirm")])

    async def resources(ctx, params):
        return ListResourcesResult(resources=[Resource(name="confirm", uri="test://confirm")])

    async def interact(ctx, params):
        if sampling:
            if not params.request_state:
                return InputRequiredResult(
                    request_state="sample-state",
                    input_requests={
                        "sample": CreateMessageRequest(
                            params=CreateMessageRequestParams(
                                messages=[
                                    SamplingMessage(role="user", content=TextContent(type="text", text="sample prompt"))
                                ],
                                max_tokens=8,
                            )
                        )
                    },
                )
            assert params.request_state == "sample-state"
            sampled = params.input_responses["sample"]
            assert "sampled response" in sampled.model_dump_json()
            return CallToolResult(content=[TextContent(type="text", text="sample confirmed")])
        if not params.request_state:
            return InputRequiredResult(
                request_state="opaque:" + ctx.method,
                input_requests={
                    "consent": ElicitRequest(
                        params=ElicitRequestFormParams(
                            message="Confirm", requested_schema={"type": "object", "properties": {}}
                        )
                    )
                },
            )
        assert params.request_state == "opaque:" + ctx.method
        assert params.input_responses["consent"].action == "accept"
        if ctx.method == "tools/call":
            return CallToolResult(content=[TextContent(type="text", text="confirmed")])
        if ctx.method == "prompts/get":
            return GetPromptResult(
                messages=[PromptMessage(role="user", content=TextContent(type="text", text="confirmed"))]
            )
        return ReadResourceResult(contents=[TextResourceContents(uri="test://confirm", text="confirmed")])

    service = Server(
        "modern-interactions",
        on_list_tools=tools,
        on_list_prompts=prompts,
        on_list_resources=resources,
        on_call_tool=interact,
        on_get_prompt=interact,
        on_read_resource=interact,
    )
    manager = StreamableHTTPSessionManager(
        service,
        stateless=True,
        json_response=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @asynccontextmanager
    async def lifespan(app):
        AppStatus.should_exit = False
        async with manager.run():
            yield
        _drain_sse_streams()
        watchers = tuple(task for task in asyncio.all_tasks() if "_shutdown_watcher" in repr(task.get_coro()))
        await asyncio.wait_for(asyncio.gather(*watchers), 2)

    app = Starlette(routes=[Mount("/mcp", app=manager.handle_request)], lifespan=lifespan)
    observed = queue.Queue()
    with asgi_server(_capturing(app, observed)) as url:
        yield McpPeer(url + "/mcp/", observed)


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


@pytest.mark.parametrize("sdk_upstream", [False, True])
@pytest.mark.parametrize("changed_target", [False, True])
def test_continuations_resume_on_another_replica_and_reject_changed_operations(
    tmp_path: Path, changed_target: bool, sdk_upstream: bool
) -> None:
    with (sdk_interaction_peer() if sdk_upstream else wire_server(interaction_peer)) as peer, httpx.Client() as client:
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


@pytest.mark.parametrize("sdk_upstream", [False, True])
def test_continuation_reauthenticates_caller_and_rechecks_revoked_permissions(tmp_path: Path, sdk_upstream: bool) -> None:
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
    with (
        (sdk_interaction_peer() if sdk_upstream else wire_server(interaction_peer)) as peer,
        (sdk_interaction_peer() if sdk_upstream else wire_server(interaction_peer)) as other_peer,
        httpx.Client() as client,
    ):
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
        ("tools", "modern-boundary"),
        ("prompts", "normal"),
        ("resources", "normal"),
        ("updated", "normal"),
        ("tools", "revoked"),
        ("tools", "identity"),
        ("tools", "team"),
        ("tools", "org"),
        ("tools", "end-user"),
        ("tools", "idle"),
        ("tools", "hidden"),
        ("tools", "upstream-ended"),
        ("tools", "legacy-caller-isolation"),
        ("tools", "legacy-caller-no-notifications"),
        ("updated", "legacy-caller-no-notifications"),
        ("tools", "legacy-caller-revoked"),
        ("tools", "legacy-sse"),
        ("updated", "legacy-sse"),
        ("prompts", "legacy-caller"),
        ("resources", "legacy-caller"),
        ("prompts", "legacy-both"),
        ("resources", "legacy-both"),
        ("tools", "legacy-caller"),
        ("updated", "legacy-caller"),
        ("tools", "legacy-both"),
        ("updated", "legacy-both"),
        ("tools", "legacy"),
        ("prompts", "legacy"),
        ("resources", "legacy"),
        ("updated", "legacy"),
        ("tools", "pages"),
        ("tools", "fanout"),
        ("tools", "setup-race"),
        ("tools", "metadata"),
        ("tools", "config-change"),
        ("tools", "renewal"),
        ("tools", "renewal-scopes"),
        ("tools", "renewal-failed"),
        ("tools", "mixed-capabilities"),
        ("tools", "probe-failure"),
        ("tools", "missing-credentials"),
        ("tools", "concurrent-probes"),
        ("tools", "probe-timeout"),
    ],
)
def test_modern_subscription_delivers_changes_between_calls_and_refreshes_tools(
    tmp_path: Path, event_kind: str, access: str
) -> None:
    import asyncio
    from collections.abc import Callable
    from contextlib import AsyncExitStack, asynccontextmanager
    from threading import Event

    import httpx2
    from integration._support.asgi import asgi_server
    from integration._support.mcp import _drain_sse_streams
    from mcp import ClientSession, MCPError
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.subscriptions import SubscriptionLost, listen
    from mcp.server.mcpserver import MCPServer
    from mcp.server.subscriptions import (
        InMemorySubscriptionBus,
        ListenHandler,
        PromptsListChanged,
        ResourcesListChanged,
        ResourceUpdated,
        ToolsListChanged,
    )
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.shared.subscriptions import ServerEvent
    from mcp.types import DiscoverResult, ListToolsResult, PaginatedRequestParams, SubscriptionsListenRequestParams
    from sse_starlette.sse import AppStatus
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    disconnected: Final = Event()
    connected: Final = Event()
    tool_executions = []
    token_grants = []
    renewal_started = Event()

    async def token_endpoint(request):
        from urllib.parse import parse_qs

        form = parse_qs((await request.body()).decode())
        assert form["grant_type"] == ["client_credentials"]
        assert form["client_id"] == ["subscription-service"]
        assert form["scope"] == ["read write"]
        token_grants.append(form)
        if renewal_started.is_set() and access == "renewal-failed":
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        return JSONResponse(
            {
                "access_token": "synthetic-renewal-" + str(len(token_grants)),
                "token_type": "Bearer",
                "expires_in": 70,
                "scope": "read" if renewal_started.is_set() and access == "renewal-scopes" else "read write",
            }
        )

    probe_fails: Final = Event()
    probe_failed: Final = Event()
    listener_credentials = []
    catalog_credentials = []
    setup_changed = Event()
    probe_arrivals = []
    catalog_reads = []
    probe_catalog_goal = []
    probe_barrier = asyncio.Event()

    active_listeners = set()

    class ObservedBus(InMemorySubscriptionBus):
        def subscribe(self, listener: Callable[[ServerEvent], None]) -> Callable[[], None]:
            unsubscribe = super().subscribe(listener)
            marker = object()
            active_listeners.add(marker)
            connected.set()

            def close() -> None:
                unsubscribe()
                active_listeners.discard(marker)
                disconnected.set()

            return close

    auth_module: Final = tmp_path / "subscription_auth.py"
    auth_module.write_text(
        "from pathlib import Path\n"
        "from fastapi import HTTPException, Request\n"
        "from litellm.proxy._types import UserAPIKeyAuth\n"
        "async def authenticate(request: Request, api_key: str) -> UserAPIKeyAuth:\n"
        "    if api_key not in ('sk-subscriptions-test', 'sk-subscriptions-other'):\n"
        "        raise HTTPException(status_code=401, detail='Unknown caller')\n"
        "    import json\n"
        "    data = json.loads(Path(__file__).with_suffix('.json').read_text())\n"
        "    change = data.pop('subscription_test_update', None)\n"
        "    if change:\n"
        "        from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager as manager\n"
        "        server = manager.config_mcp_servers['subscription-server']\n"
        "        fields = {'authorization_url': 'https://authorization.example/authorize'} if change == 'metadata' else {'url': server.url.rstrip('/') + '/changed'}\n"
        "        manager.config_mcp_servers[server.server_id] = server.model_copy(update=fields)\n"
        "    if api_key == 'sk-subscriptions-other':\n"
        "        data = {**data, 'api_key': api_key, 'user_id': 'bob', 'object_permission': {**data['object_permission'], 'mcp_tool_permissions': {'subscription-server': ['change_tools']}}}\n"
        "    return UserAPIKeyAuth.model_validate(data)\n"
    )
    identity: Final = {
        "api_key": "sk-subscriptions-test",
        "user_id": "alice",
        "user_role": "internal_user",
        "object_permission": {
            "object_permission_id": "subscription-permission",
            "mcp_servers": ["subscription-server", "second-server"] if access in ("fanout", "mixed-capabilities", "missing-credentials", "concurrent-probes") else ["subscription-server"],
        },
    }
    auth_state: Final = auth_module.with_suffix(".json")
    auth_state.write_text(json.dumps(identity))
    bus: Final = ObservedBus()
    upstream_listener: Final = ListenHandler(bus)

    async def control_listen(ctx, call_next):
        if ctx.protocol_version != "2026-07-28" and ctx.method == "notifications/initialized":
            from mcp.shared.subscriptions import event_to_notification

            queue = asyncio.Queue()
            unsubscribe = bus.subscribe(queue.put_nowait)
            connected.set()
            try:
                while True:
                    await ctx.session.send_notification(event_to_notification(await queue.get(), {}))
            finally:
                unsubscribe()
        if ctx.method in ("resources/subscribe", "resources/unsubscribe"):
            from mcp.types import EmptyResult

            return EmptyResult()

        if ctx.method == "tools/list":
            catalog_credentials.append(ctx.request.headers.get("authorization"))
            if access == "setup-race" and connected.is_set() and not setup_changed.is_set():
                setup_changed.set()
                await change_tools()
        if ctx.method == "server/discover" and probe_fails.is_set() and len(catalog_reads) >= probe_catalog_goal[0]:
            probe_failed.set()
            if access == "concurrent-probes":
                probe_arrivals.append(ctx.method)
                if len(probe_arrivals) == 2:
                    probe_barrier.set()
                await probe_barrier.wait()
            elif access == "probe-timeout":
                await probe_barrier.wait()
            else:
                raise MCPError(code=-32603, message="Discovery unavailable")
        if ctx.method == "subscriptions/listen":
            listener_credentials.append(ctx.request.headers.get("authorization"))
            if access == "legacy-caller-no-notifications":
                raise MCPError(code=-32601, message="Upstream has no subscription support")
            return await upstream_listener(ctx, SubscriptionsListenRequestParams.model_validate(ctx.params))
        result = await call_next(ctx)
        if access == "legacy-caller-no-notifications" and ctx.method in ("initialize", "server/discover"):
            from mcp.types import InitializeResult, ServerCapabilities

            parsed = (DiscoverResult if ctx.method == "server/discover" else InitializeResult).model_validate(result)
            return parsed.model_copy(update={"capabilities": ServerCapabilities()})
        if ctx.method == "initialize":
            from mcp.types import (
                InitializeResult,
                PromptsCapability,
                ResourcesCapability,
                ServerCapabilities,
                ToolsCapability,
            )

            result = InitializeResult.model_validate(result)
            return result.model_copy(
                update={
                    "capabilities": ServerCapabilities(
                        tools=ToolsCapability(list_changed=True),
                        prompts=PromptsCapability(list_changed=True),
                        resources=ResourcesCapability(list_changed=True, subscribe=True),
                    )
                }
            )
        if ctx.method in ("tools/list", "prompts/list", "resources/list", "resources/templates/list"):
            catalog_reads.append(ctx.method)
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

    upstream: Final = MCPServer(
        "subscription-peer",
        subscriptions=None if access == "legacy-caller-no-notifications" else bus,
        middleware=[control_listen],
    )

    @upstream.resource("test://value")
    def value() -> str:
        return "value"

    @upstream.tool()
    async def change_tools() -> str:
        tool_executions.append("change_tools")

        async def added() -> str:
            tool_executions.append("added")
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
            gateway_url = str(gateway.client.base_url).rstrip("/")
            connection = (
                sse_client(gateway_url + "/mcp/sse", headers=headers)
                if access == "legacy-sse"
                else streamable_http_client(gateway_url + "/mcp/", http_client=client)
            )
            async with connection as streams:
                received = asyncio.Queue()

                async def receive(notification, queue=received):
                    from mcp.shared.subscriptions import event_from_wire

                    if not isinstance(notification, Exception):
                        event = event_from_wire(
                            notification.method,
                            notification.params.model_dump(by_alias=True) if notification.params else None,
                        )
                        if event is not None:
                            await queue.put(event)

                async with (
                    ClientSession(streams[0], streams[1], message_handler=receive) as session,
                    AsyncExitStack() as additional,
                ):
                    other_received = asyncio.Queue()
                    if access == "legacy-caller-isolation":
                        other_client = await additional.enter_async_context(
                            httpx2.AsyncClient(headers={"Authorization": "Bearer sk-subscriptions-other"})
                        )
                        other_streams = await additional.enter_async_context(
                            streamable_http_client(gateway_url + "/mcp/", http_client=other_client)
                        )

                        async def receive_other(notification):
                            await receive(notification, other_received)

                        other = await additional.enter_async_context(
                            ClientSession(other_streams[0], other_streams[1], message_handler=receive_other)
                        )
                        await other.initialize()
                        assert {tool.name for tool in (await other.list_tools()).tools} == {"peer-change_tools"}

                    if access.startswith("legacy-caller") or access in ("legacy-both", "legacy-sse"):
                        await session.initialize()
                        before = await session.list_tools()
                        assert "peer-added" not in {tool.name for tool in before.tools}
                        if event_kind == "updated":
                            if access == "legacy-caller-no-notifications":
                                with pytest.raises(MCPError):
                                    await asyncio.wait_for(session.subscribe_resource("test://value"), 3)
                                assert listener_credentials == []
                                assert not active_listeners
                                assert not (await session.call_tool("peer-change_tools", {})).is_error
                                return
                            await session.subscribe_resource("test://value")
                        if access != "legacy-caller-no-notifications":
                            assert await asyncio.to_thread(connected.wait, 5)
                        if access == "legacy-caller-revoked":
                            auth_state.write_text(
                                json.dumps(
                                    {
                                        **identity,
                                        "object_permission": {
                                            "object_permission_id": "subscription-permission",
                                            "mcp_servers": ["denied"],
                                        },
                                    }
                                )
                            )
                            async with streamable_http_client(upstream_url + "/mcp") as direct_streams:
                                async with ClientSession(direct_streams[0], direct_streams[1]) as direct:
                                    direct.adopt(
                                        DiscoverResult.model_validate(await direct.send_discover("2026-07-28"))
                                    )
                                    assert not (await direct.call_tool("change_tools", {})).is_error
                            with pytest.raises(TimeoutError):
                                await asyncio.wait_for(received.get(), 0.5)
                            assert await asyncio.to_thread(disconnected.wait, 3)
                            return
                        changed = await session.call_tool("peer-change_tools", {})
                        assert not changed.is_error
                        if access == "legacy-caller-no-notifications":
                            with pytest.raises(TimeoutError):
                                await asyncio.wait_for(received.get(), 0.5)
                            assert "peer-added" in {tool.name for tool in (await session.list_tools()).tools}
                            assert listener_credentials == []
                            return
                        event = await asyncio.wait_for(received.get(), 10)
                        assert (
                            event
                            == (
                                {
                                    "tools": ToolsListChanged(),
                                    "prompts": PromptsListChanged(),
                                    "resources": ResourcesListChanged(),
                                    "updated": ResourceUpdated(uri="test://value"),
                                }[event_kind]
                            )
                        )
                        if access == "legacy-caller-isolation":
                            with pytest.raises(TimeoutError):
                                await asyncio.wait_for(other_received.get(), 0.5)
                            assert {tool.name for tool in (await other.list_tools()).tools} == {"peer-change_tools"}
                            before_denied = tuple(tool_executions)
                            denied = await other.call_tool("peer-added", {})
                            assert denied.is_error
                            assert tuple(tool_executions) == before_denied
                        if event_kind == "tools":
                            assert "peer-added" in {tool.name for tool in (await session.list_tools()).tools}
                        elif event_kind == "prompts":
                            assert "peer-added" in {prompt.name for prompt in (await session.list_prompts()).prompts}
                        elif event_kind == "resources":
                            assert "test://added" in {
                                str(resource.uri) for resource in (await session.list_resources()).resources
                            }
                        else:
                            await session.unsubscribe_resource("test://value")
                            changed = await session.call_tool("peer-change_tools", {})
                            assert not changed.is_error
                            with pytest.raises(TimeoutError):
                                await asyncio.wait_for(received.get(), 0.5)
                        return
                    discovery = DiscoverResult.model_validate(await session.send_discover("2026-07-28"))
                    session.adopt(discovery)
                    if access in ("mixed-capabilities", "missing-credentials"):
                        assert not discovery.capabilities.tools.list_changed
                    else:
                        assert discovery.capabilities.tools.list_changed
                        assert discovery.capabilities.resources.list_changed
                        assert bool(discovery.capabilities.resources.subscribe) is (access not in ("fanout", "concurrent-probes"))
                    if access in ("probe-failure", "concurrent-probes", "probe-timeout"):
                        probe_catalog_goal.append(len(catalog_reads) + 4 * len(identity["object_permission"]["mcp_servers"]))
                        probe_fails.set()
                        try:
                            degraded = DiscoverResult.model_validate(
                                await asyncio.wait_for(session.send_discover("2026-07-28"), 5)
                            )
                        finally:
                            probe_barrier.set()
                        assert probe_failed.is_set(), ("Controlled discovery probe was not exercised", catalog_reads, probe_catalog_goal)
                        assert degraded.capabilities.tools is not None
                        if access == "concurrent-probes":
                            assert len(probe_arrivals) == 2
                            assert degraded.capabilities.tools.list_changed
                        else:
                            assert not (degraded.capabilities.tools and degraded.capabilities.tools.list_changed)
                            assert not (degraded.capabilities.resources and degraded.capabilities.resources.subscribe)
                        probe_fails.clear()
                        visible = await session.list_tools()
                        assert "peer-change_tools" in {tool.name for tool in visible.tools}
                        return
                    before = await session.list_tools()
                    assert "peer-change_tools" in {tool.name for tool in before.tools}
                    assert "peer-added" not in {tool.name for tool in before.tools}
                    if access in ("mixed-capabilities", "missing-credentials"):
                        assert not discovery.capabilities.resources.list_changed
                        assert not discovery.capabilities.resources.subscribe
                        assert not connected.is_set()
                        return
                    if access == "modern-boundary":
                        for method, params in (
                            ("notifications/initialized", {}),
                            ("initialize", {"protocolVersion": "2026-07-28", "clientInfo": {"name": "boundary", "version": "1"}, "capabilities": {}}),
                            ("resources/subscribe", {"uri": "test://value"}),
                            ("resources/unsubscribe", {"uri": "test://value"}),
                        ):
                            notification = method == "notifications/initialized"
                            response = await client.post(
                                gateway_url + "/mcp/",
                                headers={
                                    "MCP-Protocol-Version": "2026-07-28",
                                    "Mcp-Method": method,
                                    "Mcp-Name": str(params.get("uri", "")),
                                    "Accept": "application/json, text/event-stream",
                                },
                                json={
                                    "jsonrpc": "2.0",
                                    **({} if notification else {"id": "unsupported"}),
                                    "method": method,
                                    "params": {**params, "_meta": _META},
                                },
                                timeout=5,
                            )
                            if notification:
                                assert response.status_code == 202
                            else:
                                assert response.status_code == 404, response.text
                                assert response.json()["error"]["code"] == -32601
                        assert listener_credentials == [], "Removed methods opened an implicit upstream listener"
                        assert not active_listeners
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
                            assert catalog_credentials and all(value is None for value in catalog_credentials), (
                                "Unscoped caller credential reached catalog listings"
                            )
                        if access in ("metadata", "config-change"):
                            auth_state.write_text(json.dumps({**identity, "subscription_test_update": access}))
                        if access in ("revoked", "identity", "team", "org", "end-user", "idle"):
                            changed_identity = (
                                {
                                    **identity,
                                    {
                                        "identity": "user_id",
                                        "team": "team_id",
                                        "org": "org_id",
                                        "end-user": "end_user_id",
                                    }[access]: "bob",
                                }
                                if access in ("identity", "team", "org", "end-user")
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
                        if access.startswith("renewal"):
                            assert token_grants
                            admitted_grants = len(token_grants)
                            renewal_started.set()
                            async with asyncio.timeout(25):
                                while len(token_grants) == admitted_grants:
                                    try:
                                        await session.list_tools()
                                    except MCPError:
                                        assert access == "renewal-failed"
                                    await asyncio.sleep(0.2)
                            assert token_grants[0] == token_grants[-1]
                        if access != "setup-race":
                            async with streamable_http_client(upstream_url + "/mcp") as direct_streams:
                                async with ClientSession(direct_streams[0], direct_streams[1]) as direct:
                                    direct.adopt(
                                        DiscoverResult.model_validate(await direct.send_discover("2026-07-28"))
                                    )
                                    changed = await direct.call_tool("change_tools", {})
                                    assert not changed.is_error
                        if access in (
                            "revoked",
                            "identity",
                            "team",
                            "org",
                            "end-user",
                            "idle",
                            "upstream-ended",
                            "config-change",
                            "renewal-scopes",
                            "renewal-failed",
                        ):
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
                        if access == "fanout":
                            assert all(value is None for value in catalog_credentials), (
                                "Unscoped caller credential reached refreshed catalogs"
                            )
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
                        if access == "modern-boundary":
                            assert len(listener_credentials) == 1
                            with pytest.raises(TimeoutError):
                                await asyncio.wait_for(anext(subscription), 0.5)
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

    hosted: Final = Starlette(routes=[Route("/token", token_endpoint, methods=["POST"]), Mount("/", app=app)], lifespan=lifespan)
    with (
        asgi_server(hosted) as upstream_url,
        (sdk_interaction_peer() if access == "mixed-capabilities" else nullcontext(None)) as plain,
        httpx.Client() as client,
    ):
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {
                        "peer": {
                            "server_id": "subscription-server",
                            **(
                                {
                                    "auth_type": "oauth2",
                                    "oauth2_flow": "client_credentials",
                                    "client_id": "subscription-service",
                                    "client_secret": "synthetic-client-secret",
                                    "token_url": upstream_url + "/token",
                                    "scopes": ["read", "write"],
                                }
                                if access.startswith("renewal")
                                else {}
                            ),
                            "url": upstream_url + "/mcp",
                            "transport": "http",
                            "protocol_version": "2025-11-25" if access in ("legacy", "legacy-both") else "2026-07-28",
                            **({"allowed_tools": ["change_tools"]} if access == "hidden" else {}),
                            **({"timeout": 2} if access == "concurrent-probes" else {}),
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
                                    "url": plain.url if plain is not None else upstream_url + "/mcp",
                                    "transport": "http",
                                    **({"timeout": 2} if access == "concurrent-probes" else {}),
                                    "protocol_version": "2026-07-28",
                                    "auth_type": "oauth_delegate",
                                    "extra_headers": ["Authorization"],
                                    **({
                                        "auth_type": "oauth2",
                                        "oauth2_flow": "authorization_code",
                                        "client_id": "subscription-client",
                                        "authorization_url": upstream_url + "/authorize",
                                        "token_url": upstream_url + "/token",
                                    } if access == "missing-credentials" else {}),
                                }
                            }
                            if access in ("fanout", "mixed-capabilities", "missing-credentials", "concurrent-probes")
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
            **({"LITELLM_MCP_TOOL_LISTING_TIMEOUT": "2"} if access == "probe-timeout" else {}),
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
            from integration._support.client import eventually

            eventually(lambda: len(active_listeners), lambda count: count == 0, seconds=10)
            assert not active_listeners, "Upstream listener survived client disconnect"


def test_subscription_partial_setup_releases_upstream_capacity() -> None:
    import asyncio
    from contextlib import asynccontextmanager
    from threading import Event

    import httpx2
    from integration._support.asgi import asgi_server
    from integration._support.mcp import _drain_sse_streams
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


@pytest.mark.parametrize("enabled", [False, True])
def test_modern_sampling_obeys_opt_in_and_completes_upstream_interaction(tmp_path: Path, enabled: bool) -> None:
    def provider(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        body = json.loads(request.body)
        assert "sample prompt" in json.dumps(body["messages"])
        return Reply(
            body=json.dumps(
                {
                    "id": "sample-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "sampling-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "sampled response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                }
            ).encode()
        )

    with sdk_interaction_peer(sampling=True) as peer, wire_server(provider) as model, httpx.Client() as client:
        config = tmp_path / "sampling.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "sampling-model",
                            "litellm_params": {
                                "model": "openai/sampling-model",
                                "api_base": model.url + "/v1",
                                "api_key": "synthetic-provider-key",
                            },
                        }
                    ],
                    "mcp_servers": {
                        "peer": {
                            "url": peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_sampling": enabled,
                        }
                    },
                    "general_settings": {
                        "master_key": _KEY,
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2026-07-28"],
                    },
                    "litellm_settings": {"default_mcp_sampling_model": "sampling-model"},
                }
            )
        )
        environment = {
            "PYTHONPATH": _PROXY_PYTHONPATH,
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": "true",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_SALT_KEY": "sampling-test-salt",
        }
        with owned_proxy(
            Gateway(client, _KEY, peer.url),
            tmp_path / "gateway",
            environment,
            config=config,
            database_setup=(),
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "REDIS_URL", "REDIS_HOST"),
        ) as gateway:
            reply = McpCaller(gateway, gateway.key, "root", protocol_version="2026-07-28").rpc(
                "tools/call", {"name": "peer-confirm", "arguments": {}}
            )
            assert reply.status_code == 200, reply.text
            response = reply.json()
            calls = tuple(request for request in model.drain() if request.method == "POST")
            if enabled:
                assert response["result"]["content"][0]["text"] == "sample confirmed", response
                assert len(calls) == 1
            else:
                assert response["result"]["isError"], response
                assert "sampling" in json.dumps(response).lower(), response
                assert calls == (), "Disabled sampling reached the model provider"
