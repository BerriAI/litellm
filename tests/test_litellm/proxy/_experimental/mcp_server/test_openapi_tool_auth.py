"""
VERIA-7 regression: OpenAPI-backed (local-registry) MCP tools must run
through `pre_call_tool_check` before dispatch, the same as managed
MCP server tools.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.mark.asyncio
async def test_openapi_local_tool_runs_pre_call_tool_check():
    """When `execute_mcp_tool` resolves a local-registry (OpenAPI) tool
    AND a server, the pre-call hook must fire before the local handler
    runs. Pre-fix this path skipped the hook entirely."""
    from litellm.proxy._experimental.mcp_server import server as mcp_module

    user = UserAPIKeyAuth(
        api_key="sk-user",
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    fake_server = MagicMock()
    fake_server.name = "openapi-petstore"
    fake_server.is_byok = False
    fake_server.auth_type = None
    fake_server.mcp_info = None
    fake_server.server_id = "srv-1"
    fake_server.server_name = "openapi-petstore"
    fake_server.alias = None
    fake_server.short_prefix = None

    fake_tool = MagicMock()
    fake_tool.name = "list_pets"

    pre_call = AsyncMock(return_value={})
    handle_local = AsyncMock(return_value=[])

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=fake_server,
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "pre_call_tool_check",
            new=pre_call,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=fake_tool,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_local_mcp_tool",
            new=handle_local,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
            return_value=True,
        ),
    ):
        await mcp_module.execute_mcp_tool(
            name="list_pets",
            arguments={"limit": 10},
            allowed_mcp_servers=[fake_server],
            start_time=datetime.now(timezone.utc),
            user_api_key_auth=user,
        )

    pre_call.assert_awaited_once()
    handle_local.assert_awaited_once()

    # The pre-call hook must run before _handle_local_mcp_tool so an
    # unauthorized tool is blocked before any work runs. AsyncMock
    # records call order indirectly — we already asserted both were
    # called; the relative ordering is enforced by the source change.
    pre_call_kwargs = pre_call.await_args.kwargs
    assert pre_call_kwargs["name"] == "list_pets"
    assert pre_call_kwargs["server"] is fake_server
    assert pre_call_kwargs["user_api_key_auth"] is user
    # `proxy_logging_obj` must be sourced from the canonical proxy_server
    # module (same as the managed path) — passing None would crash the
    # downstream `_create_mcp_request_object_from_kwargs` call with
    # AttributeError after the security checks succeed.
    assert pre_call_kwargs["proxy_logging_obj"] is not None


@pytest.mark.asyncio
async def test_openapi_local_tool_blocked_when_pre_call_check_raises():
    """If the pre-call check raises (caller not authorized for this
    tool), the local handler must NOT be invoked."""
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    user = UserAPIKeyAuth(
        api_key="sk-user",
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    fake_server = MagicMock()
    fake_server.name = "openapi-petstore"
    fake_server.is_byok = False
    fake_server.auth_type = None
    fake_server.mcp_info = None
    fake_server.server_id = "srv-1"
    fake_server.server_name = "openapi-petstore"
    fake_server.alias = None
    fake_server.short_prefix = None

    fake_tool = MagicMock()
    fake_tool.name = "delete_pet"

    pre_call = AsyncMock(
        side_effect=HTTPException(status_code=403, detail="not allowed")
    )
    handle_local = AsyncMock(return_value=[])

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=fake_server,
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "pre_call_tool_check",
            new=pre_call,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=fake_tool,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_local_mcp_tool",
            new=handle_local,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
            return_value=True,
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await mcp_module.execute_mcp_tool(
                name="delete_pet",
                arguments={},
                allowed_mcp_servers=[fake_server],
                start_time=datetime.now(timezone.utc),
                user_api_key_auth=user,
            )

    assert exc.value.status_code == 403
    pre_call.assert_awaited_once()
    handle_local.assert_not_awaited()


@pytest.mark.asyncio
async def test_openapi_local_tool_denied_when_server_not_resolvable():
    """If the local-registry tool is found but no MCP server resolves
    (startup race or orphaned registry entry), the call must be rejected
    rather than dispatched without `pre_call_tool_check`."""
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    user = UserAPIKeyAuth(
        api_key="sk-user",
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )

    fake_tool = MagicMock()
    fake_tool.name = "list_pets"

    pre_call = AsyncMock(return_value={})
    handle_local = AsyncMock(return_value=[])
    resolve_auth = MagicMock()

    # `_get_mcp_server_from_tool_name` returns None — no server context.
    with (
        patch.object(mcp_module, "_resolve_openapi_tool_auth", new=resolve_auth),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=None,
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "pre_call_tool_check",
            new=pre_call,
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=fake_tool,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_local_mcp_tool",
            new=handle_local,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
            return_value=True,
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await mcp_module.execute_mcp_tool(
                name="list_pets",
                arguments={},
                allowed_mcp_servers=[],
                start_time=datetime.now(timezone.utc),
                user_api_key_auth=user,
            )

    assert exc.value.status_code == 503
    pre_call.assert_not_awaited()
    handle_local.assert_not_awaited()
    # The credential resolver takes a non-optional server, so the 503 guard above it is what keeps
    # a missing server from ever reaching it. Pinned here so moving the guard reds this test.
    resolve_auth.assert_not_called()


@pytest.mark.asyncio
async def test_openapi_local_tool_injects_resolved_oauth_token():
    """LIT-4629: the local-registry (OpenAPI) dispatch is the primary egress for spec_path
    tools, and before the fix it dropped the gateway-resolved OAuth credential entirely, so a
    user's completed OAuth flow stored a token that never reached the upstream API. The resolved
    credential must land in the `_request_resolved_auth_headers` ContextVar the tool closure
    reads. Kills the mutant that deletes the resolve_openapi_upstream_auth call in server.py."""
    from litellm.proxy._experimental.mcp_server import server as mcp_module
    from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
        _request_resolved_auth_headers,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
        StaticHeaderAuth,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Ok
    from litellm.types.mcp import MCPAuth, MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    user = UserAPIKeyAuth(
        api_key="sk-user",
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    oauth_server = MCPServer(
        server_id="srv-sheets",
        name="google_sheets",
        server_name="google_sheets",
        url=None,
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        spec_path="https://example.com/sheets-openapi.yaml",
    )

    fake_tool = MagicMock()
    fake_tool.name = "get_values"
    captured: dict = {}

    async def handle_local(_name, _arguments):
        captured["resolved"] = _request_resolved_auth_headers.get()
        return []

    with (
        patch.object(
            mcp_module.global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=oauth_server,
        ),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "pre_call_tool_check",
            new=AsyncMock(return_value={}),
        ),
        patch.object(
            mcp_module.global_mcp_tool_registry,
            "get_tool",
            return_value=fake_tool,
        ),
        patch.object(
            mcp_module.global_mcp_server_manager._cred_provider,
            "resolve_credentials",
            new=AsyncMock(return_value=Ok(StaticHeaderAuth("Bearer stored-user-token"))),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_local_mcp_tool",
            new=handle_local,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
            return_value=True,
        ),
    ):
        await mcp_module.execute_mcp_tool(
            name="get_values",
            arguments={},
            allowed_mcp_servers=[oauth_server],
            start_time=datetime.now(timezone.utc),
            user_api_key_auth=user,
        )

    assert captured["resolved"] == {"Authorization": "Bearer stored-user-token"}
    assert _request_resolved_auth_headers.get() is None



LEGACY_SERVER_ID = "srv-legacy-petstore"
LEGACY_SERVER_NAME = "legacy_petstore"
LEGACY_TOOL = "dump_secrets"


@pytest.fixture
def legacy_local_tool():
    """A bare `mcp_tools`-style handler plus a registered server whose tools were
    never listed, which is what leaves `tool_name_to_mcp_server_name_mapping`
    cold and routes `{server}-{tool}` into `execute_mcp_tool`'s legacy fallback.

    Yields the server and the list the handler appends to, so a test can tell
    "refused" from "dispatched" by whether the handler actually ran.
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._experimental.mcp_server.tool_registry import (
        global_mcp_tool_registry,
    )

    executed: list[dict] = []
    server = MCPServer(
        server_id=LEGACY_SERVER_ID,
        name=LEGACY_SERVER_NAME,
        server_name=LEGACY_SERVER_NAME,
        alias=LEGACY_SERVER_NAME,
        url="http://127.0.0.1:1/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
    )
    global_mcp_tool_registry.register_tool(
        name=LEGACY_TOOL,
        description="bare tool registered from the mcp_tools config block",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **kwargs: executed.append(kwargs) or "legacy local tool ran",
    )
    global_mcp_server_manager.registry[LEGACY_SERVER_ID] = server
    assert (
        global_mcp_server_manager._get_mcp_server_from_tool_name(
            f"{LEGACY_SERVER_NAME}-{LEGACY_TOOL}"
        )
        is None
    ), "fixture precondition: the prefixed name must resolve to no server"
    try:
        yield server, executed
    finally:
        global_mcp_tool_registry.tools.pop(LEGACY_TOOL, None)
        global_mcp_server_manager.registry.pop(LEGACY_SERVER_ID, None)


def _caller_entitled_to(tools: list[str]) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-caller",
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-legacy-fallback",
            mcp_servers=[LEGACY_SERVER_ID],
            mcp_tool_permissions={LEGACY_SERVER_ID: tools},
        ),
    )


@pytest.mark.asyncio
async def test_legacy_local_tool_fallback_refuses_unentitled_caller(legacy_local_tool):
    """The legacy fallback dispatched into the local tool registry with no
    tool-level authorization at all: no allowed/banned check, no key/team/org
    tool permissions, no parameter validation. It must now run the same gate,
    so a caller whose entitlement excludes the tool is refused and the handler
    never runs.

    Nothing is mocked: the real registries and the real entitlement gate decide.
    """
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server import server as mcp_module
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    server, executed = legacy_local_tool
    user = _caller_entitled_to(["list_pets"])

    # The gate answers "no" for this caller/tool pair, so a dispatch below would
    # be an entitlement bypass rather than a routing quirk.
    assert (
        await MCPRequestHandler.is_tool_allowed_for_server(
            tool_name=LEGACY_TOOL,
            server_id=LEGACY_SERVER_ID,
            user_api_key_auth=user,
        )
        is False
    )

    with pytest.raises(HTTPException) as exc:
        await mcp_module.execute_mcp_tool(
            name=f"{LEGACY_SERVER_NAME}-{LEGACY_TOOL}",
            arguments={},
            allowed_mcp_servers=[server],
            start_time=datetime.now(timezone.utc),
            user_api_key_auth=user,
        )

    assert exc.value.status_code == 403
    # Pin the refusal to the ENTITLEMENT gate. The server-level check earlier in
    # execute_mcp_tool also raises 403 (with a plain-string detail), and the
    # allowed/banned-tools check raises a dict naming the server rather than the
    # key/team, so asserting on the status alone would pass for the wrong reason.
    detail = exc.value.detail
    assert isinstance(detail, dict), detail
    assert "not allowed for your key/team" in detail["error"], detail
    assert executed == []


@pytest.mark.asyncio
async def test_legacy_local_tool_fallback_still_dispatches_entitled_caller(
    legacy_local_tool,
):
    """The gate must do per-tool work rather than disabling the fallback: the
    same shape of call, from a caller entitled to the tool, still dispatches.

    This is the backwards-compatibility half. Refusing this call would trade an
    authorization hole for an outage on a configuration that worked before.
    """
    from litellm.proxy._experimental.mcp_server import server as mcp_module

    server, executed = legacy_local_tool
    user = _caller_entitled_to([LEGACY_TOOL])

    result = await mcp_module.execute_mcp_tool(
        name=f"{LEGACY_SERVER_NAME}-{LEGACY_TOOL}",
        arguments={},
        allowed_mcp_servers=[server],
        start_time=datetime.now(timezone.utc),
        user_api_key_auth=user,
    )

    assert result.isError is False
    assert executed == [{}]
    assert "legacy local tool ran" in result.content[0].text


@pytest.mark.asyncio
async def test_legacy_local_tool_fallback_fails_closed_on_empty_prefix(
    legacy_local_tool,
):
    """An empty prefix segment skips the server-level check outright:
    `split_server_prefix_from_name` yields an empty `server_name`, and `execute_mcp_tool`
    only runs `is_tool_allowed` `if server_name`. The legacy fallback then dispatched for a
    caller holding no server grant at all, so this arm of the guard is reachable rather than
    defensive. Nothing is patched here; the empty prefix segment is the whole of it.
    """
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    _server, executed = legacy_local_tool

    with pytest.raises(HTTPException) as exc:
        await mcp_module.execute_mcp_tool(
            name=f"-{LEGACY_TOOL}",
            arguments={},
            allowed_mcp_servers=[],
            start_time=datetime.now(timezone.utc),
            user_api_key_auth=_caller_entitled_to([LEGACY_TOOL]),
        )

    assert exc.value.status_code == 503
    assert executed == []


@pytest.mark.asyncio
async def test_legacy_local_tool_fallback_fails_closed_when_prefix_names_no_server(
    legacy_local_tool,
):
    """Second arm of the same guard: a non-empty prefix that named a server the caller does
    hold, but which is absent from `allowed_mcp_servers` by the time dispatch runs. Patching
    the server-level check (which would otherwise refuse first) is what makes the arm
    observable, so a later refactor cannot make the branch dispatch with no server to
    evaluate a tool ceiling against.
    """
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    _server, executed = legacy_local_tool
    other_server = MCPServer(
        server_id="srv-unrelated",
        name="unrelated_server",
        server_name="unrelated_server",
        alias="unrelated_server",
        url="http://127.0.0.1:1/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
        return_value=True,
    ):
        with pytest.raises(HTTPException) as exc:
            await mcp_module.execute_mcp_tool(
                name=f"{LEGACY_SERVER_NAME}-{LEGACY_TOOL}",
                arguments={},
                allowed_mcp_servers=[other_server],
                start_time=datetime.now(timezone.utc),
                user_api_key_auth=_caller_entitled_to([LEGACY_TOOL]),
            )

    assert exc.value.status_code == 503
    assert executed == []


@pytest.mark.asyncio
async def test_unknown_tool_name_still_reports_not_found():
    """The guard must gate dispatch, not existence. An unprefixed name that no registry
    knows cannot dispatch anything, so it has to keep reporting 404 rather than collapsing
    into the guard's 503; every typo'd tool name takes this branch.
    """
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server import server as mcp_module

    with pytest.raises(HTTPException) as exc:
        await mcp_module.execute_mcp_tool(
            name="tool_no_registry_knows",
            arguments={},
            allowed_mcp_servers=[],
            start_time=datetime.now(timezone.utc),
            user_api_key_auth=_caller_entitled_to([LEGACY_TOOL]),
        )

    assert exc.value.status_code == 404
    assert "not found" in str(exc.value.detail)


OPENAPI_PER_SERVER_TOKEN = "Bearer per-server-upstream-token"


def _spec_path_server() -> MCPServer:
    return MCPServer(
        server_id="srv-reports",
        name="report_api",
        server_name="report_api",
        alias="report_api",
        url="https://api.internal.example.com",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth_delegate,
        spec_path="https://api.internal.example.com/openapi.json",
    )


@pytest.mark.parametrize("dispatch_arm", ["local_registry", "call_tool"])
@pytest.mark.asyncio
async def test_per_server_auth_header_reaches_both_openapi_dispatch_arms(dispatch_arm: str):
    """`x-mcp-{alias}-authorization` must survive on BOTH OpenAPI dispatch arms.

    OpenAPI tools live in the local tool registry, so `execute_mcp_tool` serves MCP-protocol and REST
    tool calls while `MCPServerManager.call_tool` serves the responses-API handler. Both arms sourced
    the upstream credential only from the deprecated global / BYOK `mcp_auth_header`, so the
    per-server header was dropped and the upstream API saw no Authorization at all.

    Asserting on the resolver kwarg as well as the ContextVar is deliberate: for the client-forwarded
    modes the credential has to reach `resolve_openapi_upstream_auth`, whose passthrough arm outranks
    the ContextVar when it materializes a header.
    """
    from litellm.proxy._experimental.mcp_server import server as mcp_module
    from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
        _request_auth_header,
    )

    server = _spec_path_server()
    auth_headers = {"report_api": {"Authorization": OPENAPI_PER_SERVER_TOKEN}}
    user = UserAPIKeyAuth(api_key="sk-user", user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER.value)
    captured: dict = {}

    async def fake_resolver(**kwargs):
        captured["resolver_credential"] = kwargs["mcp_auth_header"]
        return None, kwargs["forwarded_headers"]

    async def capture_local(_name, _arguments):
        captured["injected"] = _request_auth_header.get()
        return []

    async def capture_openapi_handler(_server, _name, _arguments):
        captured["injected"] = _request_auth_header.get()
        return []

    manager = mcp_module.global_mcp_server_manager
    with (
        patch.object(manager, "resolve_openapi_upstream_auth", new=fake_resolver),
        patch.object(manager, "pre_call_tool_check", new=AsyncMock(return_value={})),
    ):
        if dispatch_arm == "local_registry":
            fake_tool = MagicMock()
            fake_tool.name = "list_reports"
            with (
                patch.object(manager, "_get_mcp_server_from_tool_name", return_value=server),
                patch.object(mcp_module.global_mcp_tool_registry, "get_tool", return_value=fake_tool),
                patch(
                    "litellm.proxy._experimental.mcp_server.server._handle_local_mcp_tool",
                    new=capture_local,
                ),
                patch(
                    "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
                    return_value=True,
                ),
            ):
                await mcp_module.execute_mcp_tool(
                    name="list_reports",
                    arguments={},
                    allowed_mcp_servers=[server],
                    start_time=datetime.now(timezone.utc),
                    user_api_key_auth=user,
                    mcp_server_auth_headers=auth_headers,
                )
        else:
            with (
                patch.object(manager, "_resolve_mcp_server_for_tool_call", return_value=server),
                patch.object(manager, "_call_openapi_tool_handler", new=capture_openapi_handler),
            ):
                await manager.call_tool(
                    server_name="report_api",
                    name="list_reports",
                    arguments={},
                    user_api_key_auth=user,
                    mcp_server_auth_headers=auth_headers,
                )

    assert captured["resolver_credential"] == {"Authorization": OPENAPI_PER_SERVER_TOKEN}
    assert captured["injected"] == OPENAPI_PER_SERVER_TOKEN
    assert _request_auth_header.get() is None


@pytest.mark.parametrize("failure", ["auth", "other"])
@pytest.mark.asyncio
async def test_local_dispatch_reports_the_outcome_instead_of_success(failure: str):
    """A failing local handler must never be reported as a successful tool result, and only an auth
    failure may propagate.

    `_handle_local_mcp_tool` used to catch every exception and return it as TextContent, and both of
    its callers then stamped `isError=False`, so an upstream rejection was served as tool output and
    `extract_mcp_tool_result_error_message` logged the request as a success.

    The two kinds are split by consequence. `MCPUpstreamAuthError` propagates because both renderers
    know it: the streamable path names the status and the REST path relays a real 401 with the
    upstream's WWW-Authenticate. Anything else is reported as `isError=True` right here, because
    `call_tool_rest_api` turns an unrecognized exception into HTTP 500 and an upstream 403 or 429 is
    not a gateway crash.
    """
    from litellm.proxy._experimental.mcp_server import server as mcp_module
    from litellm.proxy._experimental.mcp_server.exceptions import (
        MCPOpenApiUpstreamError,
        MCPUpstreamAuthError,
    )

    error = (
        MCPUpstreamAuthError(status_code=401, www_authenticate=None, server_name="report_api")
        if failure == "auth"
        else MCPOpenApiUpstreamError(429, "report_api")
    )

    async def raising_handler(**_kwargs):
        raise error

    fake_tool = MagicMock()
    fake_tool.name = "list_reports"
    fake_tool.handler = raising_handler
    server = MCPServer(
        server_id="srv-openapi",
        name="report_api",
        server_name="report_api",
        url="https://api.example.com",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth_delegate,
        spec_path="https://api.example.com/openapi.json",
    )
    user = UserAPIKeyAuth(api_key="sk-user", user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER.value)

    with (
        patch.object(mcp_module.global_mcp_server_manager, "_get_mcp_server_from_tool_name", return_value=server),
        patch.object(mcp_module.global_mcp_server_manager, "pre_call_tool_check", new=AsyncMock(return_value={})),
        patch.object(mcp_module.global_mcp_tool_registry, "get_tool", return_value=fake_tool),
        patch.object(
            mcp_module.global_mcp_server_manager,
            "resolve_openapi_upstream_auth",
            new=AsyncMock(return_value=(None, None)),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
            return_value=True,
        ),
    ):
        call = mcp_module.execute_mcp_tool(
            name="list_reports",
            arguments={},
            allowed_mcp_servers=[server],
            start_time=datetime.now(timezone.utc),
            user_api_key_auth=user,
        )
        if failure == "auth":
            with pytest.raises(MCPUpstreamAuthError):
                await call
            return
        result = await call

    # A non-auth upstream failure stays a 200 with isError, so REST does not report it as a gateway 500
    assert result.isError is True
    assert "upstream returned HTTP 429" in result.content[0].text
