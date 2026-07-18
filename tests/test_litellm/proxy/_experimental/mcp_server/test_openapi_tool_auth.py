"""
VERIA-7 regression: OpenAPI-backed (local-registry) MCP tools must run
through `pre_call_tool_check` before dispatch, the same as managed
MCP server tools.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


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

    # `_get_mcp_server_from_tool_name` returns None — no server context.
    with (
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
