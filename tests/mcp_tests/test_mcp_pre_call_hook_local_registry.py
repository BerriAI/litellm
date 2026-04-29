"""
Integration tests for pre_call_hook on local registry dispatch paths (fixes #25011).

These tests call call_mcp_tool directly (the real production code path) and
verify that pre_call_hook fires when the tool is in the local registry.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import TextContent

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _make_user_api_key_auth(**overrides) -> UserAPIKeyAuth:
    defaults = {
        "api_key": "sk-test",
        "user_id": "test-user",
        "team_id": "test-team",
        "end_user_id": None,
    }
    defaults.update(overrides)
    return UserAPIKeyAuth(**defaults)


def _make_mock_server(name="test_server"):
    return MCPServer(
        server_id="server-123",
        name=name,
        alias=name,
        server_name=name,
        url="https://test-server.com/mcp",
        transport=MCPTransport.http,
    )


@pytest.mark.asyncio
async def test_pre_call_hook_fires_for_local_registry_tool():
    """
    When call_mcp_tool dispatches to the local registry path (Path 1),
    pre_call_hook must be invoked before _handle_local_mcp_tool.
    """
    from litellm.proxy._experimental.mcp_server.server import (
        call_mcp_tool,
        global_mcp_server_manager,
    )

    proxy_logging = MagicMock(spec=ProxyLogging)
    proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
        return_value=MagicMock(tool_name="test_tool", arguments={"key": "val"})
    )
    proxy_logging._convert_mcp_to_llm_format = MagicMock(
        return_value={
            "model": "mcp-tool-call",
            "mcp_tool_name": "test_tool",
            "mcp_arguments": {"key": "val"},
        }
    )
    proxy_logging.pre_call_hook = AsyncMock(return_value=None)
    proxy_logging._convert_mcp_hook_response_to_kwargs = MagicMock(
        return_value={"arguments": {"key": "val"}}
    )

    user_auth = _make_user_api_key_auth()
    mock_server = _make_mock_server()
    local_tool_result = [TextContent(type="text", text="ok")]

    with (
        patch.object(
            global_mcp_server_manager,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[mock_server.server_id],
        ),
        patch.object(
            global_mcp_server_manager,
            "get_mcp_server_by_id",
            return_value=mock_server,
        ),
        patch.object(
            global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=None,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_tool_registry"
        ) as mock_registry,
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_local_mcp_tool",
            new_callable=AsyncMock,
            return_value=local_tool_result,
        ) as mock_handle_local,
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            proxy_logging,
        ),
    ):
        # Local registry returns a tool -> Path 1 taken
        mock_registry.get_tool.return_value = MagicMock()

        result = await call_mcp_tool(
            name="test_tool",
            arguments={"key": "val"},
            user_api_key_auth=user_auth,
            raw_headers={"Authorization": "Bearer test-jwt"},
        )

    # pre_call_hook was called
    proxy_logging.pre_call_hook.assert_called_once()
    call_kwargs = proxy_logging.pre_call_hook.call_args.kwargs
    assert call_kwargs["call_type"] == "call_mcp_tool"
    assert call_kwargs["user_api_key_dict"] == user_auth

    # _handle_local_mcp_tool was also called (tool executed)
    mock_handle_local.assert_called_once()


@pytest.mark.asyncio
async def test_pre_call_hook_blocks_local_registry_tool():
    """
    When pre_call_hook raises ValueError for a local registry tool,
    call_mcp_tool must propagate the exception and NOT call
    _handle_local_mcp_tool.
    """
    from litellm.proxy._experimental.mcp_server.server import (
        call_mcp_tool,
        global_mcp_server_manager,
    )

    proxy_logging = MagicMock(spec=ProxyLogging)
    proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
        return_value=MagicMock(tool_name="blocked_tool", arguments={})
    )
    proxy_logging._convert_mcp_to_llm_format = MagicMock(
        return_value={
            "model": "mcp-tool-call",
            "mcp_tool_name": "blocked_tool",
            "mcp_arguments": {},
        }
    )
    proxy_logging.pre_call_hook = AsyncMock(
        side_effect=ValueError("Tool 'blocked_tool' is not allowed")
    )

    user_auth = _make_user_api_key_auth()
    mock_server = _make_mock_server()

    with (
        patch.object(
            global_mcp_server_manager,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[mock_server.server_id],
        ),
        patch.object(
            global_mcp_server_manager,
            "get_mcp_server_by_id",
            return_value=mock_server,
        ),
        patch.object(
            global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=None,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_tool_registry"
        ) as mock_registry,
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_local_mcp_tool",
            new_callable=AsyncMock,
        ) as mock_handle_local,
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            proxy_logging,
        ),
    ):
        mock_registry.get_tool.return_value = MagicMock()

        with pytest.raises(ValueError, match="not allowed"):
            await call_mcp_tool(
                name="blocked_tool",
                arguments={"key": "val"},
                user_api_key_auth=user_auth,
            )

    # Hook was called
    proxy_logging.pre_call_hook.assert_called_once()
    # But _handle_local_mcp_tool was NOT called (blocked)
    mock_handle_local.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_hook_not_fired_for_managed_server_path():
    """
    When a tool is NOT in the local registry but IS on a managed server
    (Path 2), the local-path hook must NOT fire — the managed-server
    path fires its own hook via MCPServerManager.pre_call_tool_check.
    """
    from litellm.proxy._experimental.mcp_server.server import (
        call_mcp_tool,
        global_mcp_server_manager,
    )

    mock_server = MCPServer(
        server_id="server-123",
        name="test_server",
        alias="test_server",
        server_name="test_server",
        url="https://test-server.com/mcp",
        transport=MCPTransport.http,
    )

    proxy_logging = MagicMock(spec=ProxyLogging)
    proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
        return_value=MagicMock()
    )
    proxy_logging._convert_mcp_to_llm_format = MagicMock(
        return_value={"model": "mcp-tool-call"}
    )
    proxy_logging.pre_call_hook = AsyncMock(return_value=None)

    managed_result = MagicMock()

    with (
        patch.object(
            global_mcp_server_manager,
            "get_allowed_mcp_servers",
            new_callable=AsyncMock,
            return_value=[mock_server.server_id],
        ),
        patch.object(
            global_mcp_server_manager,
            "get_mcp_server_by_id",
            return_value=mock_server,
        ),
        patch.object(
            global_mcp_server_manager,
            "_get_mcp_server_from_tool_name",
            return_value=mock_server,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_tool_registry"
        ) as mock_registry,
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_managed_mcp_tool",
            new_callable=AsyncMock,
            return_value=managed_result,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.is_tool_allowed",
            return_value=True,
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
            proxy_logging,
        ),
    ):
        # Local registry returns None -> Path 2 (managed) taken
        mock_registry.get_tool.return_value = None

        await call_mcp_tool(
            name="test_server/some_tool",
            arguments={"key": "val"},
            mcp_servers=["test_server"],
        )

    # The local-path hook must NOT have fired
    proxy_logging.pre_call_hook.assert_not_called()
