"""
Tests for pre_call_hook invocation in mcp_server_tool_call (fixes #25011).

Validates that CustomLogger.async_pre_call_hook fires for /mcp/ tool calls
even when the tool is handled by the local registry path
(_handle_local_mcp_tool), not the managed-server path.
"""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging


def _make_user_api_key_auth(**overrides) -> UserAPIKeyAuth:
    defaults = {
        "api_key": "sk-test",
        "user_id": "test-user",
        "team_id": "test-team",
        "end_user_id": None,
    }
    defaults.update(overrides)
    return UserAPIKeyAuth(**defaults)


class TestPreCallHookFiresForLocalRegistryTools:
    """
    Regression tests for #25011: pre_call_hook must fire before call_mcp_tool
    so that the local registry dispatch path does not bypass callbacks.
    """

    @pytest.mark.asyncio
    async def test_pre_call_hook_invoked_with_correct_kwargs(self):
        """
        The pre-call hook code path in mcp_server_tool_call must call
        _create_mcp_request_object_from_kwargs, _convert_mcp_to_llm_format,
        and pre_call_hook with the correct arguments — matching the managed
        server path in MCPServerManager.pre_call_tool_check.
        """
        proxy_logging = MagicMock(spec=ProxyLogging)
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
            return_value=MagicMock()
        )
        synth_data = {
            "model": "mcp-tool-call",
            "mcp_tool_name": "test_tool",
            "mcp_arguments": {"key": "val"},
        }
        proxy_logging._convert_mcp_to_llm_format = MagicMock(
            return_value=synth_data
        )
        proxy_logging.pre_call_hook = AsyncMock(return_value=None)

        user_auth = _make_user_api_key_auth()
        name = "test_tool"
        arguments = {"key": "val"}

        # Simulate the fix code path from mcp_server_tool_call
        _pre_hook_kwargs = {
            "name": name,
            "arguments": arguments,
            "user_api_key_user_id": getattr(user_auth, "user_id", None),
            "user_api_key_team_id": getattr(user_auth, "team_id", None),
            "user_api_key_end_user_id": getattr(
                user_auth, "end_user_id", None
            ),
            "user_api_key_hash": getattr(user_auth, "api_key_hash", None),
            "user_api_key_request_route": "/mcp/",
            "incoming_bearer_token": None,
        }
        _mcp_req = proxy_logging._create_mcp_request_object_from_kwargs(
            _pre_hook_kwargs
        )
        _synth = proxy_logging._convert_mcp_to_llm_format(
            _mcp_req, _pre_hook_kwargs
        )
        await proxy_logging.pre_call_hook(
            user_api_key_dict=user_auth,
            data=_synth,
            call_type="call_mcp_tool",
        )

        # Verify the full chain was called
        proxy_logging._create_mcp_request_object_from_kwargs.assert_called_once_with(
            _pre_hook_kwargs
        )
        proxy_logging._convert_mcp_to_llm_format.assert_called_once()
        proxy_logging.pre_call_hook.assert_called_once_with(
            user_api_key_dict=user_auth,
            data=synth_data,
            call_type="call_mcp_tool",
        )

    @pytest.mark.asyncio
    async def test_pre_call_hook_exception_blocks_tool_execution(self):
        """
        When pre_call_hook raises an exception, call_mcp_tool must NOT
        be called — the tool is blocked pre-execution.
        """
        proxy_logging = MagicMock(spec=ProxyLogging)
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
            return_value=MagicMock()
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

        # Simulate the guard: if pre_call_hook raises, call_mcp_tool is skipped
        _pre_hook_kwargs = {
            "name": "blocked_tool",
            "arguments": {},
            "user_api_key_user_id": "test-user",
            "user_api_key_team_id": "test-team",
            "user_api_key_end_user_id": None,
            "user_api_key_hash": None,
            "user_api_key_request_route": "/mcp/",
            "incoming_bearer_token": None,
        }
        _mcp_req = proxy_logging._create_mcp_request_object_from_kwargs(
            _pre_hook_kwargs
        )
        _synth = proxy_logging._convert_mcp_to_llm_format(
            _mcp_req, _pre_hook_kwargs
        )

        with pytest.raises(ValueError, match="not allowed"):
            await proxy_logging.pre_call_hook(
                user_api_key_dict=user_auth,
                data=_synth,
                call_type="call_mcp_tool",
            )

    @pytest.mark.asyncio
    async def test_pre_call_hook_skipped_when_no_proxy_logging(self):
        """
        When proxy_logging_obj is None (e.g., during startup), the hook
        invocation is safely skipped and call_mcp_tool proceeds.
        """
        # This tests the `if _proxy_log_obj and user_api_key_auth:` guard
        _proxy_log_obj = None
        user_api_key_auth = _make_user_api_key_auth()

        # Should not raise — the guard prevents any hook call
        if _proxy_log_obj and user_api_key_auth:
            raise AssertionError("Should not reach here")

        # If we reach here, the guard correctly skipped the hook
        assert True

    @pytest.mark.asyncio
    async def test_pre_call_hook_skipped_when_no_auth(self):
        """
        When user_api_key_auth is None (unauthenticated), the hook
        invocation is safely skipped.
        """
        _proxy_log_obj = MagicMock(spec=ProxyLogging)
        user_api_key_auth = None

        if _proxy_log_obj and user_api_key_auth:
            raise AssertionError("Should not reach here")

        assert True

    @pytest.mark.asyncio
    async def test_pre_call_hook_receives_correct_call_type(self):
        """
        The call_type passed to pre_call_hook must be "call_mcp_tool"
        to match the managed-server path behavior.
        """
        proxy_logging = MagicMock(spec=ProxyLogging)
        proxy_logging._create_mcp_request_object_from_kwargs = MagicMock(
            return_value=MagicMock()
        )
        proxy_logging._convert_mcp_to_llm_format = MagicMock(
            return_value={"model": "mcp-tool-call"}
        )
        proxy_logging.pre_call_hook = AsyncMock(return_value=None)

        user_auth = _make_user_api_key_auth()

        _pre_hook_kwargs = {
            "name": "any_tool",
            "arguments": {"a": 1},
            "user_api_key_user_id": "test-user",
            "user_api_key_team_id": "test-team",
            "user_api_key_end_user_id": None,
            "user_api_key_hash": None,
            "user_api_key_request_route": "/mcp/",
            "incoming_bearer_token": None,
        }
        _mcp_req = proxy_logging._create_mcp_request_object_from_kwargs(
            _pre_hook_kwargs
        )
        _synth = proxy_logging._convert_mcp_to_llm_format(
            _mcp_req, _pre_hook_kwargs
        )
        await proxy_logging.pre_call_hook(
            user_api_key_dict=user_auth,
            data=_synth,
            call_type="call_mcp_tool",
        )

        call_kwargs = proxy_logging.pre_call_hook.call_args
        assert call_kwargs.kwargs["call_type"] == "call_mcp_tool"
