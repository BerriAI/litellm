"""
Regression tests: MCP tool calls must release the max_parallel_requests slot.

Field failure being prevented: every MCP tool call runs the parallel request
limiter's pre-call hook (+1) via ``pre_call_tool_check``, but MCP tool calls
never fire the litellm completion callbacks that normally release the slot
(``async_log_success_event`` / ``async_log_failure_event``). Without an
explicit release, every MCP tool call — including fully successful sequential
ones — leaks one slot, and a key with ``max_parallel_requests: N`` wedges
permanently after N calls until the proxy restarts.

Covers:
1. Sequential successful tool calls on a limit-1 key never 429 and leave the
   in-flight gauge at 0 (the exact field repro).
2. A tool call that raises still releases its slot (finally path).
3. An arbitrary (non-guardrail) exception from a custom pre-call hook after
   the limiter's +1 releases the slot before the exception propagates.
4. A downstream pre-call hook raising a guardrail exception after the limiter
   acquired a slot releases the slot before the exception propagates.
"""

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _make_server(name: str = "test_server") -> MCPServer:
    return MCPServer(
        server_id="test-id",
        name=name,
        server_name=name,
        url="https://example.com",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
    )


def _wire_proxy_logging(monkeypatch, extra_callbacks: Optional[list] = None):
    """
    Real ProxyLogging + real v3 limiter registered the same way the proxy
    does: in ``litellm.callbacks`` (walked by ``pre_call_hook``) and in
    ``proxy_hook_mapping`` (resolved by the slot release).
    """
    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    limiter = _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=proxy_logging.internal_usage_cache)
    proxy_logging.proxy_hook_mapping["parallel_request_limiter"] = limiter
    monkeypatch.setattr(litellm, "callbacks", [limiter] + (extra_callbacks or []))
    return proxy_logging, limiter


async def _in_flight_gauge(proxy_logging: ProxyLogging, limiter, api_key_hash: str) -> int:
    counter_key = f"{{api_key:{api_key_hash}}}:max_parallel_requests"
    raw = await proxy_logging.internal_usage_cache.dual_cache.async_get_cache(key=counter_key)
    return limiter._gauge_in_flight_from_cache_value(raw)


def _call_tool_patches(manager: MCPServerManager, server: MCPServer, tool_result: Any):
    """Patch everything network-facing; hook plumbing stays real."""
    if isinstance(tool_result, Exception):
        call_mock = AsyncMock(side_effect=tool_result)
    else:
        call_mock = AsyncMock(return_value=tool_result)
    return [
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager._resolve_byok_mcp_auth_header",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_resolve_mcp_server_for_tool_call", return_value=server),
        patch.object(manager, "check_allowed_or_banned_tools", return_value=True),
        patch.object(manager, "check_tool_permission_for_key_team", new_callable=AsyncMock),
        patch.object(manager, "validate_allowed_params"),
        patch.object(
            manager,
            "_resolve_oauth2_headers_for_tool_call",
            new=AsyncMock(return_value=None),
        ),
        patch.object(manager, "_create_during_hook_task", side_effect=lambda **_: asyncio.sleep(0)),
        patch.object(manager, "_call_regular_mcp_tool", call_mock),
    ]


@pytest.mark.asyncio
async def test_sequential_mcp_tool_calls_do_not_exhaust_parallel_slots(monkeypatch):
    """
    The field repro: a key with ``max_parallel_requests: 1`` making strictly
    sequential MCP tool calls (true concurrency never exceeds 1) must never
    be rate limited. Before the fix, call 1 leaked its slot and call 2
    rejected with a 429 — the key wedged permanently at its limit.
    """
    proxy_logging, limiter = _wire_proxy_logging(monkeypatch)
    manager = MCPServerManager()
    server = _make_server()
    _api_key = hash_token("sk-mcp-sequential")
    user_api_key_auth = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)

    patches = _call_tool_patches(manager, server, tool_result=MagicMock())
    for p in patches:
        p.start()
    try:
        for _ in range(3):
            await manager.call_tool(
                server_name="test_server",
                name="test_tool",
                arguments={"key": "val"},
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging,
            )
    finally:
        for p in patches:
            p.stop()

    assert await _in_flight_gauge(proxy_logging, limiter, _api_key) == 0


@pytest.mark.asyncio
async def test_mcp_tool_call_error_releases_parallel_slot(monkeypatch):
    """A tool call that raises must still release its slot (finally path)."""
    proxy_logging, limiter = _wire_proxy_logging(monkeypatch)
    manager = MCPServerManager()
    server = _make_server()
    _api_key = hash_token("sk-mcp-error")
    user_api_key_auth = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)

    patches = _call_tool_patches(manager, server, tool_result=RuntimeError("upstream MCP server exploded"))
    for p in patches:
        p.start()
    try:
        with pytest.raises(RuntimeError):
            await manager.call_tool(
                server_name="test_server",
                name="test_tool",
                arguments={"key": "val"},
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging,
            )
    finally:
        for p in patches:
            p.stop()

    assert await _in_flight_gauge(proxy_logging, limiter, _api_key) == 0


class _BlockingPreCallHook(CustomLogger):
    """Simulates a guardrail rejecting the call after the limiter ran."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Dict[str, Any]]:
        raise HTTPException(status_code=400, detail={"error": "blocked by guardrail"})


class _CrashingPreCallHook(CustomLogger):
    """Simulates a custom hook failing with an arbitrary (non-guardrail) error."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Dict[str, Any]]:
        raise RuntimeError("custom hook exploded")


@pytest.mark.asyncio
async def test_pre_call_arbitrary_hook_error_releases_parallel_slot(monkeypatch):
    """
    An arbitrary (non-guardrail) exception from a custom pre-call hook after
    the limiter's +1 must also release the slot: the exception escapes before
    call_tool enters its release-protected try/finally, so the release has to
    happen in pre_call_tool_check itself — for any exception type, not just
    the expected guardrail ones.
    """
    proxy_logging, limiter = _wire_proxy_logging(monkeypatch, extra_callbacks=[_CrashingPreCallHook()])
    manager = MCPServerManager()
    server = _make_server()
    _api_key = hash_token("sk-mcp-hook-crash")
    user_api_key_auth = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)

    patches = _call_tool_patches(manager, server, tool_result=MagicMock())
    for p in patches:
        p.start()
    try:
        with pytest.raises(RuntimeError):
            await manager.call_tool(
                server_name="test_server",
                name="test_tool",
                arguments={"key": "val"},
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging,
            )
    finally:
        for p in patches:
            p.stop()

    assert await _in_flight_gauge(proxy_logging, limiter, _api_key) == 0


@pytest.mark.asyncio
async def test_pre_call_guardrail_block_releases_parallel_slot(monkeypatch):
    """
    A downstream hook raising after the limiter's +1 must not strand the
    slot: ``async_post_call_failure_hook`` never runs for the MCP path, so
    ``pre_call_tool_check`` itself has to release before re-raising.
    """
    proxy_logging, limiter = _wire_proxy_logging(monkeypatch, extra_callbacks=[_BlockingPreCallHook()])
    manager = MCPServerManager()
    server = _make_server()
    _api_key = hash_token("sk-mcp-blocked")
    user_api_key_auth = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)

    patches = _call_tool_patches(manager, server, tool_result=MagicMock())
    for p in patches:
        p.start()
    try:
        with pytest.raises(HTTPException):
            await manager.call_tool(
                server_name="test_server",
                name="test_tool",
                arguments={"key": "val"},
                user_api_key_auth=user_api_key_auth,
                proxy_logging_obj=proxy_logging,
            )
    finally:
        for p in patches:
            p.stop()

    assert await _in_flight_gauge(proxy_logging, limiter, _api_key) == 0
