import asyncio
from typing import Dict, Optional

import pytest
from unittest.mock import patch

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

HOLD_SECONDS = 0.1


class _ConcurrencyTracker:
    """Records how many call_tool invocations are simultaneously in flight."""

    def __init__(self) -> None:
        self.current_by_server: Dict[str, int] = {}
        self.peak_by_server: Dict[str, int] = {}
        self.global_current = 0
        self.global_peak = 0

    def enter(self, server_id: str) -> None:
        self.current_by_server[server_id] = self.current_by_server.get(server_id, 0) + 1
        self.peak_by_server[server_id] = max(self.peak_by_server.get(server_id, 0), self.current_by_server[server_id])
        self.global_current += 1
        self.global_peak = max(self.global_peak, self.global_current)

    def exit(self, server_id: str) -> None:
        self.current_by_server[server_id] -= 1
        self.global_current -= 1


def _make_server(server_id: str, max_concurrent_requests: Optional[int]) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=server_id,
        server_name=server_id,
        url="https://example.com",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        max_concurrent_requests=max_concurrent_requests,
    )


def _patch_client_with_tracker(manager: MCPServerManager, tracker: _ConcurrencyTracker):
    async def fake_create_mcp_client(server, **kwargs):
        class _ProbeClient:
            async def call_tool(self, params, host_progress_callback=None):
                tracker.enter(server.server_id)
                try:
                    await asyncio.sleep(HOLD_SECONDS)
                    return "ok"
                finally:
                    tracker.exit(server.server_id)

        return _ProbeClient()

    return patch.object(manager, "_create_mcp_client", side_effect=fake_create_mcp_client)


async def _fire(manager: MCPServerManager, server: MCPServer, n: int) -> None:
    await asyncio.gather(
        *[
            manager._call_regular_mcp_tool(
                mcp_server=server,
                original_tool_name="tool",
                arguments={},
                tasks=[],
                mcp_auth_header=None,
                mcp_server_auth_headers=None,
                oauth2_headers=None,
                raw_headers=None,
                proxy_logging_obj=None,
            )
            for _ in range(n)
        ]
    )


@pytest.mark.asyncio
async def test_max_concurrent_requests_caps_in_flight_tool_calls():
    """A configured cap of 2 must never let more than 2 calls hit one server at once."""
    manager = MCPServerManager()
    tracker = _ConcurrencyTracker()
    server = _make_server("srv-limited", max_concurrent_requests=2)

    with _patch_client_with_tracker(manager, tracker):
        await _fire(manager, server, n=8)

    assert tracker.peak_by_server["srv-limited"] == 2


@pytest.mark.asyncio
async def test_unset_limit_allows_unbounded_concurrency():
    """With no cap, all calls run concurrently (backward-compatible default)."""
    manager = MCPServerManager()
    tracker = _ConcurrencyTracker()
    server = _make_server("srv-unbounded", max_concurrent_requests=None)

    with _patch_client_with_tracker(manager, tracker):
        await _fire(manager, server, n=6)

    assert tracker.peak_by_server["srv-unbounded"] == 6


@pytest.mark.asyncio
async def test_non_positive_limit_is_treated_as_unlimited():
    """A cap of 0 must not deadlock; it means unlimited, not a zero-permit semaphore."""
    manager = MCPServerManager()
    tracker = _ConcurrencyTracker()
    server = _make_server("srv-zero", max_concurrent_requests=0)

    with _patch_client_with_tracker(manager, tracker):
        await asyncio.wait_for(_fire(manager, server, n=5), timeout=5)

    assert tracker.peak_by_server["srv-zero"] == 5


@pytest.mark.asyncio
async def test_limit_is_scoped_per_server():
    """Each server gets its own limiter; one server's cap must not throttle another."""
    manager = MCPServerManager()
    tracker = _ConcurrencyTracker()
    server_a = _make_server("srv-a", max_concurrent_requests=1)
    server_b = _make_server("srv-b", max_concurrent_requests=1)

    with _patch_client_with_tracker(manager, tracker):
        await asyncio.gather(
            _fire(manager, server_a, n=3),
            _fire(manager, server_b, n=3),
        )

    assert tracker.peak_by_server["srv-a"] == 1
    assert tracker.peak_by_server["srv-b"] == 1
    assert tracker.global_peak == 2


@pytest.mark.asyncio
async def test_openapi_backed_server_also_respects_the_cap():
    """OpenAPI (spec_path) servers dispatch through a different handler; the cap
    must apply there too, not only on the regular MCP client path."""
    manager = MCPServerManager()
    tracker = _ConcurrencyTracker()
    server = _make_server("srv-openapi", max_concurrent_requests=2)
    server.spec_path = "/fake/openapi.json"

    async def fake_openapi_handler(mcp_server, name, arguments):
        tracker.enter(mcp_server.server_id)
        try:
            await asyncio.sleep(HOLD_SECONDS)
            return "ok"
        finally:
            tracker.exit(mcp_server.server_id)

    with (
        patch.object(manager, "_resolve_mcp_server_for_tool_call", return_value=server),
        patch.object(manager, "_call_openapi_tool_handler", side_effect=fake_openapi_handler),
    ):
        await asyncio.gather(
            *[manager.call_tool(server_name="srv-openapi", name="tool", arguments={}) for _ in range(6)]
        )

    assert tracker.peak_by_server["srv-openapi"] == 2


@pytest.mark.asyncio
async def test_edited_limit_takes_effect_without_restart():
    """Editing max_concurrent_requests must rebuild the cached semaphore so the
    new cap applies to subsequent calls immediately, not only after a restart."""
    manager = MCPServerManager()
    server = _make_server("srv-edited", max_concurrent_requests=3)

    before_edit = _ConcurrencyTracker()
    with _patch_client_with_tracker(manager, before_edit):
        await _fire(manager, server, n=6)
    assert before_edit.peak_by_server["srv-edited"] == 3

    server.max_concurrent_requests = 1
    after_edit = _ConcurrencyTracker()
    with _patch_client_with_tracker(manager, after_edit):
        await _fire(manager, server, n=6)
    assert after_edit.peak_by_server["srv-edited"] == 1


def test_semaphore_is_reused_per_server_and_distinct_across_servers():
    manager = MCPServerManager()
    server_a = _make_server("srv-a", max_concurrent_requests=3)
    server_b = _make_server("srv-b", max_concurrent_requests=3)

    sem_a_first = manager._get_call_semaphore(server_a)
    sem_a_second = manager._get_call_semaphore(server_a)
    sem_b = manager._get_call_semaphore(server_b)

    assert sem_a_first is sem_a_second
    assert sem_a_first is not sem_b


def test_no_semaphore_created_when_limit_absent():
    manager = MCPServerManager()
    server = _make_server("srv-none", max_concurrent_requests=None)

    assert manager._get_call_semaphore(server) is None
