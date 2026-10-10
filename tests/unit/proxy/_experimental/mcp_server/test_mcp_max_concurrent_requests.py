import asyncio
from collections.abc import Awaitable, Callable
from typing import Dict, Final, Optional, TypeVar, cast
from unittest.mock import AsyncMock, patch
from mcp import ClientSession
from mcp.server.context import ServerRequestContext
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from litellm.experimental_mcp_client.client import MCPClient, PersistentMCPSession
from litellm.proxy._experimental.mcp_server.mcp_context import active_mcp_request_ctx_var

import pytest
from mcp.types import CallToolResult, TextContent

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

HOLD_SECONDS = 0.1


_SessionResult = TypeVar("_SessionResult")


class _AdmissionProbeClient(MCPClient):
    def __init__(
        self,
        server_id: str,
        fingerprint: str,
        tracker: "_ConcurrencyTracker",
        submitted: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        super().__init__(server_url="http://upstream/mcp", transport_type=MCPTransport.http)
        self._server_id: Final = server_id
        self._fingerprint: Final = fingerprint
        self._tracker: Final = tracker
        self._submitted: Final = submitted
        self._release: Final = release

    async def discovery_auth_fingerprint(self) -> str:
        return self._fingerprint

    async def run_with_session(
        self,
        operation: Callable[[ClientSession], Awaitable[_SessionResult]],
        *,
        quiet_on_error: bool = False,
        on_stream_error: Callable[[asyncio.Future[Exception]], None] | None = None,
        on_cleanup: Callable[[], None] | None = None,
    ) -> _SessionResult:
        del quiet_on_error, on_stream_error, on_cleanup
        return await operation(cast(ClientSession, object()))

    async def _perform_call(self) -> CallToolResult:
        self._tracker.enter(self._server_id)
        try:
            if self._release is not None:
                await self._release.wait()
            return CallToolResult(content=[], isError=False)
        finally:
            self._tracker.exit(self._server_id)

    async def call_tool(
        self,
        call_tool_request_params: MCPCallToolRequestParams,
        host_progress_callback: Callable | None = None,
        raise_on_error: bool = False,
        allow_input_required: bool = False,
        persistent_session: PersistentMCPSession | None = None,
    ) -> CallToolResult:
        del call_tool_request_params, host_progress_callback, raise_on_error, allow_input_required
        if self._submitted is not None:
            self._submitted.set()
        if persistent_session is None:
            return await self._perform_call()
        return await persistent_session.run(lambda _: self._perform_call())


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


def _make_server(
    server_id: str,
    max_concurrent_requests: Optional[int],
    auth_type: MCPAuth = MCPAuth.none,
) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=server_id,
        server_name=server_id,
        url="https://example.com",
        transport=MCPTransport.http,
        auth_type=auth_type,
        max_concurrent_requests=max_concurrent_requests,
    )


def _patch_client_with_tracker(manager: MCPServerManager, tracker: _ConcurrencyTracker):
    async def fake_create_mcp_client(server, **kwargs):
        class _ProbeClient:
            async def call_tool(
                self,
                params,
                host_progress_callback=None,
                allow_input_required=False,
                persistent_session: PersistentMCPSession | None = None,
            ):
                tracker.enter(server.server_id)
                try:
                    await asyncio.sleep(HOLD_SECONDS)
                    return "ok"
                finally:
                    tracker.exit(server.server_id)

        return _ProbeClient()

    return patch.object(manager, "create_mcp_client", side_effect=fake_create_mcp_client)


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


@pytest.mark.parametrize("auth_type", [MCPAuth.none, MCPAuth.oauth2_token_exchange])
@pytest.mark.asyncio
async def test_persistent_tool_call_with_limit_one_completes_without_nested_admission(
    auth_type: MCPAuth,
) -> None:
    manager: Final = MCPServerManager()
    server: Final = _make_server("persistent-limit-one", 1, auth_type)
    tracker: Final = _ConcurrencyTracker()
    client: Final = _AdmissionProbeClient(server.server_id, "auth-fingerprint", tracker)
    gateway_session_id: Final = "gw-limit-one"
    oauth2_headers: Final = (
        {"Authorization": "Bearer subject-token"} if auth_type == MCPAuth.oauth2_token_exchange else None
    )
    context_token: Final = active_mcp_request_ctx_var.set(cast(ServerRequestContext, object()))
    manager.track_gateway_session(gateway_session_id)
    try:
        with patch.object(manager, "create_mcp_client", new=AsyncMock(return_value=client)):
            response: Final = await asyncio.wait_for(
                manager._call_regular_mcp_tool(
                    mcp_server=server,
                    original_tool_name="tool",
                    arguments={},
                    mcp_auth_header=None,
                    mcp_server_auth_headers=None,
                    oauth2_headers=oauth2_headers,
                    raw_headers={"mcp-session-id": gateway_session_id},
                    proxy_logging_obj=None,
                    tasks=[],
                ),
                1,
            )
            assert response.is_error is False
    finally:
        sessions_to_close: Final = tuple(manager._upstream_sessions.values())
        manager.release_upstream_sessions(gateway_session_id)
        await asyncio.gather(*(session.wait_closed() for session in sessions_to_close))
        active_mcp_request_ctx_var.reset(context_token)


@pytest.mark.asyncio
async def test_persistent_session_admission_uses_updated_server_limit() -> None:
    manager: Final = MCPServerManager()
    server: Final = _make_server("persistent-edited-limit", 2)
    manager.registry = {server.server_id: server}
    tracker: Final = _ConcurrencyTracker()
    gateway_session_id: Final = "gw-edited-limit"
    second_submitted: Final = asyncio.Event()
    release_second: Final = asyncio.Event()
    first_client: Final = _AdmissionProbeClient(server.server_id, "auth-fingerprint", tracker)
    second_client: Final = _AdmissionProbeClient(
        server.server_id,
        "auth-fingerprint",
        tracker,
        submitted=second_submitted,
        release=release_second,
    )
    context_token: Final = active_mcp_request_ctx_var.set(cast(ServerRequestContext, object()))
    manager.track_gateway_session(gateway_session_id)

    async def dispatch() -> CallToolResult:
        return await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={"mcp-session-id": gateway_session_id},
            proxy_logging_obj=None,
            tasks=[],
        )

    try:
        with patch.object(manager, "create_mcp_client", new=AsyncMock(side_effect=(first_client, second_client))):
            first_response: Final = await asyncio.wait_for(dispatch(), 1)
            assert first_response.is_error is False

            edited_server: Final = server.model_copy(update={"max_concurrent_requests": 1})
            manager.registry = {server.server_id: edited_server}
            edited_semaphore: Final = manager._get_call_semaphore(edited_server)
            assert edited_semaphore is not None
            await edited_semaphore.acquire()

            second_call: Final = asyncio.create_task(dispatch())
            try:
                await asyncio.wait_for(second_submitted.wait(), 1)
                await asyncio.sleep(HOLD_SECONDS)
                assert not second_call.done()
                assert tracker.current_by_server.get(server.server_id, 0) == 0
            finally:
                edited_semaphore.release()
                release_second.set()
                await asyncio.gather(second_call, return_exceptions=True)

            second_response: Final = await asyncio.wait_for(second_call, 1)
            assert second_response.is_error is False
    finally:
        release_second.set()
        sessions_to_close: Final = tuple(manager._upstream_sessions.values()) + tuple(
            session for _, session in manager._retiring_upstream_sessions
        )
        manager.release_upstream_sessions(gateway_session_id)
        await asyncio.gather(*(session.wait_closed() for session in sessions_to_close), return_exceptions=True)
        active_mcp_request_ctx_var.reset(context_token)


@pytest.mark.asyncio
async def test_queued_persistent_call_survives_header_change_while_waiting_for_admission() -> None:
    manager: Final = MCPServerManager()
    server: Final = _make_server("persistent-header-change", 1)
    tracker: Final = _ConcurrencyTracker()
    gateway_session_id: Final = "gw-header-change"
    holder_started: Final = asyncio.Event()
    release_holder: Final = asyncio.Event()
    first_submitted: Final = asyncio.Event()
    second_submitted: Final = asyncio.Event()
    first_client: Final = _AdmissionProbeClient(
        server.server_id, "first-fingerprint", tracker, submitted=first_submitted
    )
    second_client: Final = _AdmissionProbeClient(
        server.server_id, "second-fingerprint", tracker, submitted=second_submitted
    )
    context_token: Final = active_mcp_request_ctx_var.set(cast(ServerRequestContext, object()))
    manager.track_gateway_session(gateway_session_id)

    async def hold_slot() -> None:
        async with manager._limit_outbound_concurrency(server):
            holder_started.set()
            await release_holder.wait()

    async def dispatch(authorization: str) -> CallToolResult:
        return await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers={"mcp-session-id": gateway_session_id, "Authorization": authorization},
            proxy_logging_obj=None,
            tasks=[],
        )

    holder_task: Final = asyncio.create_task(hold_slot())
    call_tasks: Final[list[asyncio.Task[CallToolResult]]] = []
    try:
        await asyncio.wait_for(holder_started.wait(), 1)
        with patch.object(manager, "create_mcp_client", new=AsyncMock(side_effect=(first_client, second_client))):
            call_tasks.append(asyncio.create_task(dispatch("Bearer old")))
            await asyncio.wait_for(first_submitted.wait(), 1)
            old_session: Final = next(iter(manager._upstream_sessions.values()))
            call_tasks.append(asyncio.create_task(dispatch("Bearer new")))
            await asyncio.wait_for(second_submitted.wait(), 1)

            assert (gateway_session_id, old_session) in manager._retiring_upstream_sessions
            assert not old_session.closed
            assert tracker.current_by_server.get(server.server_id, 0) == 0

            release_holder.set()
            responses: Final = await asyncio.wait_for(asyncio.gather(*call_tasks), 1)

        assert all(response.is_error is False for response in responses)
        await asyncio.wait_for(old_session.wait_closed(), 1)
    finally:
        release_holder.set()
        sessions_to_close: Final = tuple(
            manager._upstream_sessions.values()
        ) + tuple(session for _, session in manager._retiring_upstream_sessions)
        manager.release_upstream_sessions(gateway_session_id)
        await asyncio.gather(*(session.wait_closed() for session in sessions_to_close), return_exceptions=True)
        await asyncio.gather(*call_tasks, return_exceptions=True)
        await holder_task
        active_mcp_request_ctx_var.reset(context_token)


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

    async def fake_openapi_handler(mcp_server, name, arguments, wire_compat):
        tracker.enter(mcp_server.server_id)
        try:
            await asyncio.sleep(HOLD_SECONDS)
            return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
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
