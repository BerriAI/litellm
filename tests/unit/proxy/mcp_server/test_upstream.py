import queue
from collections.abc import Iterator, Mapping
from collections.abc import Set as AbstractSet
from typing import Final
from unittest.mock import MagicMock

import httpx2
import pytest
from mcp.server.context import ServerRequestContext
from mcp.types import CallToolResult

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server.mcp_context import active_mcp_request_ctx_var
from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy.mcp_server.outbound_credentials.resolver import UpstreamCredentialProvider
from litellm.proxy.mcp_server.upstream import prepare_upstream_client
from litellm.proxy.utils import ProxyLogging
from litellm.types.mcp import MCPAuth, MCPTransport, MCPUpstreamProtocol
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from tests._support.mcp_auth_profiles import (
    MINTED_PROFILE_TOKEN,
    ProfileUpstream,
    call_tickets_add,
    caller_granted,
    install_profile_manager,
    profile_manager,
    tickets_server,
)
from tests._support.mcp_count_tool_upstream import count_tool_upstream, methods_received

_HTTP_URL: Final = "https://upstream.example/mcp"
_SERVER_SHAPES: Final = pytest.mark.parametrize(
    ("server_fields", "per_request_token", "forwarded_token"),
    [
        ({"transport": MCPTransport.http, "url": _HTTP_URL}, None, None),
        (
            {
                "transport": MCPTransport.http,
                "url": _HTTP_URL,
                "auth_type": MCPAuth.bearer_token,
                "authentication_token": "static-token",
            },
            None,
            None,
        ),
        ({"transport": MCPTransport.http, "url": _HTTP_URL, "auth_type": MCPAuth.bearer_token}, "caller-key", None),
        ({"transport": MCPTransport.http, "url": _HTTP_URL, "auth_type": MCPAuth.oauth_delegate}, None, "caller-token"),
    ],
    ids=["no-auth", "static-bearer-token", "per-request-token", "forwarded-caller-token"],
)


@pytest.fixture
def _active_mcp_request_context() -> Iterator[None]:
    context: Final = ServerRequestContext(
        session=MagicMock(), lifespan_context={}, protocol_version="2025-11-25", method="tools/call", request=None
    )
    token: Final = active_mcp_request_ctx_var.set(context)
    try:
        yield
    finally:
        active_mcp_request_ctx_var.reset(token)


def _server(server_fields: Mapping[str, object], skip_setting: bool | None) -> MCPServer:
    return MCPServer.model_validate(
        {
            **server_fields,
            "server_id": "skip-validation",
            "name": "skip_validation",
            "mcp_info": {} if skip_setting is None else {"skip_tool_output_validation": skip_setting},
        }
    )


async def _call_count_tool(
    manager: MCPServerManager,
    server: MCPServer,
    *,
    per_request_token: str | None = None,
    forwarded_token: str | None = None,
    gateway_session_id: str | None = None,
) -> CallToolResult:
    result: Final = await manager._call_regular_mcp_tool(
        mcp_server=server,
        original_tool_name="count",
        arguments={},
        tasks=[],
        mcp_auth_header=per_request_token,
        mcp_server_auth_headers=None,
        oauth2_headers=None if forwarded_token is None else {"Authorization": f"Bearer {forwarded_token}"},
        raw_headers=None if gateway_session_id is None else {"mcp-session-id": gateway_session_id},
        proxy_logging_obj=None,
    )
    assert isinstance(result, CallToolResult)
    return result


@pytest.mark.asyncio
@_SERVER_SHAPES
@pytest.mark.parametrize(
    ("skip_setting", "rejected"), [(None, True), (False, True), (True, False)], ids=["unset", "off", "on"]
)
async def test_the_setting_alone_decides_whether_a_tool_result_is_checked(
    server_fields: Mapping[str, object],
    per_request_token: str | None,
    forwarded_token: str | None,
    skip_setting: bool | None,
    rejected: bool,
) -> None:
    upstream: Final = count_tool_upstream({"n": "not-an-integer"}, queue.SimpleQueue())
    manager: Final = MCPServerManager(upstream_transport=httpx2.MockTransport(upstream))

    result: Final = await _call_count_tool(
        manager,
        _server(server_fields, skip_setting),
        per_request_token=per_request_token,
        forwarded_token=forwarded_token,
    )

    assert result.is_error is rejected


@pytest.mark.asyncio
@_SERVER_SHAPES
async def test_a_skipping_server_is_never_asked_for_its_tools(
    server_fields: Mapping[str, object], per_request_token: str | None, forwarded_token: str | None
) -> None:
    methods_seen: Final[queue.SimpleQueue[str]] = queue.SimpleQueue()
    manager: Final = MCPServerManager(
        upstream_transport=httpx2.MockTransport(count_tool_upstream({"n": 1}, methods_seen))
    )

    result: Final = await _call_count_tool(
        manager,
        _server(server_fields, True),
        per_request_token=per_request_token,
        forwarded_token=forwarded_token,
    )

    assert result.is_error is False
    assert "tools/list" not in methods_received(methods_seen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_setting", "second_setting", "rejected_in_order"),
    [(True, False, (False, True)), (False, True, (True, False))],
    ids=["turned-off", "turned-on"],
)
async def test_a_changed_setting_applies_to_the_next_call_in_a_live_gateway_session(
    _active_mcp_request_context: None,
    first_setting: bool,
    second_setting: bool,
    rejected_in_order: tuple[bool, bool],
) -> None:
    upstream: Final = count_tool_upstream({"n": "not-an-integer"}, queue.SimpleQueue())
    manager: Final = MCPServerManager(upstream_transport=httpx2.MockTransport(upstream))
    manager.track_gateway_session("gateway-1")
    fields: Final = {"transport": MCPTransport.http, "url": _HTTP_URL}
    try:
        first: Final = await _call_count_tool(manager, _server(fields, first_setting), gateway_session_id="gateway-1")
        second: Final = await _call_count_tool(manager, _server(fields, second_setting), gateway_session_id="gateway-1")
    finally:
        await manager.close_all_upstream_sessions()

    assert (first.is_error, second.is_error) == rejected_in_order


@pytest.mark.asyncio
async def test_a_stdio_upstream_client_carries_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_ENABLE_MCP_STDIO", "true")
    server: Final = _server({"transport": MCPTransport.stdio, "command": "python", "args": ["server.py"]}, True)

    client: Final = await prepare_upstream_client(
        server, provider=UpstreamCredentialProvider(), root_path="/mcp", protocol_version="auto"
    )

    assert client.skip_tool_output_validation is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol_version", "expected"), [("auto", True), ("2025-11-25", True), ("2026-07-28", False)]
)
async def test_upstream_client_ignores_the_setting_where_the_call_already_lists_tools(
    protocol_version: MCPUpstreamProtocol, expected: bool
) -> None:
    server: Final = MCPServer.model_validate(
        {
            "server_id": "skip-validation",
            "name": "skip_validation",
            "url": _HTTP_URL,
            "transport": MCPTransport.http,
            "protocol_version": protocol_version,
            "mcp_info": {"skip_tool_output_validation": True},
        }
    )

    client: Final = await prepare_upstream_client(
        server, provider=UpstreamCredentialProvider(), root_path="/mcp", protocol_version=protocol_version
    )

    assert client.skip_tool_output_validation is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("granted", "expected"),
    [
        pytest.param(None, "Bearer caller-header", id="without-a-grant-the-callers-header-still-wins"),
        pytest.param("bots", "Bearer bots-token", id="a-selected-profile-is-not-overridden-by-the-callers-header"),
        pytest.param(
            "ops", f"Bearer {MINTED_PROFILE_TOKEN}", id="a-minted-token-is-not-overridden-by-the-callers-header"
        ),
    ],
)
@pytest.mark.parametrize("per_server", [False, True], ids=["global-header", "per-server-header"])
async def test_a_callers_auth_header_does_not_replace_a_selected_profiles_credential(
    monkeypatch: pytest.MonkeyPatch, granted: str | None, expected: str, per_server: bool
) -> None:
    upstream: Final = ProfileUpstream()
    server: Final = tickets_server()
    install_profile_manager(monkeypatch, profile_manager(server, upstream))

    await call_tickets_add(
        server,
        caller_granted(granted),
        mcp_auth_header=None if per_server else "Bearer caller-header",
        mcp_server_auth_headers={"tickets": {"Authorization": "Bearer caller-header"}} if per_server else None,
    )

    assert upstream.credentials_seen() == {expected}


class _ApiKeyUpstream(ProfileUpstream):
    def __init__(self) -> None:
        super().__init__()
        self.api_keys: Final[list[str | None]] = []  # mutable-ok: the fake records what it was sent

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            self.api_keys.append(request.headers.get("x-api-key"))
        return super().__call__(request)


class _HookAddingAnApiKey(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
        return {**data, "extra_headers": {"X-API-Key": "hook-key"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("granted", "expected"),
    [
        pytest.param(None, {"hook-key"}, id="without-a-grant-the-hook-header-reaches-the-upstream"),
        pytest.param("keys", {"profile-key"}, id="a-selected-api-key-profile-is-not-replaced-by-the-hook-header"),
    ],
)
async def test_a_hook_header_does_not_replace_a_selected_api_key_profiles_credential(
    monkeypatch: pytest.MonkeyPatch, granted: str | None, expected: AbstractSet[str]
) -> None:
    upstream: Final = _ApiKeyUpstream()
    server: Final = tickets_server(profiles=({"id": "keys", "auth_type": "api_key", "auth_value": "profile-key"},))
    manager: Final = profile_manager(server, upstream)
    install_profile_manager(monkeypatch, manager)
    monkeypatch.setattr(litellm, "callbacks", [_HookAddingAnApiKey()])

    await manager.call_tool(
        server_name="tickets",
        name="add",
        arguments={},
        user_api_key_auth=caller_granted(granted),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=UserApiKeyCache()),
    )

    assert set(upstream.api_keys) == expected
