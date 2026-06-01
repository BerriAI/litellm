"""
Regression tests for MCP auth/redaction hardening:
- public discovery routes vs path smuggling
- M2M (client_credentials) exclusion from the anonymous OAuth2 fallback
- case-insensitive tools/call allow/deny enforcement
- public MCP hub credential redaction
- client-IP enforcement on tools/call (not just discovery)
"""

import os
import sys

import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.fixture(autouse=True)
def cleanup_mcp_global_state():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    global_mcp_server_manager.registry.clear()
    yield
    global_mcp_server_manager.registry.clear()


# --------------------------------------------------------------------------- #
# Path smuggling: only the registered discovery routes are public (VERIA-158)
# --------------------------------------------------------------------------- #


def test_is_public_mcp_discovery_route_allows_only_known_families():
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    f = MCPRequestHandler._is_public_mcp_discovery_route

    # Exactly the registered discovery templates are public.
    assert f("/.well-known/oauth-authorization-server")
    assert f("/.well-known/oauth-authorization-server/my-server")  # /{server}
    assert f("/.well-known/oauth-protected-resource/mcp/my-server")  # /mcp/{server}
    assert f("/.well-known/oauth-protected-resource/my-server/mcp")  # /{server}/mcp
    assert f("/.well-known/openid-configuration")
    assert f("/.well-known/jwks.json")

    # Smuggled / non-discovery paths must NOT get the anonymous grant.
    assert not f("/.well-known/smuggled")
    assert not f("/.well-known/")
    assert not f("/.well-known/oauth-authorization-server-evil")  # prefix boundary
    assert not f("/mcp/tools/call")
    # Static endpoints have no sub-paths; a sub-path is a smuggle attempt.
    assert not f("/.well-known/jwks.json/anything")
    assert not f("/.well-known/openid-configuration/anything")
    # Extra/unregistered segments under a parameterized prefix are rejected.
    assert not f("/.well-known/oauth-authorization-server/foo/bar")  # 2 segs, no mcp
    assert not f("/.well-known/oauth-authorization-server/mcp/srv/extra")  # too deep
    assert not f("/.well-known/oauth-protected-resource/srv/")  # trailing slash


@pytest.mark.asyncio
async def test_public_discovery_anonymous_grant_is_get_only():
    """The anonymous grant for a discovery-shaped path is GET/HEAD only. A POST
    (an MCP JSON-RPC tool call smuggled onto a discovery path) must fall through
    to normal auth instead of receiving an anonymous session."""
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )
    from litellm.proxy._types import UserAPIKeyAuth

    discovery_path = "/.well-known/oauth-authorization-server"

    async def fake_auth(api_key, request):
        # Distinct marker proving normal auth ran (not the anonymous short-circuit).
        return UserAPIKeyAuth(user_id="authed-via-normal-path")

    async def run(method):
        scope = {
            "type": "http",
            "method": method,
            "path": discovery_path,
            "headers": [],
        }
        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth",
            side_effect=fake_auth,
        ) as mock_auth:
            auth_result = (await MCPRequestHandler.process_mcp_request(scope))[0]
            return auth_result, mock_auth

    # GET → anonymous short-circuit; normal auth is never consulted.
    get_result, get_mock = await run("GET")
    assert get_result.user_id is None
    get_mock.assert_not_called()

    # POST → no anonymous grant; falls through to normal auth.
    post_result, post_mock = await run("POST")
    post_mock.assert_called_once()
    assert post_result.user_id == "authed-via-normal-path"


# --------------------------------------------------------------------------- #
# Anonymous OAuth2 fallback must exclude M2M servers (VERIA-116)
# --------------------------------------------------------------------------- #


def _oauth2_server(server_id, m2m):
    return MCPServer(
        server_id=server_id,
        name=server_id,
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow="client_credentials" if m2m else None,
    )


@pytest.mark.parametrize("m2m,expected", [(True, False), (False, True)])
def test_target_servers_use_oauth2_excludes_m2m(m2m, expected):
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )
    import litellm.proxy._experimental.mcp_server.mcp_server_manager as mgr_mod

    server = _oauth2_server("s1", m2m=m2m)
    assert server.has_client_credentials is m2m  # guard our own assumption

    with (
        patch.object(
            MCPRequestHandler, "_resolve_target_server_names", return_value=["s1"]
        ),
        patch.object(
            mgr_mod.global_mcp_server_manager,
            "get_mcp_server_by_name",
            return_value=server,
        ),
    ):
        # M2M server -> False (no anonymous passthrough); interactive -> True.
        assert (
            MCPRequestHandler._target_servers_use_oauth2(
                path="/mcp", mcp_servers=["s1"]
            )
            is expected
        )


# --------------------------------------------------------------------------- #
# tools/call allow/deny is case-insensitive (VERIA-184)
# --------------------------------------------------------------------------- #


def test_check_allowed_or_banned_tools_denylist_case_insensitive_allowlist_exact():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    f = global_mcp_server_manager.check_allowed_or_banned_tools

    # Denylist stays case-insensitive: a camelCase call to a lowercase-banned
    # tool must still be blocked.
    denied = MCPServer(
        server_id="s1",
        name="srv",
        transport=MCPTransport.http,
        disallowed_tools=["dangeroustool"],
    )
    assert f("DangerousTool", denied) is False
    assert f("other", denied) is True

    # Allowlist is exact: a case-variant of an allowed tool must be rejected, or a
    # caller could invoke a different case-sensitive upstream tool whose name
    # differs only by case (tool allowlist bypass by case collision).
    allowed = MCPServer(
        server_id="s2",
        name="srv",
        transport=MCPTransport.http,
        allowed_tools=["safetool"],
    )
    assert f("safetool", allowed) is True  # exact configured name
    assert f("SafeTool", allowed) is False  # case collision -> rejected
    assert f("nope", allowed) is False

    # The exact server-prefixed form is accepted; its case-variant is not.
    prefixed = MCPServer(
        server_id="s3",
        name="srv",
        transport=MCPTransport.http,
        allowed_tools=["srv-safetool"],
    )
    assert f("safetool", prefixed) is True  # matches the prefixed allowlist entry
    assert f("SafeTool", prefixed) is False  # case collision on prefixed form


# --------------------------------------------------------------------------- #
# Public MCP hub redacts credentials (VERIA-159)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_public_mcp_hub_redacts_secrets():
    import litellm.proxy._experimental.mcp_server.mcp_server_manager as mgr_mod
    from litellm.proxy.public_endpoints.public_endpoints import get_mcp_servers

    server = MCPServer(
        server_id="s1",
        name="srv",
        transport=MCPTransport.http,
        url="https://user:secretpw@upstream.example.com/path",
        spec_path="https://upstream.example.com/openapi.json?api_key=spec-secret",
        mcp_info={"description": "a server", "api_key": "sk-mcp-info-secret"},
    )

    with patch.object(
        mgr_mod.global_mcp_server_manager,
        "get_public_mcp_servers",
        return_value=[server],
    ):
        result = await get_mcp_servers()

    pub = result[0]
    # Credential carriers and free-form metadata are dropped entirely on the
    # unauthenticated hub.
    assert pub.url is None
    assert pub.spec_path is None
    assert pub.mcp_info is None


# --------------------------------------------------------------------------- #
# tools/call enforces client-IP filtering, not just discovery (VERIA-183/133)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_call_mcp_tool_blocks_internal_only_server_from_external_ip():
    import litellm.proxy._experimental.mcp_server.server as server_mod
    from litellm.proxy._experimental.mcp_server.server import call_mcp_tool

    internal_only = MCPServer(
        server_id="internal-1",
        name="internal",
        transport=MCPTransport.http,
        available_on_public_internet=False,
    )
    server_mod.global_mcp_server_manager.registry["internal-1"] = internal_only

    auth = MagicMock()

    async def _passthrough(mcp_servers, allowed_mcp_servers):
        return allowed_mcp_servers

    # execute_mcp_tool raises a distinctive 418 so "reached execution" is
    # observable without exercising the response/logging post-processing.
    reached = HTTPException(status_code=418, detail="reached-execute")

    with (
        patch.object(
            server_mod.global_mcp_server_manager,
            "get_allowed_mcp_servers",
            AsyncMock(return_value=["internal-1"]),
        ),
        patch.object(
            server_mod,
            "_get_allowed_mcp_servers_from_mcp_server_names",
            _passthrough,
        ),
        patch.object(
            server_mod, "execute_mcp_tool", AsyncMock(side_effect=reached)
        ) as mock_exec,
    ):
        # External IP -> internal-only server filtered out -> 403, never executes.
        with pytest.raises(HTTPException) as exc:
            await call_mcp_tool(
                name="internal-tool",
                arguments={},
                user_api_key_auth=auth,
                client_ip="8.8.8.8",
            )
        assert exc.value.status_code == 403
        mock_exec.assert_not_called()

        # Unknown IP (internal/trusted call) -> filter fails open -> reaches execute.
        with pytest.raises(HTTPException) as exc2:
            await call_mcp_tool(
                name="internal-tool",
                arguments={},
                user_api_key_auth=auth,
                client_ip=None,
            )
        assert exc2.value.status_code == 418
        mock_exec.assert_awaited()
