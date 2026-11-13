"""
Team MCP permissions are enforced when a caller authenticates with a JWT whose groups claim names
the team: JWT auth assigns the team on MCP routes, and the MCP handler then serves the team's servers.
"""

import time
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
)
from litellm.proxy.auth import core
from litellm.proxy.auth import jwt as proxy_jwt
from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler

SIGNING_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _Jwks:
    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        key: Final = jwt.algorithms.RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(200, json={"keys": [{**key, "kid": "k1"}]}, request=httpx.Request("GET", url))


def _signed(groups: list[str]) -> str:
    return jwt.encode(
        {"sub": "user-123", "groups": groups, "scope": "", "exp": int(time.time()) + 600},
        SIGNING_KEY,
        algorithm="RS256",
        headers={"kid": "k1"},
    )


@pytest.fixture(autouse=True)
def groups_jwt_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://idp.example/mcp-enforcement/jwks")
    monkeypatch.setattr(proxy_server, "jwt_http_handler", _Jwks())
    monkeypatch.setattr(proxy_server, "general_settings", {"enable_jwt_auth": True})
    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(proxy_server, "litellm_jwtauth", LiteLLM_JWTAuth(team_ids_jwt_field="groups"))


def _serve_teams(monkeypatch: pytest.MonkeyPatch, *teams: LiteLLM_TeamTable) -> None:
    by_id: Final = {team.team_id: team for team in teams}

    async def get_team_object(team_id: str, **_: object) -> LiteLLM_TeamTable | None:
        return by_id.get(team_id)

    monkeypatch.setattr("litellm.proxy.auth.host.python.get_team_object", get_team_object)
    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_team_object", get_team_object)


def test_mcp_routes_are_in_the_default_team_allowed_routes():
    assert "mcp_routes" in LiteLLM_JWTAuth().team_allowed_routes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    ["/mcp/tools/list", "/v1/mcp/server/register", "/v1/mcp/server/health", "/v1/mcp/server/abc/approve"],
)
async def test_jwt_groups_team_is_assigned_on_mcp_routes_without_a_model(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    team: Final = LiteLLM_TeamTable(team_id="ABC", models=["gpt-4", "gpt-3.5-turbo"])
    _serve_teams(monkeypatch, team)

    result: Final = await proxy_jwt.authorize(core.proxy_settings(), _signed(["ABC"]), route, "POST", provisioning=True)

    assert result["team_id"] == "ABC"
    assert result["team_object"] == team


@pytest.mark.asyncio
async def test_jwt_team_mcp_servers_are_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    team_mcp_servers: Final = ["mcp-server-1", "mcp-server-2"]
    team_object_permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm-abc-123", mcp_servers=team_mcp_servers, mcp_access_groups=[], vector_stores=[]
    )
    _serve_teams(
        monkeypatch,
        LiteLLM_TeamTable(
            team_id="ABC",
            models=["gpt-4"],
            object_permission=team_object_permission,
            object_permission_id="perm-abc-123",
        ),
    )

    result: Final = await proxy_jwt.authorize(
        core.proxy_settings(), _signed(["ABC"]), "/mcp/tools/list", "POST", provisioning=True
    )
    auth: Final = proxy_jwt.auth_from_result(result)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())

    with (
        patch.object(MCPRequestHandler, "_get_allowed_mcp_servers_for_key", return_value=[]),
        patch.object(MCPRequestHandler, "get_mcp_servers_from_access_groups", return_value=[]),
    ):
        allowed_servers: Final = await MCPRequestHandler.get_allowed_mcp_servers(auth)

    assert set(allowed_servers) == set(team_mcp_servers)


@pytest.mark.asyncio
async def test_jwt_without_a_team_gets_no_team_mcp_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_teams(monkeypatch)

    result: Final = await proxy_jwt.authorize(
        core.proxy_settings(), _signed([]), "/mcp/tools/list", "POST", provisioning=True
    )

    assert result["team_id"] is None
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    assert await MCPRequestHandler._get_allowed_mcp_servers_for_team(proxy_jwt.auth_from_result(result)) == []


@pytest.mark.asyncio
async def test_jwt_team_mcp_servers_intersect_with_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_teams(
        monkeypatch,
        LiteLLM_TeamTable(
            team_id="TEAM-X",
            models=["gpt-4"],
            object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="team-perm", mcp_servers=["server-1", "server-2", "server-3"]
            ),
        ),
    )
    key_object_permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="key-perm", mcp_servers=["server-2", "server-4"]
    )

    result: Final = await proxy_jwt.authorize(
        core.proxy_settings(), _signed(["TEAM-X"]), "/mcp/tools/list", "POST", provisioning=True
    )
    auth: Final = proxy_jwt.auth_from_result(result).model_copy(update={"object_permission": key_object_permission})
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())

    with (
        patch.object(MCPRequestHandler, "_get_key_object_permission", return_value=key_object_permission),
        patch.object(MCPRequestHandler, "get_mcp_servers_from_access_groups", new_callable=AsyncMock, return_value=[]),
    ):
        allowed_servers: Final = await MCPRequestHandler.get_allowed_mcp_servers(auth)

    assert sorted(allowed_servers) == ["server-2"]
