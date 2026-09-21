"""Real OAuth consent, immediate MCP operations and cold-restart persistence.

Aggregate SSO uses the SDK's normal authentication. The per-server variant is
explicitly a configured two-header client, not an Authorization-only OAuth host.
The observed variants forward to the same real Linear upstream and compare its
bearer at the forwarding boundary; direct variants retain unmodified discovery.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import pytest
from e2e_config import LINEAR_MCP_URL, LINEAR_READONLY_TOOL, LINEAR_STORAGE_STATE, unique_marker
from e2e_http import AuthHeaders, NoBody, get_external, unwrap
from idp import Identity, Keycloak
from lifecycle import ResourceManager
from models import (
    McpOauthCredentials,
    McpServerCreateBody,
    ObjectPermission,
    TeamMemberAddBody,
    TeamMemberEntry,
    TeamUpdateBody,
)
from oauth_chat_client import ChatMcpClient, InMemoryTokenStorage, OauthToolRun, build_chat_client
from oauth_gateway import OAuthGateway, OAuthObservation, owned_gateway, stored_oauth
from provider_edge import LiveEdge, start_provider_edge
from proxy_client import ProxyClient
from pydantic import BaseModel, ValidationError

pytestmark = [pytest.mark.e2e, pytest.mark.mcp_oauth_live, pytest.mark.provider_live]


class OAuthMetadata(BaseModel):
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str


class LinearTeam(BaseModel):
    id: str
    name: str


class LinearTeams(BaseModel):
    teams: tuple[LinearTeam, ...]


def assert_tool_result(run: OauthToolRun, tool: str) -> None:
    assert tool in run.tools
    assert run.is_error is False
    try:
        result: Final = LinearTeams.model_validate_json(run.text)
    except ValidationError:
        raise AssertionError("list_teams did not return the expected teams payload") from None
    assert result.teams, "the test workspace must contain at least one team"
    assert all(team.id and team.name for team in result.teams), "team results must contain identifiers and names"


@pytest.fixture(scope="module")
def oauth_gateway(idp: Keycloak, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OAuthGateway]:
    assert LINEAR_STORAGE_STATE and Path(LINEAR_STORAGE_STATE).is_file(), (
        "E2E_LINEAR_STORAGE_STATE must name a captured Linear login; see mcp/linear_session_capture.py"
    )
    assert os.environ.get("E2E_FIXTURE_MODE", "live") == "live", "OAuth acceptance cannot use replay"
    with ExitStack() as cleanup:
        yield owned_gateway(idp, tmp_path_factory.mktemp("mcp-oauth"), cleanup)


@pytest.fixture(scope="module")
def proxy(oauth_gateway: OAuthGateway) -> ProxyClient:
    return oauth_gateway.proxy


@pytest.fixture(scope="module")
def client(proxy: ProxyClient) -> ChatMcpClient:
    return build_chat_client(proxy)


class TestMcpOauthHappyPath:
    @pytest.mark.covers("mcp.list_tools.oauth.succeeds")
    @pytest.mark.covers("mcp.call_tool.oauth.succeeds")
    @pytest.mark.covers("mcp.call_tool.oauth.persists_across_processes")
    @pytest.mark.parametrize("route", ("aggregate_sso", "explicit_header_jwt"))
    @pytest.mark.parametrize("observed", (False, True), ids=("direct", "observed"))
    def test_consent_list_call_and_cold_restart(
        self,
        client: ChatMcpClient,
        resources: ResourceManager,
        jwt_identity: Identity,
        idp: Keycloak,
        oauth_gateway: OAuthGateway,
        route: Literal["aggregate_sso", "explicit_header_jwt"],
        observed: bool,
    ) -> None:
        alias: Final = f"e2elinear{unique_marker()}"
        tool: Final = f"{alias}-{LINEAR_READONLY_TOOL}"
        token: Final = idp.access_token(jwt_identity)
        observation: Final = OAuthObservation(gateway_token=token)
        edge: Final = (
            start_provider_edge(
                LiveEdge(observe_request=observation.observe),
                mounts=MappingProxyType(
                    {"linear": "https://mcp.linear.app", ".well-known": "https://mcp.linear.app/.well-known"}
                ),
            )
            if observed
            else None
        )
        if edge is not None:
            resources.defer(edge.shutdown)
        metadata: Final = (
            unwrap(
                get_external(
                    "https://mcp.linear.app/.well-known/oauth-authorization-server",
                    response_type=OAuthMetadata,
                )
            )
            if observed
            else None
        )
        created: Final = client.create_server(
            McpServerCreateBody(
                alias=alias,
                server_name=alias,
                url=f"{edge.edge.api_base('linear')}/mcp" if edge is not None else LINEAR_MCP_URL,
                transport="http",
                allow_all_keys=False,
                auth_type="oauth2",
                oauth2_flow="authorization_code",
                per_server_oauth_discovery=route == "explicit_header_jwt",
                authorization_url=metadata.authorization_endpoint if metadata else None,
                token_url=metadata.token_endpoint if metadata else None,
                registration_url=metadata.registration_endpoint if metadata else None,
                credentials=McpOauthCredentials(upstream_resource=LINEAR_MCP_URL) if observed else None,
            )
        )
        resources.defer(lambda: client.delete_server(created.server_id))
        assert client.server_user_credentials(created.server_id) == (), (
            "scenario must start without upstream credentials"
        )
        unwrap(
            client.proxy.transport.post(
                "/team/member_add",
                headers=client.proxy.transport.master,
                json=TeamMemberAddBody(
                    team_id=jwt_identity.group, member=TeamMemberEntry(user_id=jwt_identity.user_id, role="user")
                ),
                response_type=NoBody,
            )
        )
        client.proxy.update_team(
            TeamUpdateBody(
                team_id=jwt_identity.group,
                object_permission=ObjectPermission(mcp_servers=[created.server_id]),
            )
        )
        headers: Final = {"x-litellm-api-key": f"Bearer {token}"} if route == "explicit_header_jwt" else {}
        resources.defer(
            lambda: client.revoke_user_token(
                created.server_id,
                AuthHeaders(authorization=f"Bearer {idp.access_token(jwt_identity)}"),
            )
        )
        identity: Final = jwt_identity if route == "aggregate_sso" else None
        first: Final = client.list_and_call(
            alias,
            headers,
            InMemoryTokenStorage(),
            LINEAR_STORAGE_STATE,
            tool,
            {},
            base_url=oauth_gateway.base_url,
            identity=identity,
        )
        assert_tool_result(first, tool)
        credentials: Final = client.server_user_credentials(created.server_id)
        assert len(credentials) == 1
        assert credentials[0].user_id == jwt_identity.user_id
        assert credentials[0].credential_type == "oauth2"
        first_stored_oauth: Final = stored_oauth(jwt_identity.user_id, created.server_id)
        if observed:
            observation.assert_forwarded(first_stored_oauth)
        oauth_gateway.restart()
        fresh_token: Final = idp.access_token(jwt_identity)
        observation.gateway_token = fresh_token
        second: Final = client.list_and_call(
            alias,
            {"Authorization": f"Bearer {fresh_token}"} if identity is None else {},
            InMemoryTokenStorage(),
            LINEAR_STORAGE_STATE if identity is not None else None,
            tool,
            {},
            base_url=oauth_gateway.base_url,
            identity=identity,
            allow_upstream_consent=False,
        )
        assert_tool_result(second, tool)
        second_stored_oauth: Final = stored_oauth(jwt_identity.user_id, created.server_id)
        if observed:
            observation.assert_forwarded(second_stored_oauth)
