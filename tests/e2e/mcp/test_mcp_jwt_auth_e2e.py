from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final, Literal

import pytest
from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import DD_SEARCH_FROM, unique_marker
from e2e_http import AuthHeaders, UnauthorizedError, unwrap
from idp import SHORT_LIVED_CLIENT_ID, Identity, Keycloak, token_claims
from lifecycle import ResourceManager
from management.management_client import build_client as build_management_client
from mcp_client import McpClient
from models import KeyGenerateBody, ObjectPermission, TeamUpdateBody, UserScimMetadata, UserUpdateBody

pytestmark = pytest.mark.e2e


@dataclass(frozen=True, slots=True)
class GrantedMcp:
    server_id: str
    other_server_id: str
    tool: str
    token: str = field(repr=False)


def _allowed(client: McpClient, access: GrantedMcp, headers: AuthHeaders) -> None:
    listing: Final = unwrap(client.list_tools(access.token, headers=headers))
    assert listing.tool_names_for_server(access.server_id) == frozenset((access.tool,))
    assert listing.tool_names_for_server(access.other_server_id) == frozenset()
    result: Final = client.await_call_tool(
        access.token,
        server_id=access.server_id,
        name=access.tool,
        arguments={"query": f"lit4506-{unique_marker()}", "from": DD_SEARCH_FROM, "to": "now", "max_tokens": 1000},
        headers=headers,
    )
    assert result.is_error is False, result
    assert result.all_text.strip(), "The successful control must return a real tool result"


def _denied(client: McpClient, access: GrantedMcp, headers: AuthHeaders, reason: str) -> None:
    listing: Final = client.list_tools(access.token, headers=headers)
    assert isinstance(listing, UnauthorizedError), listing
    assert reason in listing.body.lower(), listing.body
    execution: Final = client.call_tool(
        access.token,
        server_id=access.server_id,
        name=access.tool,
        arguments={"query": f"lit4506-{unique_marker()}", "from": DD_SEARCH_FROM, "to": "now", "max_tokens": 1000},
        headers=headers,
    )
    assert isinstance(execution, UnauthorizedError), execution
    assert reason in execution.body.lower(), execution.body


@pytest.fixture
def granted_mcp(client: McpClient, resources: ResourceManager, jwt_identity: Identity, idp: Keycloak) -> GrantedMcp:
    server_id: Final = register_datadog_mcp(client, resources)
    other_server_id: Final = register_datadog_mcp(client, resources)
    client.await_registered(server_id)
    client.await_registered(other_server_id)
    management: Final = build_management_client(client.proxy)
    management.update_team(
        TeamUpdateBody(
            team_id=jwt_identity.group,
            team_alias=jwt_identity.group,
            object_permission=ObjectPermission(mcp_servers=[server_id]),
        )
    )
    management.update_user(
        UserUpdateBody(
            user_id=jwt_identity.user_id,
            user_role="internal_user",
            object_permission=ObjectPermission(mcp_servers=[server_id]),
        )
    )
    stored: Final = management.user_info(jwt_identity.user_id)
    assert stored.user_info.user_id == jwt_identity.user_id
    assert stored.user_info.user_role == "internal_user"
    token: Final = idp.access_token(jwt_identity)
    assert token_claims(token).sub == jwt_identity.user_id
    tool: Final = client.await_tool(token, server_id, SEARCH_LOGS_TOOL)
    return GrantedMcp(server_id, other_server_id, tool, token)


class TestMcpGatewayJwt:
    @pytest.mark.covers("mcp.list_tools.bearer.scoped", "mcp.call_tool.bearer.scoped")
    @pytest.mark.parametrize("header", ("authorization", "x_litellm_api_key"))
    def test_valid_scoped_jwt_lists_and_calls(
        self, client: McpClient, granted_mcp: GrantedMcp, header: Literal["authorization", "x_litellm_api_key"]
    ) -> None:
        headers: Final = (
            AuthHeaders(authorization=f"Bearer {granted_mcp.token}")
            if header == "authorization"
            else AuthHeaders.model_validate({"x-litellm-api-key": granted_mcp.token})
        )
        _allowed(client, granted_mcp, headers)

    @pytest.mark.covers(
        "mcp.list_tools.bearer.denied_invalid_signature", "mcp.call_tool.bearer.denied_invalid_signature"
    )
    def test_tampered_signature_denies_both_operations(self, client: McpClient, granted_mcp: GrantedMcp) -> None:
        valid: Final = AuthHeaders(authorization=f"Bearer {granted_mcp.token}")
        _allowed(client, granted_mcp, valid)
        header, payload, signature = granted_mcp.token.split(".")
        flipped: Final = "A" if signature[10] != "A" else "B"
        tampered: Final = f"{header}.{payload}.{signature[:10]}{flipped}{signature[11:]}"
        _denied(client, granted_mcp, AuthHeaders(authorization=f"Bearer {tampered}"), "signature")
        _allowed(client, granted_mcp, valid)

    @pytest.mark.covers("mcp.list_tools.bearer.denied_expired", "mcp.call_tool.bearer.denied_expired")
    def test_expired_signed_jwt_denies_both_operations(
        self, client: McpClient, granted_mcp: GrantedMcp, jwt_identity: Identity, idp: Keycloak
    ) -> None:
        valid: Final = AuthHeaders(authorization=f"Bearer {granted_mcp.token}")
        _allowed(client, granted_mcp, valid)
        expiring: Final = idp.access_token(jwt_identity, client_id=SHORT_LIVED_CLIENT_ID)
        claims: Final = token_claims(expiring)
        assert claims.sub == jwt_identity.user_id
        delay: Final = claims.exp - time.time() + 1
        assert delay <= 5, "Short-lived token configuration or IdP clock drifted"
        time.sleep(max(0, delay))
        _denied(client, granted_mcp, AuthHeaders(authorization=f"Bearer {expiring}"), "expired")
        _allowed(client, granted_mcp, valid)

    @pytest.mark.covers("mcp.list_tools.bearer.denied_inactive_user", "mcp.call_tool.bearer.denied_inactive_user")
    def test_deactivated_user_cannot_reuse_warm_jwt(
        self, client: McpClient, granted_mcp: GrantedMcp, jwt_identity: Identity
    ) -> None:
        headers: Final = AuthHeaders(authorization=f"Bearer {granted_mcp.token}")
        _allowed(client, granted_mcp, headers)
        management: Final = build_management_client(client.proxy)
        management.update_user(
            UserUpdateBody(
                user_id=jwt_identity.user_id, user_role="internal_user", metadata=UserScimMetadata(scim_active=False)
            )
        )
        stored: Final = management.user_info(jwt_identity.user_id)
        assert stored.user_info.metadata is not None and stored.user_info.metadata.scim_active is False
        _denied(client, granted_mcp, headers, "deactivated")
        management.update_user(
            UserUpdateBody(
                user_id=jwt_identity.user_id, user_role="internal_user", metadata=UserScimMetadata(scim_active=True)
            )
        )
        _allowed(client, granted_mcp, headers)

    @pytest.mark.covers("mcp.list_tools.bearer.denied_inactive_user", "mcp.call_tool.bearer.denied_inactive_user")
    def test_deactivated_user_is_denied_on_a_cold_jwt(
        self, client: McpClient, granted_mcp: GrantedMcp, jwt_identity: Identity, idp: Keycloak
    ) -> None:
        management: Final = build_management_client(client.proxy)
        management.update_user(
            UserUpdateBody(
                user_id=jwt_identity.user_id, user_role="internal_user", metadata=UserScimMetadata(scim_active=False)
            )
        )
        cold: Final = idp.access_token(jwt_identity)
        assert cold != granted_mcp.token
        access: Final = GrantedMcp(granted_mcp.server_id, granted_mcp.other_server_id, granted_mcp.tool, cold)
        _denied(client, access, AuthHeaders(authorization=f"Bearer {cold}"), "deactivated")
        management.update_user(
            UserUpdateBody(
                user_id=jwt_identity.user_id, user_role="internal_user", metadata=UserScimMetadata(scim_active=True)
            )
        )
        _allowed(client, access, AuthHeaders(authorization=f"Bearer {cold}"))

    @pytest.mark.covers(
        "mcp.list_tools.bearer.explicit_header_precedence", "mcp.call_tool.bearer.explicit_header_precedence"
    )
    def test_explicit_gateway_header_wins_without_fallback(
        self, client: McpClient, granted_mcp: GrantedMcp, resources: ResourceManager
    ) -> None:
        _allowed(
            client,
            granted_mcp,
            AuthHeaders.model_validate(
                {"x-litellm-api-key": granted_mcp.token, "authorization": "Bearer invalid-secondary-token"}
            ),
        )
        _denied(
            client,
            granted_mcp,
            AuthHeaders.model_validate(
                {"x-litellm-api-key": "sk-invalid-primary-token", "authorization": f"Bearer {granted_mcp.token}"}
            ),
            "key",
        )
        sibling_key: Final = client.proxy.generate_key(
            KeyGenerateBody(object_permission=ObjectPermission(mcp_servers=[granted_mcp.other_server_id]))
        )
        resources.defer(lambda: client.proxy.delete_key(sibling_key))
        sibling_tool: Final = client.await_tool(sibling_key, granted_mcp.other_server_id, SEARCH_LOGS_TOOL)
        sibling: Final = GrantedMcp(granted_mcp.other_server_id, granted_mcp.server_id, sibling_tool, sibling_key)
        _allowed(
            client,
            sibling,
            AuthHeaders.model_validate(
                {"x-litellm-api-key": sibling_key, "authorization": f"Bearer {granted_mcp.token}"}
            ),
        )
