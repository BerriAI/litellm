"""
Tests covering the MCP creation fixes from the fix_mcp_creation branch:

1. _prepare_mcp_server_data includes approval_status when set (was silently
   excluded, triggering a Prisma "Could not find field" error).
2. NewMCPServerRequest accepts oauth2_flow and stores the correct value.
3. admin direct-create endpoint enforces approval_status=active.
4. non-admin users can create session (ephemeral) MCP servers.
"""

import pytest
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    LitellmUserRoles,
    MCPApprovalStatus,
    MCPTransport,
    NewMCPServerRequest,
    UserAPIKeyAuth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_admin_auth(user_id: str = "admin-user") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id=user_id,
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )


def _make_internal_user_auth(
    user_id: str = "user-abc", team_id: Optional[str] = None
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id=user_id,
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id=team_id,
    )


def _make_db_record(
    server_id: str = "srv-1",
    alias: str = "Test Server",
    approval_status: Optional[str] = "active",
) -> LiteLLM_MCPServerTable:
    now = datetime.now()
    record = LiteLLM_MCPServerTable(
        server_id=server_id,
        alias=alias,
        url="https://example.com/mcp",
        transport=MCPTransport.http,
        created_at=now,
        updated_at=now,
        created_by="test",
        updated_by="test",
    )
    record.approval_status = approval_status
    return record


# ---------------------------------------------------------------------------
# _prepare_mcp_server_data: approval_status must be included in data dict
# ---------------------------------------------------------------------------


class TestPrepareDataIncludesApprovalStatus:
    """Ensure _prepare_mcp_server_data forwards approval_status to Prisma.

    Before the fix, model_dump(exclude_none=True) would drop approval_status
    when the endpoint forgot to set it, and even when it was set to an enum
    value the Prisma engine rejected it with "Could not find field" because the
    root schema.prisma was missing the field.  These tests guard both paths.
    """

    def test_active_status_included_in_dict(self):
        from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data

        req = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
        )
        req.approval_status = MCPApprovalStatus.active

        data = _prepare_mcp_server_data(req)

        assert "approval_status" in data
        assert data["approval_status"] == MCPApprovalStatus.active

    def test_pending_review_status_included_in_dict(self):
        from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data

        req = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
        )
        req.approval_status = MCPApprovalStatus.pending_review
        req.submitted_by = "user-abc"

        data = _prepare_mcp_server_data(req)

        assert data["approval_status"] == MCPApprovalStatus.pending_review
        assert data["submitted_by"] == "user-abc"

    def test_submitted_at_included_when_set(self):
        from datetime import timezone
        from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data

        req = NewMCPServerRequest(
            alias="My Server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
        )
        req.approval_status = MCPApprovalStatus.pending_review
        req.submitted_at = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

        data = _prepare_mcp_server_data(req)

        assert "submitted_at" in data
        assert data["submitted_at"] == req.submitted_at


# ---------------------------------------------------------------------------
# oauth2_flow: field accepted and stored correctly
# ---------------------------------------------------------------------------


class TestOAuth2FlowField:
    """NewMCPServerRequest must accept oauth2_flow values that the UI now maps to."""

    def test_client_credentials_accepted(self):
        req = NewMCPServerRequest(
            alias="M2M Server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            oauth2_flow="client_credentials",
        )
        assert req.oauth2_flow == "client_credentials"

    def test_authorization_code_accepted(self):
        req = NewMCPServerRequest(
            alias="Interactive Server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            oauth2_flow="authorization_code",
        )
        assert req.oauth2_flow == "authorization_code"

    def test_oauth2_flow_included_in_prepare_data(self):
        from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data

        req = NewMCPServerRequest(
            alias="M2M Server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
            oauth2_flow="client_credentials",
        )
        req.approval_status = MCPApprovalStatus.active

        data = _prepare_mcp_server_data(req)

        assert data.get("oauth2_flow") == "client_credentials"

    def test_none_oauth2_flow_not_forwarded(self):
        """When oauth2_flow is not set, it should be absent from the data dict
        (exclude_none=True keeps the dict clean)."""
        from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data

        req = NewMCPServerRequest(
            alias="API Key Server",
            url="https://example.com/mcp",
            transport=MCPTransport.http,
        )
        req.approval_status = MCPApprovalStatus.active

        data = _prepare_mcp_server_data(req)

        assert "oauth2_flow" not in data


# ---------------------------------------------------------------------------
# add_mcp_server: admin endpoint always overrides to active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_mcp_server_overrides_approval_status_to_active():
    """Even if the caller supplies approval_status='pending_review', the admin
    create endpoint must override it to 'active' and clear submission fields."""
    try:
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            add_mcp_server,
        )
    except ImportError:
        pytest.skip("MCP management endpoints not available")

    payload = NewMCPServerRequest(
        alias="My Server",
        url="https://example.com/mcp",
        transport=MCPTransport.http,
        # Caller attempts to sneak in a pending status — must be overridden.
        approval_status="pending_review",
        submitted_by="attacker",
    )
    admin = _make_admin_auth()
    created_record = _make_db_record(approval_status="active")

    mock_manager = MagicMock()
    mock_manager.add_server = AsyncMock()
    mock_manager.reload_servers_from_database = AsyncMock()

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=MagicMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
            MagicMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_mcp_server",
            AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.create_mcp_server",
            AsyncMock(return_value=created_record),
        ) as mock_create,
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ),
    ):
        await add_mcp_server(payload=payload, user_api_key_dict=admin)

    call_payload: NewMCPServerRequest = mock_create.call_args[0][1]
    assert call_payload.approval_status == MCPApprovalStatus.active
    assert call_payload.submitted_by is None
    assert call_payload.submitted_at is None


# ---------------------------------------------------------------------------
# add_session_mcp_server: non-admin users now allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_session_mcp_server_allowed_for_non_admin():
    """Any authenticated user (not just PROXY_ADMIN) may create an ephemeral
    session server.  The session endpoint writes nothing to the database so
    there is no meaningful security risk."""
    try:
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            TEMPORARY_MCP_SERVER_TTL_SECONDS,
            add_session_mcp_server,
        )
    except ImportError:
        pytest.skip("MCP management endpoints not available")

    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    payload = NewMCPServerRequest(
        alias="Temp Server",
        server_id="temp-1",
        url="https://temp.example.com/mcp",
        transport=MCPTransport.http,
    )
    non_admin = _make_internal_user_auth(user_id="submitter-user")

    built_server = MCPServer(
        server_id="temp-1",
        name="Temp Server",
        url="https://temp.example.com/mcp",
        transport=MCPTransport.http,
    )
    mock_manager = MagicMock()
    mock_manager.get_mcp_server_by_id.return_value = None
    mock_manager.build_mcp_server_from_table = AsyncMock(return_value=built_server)

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
            MagicMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._cache_temporary_mcp_server",
            MagicMock(),
        ) as cache_mock,
    ):
        response = await add_session_mcp_server(
            payload=payload,
            user_api_key_dict=non_admin,
        )

    # Should succeed and cache the server
    cache_mock.assert_called_once()
    assert response is not None


@pytest.mark.asyncio
async def test_add_session_mcp_server_created_by_reflects_non_admin_user_id():
    """created_by on the temp record uses the user's ID, not LITELLM_PROXY_ADMIN_NAME."""
    try:
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            add_session_mcp_server,
        )
    except ImportError:
        pytest.skip("MCP management endpoints not available")

    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    payload = NewMCPServerRequest(
        alias="Temp Server",
        server_id="temp-2",
        url="https://temp.example.com/mcp",
        transport=MCPTransport.http,
    )
    non_admin = _make_internal_user_auth(user_id="non-admin-123")

    built_server = MCPServer(
        server_id="temp-2",
        name="Temp Server",
        url="https://temp.example.com/mcp",
        transport=MCPTransport.http,
    )
    mock_manager = MagicMock()
    mock_manager.get_mcp_server_by_id.return_value = None
    mock_manager.build_mcp_server_from_table = AsyncMock(return_value=built_server)
    captured_temp_record = {}

    def capture_cache(server, ttl_seconds):
        pass

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.validate_and_normalize_mcp_server_payload",
            MagicMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.global_mcp_server_manager",
            mock_manager,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints._cache_temporary_mcp_server",
            MagicMock(side_effect=capture_cache),
        ),
    ):
        await add_session_mcp_server(
            payload=payload,
            user_api_key_dict=non_admin,
        )

    # build_mcp_server_from_table was called with the temp record
    args, _ = mock_manager.build_mcp_server_from_table.call_args
    temp_record = args[0]
    assert temp_record.created_by == "non-admin-123"
