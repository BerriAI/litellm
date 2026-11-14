from datetime import datetime, timezone

from litellm.proxy._types import LiteLLM_MCPServerTable, LitellmUserRoles
from litellm.proxy.management_endpoints.mcp_management_endpoints import (
    _is_restricted_virtual_key_request,
    _sanitize_mcp_server_for_non_admin,
    _sanitize_mcp_server_for_virtual_key,
    _user_is_full_admin,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


def _mock_mcp_server(server_id: str = "srv-test-1", **kwargs) -> LiteLLM_MCPServerTable:
    defaults = {
        "alias": "Test MCP Server",
        "url": "https://upstream.internal/mcp/secret-token",
        "transport": "sse",
    }
    defaults.update(kwargs)
    return LiteLLM_MCPServerTable(
        server_id=server_id,
        **defaults,
    )


def test_is_restricted_virtual_key_request_admin_roles():
    """Route-restricted keys held by admin roles (PROXY_ADMIN, PROXY_ADMIN_VIEW_ONLY)
    with explicit MCP management routes must NOT be flagged as restricted virtual keys,
    so they receive non-admin governance metadata rather than the stripped virtual-key view.
    """
    admin_viewer = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        allowed_routes=["/v1/mcp/server"],
    )
    assert _is_restricted_virtual_key_request(admin_viewer) is False

    admin_viewer_group = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        allowed_routes=["mcp_management_routes"],
    )
    assert _is_restricted_virtual_key_request(admin_viewer_group) is False

    proxy_admin_mgmt = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        allowed_routes=["/v1/mcp/server"],
    )
    assert _is_restricted_virtual_key_request(proxy_admin_mgmt) is False


def test_is_restricted_virtual_key_request_inference_scoped_admin_keys():
    """An inference-scoped key associated with PROXY_ADMIN (e.g. allowed_routes=['llm_api_routes']
    or ['llm_api_routes', '/v1/mcp/tools']) reaches discovery endpoints via the route carve-out
    and MUST remain on the restricted virtual key path so credentials are fully sanitized.
    """
    inference_admin_key = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        allowed_routes=["llm_api_routes"],
    )
    assert _is_restricted_virtual_key_request(inference_admin_key) is True

    mcp_tools_admin_key = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        allowed_routes=["llm_api_routes", "/v1/mcp/tools"],
    )
    assert _is_restricted_virtual_key_request(mcp_tools_admin_key) is True

    mcp_tools_only_admin_key = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        allowed_routes=["/v1/mcp/tools"],
    )
    assert _is_restricted_virtual_key_request(mcp_tools_only_admin_key) is True

    chat_admin_key = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        allowed_routes=["/chat/completions"],
    )
    assert _is_restricted_virtual_key_request(chat_admin_key) is True


def test_user_is_full_admin_route_restricted_safety():
    """Only an unrestricted PROXY_ADMIN receives full admin access. Route-restricted keys
    (even if PROXY_ADMIN) must not be treated as full admin.
    """
    unrestricted_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        allowed_routes=None,
    )
    assert _user_is_full_admin(unrestricted_admin) is True

    empty_routes_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        allowed_routes=[],
    )
    assert _user_is_full_admin(empty_routes_admin) is True

    route_restricted_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        allowed_routes=["/v1/mcp/server"],
    )
    assert _user_is_full_admin(route_restricted_admin) is False

    admin_viewer = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        allowed_routes=None,
    )
    assert _user_is_full_admin(admin_viewer) is False


def test_is_restricted_virtual_key_request_non_admin_roles():
    """Non-admin keys with allowed_routes must be recognized as restricted virtual keys."""
    client_key = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        allowed_routes=["/v1/mcp/server"],
    )
    assert _is_restricted_virtual_key_request(client_key) is True

    unspecified_role = UserAPIKeyAuth(
        allowed_routes=["/v1/mcp/server"],
    )
    assert _is_restricted_virtual_key_request(unspecified_role) is True

    unrestricted_key = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        allowed_routes=[],
    )
    assert _is_restricted_virtual_key_request(unrestricted_key) is False


def test_admin_viewer_preserves_mcp_info_and_timestamps():
    """Admin viewer requests go through _sanitize_mcp_server_for_non_admin,
    which preserves governance metadata in mcp_info and creation/update timestamps
    while stripping credentials.
    """
    created_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    updated_time = datetime(2026, 11, 12, 15, 30, 0, tzinfo=timezone.utc)

    governance_mcp_info = {
        "server_name": "production-analytics-mcp",
        "description": "Internal data analytics connector",
        "owners": ["data-team@company.internal"],
        "data_classification": "restricted",
        "runtime_unit": "k8s-pod-eu-central",
        "documentation_url": "https://wiki.company.internal/mcp/analytics",
        "is_public": False,
        "is_public_explicit": False,
    }

    server = _mock_mcp_server(
        server_id="srv-analytics-1",
        alias="Analytics MCP Server",
        url="https://upstream.internal/mcp/secret-token",
        credentials={"auth_value": "secret-bearer-token"},
        static_headers={"Authorization": "Bearer secret-bearer-token"},
        mcp_info=governance_mcp_info,
        created_at=created_time,
        updated_at=updated_time,
        created_by="user-admin-123",
        updated_by="user-admin-123",
    )

    # Admin viewer route uses _sanitize_mcp_server_for_non_admin
    sanitized = _sanitize_mcp_server_for_non_admin(server)

    # Credentials must be stripped
    assert sanitized.credentials is None
    assert sanitized.static_headers is None
    assert sanitized.url is None

    # Governance mcp_info and timestamps must be PRESERVED
    assert sanitized.mcp_info == governance_mcp_info
    assert sanitized.created_at == created_time
    assert sanitized.updated_at == updated_time
    assert sanitized.created_by == "user-admin-123"
    assert sanitized.updated_by == "user-admin-123"


def test_restricted_virtual_key_strips_governance_mcp_info_and_timestamps():
    """Restricted virtual keys go through _sanitize_mcp_server_for_virtual_key,
    which restricts mcp_info to visibility flags only and blanks timestamps.
    """
    created_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    updated_time = datetime(2026, 11, 12, 15, 30, 0, tzinfo=timezone.utc)

    governance_mcp_info = {
        "server_name": "production-analytics-mcp",
        "owners": ["data-team@company.internal"],
        "is_public": False,
        "is_public_explicit": False,
    }

    server = _mock_mcp_server(
        server_id="srv-analytics-2",
        mcp_info=governance_mcp_info,
        created_at=created_time,
        updated_at=updated_time,
        created_by="user-admin-123",
    )

    sanitized = _sanitize_mcp_server_for_virtual_key(server)

    # Virtual key sanitizer strips created_at and keeps only visibility flags
    assert sanitized.created_at is None
    assert sanitized.created_by is None
    assert sanitized.mcp_info == {"is_public": False, "is_public_explicit": False}
