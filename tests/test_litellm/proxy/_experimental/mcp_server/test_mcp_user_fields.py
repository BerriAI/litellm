"""Unit tests for admin-declared MCP user-fields.

Covers the per-user fields feature end-to-end:
  - Pydantic shapes for MCPUserField on the create/update/read models
  - Encrypted JSON round-trip in LiteLLM_MCPUserCredentials.credential_b64
  - Storage helpers honour the type discriminator (don't collide with BYOK / OAuth2)
  - Pure helpers in user_fields.py compute the right missing-field list and
    inject the right headers / env vars
  - execute_mcp_tool raises the friendly 401 when required fields are absent,
    and proceeds when they're present
"""

import json
import os
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Tests run with a fixed salt so encrypt_value_helper has a stable key.
os.environ.setdefault("LITELLM_SALT_KEY", "test-salt-for-user-fields")

from litellm.proxy._experimental.mcp_server.db import (
    _decode_user_fields_payload,
)
from litellm.proxy._experimental.mcp_server.user_fields import (
    build_user_fields_missing_error,
    coerce_user_fields,
    compute_missing_user_fields,
    resolve_user_field_env,
    resolve_user_field_headers,
    server_has_user_fields,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPTransport,
    MCPUserField,
    NewMCPServerRequest,
    UpdateMCPServerRequest,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.types.mcp_server.mcp_server_manager import MCPServer


# ---------------------------------------------------------------------------
# Pydantic shapes
# ---------------------------------------------------------------------------


def test_new_mcp_server_request_accepts_user_fields():
    """user_fields should serialize as a list of MCPUserField models."""
    req = NewMCPServerRequest(
        server_name="gmail",
        url="https://gmail.example.com",
        transport=MCPTransport.http,
        user_fields=[
            MCPUserField(
                field_key="TOKEN",
                display_name="Gmail OAuth Token",
                header_name="Authorization",
                header_value_template="Bearer {value}",
                required=True,
            ),
        ],
    )
    assert len(req.user_fields) == 1
    field = req.user_fields[0]
    assert field.field_key == "TOKEN"
    assert field.required is True
    assert field.header_value_template == "Bearer {value}"


def test_update_mcp_server_request_user_fields_default_empty():
    req = UpdateMCPServerRequest(
        server_id="s1",
        server_name="gmail",
        url="https://gmail.example.com",
        transport=MCPTransport.http,
    )
    assert req.user_fields == []


def test_litellm_mcp_server_table_user_fields_default_empty():
    row = LiteLLM_MCPServerTable(
        server_id="s1",
        server_name="gmail",
        transport=MCPTransport.http,
    )
    assert row.user_fields == []
    # missing_user_field_keys is opt-in (set per-request by the list endpoint)
    assert row.missing_user_field_keys is None


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------


def _encrypt_user_fields_blob(values: Dict[str, str]) -> str:
    """Mirror what store_user_field_values writes into credential_b64."""
    payload = json.dumps({"type": "user_fields", "values": values})
    return encrypt_value_helper(payload)


def test_decode_user_fields_payload_round_trip():
    encoded = _encrypt_user_fields_blob({"BEARER": "tok", "WS": "ws1"})
    decoded = _decode_user_fields_payload(encoded)
    assert decoded == {"BEARER": "tok", "WS": "ws1"}


def test_decode_user_fields_payload_rejects_oauth2_blob():
    """OAuth2 rows live in the same column — must not be misread as user-fields."""
    oauth_blob = encrypt_value_helper(
        json.dumps({"type": "oauth2", "access_token": "abc"})
    )
    assert _decode_user_fields_payload(oauth_blob) is None


def test_decode_user_fields_payload_rejects_byok_string():
    """Plain BYOK strings must not parse as a user-fields payload."""
    byok_blob = encrypt_value_helper("plain-api-key")
    assert _decode_user_fields_payload(byok_blob) is None


def test_decode_user_fields_payload_handles_garbage():
    assert _decode_user_fields_payload("not-base64-at-all") is None


# ---------------------------------------------------------------------------
# user_fields helpers
# ---------------------------------------------------------------------------


def _gmail_server(user_fields: Optional[List[Dict[str, Any]]] = None) -> MCPServer:
    """Build a minimal MCPServer fixture for the helper tests."""
    return MCPServer(
        server_id="s1",
        name="Gmail",
        alias="gmail-prod",
        transport=MCPTransport.http,
        user_fields=(
            user_fields
            if user_fields is not None
            else [
                {
                    "field_key": "GMAIL_TOKEN",
                    "display_name": "Gmail Token",
                    "description": "Your Gmail OAuth bearer token",
                    "header_name": "Authorization",
                    "header_value_template": "Bearer {value}",
                    "required": True,
                },
                {
                    "field_key": "WORKSPACE",
                    "display_name": "Workspace ID",
                    "header_name": "X-Workspace",
                    "required": False,
                },
            ]
        ),
    )


def test_coerce_user_fields_accepts_list():
    srv = _gmail_server()
    assert len(coerce_user_fields(srv)) == 2


def test_coerce_user_fields_accepts_json_string():
    srv = _gmail_server(user_fields=None)
    # Prisma occasionally hands JSONB columns back as a string.
    srv.user_fields = json.dumps([{"field_key": "X", "required": True}])
    assert coerce_user_fields(srv) == [{"field_key": "X", "required": True}]


def test_coerce_user_fields_returns_empty_for_invalid_json_string():
    srv = _gmail_server(user_fields=None)
    srv.user_fields = "not-valid-json{"
    assert coerce_user_fields(srv) == []


def test_coerce_user_fields_returns_empty_for_non_list_json_string():
    srv = _gmail_server(user_fields=None)
    # Well-formed JSON but not a list — must not crash and must yield [].
    srv.user_fields = json.dumps({"field_key": "X"})
    assert coerce_user_fields(srv) == []


def test_coerce_user_fields_returns_empty_for_unsupported_type():
    """Anything that isn't a list or a string (e.g. a stray int from a bad
    migration) must drop to the safe empty fallback rather than raise."""
    srv = _gmail_server(user_fields=None)
    srv.user_fields = 42  # type: ignore[assignment]
    assert coerce_user_fields(srv) == []


def test_coerce_user_fields_drops_entries_without_model_dump():
    """Entries that are neither dicts nor model_dump-able are silently dropped."""
    srv = _gmail_server(user_fields=None)
    srv.user_fields = ["a-bare-string", 7]  # type: ignore[list-item]
    assert coerce_user_fields(srv) == []


def test_coerce_user_fields_drops_entries_when_model_dump_raises():
    """A model_dump that blows up should not bring down request dispatch."""

    class _Boom:
        def model_dump(self):
            raise RuntimeError("kaboom")

    srv = _gmail_server(user_fields=None)
    srv.user_fields = [_Boom()]  # type: ignore[list-item]
    assert coerce_user_fields(srv) == []


def test_coerce_user_fields_empty_when_missing():
    srv = MCPServer(server_id="s2", name="empty", transport=MCPTransport.http)
    assert coerce_user_fields(srv) == []
    assert server_has_user_fields(srv) is False


def test_coerce_user_fields_accepts_litellm_mcp_server_table():
    """LiteLLM_MCPServerTable.user_fields is List[MCPUserField] (Pydantic
    instances), not List[dict]. The helper must normalise both shapes so
    the management-layer annotation / enforcement paths don't silently
    return empty results.
    """
    table = LiteLLM_MCPServerTable(
        server_id="s3",
        transport=MCPTransport.http,
        user_fields=[
            {"field_key": "TOKEN", "header_name": "Authorization", "required": True},
            {"field_key": "WS", "header_name": "X-Workspace", "required": False},
        ],
    )
    assert isinstance(table.user_fields[0], MCPUserField)
    coerced = coerce_user_fields(table)
    assert [f["field_key"] for f in coerced] == ["TOKEN", "WS"]
    assert server_has_user_fields(table) is True
    missing = compute_missing_user_fields(table, None)
    assert [f["field_key"] for f in missing] == ["TOKEN"]


def test_compute_missing_required_only():
    srv = _gmail_server()
    missing = compute_missing_user_fields(srv, None)
    # Only the required field is reported as missing.
    assert [f["field_key"] for f in missing] == ["GMAIL_TOKEN"]


def test_compute_missing_empty_when_all_filled():
    srv = _gmail_server()
    missing = compute_missing_user_fields(
        srv, {"GMAIL_TOKEN": "tok", "WORKSPACE": "ws1"}
    )
    assert missing == []


def test_compute_missing_optional_field_unaffected():
    """Optional fields without a value should not appear in missing."""
    srv = _gmail_server()
    missing = compute_missing_user_fields(srv, {"GMAIL_TOKEN": "tok"})
    assert missing == []


def test_compute_missing_skips_entries_without_valid_field_key():
    """Malformed entries (no field_key, non-string field_key, empty string) are
    dropped instead of being surfaced as missing."""
    srv = _gmail_server(
        user_fields=[
            {"field_key": "", "required": True},
            {"required": True},
            {"field_key": 5, "required": True},
            {"field_key": "REAL", "required": True},
        ]
    )
    missing = compute_missing_user_fields(srv, None)
    assert [f["field_key"] for f in missing] == ["REAL"]


def test_resolve_user_field_headers_applies_template():
    srv = _gmail_server()
    headers = resolve_user_field_headers(
        srv, {"GMAIL_TOKEN": "tok", "WORKSPACE": "ws1"}
    )
    assert headers == {"Authorization": "Bearer tok", "X-Workspace": "ws1"}


def test_resolve_user_field_headers_skips_unset_values():
    srv = _gmail_server()
    headers = resolve_user_field_headers(srv, {"WORKSPACE": "ws1"})
    # GMAIL_TOKEN has no stored value → no Authorization header.
    assert headers == {"X-Workspace": "ws1"}


def test_resolve_user_field_headers_passes_unknown_placeholders_through():
    """Templates with non-``{value}`` placeholders are preserved literally
    instead of triggering format-string evaluation. ``str.replace`` only
    matches the literal ``{value}`` token, so unrelated braces flow through
    untouched and never crash the request path."""
    srv = _gmail_server(
        user_fields=[
            {
                "field_key": "TOKEN",
                "header_name": "Authorization",
                "header_value_template": "Bearer {unknown_placeholder}",
                "required": True,
            }
        ]
    )
    headers = resolve_user_field_headers(srv, {"TOKEN": "raw"})
    assert headers == {"Authorization": "Bearer {unknown_placeholder}"}


def test_resolve_user_field_headers_does_not_evaluate_attribute_access():
    """A template containing ``{value.__class__}`` must NOT evaluate the
    attribute traversal — that would leak Python internals into outbound
    HTTP headers. With ``str.replace`` it is preserved as a literal token."""
    srv = _gmail_server(
        user_fields=[
            {
                "field_key": "TOKEN",
                "header_name": "Authorization",
                "header_value_template": "Bearer {value.__class__}",
                "required": True,
            }
        ]
    )
    headers = resolve_user_field_headers(srv, {"TOKEN": "raw"})
    assert headers == {"Authorization": "Bearer {value.__class__}"}


def test_resolve_user_field_headers_falls_back_on_non_string_template():
    """A corrupt JSONB row may carry a non-string ``header_value_template``
    (e.g. an int from an external write that bypassed the API). The resolver
    must catch the resulting ``AttributeError`` from ``int.replace`` and fall
    back to the raw user value rather than crashing the request path."""
    srv = _gmail_server(
        user_fields=[
            {
                "field_key": "TOKEN",
                "header_name": "Authorization",
                "header_value_template": 42,
                "required": True,
            }
        ]
    )
    headers = resolve_user_field_headers(srv, {"TOKEN": "raw-tok"})
    assert headers == {"Authorization": "raw-tok"}


def test_resolve_user_field_headers_skips_entries_missing_header_name():
    """Stdio-only fields (no header_name) must not produce empty-name headers."""
    srv = MCPServer(
        server_id="s4",
        name="local",
        transport=MCPTransport.http,
        user_fields=[
            {"field_key": "ONLY_ENV", "env_var_name": "SECRET", "required": True},
            {"field_key": "WS", "header_name": "X-Workspace", "required": False},
        ],
    )
    headers = resolve_user_field_headers(srv, {"ONLY_ENV": "x", "WS": "ws1"})
    assert headers == {"X-Workspace": "ws1"}


def test_resolve_user_field_env_for_stdio():
    srv = MCPServer(
        server_id="s3",
        name="local",
        transport=MCPTransport.stdio,
        user_fields=[
            {"field_key": "GH_TOKEN", "env_var_name": "GITHUB_TOKEN", "required": True}
        ],
    )
    env = resolve_user_field_env(srv, {"GH_TOKEN": "ghs_xxx"})
    assert env == {"GITHUB_TOKEN": "ghs_xxx"}


def test_resolve_user_field_env_skips_entries_without_env_var_name():
    """HTTP-only fields (no env_var_name) and empty stored values are skipped."""
    srv = MCPServer(
        server_id="s5",
        name="local",
        transport=MCPTransport.stdio,
        user_fields=[
            {
                "field_key": "HTTP_ONLY",
                "header_name": "Authorization",
                "required": True,
            },
            {"field_key": "GH_TOKEN", "env_var_name": "GITHUB_TOKEN", "required": True},
            {"field_key": "EMPTY", "env_var_name": "EMPTY_VAR", "required": False},
        ],
    )
    env = resolve_user_field_env(
        srv, {"HTTP_ONLY": "x", "GH_TOKEN": "ghs", "EMPTY": ""}
    )
    assert env == {"GITHUB_TOKEN": "ghs"}


def test_build_user_fields_missing_error_uses_proxy_base_url():
    srv = _gmail_server()
    missing = compute_missing_user_fields(srv, None)
    err = build_user_fields_missing_error(srv, missing, "https://proxy.example.com/")
    assert err["error"] == "user_fields_missing"
    assert err["server_id"] == "s1"
    assert (
        err["config_url"]
        == "https://proxy.example.com/ui?page=mcp-servers&server_id=s1"
    )
    # Friendly message contains the URL and the field display name.
    assert "Gmail Token" in err["message"]
    assert err["config_url"] in err["message"]
    assert err["missing_fields"][0]["field_key"] == "GMAIL_TOKEN"


def test_build_user_fields_missing_error_falls_back_to_relative_path():
    srv = _gmail_server()
    err = build_user_fields_missing_error(
        srv, compute_missing_user_fields(srv, None), None
    )
    # When no base URL is known, the config_url is a relative path so it
    # still surfaces something useful in error logs.
    assert err["config_url"] == "/ui?page=mcp-servers&server_id=s1"


# ---------------------------------------------------------------------------
# execute_mcp_tool enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enforce_user_fields_raises_when_missing():
    """When a required field is missing, the helper raises HTTP 401 with the
    structured error payload Claude Code can render to the end-user."""
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.server import (
        _enforce_user_fields,
        _user_fields_cache,
    )
    from litellm.proxy._types import UserAPIKeyAuth

    _user_fields_cache.clear()
    srv = _gmail_server()
    user = UserAPIKeyAuth(api_key="hashed", user_id="user-1")

    # Simulate "no stored values" by patching the cache directly so we don't
    # need a live prisma_client.
    _user_fields_cache[("user-1", "s1")] = (None, 1e18)  # fresh entry, value=None

    with pytest.raises(HTTPException) as exc_info:
        await _enforce_user_fields(srv, user)
    assert exc_info.value.status_code == 401
    detail = exc_info.value.detail
    assert detail["error"] == "user_fields_missing"
    assert detail["server_id"] == "s1"
    assert "GMAIL_TOKEN" in [m["field_key"] for m in detail["missing_fields"]]
    assert "config_url" in detail


@pytest.mark.asyncio
async def test_enforce_user_fields_passes_when_all_required_present():
    from litellm.proxy._experimental.mcp_server.server import (
        _enforce_user_fields,
        _user_fields_cache,
    )
    from litellm.proxy._types import UserAPIKeyAuth

    _user_fields_cache.clear()
    srv = _gmail_server()
    user = UserAPIKeyAuth(api_key="hashed", user_id="user-2")
    _user_fields_cache[("user-2", "s1")] = ({"GMAIL_TOKEN": "tok"}, 1e18)

    # Should not raise.
    await _enforce_user_fields(srv, user)


@pytest.mark.asyncio
async def test_enforce_user_fields_no_user_id_raises():
    """A request without a user identity cannot be satisfied — surface the
    same friendly error so the client knows where to send the user."""
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.server import _enforce_user_fields
    from litellm.proxy._types import UserAPIKeyAuth

    srv = _gmail_server()
    user = UserAPIKeyAuth(api_key="hashed")  # no user_id

    with pytest.raises(HTTPException) as exc_info:
        await _enforce_user_fields(srv, user)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"] == "user_fields_missing"


# ---------------------------------------------------------------------------
# Manager wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manager_call_tool_enforces_required_user_fields():
    """``MCPServerManager.call_tool`` is invoked directly by the Responses
    API path, so it must raise the same friendly 401 as ``execute_mcp_tool``
    when a required user-field has no stored value — otherwise a caller
    can bypass enforcement and dispatch with the value silently dropped.
    """
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.proxy._experimental.mcp_server.server import _user_fields_cache
    from litellm.proxy._types import UserAPIKeyAuth

    _user_fields_cache.clear()
    srv = _gmail_server()
    mgr = MCPServerManager()
    mgr.registry["s1"] = srv
    mgr.tool_name_to_mcp_server_name_mapping["gmail-prod-send"] = srv.name
    mgr.tool_name_to_mcp_server_name_mapping["send"] = srv.name
    user = UserAPIKeyAuth(api_key="hashed", user_id="responses-user")
    _user_fields_cache[("responses-user", "s1")] = (None, 1e18)

    with pytest.raises(HTTPException) as exc_info:
        await mgr.call_tool(
            server_name="gmail-prod",
            name="send",
            arguments={},
            user_api_key_auth=user,
        )
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"] == "user_fields_missing"
    assert "GMAIL_TOKEN" in [
        m["field_key"] for m in exc_info.value.detail["missing_fields"]
    ]


@pytest.mark.asyncio
async def test_manager_call_tool_enforces_when_no_user_id():
    """Anonymous callers cannot satisfy required user-fields, so the manager
    must still surface ``user_fields_missing`` rather than silently dispatch
    without the configured headers."""
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.proxy._types import UserAPIKeyAuth

    srv = _gmail_server()
    mgr = MCPServerManager()
    mgr.registry["s1"] = srv
    mgr.tool_name_to_mcp_server_name_mapping["gmail-prod-send"] = srv.name
    mgr.tool_name_to_mcp_server_name_mapping["send"] = srv.name
    user = UserAPIKeyAuth(api_key="hashed")  # no user_id

    with pytest.raises(HTTPException) as exc_info:
        await mgr.call_tool(
            server_name="gmail-prod",
            name="send",
            arguments={},
            user_api_key_auth=user,
        )
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["error"] == "user_fields_missing"


@pytest.mark.asyncio
async def test_build_stdio_env_merges_user_field_env_over_static():
    """Stored user_field env vars must take precedence over static server.env."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    srv = MCPServer(
        server_id="s4",
        name="local",
        transport=MCPTransport.stdio,
        env={"GITHUB_TOKEN": "static-default", "OTHER": "keep-me"},
    )
    mgr = MCPServerManager()
    merged = mgr._build_stdio_env(
        srv, raw_headers=None, user_field_env={"GITHUB_TOKEN": "user-value"}
    )
    assert merged == {"GITHUB_TOKEN": "user-value", "OTHER": "keep-me"}


@pytest.mark.asyncio
async def test_build_stdio_env_returns_user_env_when_no_static_env():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    srv = MCPServer(server_id="s5", name="local", transport=MCPTransport.stdio)
    mgr = MCPServerManager()
    merged = mgr._build_stdio_env(srv, raw_headers=None, user_field_env={"X": "1"})
    assert merged == {"X": "1"}


def test_openapi_tool_merge_injects_user_field_headers_over_static():
    """User-field headers override static_headers in the OpenAPI/local-tool
    dispatch path, matching managed-MCP precedence.

    Regression: stored user-field values were validated at enforcement time
    but silently dropped for OpenAPI tools, so upstream calls failed even
    after users provided required values.
    """
    from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
        _merge_openapi_tool_request_headers,
        _request_auth_header,
        _request_extra_headers,
        _request_user_field_headers,
    )

    extra_token = _request_extra_headers.set({"X-Trace": "abc"})
    user_token = _request_user_field_headers.set(
        {"Authorization": "Bearer user-tok", "X-Workspace": "ws1"}
    )
    auth_token = _request_auth_header.set(None)
    try:
        merged = _merge_openapi_tool_request_headers(
            {"Authorization": "Bearer static-default", "X-Static": "keep"}
        )
    finally:
        _request_auth_header.reset(auth_token)
        _request_user_field_headers.reset(user_token)
        _request_extra_headers.reset(extra_token)

    # User-field Authorization wins over static.
    assert merged["Authorization"] == "Bearer user-tok"
    # Other user-field headers are added.
    assert merged["X-Workspace"] == "ws1"
    # Static headers without a user-field override survive.
    assert merged["X-Static"] == "keep"
    # Forwarded extras still come through when not colliding with static.
    assert merged["X-Trace"] == "abc"


def test_openapi_tool_merge_byok_auth_overrides_user_field_authorization():
    """BYOK `_request_auth_header` is the final word on Authorization,
    even when a user_field also targets Authorization. Matches the managed
    path's `mcp_auth_header` parameter precedence."""
    from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
        _merge_openapi_tool_request_headers,
        _request_auth_header,
        _request_extra_headers,
        _request_user_field_headers,
    )

    extra_token = _request_extra_headers.set(None)
    user_token = _request_user_field_headers.set({"Authorization": "Bearer user-tok"})
    auth_token = _request_auth_header.set("Bearer byok-tok")
    try:
        merged = _merge_openapi_tool_request_headers({})
    finally:
        _request_auth_header.reset(auth_token)
        _request_user_field_headers.reset(user_token)
        _request_extra_headers.reset(extra_token)

    assert merged["Authorization"] == "Bearer byok-tok"


# ---------------------------------------------------------------------------
# HTTP endpoints — POST / GET / DELETE /v1/mcp/server/{id}/user-field-values
# ---------------------------------------------------------------------------


def _server_row_with_user_fields() -> Any:
    """Mock Prisma row matching what get_mcp_server returns.

    The endpoint sees `user_fields` as a list, mirroring how Prisma decodes
    JSONB in the happy path.
    """
    now = "2026-05-19T00:00:00"
    return MagicMock(
        server_id="srv-1",
        server_name="Gmail",
        alias="gmail",
        transport=MCPTransport.http,
        url="https://gmail.example.com",
        created_at=now,
        updated_at=now,
        is_byok=False,
        byok_description=[],
        byok_api_key_help_url=None,
        user_fields=[
            {
                "field_key": "GMAIL_TOKEN",
                "display_name": "Gmail Token",
                "header_name": "Authorization",
                "header_value_template": "Bearer {value}",
                "required": True,
            },
            {
                "field_key": "WORKSPACE",
                "display_name": "Workspace",
                "header_name": "X-Workspace",
                "required": False,
            },
        ],
        credentials=None,
        mcp_info={},
        mcp_access_groups=[],
        allowed_tools=[],
        tool_name_to_display_name={},
        tool_name_to_description={},
        extra_headers=[],
        static_headers={},
        status="unknown",
        last_health_check=None,
        health_check_error=None,
        command=None,
        args=[],
        env={},
        authorization_url=None,
        token_url=None,
        registration_url=None,
        allow_all_keys=False,
        available_on_public_internet=True,
        delegate_auth_to_upstream=False,
        source_url=None,
        approval_status="active",
        submitted_by=None,
        submitted_at=None,
        reviewed_at=None,
        review_notes=None,
        spec_path=None,
        instructions=None,
        created_by="admin",
        updated_by="admin",
        auth_type=None,
    )


@pytest.mark.asyncio
async def test_get_user_field_values_endpoint_reports_missing():
    """Initial GET (no stored row) should show all required fields as missing."""
    from litellm.proxy._types import UserAPIKeyAuth

    server_row = _server_row_with_user_fields()
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(
        return_value=server_row
    )
    prisma_client.db.litellm_mcpusercredentials.find_unique = AsyncMock(
        return_value=None
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=prisma_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.db.get_user_field_values",
            AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            AsyncMock(return_value=[server_row]),
        ),
    ):
        # Call the endpoint function directly to skip FastAPI auth wiring.
        from litellm.proxy.management_endpoints import mcp_management_endpoints as mod

        # The endpoint is closed over `router` inside `setup_mcp_management_routes`;
        # exercise it via a TestClient.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = FastAPI()
        app.include_router(mod.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="hashed", user_id="user-1"
        )
        client = TestClient(app)
        res = client.get("/v1/mcp/server/srv-1/user-field-values")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["server_id"] == "srv-1"
        assert "GMAIL_TOKEN" in body["missing_field_keys"]
        # Optional field is never reported as missing.
        assert "WORKSPACE" not in body["missing_field_keys"]
        assert body["stored_field_keys"] == []
        # Declared field descriptors are echoed back so the UI can render
        # input fields without a second round-trip.
        keys = {f["field_key"] for f in body["user_fields"]}
        assert keys == {"GMAIL_TOKEN", "WORKSPACE"}


@pytest.mark.asyncio
async def test_post_user_field_values_rejects_undeclared_keys():
    """Attempting to save a key the server didn't declare must be dropped silently."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints import mcp_management_endpoints as mod

    server_row = _server_row_with_user_fields()
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(
        return_value=server_row
    )

    captured = {}

    async def fake_store(
        prisma, user_id, server_id, values=None, *, merge_fn=None, **kwargs
    ):
        captured["user_id"] = user_id
        captured["server_id"] = server_id
        # The endpoint passes a ``merge_fn`` that closes over the request
        # payload + declared_keys; resolve it against an empty existing dict
        # (mirroring "no row yet") so the test can assert on the final values.
        if merge_fn is not None:
            result = merge_fn({})
        else:
            result = values or {}
        captured["values"] = result
        return result

    async def fake_get(prisma, user_id, server_id):
        return None  # no existing values

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=prisma_client,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.store_user_field_values",
            new=fake_store,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_user_field_values",
            new=fake_get,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._invalidate_byok_cred_cache"
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._invalidate_user_fields_cache"
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            AsyncMock(return_value=[server_row]),
        ),
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = FastAPI()
        app.include_router(mod.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="hashed", user_id="user-2"
        )
        client = TestClient(app)
        res = client.post(
            "/v1/mcp/server/srv-1/user-field-values",
            json={
                "values": {
                    "GMAIL_TOKEN": "tok",
                    "WORKSPACE": "ws1",
                    "EVIL_KEY": "should-be-dropped",
                }
            },
        )
        assert res.status_code == 200, res.text
        assert captured["values"] == {"GMAIL_TOKEN": "tok", "WORKSPACE": "ws1"}
        body = res.json()
        assert body["missing_field_keys"] == []


@pytest.mark.asyncio
async def test_post_user_field_values_rejects_server_with_no_declared_fields():
    """Storing values on a server that doesn't use user_fields should 400."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints import mcp_management_endpoints as mod

    server_row = _server_row_with_user_fields()
    server_row.user_fields = []  # no declared fields
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(
        return_value=server_row
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=prisma_client,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            AsyncMock(return_value=[server_row]),
        ),
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = FastAPI()
        app.include_router(mod.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="hashed", user_id="user-3"
        )
        client = TestClient(app)
        res = client.post(
            "/v1/mcp/server/srv-1/user-field-values", json={"values": {"X": "y"}}
        )
        assert res.status_code == 400


@pytest.mark.asyncio
async def test_user_field_value_endpoints_404_when_server_not_in_allowed_set():
    """Non-admin callers must not be able to probe servers outside their allowed set.

    Without this gate, an authenticated user who knows another team's
    ``server_id`` could call GET/POST/DELETE on the user-field-values
    endpoints and leak the admin-declared field descriptors (header names,
    env var names) for that server.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints import mcp_management_endpoints as mod

    server_row = _server_row_with_user_fields()
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(
        return_value=server_row
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=prisma_client,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            AsyncMock(return_value=[]),
        ),
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = FastAPI()
        app.include_router(mod.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="hashed", user_id="outsider"
        )
        client = TestClient(app)

        get_res = client.get("/v1/mcp/server/srv-1/user-field-values")
        assert get_res.status_code == 404
        post_res = client.post(
            "/v1/mcp/server/srv-1/user-field-values",
            json={"values": {"GMAIL_TOKEN": "tok"}},
        )
        assert post_res.status_code == 404
        del_res = client.delete("/v1/mcp/server/srv-1/user-field-values")
        assert del_res.status_code == 404

        # find_unique must never be reached — the access gate fails first,
        # before any server metadata is read or returned.
        prisma_client.db.litellm_mcpservertable.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_delete_user_field_values_clears_stored_and_returns_status():
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints import mcp_management_endpoints as mod

    server_row = _server_row_with_user_fields()
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(
        return_value=server_row
    )

    delete_calls: List[tuple] = []

    async def fake_delete(prisma, user_id, server_id):
        delete_calls.append((user_id, server_id))

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=prisma_client,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.delete_user_field_values",
            new=fake_delete,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            AsyncMock(return_value=[server_row]),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._invalidate_byok_cred_cache"
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._invalidate_user_fields_cache"
        ),
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = FastAPI()
        app.include_router(mod.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="hashed", user_id="user-del"
        )
        client = TestClient(app)
        res = client.delete("/v1/mcp/server/srv-1/user-field-values")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["server_id"] == "srv-1"
        assert body["stored_field_keys"] == []
        assert "GMAIL_TOKEN" in body["missing_field_keys"]
        assert delete_calls == [("user-del", "srv-1")]


@pytest.mark.asyncio
async def test_delete_user_field_values_swallows_record_not_found():
    """DELETE on a row that was already removed should still succeed."""
    from prisma.errors import RecordNotFoundError

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints import mcp_management_endpoints as mod

    server_row = _server_row_with_user_fields()
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(
        return_value=server_row
    )

    async def fake_delete(prisma, user_id, server_id):
        raise RecordNotFoundError(data={"error": {"message": "no rows", "meta": {}}})

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=prisma_client,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.delete_user_field_values",
            new=fake_delete,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            AsyncMock(return_value=[server_row]),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._invalidate_byok_cred_cache"
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._invalidate_user_fields_cache"
        ),
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = FastAPI()
        app.include_router(mod.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="hashed", user_id="user-del2"
        )
        client = TestClient(app)
        res = client.delete("/v1/mcp/server/srv-1/user-field-values")
        assert res.status_code == 200, res.text


@pytest.mark.asyncio
async def test_list_user_field_values_aggregates_per_server_status():
    """GET /v1/mcp/user-field-values returns one entry per server with user_fields."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
    from litellm.proxy.management_endpoints import mcp_management_endpoints as mod

    gmail_row = _server_row_with_user_fields()
    no_fields_row = _server_row_with_user_fields()
    no_fields_row.server_id = "srv-2"
    no_fields_row.user_fields = []

    other_row = _server_row_with_user_fields()
    other_row.server_id = "srv-3"
    other_row.user_fields = [
        {
            "field_key": "JIRA_TOKEN",
            "header_name": "Authorization",
            "header_value_template": "Bearer {value}",
            "required": True,
        }
    ]

    encoded = encrypt_value_helper(
        json.dumps({"type": "user_fields", "values": {"JIRA_TOKEN": "jt"}})
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpusercredentials.find_many = AsyncMock(
        return_value=[MagicMock(server_id="srv-3", credential_b64=encoded)]
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=prisma_client,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            AsyncMock(return_value=[gmail_row, no_fields_row, other_row]),
        ),
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = FastAPI()
        app.include_router(mod.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="hashed", user_id="user-list"
        )
        client = TestClient(app)
        res = client.get("/v1/mcp/user-field-values")
        assert res.status_code == 200, res.text
        body = res.json()
        # Only servers with declared user_fields are returned.
        ids = {entry["server_id"] for entry in body}
        assert ids == {"srv-1", "srv-3"}

        by_id = {entry["server_id"]: entry for entry in body}
        # srv-1 has no stored values → required field missing.
        assert "GMAIL_TOKEN" in by_id["srv-1"]["missing_field_keys"]
        # srv-3 has the stored value → nothing missing.
        assert by_id["srv-3"]["missing_field_keys"] == []
        assert by_id["srv-3"]["stored_field_keys"] == ["JIRA_TOKEN"]


@pytest.mark.asyncio
async def test_list_user_field_values_returns_empty_when_no_servers_declare_fields():
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.management_endpoints import mcp_management_endpoints as mod

    plain_row = _server_row_with_user_fields()
    plain_row.user_fields = []

    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpusercredentials.find_many = AsyncMock(return_value=[])

    with (
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=prisma_client,
        ),
        patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_all_mcp_servers_for_user",
            AsyncMock(return_value=[plain_row]),
        ),
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app = FastAPI()
        app.include_router(mod.router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            api_key="hashed", user_id="user-empty"
        )
        client = TestClient(app)
        res = client.get("/v1/mcp/user-field-values")
        assert res.status_code == 200, res.text
        assert res.json() == []
        # Should short-circuit before hitting the credentials table.
        prisma_client.db.litellm_mcpusercredentials.find_many.assert_not_called()
