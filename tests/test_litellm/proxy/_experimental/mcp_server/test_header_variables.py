"""Tests for MCP header-variable interpolation and per-user value storage."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._experimental.mcp_server.header_variables import (
    HEADER_VARIABLE_CREDENTIAL_TYPE,
    MissingPerUserHeaderVariablesError,
    _decode_header_variables_payload,
    _normalize_header_variables,
    build_dashboard_url,
    find_missing_per_user_variables,
    get_global_variable_values,
    get_per_user_variable_names,
    get_user_header_variables,
    interpolate_static_headers,
    interpolate_string,
    resolve_static_headers_for_user,
    store_user_header_variables,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.types.mcp_server.mcp_server_manager import MCPServer


SALT_KEY = "test-salt-key-for-header-variable-tests-12345"


@pytest.fixture(autouse=True)
def _set_salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", SALT_KEY)


def _make_server(
    *,
    server_id: str = "srv-1",
    static_headers=None,
    header_variables=None,
) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=server_id,
        transport="http",
        url="https://upstream.example/mcp",
        static_headers=static_headers,
        header_variables=header_variables,
    )


def test_normalize_drops_invalid_entries():
    raw = [
        {"name": "A", "value": "x", "scope": "global"},
        {"name": "B", "scope": "per_user"},
        {"name": "C", "scope": "bogus"},  # invalid scope → defaults to global
        {"value": "no-name"},  # missing name → dropped
        "not-a-dict",  # dropped
    ]
    out = _normalize_header_variables(raw)
    assert out == [
        {"name": "A", "value": "x", "scope": "global"},
        {"name": "B", "value": None, "scope": "per_user"},
        {"name": "C", "value": None, "scope": "global"},
    ]


def test_normalize_accepts_json_string():
    raw = json.dumps([{"name": "X", "scope": "global", "value": "v"}])
    assert _normalize_header_variables(raw) == [
        {"name": "X", "value": "v", "scope": "global"}
    ]


def test_get_per_user_and_global_helpers():
    server = _make_server(
        header_variables=[
            {"name": "G1", "value": "g1-val", "scope": "global"},
            {"name": "U1", "scope": "per_user"},
            {"name": "U2", "scope": "per_user"},
        ]
    )
    assert get_per_user_variable_names(server) == ["U1", "U2"]
    assert get_global_variable_values(server) == {"G1": "g1-val"}


def test_interpolate_string_resolves_and_records_missing():
    out, missing = interpolate_string("${A}://${B}@host", {"A": "https", "B": "me"})
    assert out == "https://me@host"
    assert missing == []

    out, missing = interpolate_string("${X}/${A}", {"A": "ok"})
    assert out == "${X}/ok"
    assert missing == ["X"]


def test_interpolate_static_headers_deduplicates_missing():
    result, missing = interpolate_static_headers(
        {"Authorization": "Bearer ${TOK}", "X-User": "${USER}"},
        {"TOK": "abc"},
    )
    assert result == {"Authorization": "Bearer abc", "X-User": "${USER}"}
    assert missing == ["USER"]


def test_find_missing_returns_only_referenced_variables():
    server = _make_server(
        static_headers={"X": "${U1}"},
        header_variables=[
            {"name": "U1", "scope": "per_user"},
            {"name": "U2", "scope": "per_user"},  # never referenced
        ],
    )
    # U1 is referenced but the user has not provided it
    assert find_missing_per_user_variables(server, {}) == ["U1"]
    # U2 is declared but not referenced — should NOT be flagged
    assert find_missing_per_user_variables(server, {"U1": "alice"}) == []


def test_find_missing_blanks_count_as_missing():
    server = _make_server(
        static_headers={"X": "${U1}"},
        header_variables=[{"name": "U1", "scope": "per_user"}],
    )
    assert find_missing_per_user_variables(server, {"U1": ""}) == ["U1"]


def test_build_dashboard_url_relative_and_absolute():
    assert (
        build_dashboard_url("srv-1")
        == "/ui/?page=mcp-servers&fill_variables_for=srv-1"
    )
    assert (
        build_dashboard_url("srv-1", "https://proxy.example/")
        == "https://proxy.example/ui/?page=mcp-servers&fill_variables_for=srv-1"
    )


# ── Per-user value persistence ────────────────────────────────────────────────


def _row_with_header_variables(values: dict) -> MagicMock:
    payload = json.dumps({"type": HEADER_VARIABLE_CREDENTIAL_TYPE, "values": values})
    encoded = encrypt_value_helper(payload)
    row = MagicMock()
    row.credential_b64 = encoded
    row.user_id = "alice"
    row.server_id = "srv-1"
    return row


def _prisma_with(row):
    prisma = MagicMock()
    prisma.db.litellm_mcpusercredentials.find_unique = AsyncMock(return_value=row)
    prisma.db.litellm_mcpusercredentials.upsert = AsyncMock()
    return prisma


@pytest.mark.asyncio
async def test_get_user_header_variables_round_trip():
    row = _row_with_header_variables({"CORP_USERNAME": "alice", "CORP_PASSWORD": "p1"})
    prisma = _prisma_with(row)
    out = await get_user_header_variables(prisma, "alice", "srv-1")
    assert out == {"CORP_USERNAME": "alice", "CORP_PASSWORD": "p1"}


@pytest.mark.asyncio
async def test_get_user_header_variables_missing_row():
    prisma = _prisma_with(row=None)
    assert await get_user_header_variables(prisma, "alice", "srv-1") == {}


@pytest.mark.asyncio
async def test_store_user_header_variables_writes_encrypted_payload():
    prisma = _prisma_with(row=None)
    await store_user_header_variables(
        prisma, "alice", "srv-1", {"CORP_USERNAME": "alice"}
    )
    call = prisma.db.litellm_mcpusercredentials.upsert.call_args
    stored = call.kwargs["data"]["create"]["credential_b64"]
    decoded = _decode_header_variables_payload(stored)
    assert decoded == {"CORP_USERNAME": "alice"}
    # Must not be plaintext JSON sitting in the DB
    assert "CORP_USERNAME" not in stored


@pytest.mark.asyncio
async def test_store_refuses_to_overwrite_non_header_variable_row():
    # Existing row holds a BYOK API key (just a plaintext string after decryption).
    row = MagicMock()
    row.credential_b64 = encrypt_value_helper("sk-some-byok-key")
    prisma = _prisma_with(row)
    with pytest.raises(ValueError, match="is not a header_variables payload"):
        await store_user_header_variables(prisma, "alice", "srv-1", {"A": "b"})


# ── resolve_static_headers_for_user ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_substitutes_global_only_when_no_per_user():
    server = _make_server(
        static_headers={"X-Env": "${ENV}"},
        header_variables=[{"name": "ENV", "value": "prod", "scope": "global"}],
    )
    out = await resolve_static_headers_for_user(
        server, user_id=None, prisma_client=None
    )
    assert out == {"X-Env": "prod"}


@pytest.mark.asyncio
async def test_resolve_raises_when_per_user_missing():
    server = _make_server(
        static_headers={"Authorization": "Basic ${USER}:${PASSWORD}"},
        header_variables=[
            {"name": "USER", "scope": "per_user"},
            {"name": "PASSWORD", "scope": "per_user"},
        ],
    )
    prisma = _prisma_with(row=None)
    with pytest.raises(MissingPerUserHeaderVariablesError) as excinfo:
        await resolve_static_headers_for_user(
            server, user_id="alice", prisma_client=prisma
        )
    err = excinfo.value
    assert sorted(err.missing_variables) == ["PASSWORD", "USER"]
    assert "/ui/?page=mcp-servers" in err.dashboard_url
    assert err.server_id == "srv-1"


@pytest.mark.asyncio
async def test_resolve_succeeds_after_user_fills_in_values():
    server = _make_server(
        static_headers={
            "Authorization": "Basic ${USER}:${PASSWORD}",
            "X-Env": "${ENV}",
        },
        header_variables=[
            {"name": "ENV", "value": "prod", "scope": "global"},
            {"name": "USER", "scope": "per_user"},
            {"name": "PASSWORD", "scope": "per_user"},
        ],
    )
    row = _row_with_header_variables({"USER": "alice", "PASSWORD": "p1"})
    prisma = _prisma_with(row)
    out = await resolve_static_headers_for_user(
        server, user_id="alice", prisma_client=prisma
    )
    assert out == {
        "Authorization": "Basic alice:p1",
        "X-Env": "prod",
    }


@pytest.mark.asyncio
async def test_resolve_with_no_header_variables_passes_through():
    server = _make_server(
        static_headers={"X-Plain": "static-value"},
        header_variables=None,
    )
    out = await resolve_static_headers_for_user(
        server, user_id="alice", prisma_client=None
    )
    assert out == {"X-Plain": "static-value"}


# ── End-to-end via MCPServerManager._call_regular_mcp_tool ────────────────────


@pytest.mark.asyncio
async def test_call_regular_mcp_tool_raises_friendly_error_on_missing_per_user_var(
    monkeypatch,
):
    """When ``static_headers`` references a per-user variable the caller has
    not configured, ``_call_regular_mcp_tool`` raises a 400 HTTPException whose
    detail contains a dashboard link the user can click."""
    from fastapi import HTTPException
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp import MCPAuth

    manager = MCPServerManager()
    server = _make_server(
        static_headers={"Authorization": "Basic ${CORP_USERNAME}:${CORP_PASSWORD}"},
        header_variables=[
            {"name": "CORP_USERNAME", "scope": "per_user"},
            {"name": "CORP_PASSWORD", "scope": "per_user"},
        ],
    )
    server.auth_type = MCPAuth.none

    prisma = _prisma_with(row=None)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        prisma,
        raising=False,
    )

    manager._create_mcp_client = AsyncMock()  # type: ignore[assignment]

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="alice")

    with pytest.raises(HTTPException) as excinfo:
        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="tool",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
            proxy_logging_obj=None,
            user_api_key_auth=user_auth,
        )
    assert excinfo.value.status_code == 400
    detail = str(excinfo.value.detail)
    assert "CORP_USERNAME" in detail
    assert "CORP_PASSWORD" in detail
    assert "/ui/?page=mcp-servers" in detail


@pytest.mark.asyncio
async def test_call_regular_mcp_tool_succeeds_after_per_user_vars_filled(monkeypatch):
    """Once the user has stored values, ``_call_regular_mcp_tool`` interpolates
    them into ``static_headers`` and forwards the resolved values upstream."""
    from unittest.mock import AsyncMock

    from mcp.types import CallToolResult

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp import MCPAuth

    manager = MCPServerManager()
    server = _make_server(
        static_headers={"Authorization": "Basic ${CORP_USERNAME}:${CORP_PASSWORD}"},
        header_variables=[
            {"name": "CORP_USERNAME", "scope": "per_user"},
            {"name": "CORP_PASSWORD", "scope": "per_user"},
        ],
    )
    server.auth_type = MCPAuth.none

    row = _row_with_header_variables({"CORP_USERNAME": "alice", "CORP_PASSWORD": "p1"})
    prisma = _prisma_with(row)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        prisma,
        raising=False,
    )

    captured = {}
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(
        return_value=CallToolResult(content=[], isError=False)
    )

    async def capture(server, mcp_auth_header, extra_headers, stdio_env, subject_token=None):
        captured["extra_headers"] = extra_headers
        return mock_client

    manager._create_mcp_client = AsyncMock(side_effect=capture)  # type: ignore[assignment]

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="alice")
    result = await manager._call_regular_mcp_tool(
        mcp_server=server,
        original_tool_name="tool",
        arguments={},
        tasks=[],
        mcp_auth_header=None,
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
        proxy_logging_obj=None,
        user_api_key_auth=user_auth,
    )
    assert isinstance(result, CallToolResult)
    assert captured["extra_headers"] == {"Authorization": "Basic alice:p1"}
