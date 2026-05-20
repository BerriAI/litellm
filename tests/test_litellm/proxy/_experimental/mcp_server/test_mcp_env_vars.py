"""Tests for MCP env-var interpolation utilities.

These cover the pure helpers in
``litellm.proxy._experimental.mcp_server.utils`` and do not require a DB
connection. The DB-backed per-user flow is exercised in higher-level
tests in tests/mcp_tests.
"""

import pytest

# Look up these names lazily on every access. Tests in this directory call
# ``importlib.reload`` on the utils module to exercise registration logic,
# which replaces ``MCPMissingUserEnvVarsError`` with a freshly-constructed
# class. A direct ``from ... import`` at module load time would freeze the
# old class object and ``pytest.raises(_u("MCPMissingUserEnvVarsError"))`` would
# stop matching the new class. Accessing the attribute through the module
# always picks up the current version.
import litellm.proxy._experimental.mcp_server.utils as _mcp_utils


def _u(name: str):
    return getattr(_mcp_utils, name)


def test_parse_admin_env_vars_splits_global_and_user():
    g, u = _u("parse_admin_env_vars")(
        [
            {"name": "DB_PROTOCOL", "value": "postgres", "scope": "global"},
            {"name": "DB_HOST", "value": "localhost", "scope": "global"},
            {
                "name": "CORP_USERNAME",
                "value": "",
                "scope": "user",
                "description": "Your DB username",
            },
            {"name": "CORP_PASSWORD", "value": "", "scope": "user"},
        ]
    )
    assert g == {"DB_PROTOCOL": "postgres", "DB_HOST": "localhost"}
    assert u == [
        {"name": "CORP_USERNAME", "description": "Your DB username"},
        {"name": "CORP_PASSWORD", "description": None},
    ]


def test_parse_admin_env_vars_handles_none_and_empty():
    assert _u("parse_admin_env_vars")(None) == ({}, [])
    assert _u("parse_admin_env_vars")([]) == ({}, [])


def test_parse_admin_env_vars_skips_malformed_entries():
    g, u = _u("parse_admin_env_vars")(
        [
            None,
            {"name": "", "value": "x"},
            {"value": "no_name"},
            {"name": "OK", "value": "v"},
        ]
    )
    assert g == {"OK": "v"}
    assert u == []


def test_find_env_var_references():
    assert _u("find_env_var_references")("") == set()
    assert _u("find_env_var_references")("plain") == set()
    assert _u("find_env_var_references")("${A}") == {"A"}
    assert _u("find_env_var_references")("${A}/${B}/${A}") == {"A", "B"}
    # Invalid identifier patterns should not match
    assert _u("find_env_var_references")("${1abc}") == set()
    assert _u("find_env_var_references")("${a-b}") == set()


def test_collect_env_var_references():
    refs = _u("collect_env_var_references")(strings=["${A}", "static", "${B}-${C}", None])
    assert refs == {"A", "B", "C"}


def test_interpolate_env_vars_replaces_known_and_leaves_unknown():
    assert _u("interpolate_env_vars")("${A}://${B}/${C}", {"A": "https", "B": "host"}) == (
        "https://host/${C}"
    )


def test_interpolate_headers_returns_independent_copy():
    headers = {"X-Url": "${A}://x"}
    out = _u("interpolate_headers")(headers, {"A": "https"})
    assert out == {"X-Url": "https://x"}
    # original untouched
    assert headers == {"X-Url": "${A}://x"}


def test_build_env_var_setup_url_includes_server_id(monkeypatch):
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    url = _u("build_env_var_setup_url")("abc-123")
    assert url.startswith("/ui/?page=mcp-servers")
    assert "fill_env_vars=abc-123" in url


def test_build_env_var_setup_url_prepends_proxy_base_url(monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com/")
    url = _u("build_env_var_setup_url")("abc-123")
    assert url.startswith("https://proxy.example.com/ui/")
    assert "fill_env_vars=abc-123" in url


def test_missing_user_env_vars_error_message_is_friendly():
    with pytest.raises(_u("MCPMissingUserEnvVarsError")) as exc_info:
        raise _u("MCPMissingUserEnvVarsError")(
            server_id="abc-123",
            server_name="CorporateDB",
            missing=["CORP_USERNAME", "CORP_PASSWORD"],
            setup_url="https://proxy.example.com/ui/?page=mcp-servers&fill_env_vars=abc-123",
        )
    err = exc_info.value
    text = str(err)
    assert "CorporateDB" in text
    assert "CORP_USERNAME" in text
    assert "CORP_PASSWORD" in text
    assert "fill_env_vars=abc-123" in text
    assert err.server_id == "abc-123"
    assert err.missing == ["CORP_USERNAME", "CORP_PASSWORD"]


def test_missing_user_env_vars_error_singular_message():
    err = _u("MCPMissingUserEnvVarsError")(
        server_id="abc",
        server_name=None,
        missing=["X"],
        setup_url="/ui/",
    )
    text = str(err)
    # Singular "variable" rather than "variables" when only one is missing
    assert "variable that you need to fill in" in text
    # Falls back to server_id when server_name is missing
    assert "abc" in text


# ── _resolve_static_headers_with_env_vars ────────────────────────────────


@pytest.fixture
def mock_server():
    """A minimal MCPServer-like object for the static-headers resolver."""
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id="srv-1",
        name="srv",
        server_name="srv",
        transport="http",
        url="https://example.com",
        static_headers={
            "X-DB-URL": "${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@${DB_HOST}/db",
            "X-Other": "literal",
        },
        env_vars=[
            {"name": "DB_PROTOCOL", "value": "postgres", "scope": "global"},
            {"name": "DB_HOST", "value": "db.local", "scope": "global"},
            {
                "name": "CORP_USERNAME",
                "value": "",
                "scope": "user",
                "description": "Your DB username",
            },
            {"name": "CORP_PASSWORD", "value": "", "scope": "user"},
        ],
    )


@pytest.mark.asyncio
async def test_resolve_static_headers_interpolates_globals_and_user(
    mock_server, monkeypatch
):
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()

    # Stub the per-user lookup so we don't need a real DB.
    async def fake_load_user_env_vars(server, user_api_key_auth):
        return {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    headers = await manager._resolve_static_headers_with_env_vars(
        mock_server, user_api_key_auth=object()
    )
    assert headers == {
        "X-DB-URL": "postgres://alice:s3cret@db.local/db",
        "X-Other": "literal",
    }


@pytest.mark.asyncio
async def test_resolve_static_headers_raises_when_user_vars_missing(
    mock_server, monkeypatch
):
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()

    async def fake_load_user_env_vars(server, user_api_key_auth):
        # User has only filled in one of the two required vars
        return {"CORP_USERNAME": "alice"}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    with pytest.raises(_u("MCPMissingUserEnvVarsError")) as exc:
        await manager._resolve_static_headers_with_env_vars(
            mock_server, user_api_key_auth=object()
        )
    assert exc.value.missing == ["CORP_PASSWORD"]
    assert exc.value.server_id == "srv-1"
    assert "fill_env_vars=srv-1" in exc.value.setup_url


@pytest.mark.asyncio
async def test_resolve_static_headers_passthrough_when_no_env_vars():
    """Servers without env_vars should keep static_headers untouched."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-2",
        name="srv2",
        transport="http",
        url="https://example.com",
        static_headers={"Authorization": "Bearer admin-static"},
        env_vars=None,
    )
    headers = await manager._resolve_static_headers_with_env_vars(server, None)
    assert headers == {"Authorization": "Bearer admin-static"}


@pytest.mark.asyncio
async def test_resolve_static_headers_unreferenced_user_var_is_not_blocking(
    monkeypatch,
):
    """A per-user var declared by the admin but never referenced in
    static_headers must not block the request — only blocking-by-use is
    enforced."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-3",
        name="srv3",
        transport="http",
        url="https://example.com",
        static_headers={"X-Static": "${GLOBAL_VAR}"},
        env_vars=[
            {"name": "GLOBAL_VAR", "value": "ok", "scope": "global"},
            # User var declared but not referenced anywhere — should be ignored.
            {"name": "UNUSED_USER_VAR", "value": "", "scope": "user"},
        ],
    )

    async def fake_load_user_env_vars(server, user_api_key_auth):
        return {}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    headers = await manager._resolve_static_headers_with_env_vars(server, object())
    assert headers == {"X-Static": "ok"}
