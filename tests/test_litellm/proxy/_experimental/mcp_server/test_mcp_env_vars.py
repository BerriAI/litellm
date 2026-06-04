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
    refs = _u("collect_env_var_references")(
        strings=["${A}", "static", "${B}-${C}", None]
    )
    assert refs == {"A", "B", "C"}


def test_interpolate_env_vars_replaces_known_and_leaves_unknown():
    assert _u("interpolate_env_vars")(
        "${A}://${B}/${C}", {"A": "https", "B": "host"}
    ) == ("https://host/${C}")


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


def test_build_env_var_setup_url_encodes_unsafe_server_id(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    server_id = "a&b=c #d/e"
    url = _u("build_env_var_setup_url")(server_id)
    assert "a&b=c #d/e" not in url
    parsed = parse_qs(urlsplit(url).query)
    assert parsed["fill_env_vars"] == [server_id]


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
    assert 'Cannot connect to MCP server "CorporateDB".' in text
    assert "- CORP_USERNAME" in text
    assert "- CORP_PASSWORD" in text
    assert "fill_env_vars=abc-123" in text
    assert "Set your credentials here:" in text
    assert err.server_id == "abc-123"
    assert err.missing == ["CORP_USERNAME", "CORP_PASSWORD"]


def test_missing_user_env_vars_error_falls_back_to_server_id():
    err = _u("MCPMissingUserEnvVarsError")(
        server_id="abc",
        server_name=None,
        missing=["X"],
        setup_url="/ui/",
    )
    text = str(err)
    # Falls back to server_id when server_name is missing
    assert 'Cannot connect to MCP server "abc".' in text
    assert "- X" in text


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

    async def fake_load_user_env_vars(
        server, user_api_key_auth, *, force_refresh=False
    ):
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
async def test_resolve_static_headers_rechecks_db_before_raising_412(
    mock_server, monkeypatch
):
    """A stale cached negative must not produce a 412 on the tool-call path.

    Cache invalidation is process-local, so a user who stored values on another
    worker can have a stale (incomplete) entry on this one. Before raising
    MCPMissingUserEnvVarsError the resolver must re-read with force_refresh and
    honor the fresh DB values.
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()

    calls = []

    async def fake_load_user_env_vars(
        server, user_api_key_auth, *, force_refresh=False
    ):
        calls.append(force_refresh)
        if force_refresh:
            # Fresh DB read sees the values the user stored on another worker.
            return {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}
        # Stale, process-local cached entry is still missing CORP_PASSWORD.
        return {"CORP_USERNAME": "alice"}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    headers = await manager._resolve_static_headers_with_env_vars(
        mock_server, user_api_key_auth=object()
    )
    assert headers == {
        "X-DB-URL": "postgres://alice:s3cret@db.local/db",
        "X-Other": "literal",
    }
    # The cached read happened first, then exactly one forced DB re-read.
    assert calls == [False, True]


@pytest.mark.asyncio
async def test_resolve_static_headers_missing_is_non_blocking_for_listing(
    mock_server, monkeypatch
):
    """With raise_on_missing=False (the tool-list path), missing per-user vars
    must NOT raise. Available vars interpolate; unfilled ${NAME} refs are left
    untouched so the server's tools still appear in the listing."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()

    async def fake_load_user_env_vars(server, user_api_key_auth):
        # User has only filled in one of the two required vars.
        return {"CORP_USERNAME": "alice"}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    headers = await manager._resolve_static_headers_with_env_vars(
        mock_server, user_api_key_auth=object(), raise_on_missing=False
    )
    # Globals + the supplied user var are interpolated; the still-missing
    # CORP_PASSWORD reference is left as a literal rather than blocking listing.
    assert headers == {
        "X-DB-URL": "postgres://alice:${CORP_PASSWORD}@db.local/db",
        "X-Other": "literal",
    }


@pytest.mark.asyncio
async def test_resolve_static_headers_propagates_db_error_on_tool_call(
    mock_server, monkeypatch
):
    """A DB failure on the tool-call path must surface as a real error, not be
    masked as a "missing credentials" MCPMissingUserEnvVarsError (412)."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()

    async def boom(server, user_api_key_auth):
        raise RuntimeError("db down")

    monkeypatch.setattr(manager, "_load_user_env_vars", boom)

    with pytest.raises(RuntimeError, match="db down"):
        await manager._resolve_static_headers_with_env_vars(
            mock_server, user_api_key_auth=object()
        )


@pytest.mark.asyncio
async def test_resolve_static_headers_swallows_db_error_on_listing(
    mock_server, monkeypatch
):
    """On the listing path a DB failure is non-blocking: globals interpolate
    and unfilled per-user ${NAME} refs are left untouched."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()

    async def boom(server, user_api_key_auth):
        raise RuntimeError("db down")

    monkeypatch.setattr(manager, "_load_user_env_vars", boom)

    headers = await manager._resolve_static_headers_with_env_vars(
        mock_server, user_api_key_auth=object(), raise_on_missing=False
    )
    assert headers == {
        "X-DB-URL": "postgres://${CORP_USERNAME}:${CORP_PASSWORD}@db.local/db",
        "X-Other": "literal",
    }


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


@pytest.mark.asyncio
async def test_resolve_static_headers_stale_user_value_cannot_override_global(
    monkeypatch,
):
    """A var that used to be user-scoped (so the user has a stored value) but is
    now global must resolve to the admin's global value, not the stale per-user
    row. Otherwise a user could override admin-configured headers indefinitely."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-4",
        name="srv4",
        transport="http",
        url="https://example.com",
        static_headers={"X-DB-URL": "${DB_HOST}/${CORP_USERNAME}"},
        env_vars=[
            # DB_HOST is now global; it used to be user-scoped.
            {"name": "DB_HOST", "value": "admin-db", "scope": "global"},
            {"name": "CORP_USERNAME", "value": "", "scope": "user"},
        ],
    )

    async def fake_load_user_env_vars(server, user_api_key_auth):
        # Stale DB_HOST row left over from when it was user-scoped.
        return {"DB_HOST": "evil-db", "CORP_USERNAME": "alice"}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    headers = await manager._resolve_static_headers_with_env_vars(server, object())
    assert headers == {"X-DB-URL": "admin-db/alice"}


# ── _load_user_env_vars guard paths ────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_user_env_vars_returns_empty_without_user():
    """No user auth → no per-user lookup is attempted."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="s", name="s", transport="http", url="https://example.com"
    )
    assert await manager._load_user_env_vars(server, None) == {}


@pytest.mark.asyncio
async def test_load_user_env_vars_returns_empty_without_user_id():
    """User auth without a user_id (e.g. anonymous virtual key) → empty dict."""
    from unittest.mock import MagicMock

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="s", name="s", transport="http", url="https://example.com"
    )
    fake_auth = MagicMock()
    fake_auth.user_id = None
    assert await manager._load_user_env_vars(server, fake_auth) == {}


@pytest.mark.asyncio
async def test_load_user_env_vars_returns_empty_when_db_unavailable(monkeypatch):
    """If prisma_client is None, the lookup short-circuits rather than crashing."""
    from unittest.mock import MagicMock

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="s", name="s", transport="http", url="https://example.com"
    )
    fake_auth = MagicMock()
    fake_auth.user_id = "alice"
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    assert await manager._load_user_env_vars(server, fake_auth) == {}


@pytest.mark.asyncio
async def test_load_user_env_vars_caches_within_ttl(env_vars_salt_key, monkeypatch):
    """A second load within the TTL window is served from the in-memory cache,
    keeping the hot tool-call/tool-listing path off the DB."""
    from unittest.mock import MagicMock

    from litellm.proxy._experimental.mcp_server import mcp_server_manager as mgr_mod
    from litellm.proxy._experimental.mcp_server.db import store_user_env_vars
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mgr_mod._user_env_vars_cache.clear()

    blob_prisma = _mock_env_vars_prisma()
    await store_user_env_vars(blob_prisma, "alice", "srv-1", {"TOKEN": "t0p"})
    row = MagicMock()
    row.values_b64 = _captured_values_blob(blob_prisma)

    prisma = _mock_env_vars_prisma(row=row)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-1", name="s", transport="http", url="https://example.com"
    )
    fake_auth = MagicMock()
    fake_auth.user_id = "alice"

    first = await manager._load_user_env_vars(server, fake_auth)
    second = await manager._load_user_env_vars(server, fake_auth)
    assert first == {"TOKEN": "t0p"} == second
    assert prisma.db.litellm_mcpuserenvvars.find_unique.await_count == 1

    mgr_mod._user_env_vars_cache.clear()


@pytest.mark.asyncio
async def test_load_user_env_vars_force_refresh_bypasses_cache(
    env_vars_salt_key, monkeypatch
):
    """force_refresh re-reads from the DB even with a fresh cached entry, so a
    process-local stale value cannot mask credentials stored on another worker."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server import mcp_server_manager as mgr_mod
    from litellm.proxy._experimental.mcp_server.db import store_user_env_vars
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mgr_mod._user_env_vars_cache.clear()

    blob_prisma = _mock_env_vars_prisma()
    await store_user_env_vars(blob_prisma, "alice", "srv-1", {"TOKEN": "old"})
    old_row = MagicMock()
    old_row.values_b64 = _captured_values_blob(blob_prisma)
    await store_user_env_vars(blob_prisma, "alice", "srv-1", {"TOKEN": "new"})
    new_row = MagicMock()
    new_row.values_b64 = _captured_values_blob(blob_prisma)

    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpuserenvvars.find_unique = AsyncMock(
        side_effect=[old_row, new_row]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-1", name="s", transport="http", url="https://example.com"
    )
    fake_auth = MagicMock()
    fake_auth.user_id = "alice"

    assert await manager._load_user_env_vars(server, fake_auth) == {"TOKEN": "old"}
    # A normal load is served from cache (still "old"); force_refresh re-reads.
    assert await manager._load_user_env_vars(server, fake_auth) == {"TOKEN": "old"}
    assert await manager._load_user_env_vars(server, fake_auth, force_refresh=True) == {
        "TOKEN": "new"
    }
    assert prisma.db.litellm_mcpuserenvvars.find_unique.await_count == 2

    mgr_mod._user_env_vars_cache.clear()


@pytest.mark.asyncio
async def test_load_user_env_vars_invalidation_forces_refetch(
    env_vars_salt_key, monkeypatch
):
    """After invalidation (store/clear) the next load reads fresh from the DB
    instead of serving the stale cached value."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server import mcp_server_manager as mgr_mod
    from litellm.proxy._experimental.mcp_server.db import store_user_env_vars
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
        invalidate_user_env_vars_cache,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mgr_mod._user_env_vars_cache.clear()

    blob_prisma = _mock_env_vars_prisma()
    await store_user_env_vars(blob_prisma, "alice", "srv-1", {"TOKEN": "old"})
    old_row = MagicMock()
    old_row.values_b64 = _captured_values_blob(blob_prisma)
    await store_user_env_vars(blob_prisma, "alice", "srv-1", {"TOKEN": "new"})
    new_row = MagicMock()
    new_row.values_b64 = _captured_values_blob(blob_prisma)

    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpuserenvvars.find_unique = AsyncMock(
        side_effect=[old_row, new_row]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-1", name="s", transport="http", url="https://example.com"
    )
    fake_auth = MagicMock()
    fake_auth.user_id = "alice"

    assert await manager._load_user_env_vars(server, fake_auth) == {"TOKEN": "old"}
    invalidate_user_env_vars_cache("alice", "srv-1")
    assert await manager._load_user_env_vars(server, fake_auth) == {"TOKEN": "new"}
    assert prisma.db.litellm_mcpuserenvvars.find_unique.await_count == 2

    mgr_mod._user_env_vars_cache.clear()


# ── DB helpers: per-user env vars ─────────────────────────────────────────

_SALT_KEY = "test-salt-key-for-env-vars-tests-1234"


@pytest.fixture
def env_vars_salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", _SALT_KEY)


def _mock_env_vars_prisma(row=None):
    """Build a MagicMock prisma_client whose env-vars table returns ``row``."""
    from unittest.mock import AsyncMock, MagicMock

    prisma = MagicMock()
    prisma.db.litellm_mcpuserenvvars.find_unique = AsyncMock(return_value=row)
    prisma.db.litellm_mcpuserenvvars.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_mcpuserenvvars.upsert = AsyncMock()
    prisma.db.litellm_mcpuserenvvars.delete_many = AsyncMock()
    return prisma


def _captured_values_blob(prisma) -> str:
    """Pull the values_b64 value passed to the most recent upsert."""
    call = prisma.db.litellm_mcpuserenvvars.upsert.call_args
    data = call.kwargs["data"]
    create_value = data["create"]["values_b64"]
    update_value = data["update"]["values_b64"]
    assert create_value == update_value
    return create_value


@pytest.mark.asyncio
async def test_store_user_env_vars_does_not_persist_plaintext(env_vars_salt_key):
    from litellm.proxy._experimental.mcp_server.db import store_user_env_vars

    prisma = _mock_env_vars_prisma()
    await store_user_env_vars(
        prisma, "alice", "srv-1", {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}
    )
    stored = _captured_values_blob(prisma)
    # Stored blob must not contain plaintext values
    assert "s3cret" not in stored
    assert "alice" not in stored or stored.count("alice") == 0  # encrypted form


@pytest.mark.asyncio
async def test_get_user_env_vars_round_trip(env_vars_salt_key):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server.db import (
        get_user_env_vars,
        store_user_env_vars,
    )

    prisma = _mock_env_vars_prisma()
    payload = {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}
    await store_user_env_vars(prisma, "alice", "srv-1", payload)
    stored = _captured_values_blob(prisma)

    # Now simulate the read returning that blob.
    row = MagicMock()
    row.values_b64 = stored
    prisma.db.litellm_mcpuserenvvars.find_unique = AsyncMock(return_value=row)

    result = await get_user_env_vars(prisma, "alice", "srv-1")
    assert result == payload


@pytest.mark.asyncio
async def test_get_user_env_vars_returns_empty_for_missing_row():
    from litellm.proxy._experimental.mcp_server.db import get_user_env_vars

    prisma = _mock_env_vars_prisma(row=None)
    assert await get_user_env_vars(prisma, "alice", "srv-1") == {}


@pytest.mark.asyncio
async def test_get_user_env_vars_bulk_distributes_results(env_vars_salt_key):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server.db import (
        get_user_env_vars_bulk,
        store_user_env_vars,
    )

    # Use store_user_env_vars to get correctly-encrypted blobs.
    prisma1 = _mock_env_vars_prisma()
    await store_user_env_vars(prisma1, "alice", "srv-1", {"A": "1"})
    blob1 = _captured_values_blob(prisma1)
    await store_user_env_vars(prisma1, "alice", "srv-2", {"B": "2"})
    blob2 = _captured_values_blob(prisma1)

    row1 = MagicMock()
    row1.server_id = "srv-1"
    row1.values_b64 = blob1
    row2 = MagicMock()
    row2.server_id = "srv-2"
    row2.values_b64 = blob2

    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpuserenvvars.find_many = AsyncMock(return_value=[row1, row2])
    result = await get_user_env_vars_bulk(prisma, "alice", ["srv-1", "srv-2", "srv-3"])
    assert result == {"srv-1": {"A": "1"}, "srv-2": {"B": "2"}}


@pytest.mark.asyncio
async def test_get_user_env_vars_bulk_empty_ids_short_circuits():
    from litellm.proxy._experimental.mcp_server.db import get_user_env_vars_bulk

    prisma = _mock_env_vars_prisma()
    assert await get_user_env_vars_bulk(prisma, "alice", []) == {}
    # find_many should never have been called
    assert prisma.db.litellm_mcpuserenvvars.find_many.await_count == 0


@pytest.mark.asyncio
async def test_delete_user_env_vars_is_idempotent_delete_many():
    """Delete must use ``delete_many`` so a missing row is a no-op rather than
    raising RecordNotFound; real DB errors are left to propagate."""
    from litellm.proxy._experimental.mcp_server.db import delete_user_env_vars

    prisma = _mock_env_vars_prisma()
    await delete_user_env_vars(prisma, "alice", "srv-1")
    prisma.db.litellm_mcpuserenvvars.delete_many.assert_awaited_once()
    call = prisma.db.litellm_mcpuserenvvars.delete_many.call_args
    assert call.kwargs["where"] == {"user_id": "alice", "server_id": "srv-1"}


@pytest.mark.asyncio
async def test_delete_mcp_server_removes_orphaned_user_env_vars():
    """Deleting a server must also drop every user's per-user env var rows for
    it; there is no FK cascade, so skipping this leaves orphaned credentials."""
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.db import delete_mcp_server

    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpservertable.delete = AsyncMock(return_value=object())

    await delete_mcp_server(prisma, "srv-1")

    prisma.db.litellm_mcpuserenvvars.delete_many.assert_awaited_once()
    call = prisma.db.litellm_mcpuserenvvars.delete_many.call_args
    assert call.kwargs["where"] == {"server_id": "srv-1"}


@pytest.mark.asyncio
async def test_delete_mcp_server_skips_env_var_cleanup_when_server_missing():
    """A no-op delete (server not found) must not touch the env var table."""
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.db import delete_mcp_server

    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpservertable.delete = AsyncMock(return_value=None)

    result = await delete_mcp_server(prisma, "srv-1")

    assert result is None
    prisma.db.litellm_mcpuserenvvars.delete_many.assert_not_awaited()


# ── DB helpers: global env vars encrypted at rest ─────────────────────────


def _global_env_var_server_request(env_vars):
    from litellm.proxy._types import NewMCPServerRequest

    return NewMCPServerRequest(
        alias="echo",
        url="https://upstream.example.com/mcp",
        transport="http",
        auth_type="none",
        static_headers={"X-Db": "${DB_PASSWORD}"},
        env_vars=env_vars,
    )


def test_prepare_mcp_server_data_encrypts_global_env_var_values(env_vars_salt_key):
    """``scope="global"`` secrets must be encrypted before they reach the JSON
    column, while ``scope="user"`` placeholders (not secrets) stay verbatim."""
    import json

    from litellm.proxy._experimental.mcp_server.db import (
        _prepare_mcp_server_data,
        decrypt_global_env_var_values,
    )
    from litellm.proxy._types import MCPEnvVar

    req = _global_env_var_server_request(
        [
            MCPEnvVar(name="DB_PASSWORD", value="s3cr3t-p@ss", scope="global"),
            MCPEnvVar(
                name="CORP_USER",
                value="placeholder-hint",
                scope="user",
                description="your db user",
            ),
        ]
    )

    stored = _prepare_mcp_server_data(req)["env_vars"]
    entries = {e["name"]: e for e in json.loads(stored)}

    # The global secret is unrecoverable from the stored JSON ...
    assert "s3cr3t-p@ss" not in stored
    assert entries["DB_PASSWORD"]["value"] != "s3cr3t-p@ss"
    # ... but the per-user placeholder is stored as-is.
    assert entries["CORP_USER"]["value"] == "placeholder-hint"

    # And the encrypted global decrypts back to the original secret.
    decrypt_global_env_var_values(list(entries.values()))
    assert entries["DB_PASSWORD"]["value"] == "s3cr3t-p@ss"
    assert entries["CORP_USER"]["value"] == "placeholder-hint"


@pytest.mark.asyncio
async def test_build_mcp_server_from_table_decrypts_global_env_vars(env_vars_salt_key):
    """End-to-end: an encrypted global value persisted in the DB must be
    decrypted when the server is built into the runtime registry, so ``${NAME}``
    headers interpolate to the real secret instead of forwarding ciphertext."""
    import json

    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.proxy._types import LiteLLM_MCPServerTable, MCPEnvVar

    req = _global_env_var_server_request(
        [MCPEnvVar(name="DB_PASSWORD", value="s3cr3t-p@ss", scope="global")]
    )
    prepared = _prepare_mcp_server_data(req)

    table = LiteLLM_MCPServerTable(
        server_id="srv-global",
        alias="echo",
        url="https://upstream.example.com/mcp",
        transport="http",
        auth_type="none",
        static_headers={"X-Db": "${DB_PASSWORD}"},
        env_vars=json.loads(prepared["env_vars"]),
    )

    manager = MCPServerManager()
    server = await manager.build_mcp_server_from_table(table)

    headers = await manager._resolve_static_headers_with_env_vars(server, None)
    assert headers == {"X-Db": "s3cr3t-p@ss"}


def test_decrypt_global_env_var_drops_undecryptable_value(
    env_vars_salt_key, monkeypatch
):
    """A global value encrypted under a previous salt key must be dropped (not
    forwarded as ciphertext) and surfaced as a warning, so a rotated
    ``LITELLM_SALT_KEY`` can't silently leak ciphertext into ``${NAME}`` headers."""
    import json
    from unittest.mock import MagicMock

    import litellm.proxy._experimental.mcp_server.db as mcp_db
    from litellm.proxy._experimental.mcp_server.db import (
        _prepare_mcp_server_data,
        decrypt_global_env_var_values,
    )
    from litellm.proxy._types import MCPEnvVar

    req = _global_env_var_server_request(
        [MCPEnvVar(name="DB_PASSWORD", value="s3cr3t-p@ss", scope="global")]
    )
    entries = json.loads(_prepare_mcp_server_data(req)["env_vars"])
    ciphertext = entries[0]["value"]
    assert ciphertext != "s3cr3t-p@ss"  # encrypted under the original salt key

    # Rotate the salt key so the stored ciphertext no longer decrypts.
    monkeypatch.setenv("LITELLM_SALT_KEY", "a-totally-different-salt-key-0000")
    logger = MagicMock()
    monkeypatch.setattr(mcp_db, "verbose_proxy_logger", logger)

    decrypt_global_env_var_values(entries)

    assert entries[0]["value"] == ""
    assert ciphertext not in json.dumps(entries)
    logger.warning.assert_called_once()
    assert "DB_PASSWORD" in logger.warning.call_args.args


# ── REST exception handling ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_user_env_vars_error_renders_in_mcp_call_tool():
    """The MCP ``call_tool`` handler must turn ``MCPMissingUserEnvVarsError``
    into a friendly ``CallToolResult`` with ``isError=True`` so Claude Code
    surfaces the setup URL instead of an opaque internal error."""
    from mcp.types import TextContent

    err = _u("MCPMissingUserEnvVarsError")(
        server_id="srv-99",
        server_name="CorporateDB",
        missing=["CORP_USERNAME"],
        setup_url="/ui/?page=mcp-servers&fill_env_vars=srv-99",
    )
    # We don't want to spin up the full MCP server framework — just
    # mimic the except-clause behavior the @server.call_tool handler uses.
    from mcp.types import CallToolResult

    result = CallToolResult(
        content=[TextContent(text=str(err), type="text")],
        isError=True,
    )
    assert result.isError is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "CorporateDB" in text
    assert "CORP_USERNAME" in text
    assert "fill_env_vars=srv-99" in text
