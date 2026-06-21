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


@pytest.mark.asyncio
async def test_resolve_static_headers_dual_scope_var_uses_global_without_412(
    monkeypatch,
):
    """A var declared with both ``global`` and ``user`` scope is covered by the
    global value (globals win in the merge), so the tool-call path must resolve
    it from the global instead of raising a 412 when the user hasn't filled it
    in. This happens during a global-to-user (or user-to-global) migration."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-5",
        name="srv5",
        transport="http",
        url="https://example.com",
        static_headers={"Authorization": "Bearer ${SHARED_TOKEN}"},
        env_vars=[
            {"name": "SHARED_TOKEN", "value": "global-secret", "scope": "global"},
            {"name": "SHARED_TOKEN", "value": "", "scope": "user"},
        ],
    )

    load_calls = []

    async def fake_load_user_env_vars(
        server, user_api_key_auth, *, force_refresh=False
    ):
        load_calls.append(force_refresh)
        return {}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    headers = await manager._resolve_static_headers_with_env_vars(
        server, user_api_key_auth=object()
    )
    assert headers == {"Authorization": "Bearer global-secret"}
    # The global fully covers the reference, so no per-user lookup is needed.
    assert load_calls == []


@pytest.mark.asyncio
async def test_resolve_static_headers_empty_global_does_not_cover_user_var(
    monkeypatch,
):
    """An empty-valued global must not cover a referenced per-user var. The
    global carries no usable value, so the tool-call path still raises a 412
    when the user hasn't supplied one, instead of silently interpolating an
    empty string into the header."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-6",
        name="srv6",
        transport="http",
        url="https://example.com",
        static_headers={"Authorization": "Bearer ${SHARED_TOKEN}"},
        env_vars=[
            {"name": "SHARED_TOKEN", "value": "", "scope": "global"},
            {"name": "SHARED_TOKEN", "value": "", "scope": "user"},
        ],
    )

    async def fake_load_user_env_vars(
        server, user_api_key_auth, *, force_refresh=False
    ):
        return {}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    with pytest.raises(_u("MCPMissingUserEnvVarsError")) as exc:
        await manager._resolve_static_headers_with_env_vars(
            server, user_api_key_auth=object()
        )
    assert exc.value.missing == ["SHARED_TOKEN"]


@pytest.mark.asyncio
async def test_resolve_static_headers_user_value_wins_over_empty_global(
    monkeypatch,
):
    """When a global is empty, a value the user did supply must win the merge
    rather than being clobbered by the empty global. The header resolves to the
    user's value, not an empty string."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-7",
        name="srv7",
        transport="http",
        url="https://example.com",
        static_headers={"Authorization": "Bearer ${SHARED_TOKEN}"},
        env_vars=[
            {"name": "SHARED_TOKEN", "value": "", "scope": "global"},
            {"name": "SHARED_TOKEN", "value": "", "scope": "user"},
        ],
    )

    async def fake_load_user_env_vars(
        server, user_api_key_auth, *, force_refresh=False
    ):
        return {"SHARED_TOKEN": "user-secret"}

    monkeypatch.setattr(manager, "_load_user_env_vars", fake_load_user_env_vars)

    headers = await manager._resolve_static_headers_with_env_vars(
        server, user_api_key_auth=object()
    )
    assert headers == {"Authorization": "Bearer user-secret"}


# ── health-check skip for per-user-env-var-backed headers ──────────────────


@pytest.mark.parametrize(
    "static_headers, env_vars, expected",
    [
        (
            {"Authorization": "Bearer ${GITHUB_TOKEN}"},
            [{"name": "GITHUB_TOKEN", "value": "", "scope": "user"}],
            True,
        ),
        (
            {"Authorization": "Bearer ${SHARED_TOKEN}"},
            [{"name": "SHARED_TOKEN", "value": "abc", "scope": "global"}],
            False,
        ),
        (
            {"X-Static": "literal"},
            [{"name": "GITHUB_TOKEN", "value": "", "scope": "user"}],
            False,
        ),
        (None, [{"name": "GITHUB_TOKEN", "value": "", "scope": "user"}], False),
        ({"Authorization": "Bearer ${GITHUB_TOKEN}"}, None, False),
    ],
)
def test_references_per_user_env_var(static_headers, env_vars, expected):
    """Only headers that actually reference a *per-user* var count: globals and
    declared-but-unreferenced user vars do not, since the userless probe can
    still resolve (or simply not need) them."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-x",
        name="srv",
        transport="http",
        url="https://example.com",
        static_headers=static_headers,
        env_vars=env_vars,
    )
    assert manager._references_per_user_env_var(server) is expected


@pytest.mark.asyncio
async def test_health_check_skips_servers_referencing_per_user_env_var(
    mock_server, monkeypatch
):
    """A userless health probe cannot fill per-user ${NAME} placeholders, so a
    server whose static_headers reference one must report 'unknown' without
    connecting. Otherwise it forwards the literal placeholder upstream, gets a
    401, and flips to 'unhealthy' even though real user calls succeed."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()
    manager.registry[mock_server.server_id] = mock_server

    created = []

    async def fake_create_client(*args, **kwargs):
        created.append((args, kwargs))
        raise RuntimeError("upstream rejected literal ${NAME}")

    monkeypatch.setattr(manager, "_create_mcp_client", fake_create_client)

    result = await manager.health_check_server(mock_server.server_id)

    assert created == []
    assert result.status == "unknown"
    assert result.health_check_error is None


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
async def test_load_user_env_vars_raises_when_db_unavailable(monkeypatch):
    """A missing DB connection must raise, not return ``{}``. Returning ``{}``
    would be indistinguishable from "user has no values" and would mislead the
    tool-call path into a "set up your credentials" 412 the user can never
    satisfy (per-user env vars are unusable without a DB)."""
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
    with pytest.raises(RuntimeError, match="database connection"):
        await manager._load_user_env_vars(server, fake_auth)


@pytest.mark.asyncio
async def test_resolve_static_headers_db_unavailable_is_not_missing_412(
    mock_server, monkeypatch
):
    """On the tool-call path, an unavailable DB must surface as a real error
    rather than a misleading MCPMissingUserEnvVarsError (412). This guards the
    regression where ``_load_user_env_vars`` returned ``{}`` when prisma_client
    was None, making a DB outage look like "user has no credentials"."""
    from unittest.mock import MagicMock

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    manager = MCPServerManager()
    fake_auth = MagicMock()
    fake_auth.user_id = "alice"
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    with pytest.raises(RuntimeError, match="database connection"):
        await manager._resolve_static_headers_with_env_vars(
            mock_server, user_api_key_auth=fake_auth
        )


@pytest.mark.asyncio
async def test_load_user_env_vars_caches_within_ttl(env_vars_salt_key, monkeypatch):
    """A second load within the TTL window is served from the in-memory cache,
    keeping the hot tool-call/tool-listing path off the DB."""
    from unittest.mock import MagicMock

    from litellm.proxy._experimental.mcp_server import mcp_server_manager as mgr_mod
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mgr_mod._user_env_vars_cache.clear()

    row = MagicMock()
    row.values_b64 = _encrypted_user_env_blob({"TOKEN": "t0p"})

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
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mgr_mod._user_env_vars_cache.clear()

    old_row = MagicMock()
    old_row.values_b64 = _encrypted_user_env_blob({"TOKEN": "old"})
    new_row = MagicMock()
    new_row.values_b64 = _encrypted_user_env_blob({"TOKEN": "new"})

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
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
        invalidate_user_env_vars_cache,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mgr_mod._user_env_vars_cache.clear()

    old_row = MagicMock()
    old_row.values_b64 = _encrypted_user_env_blob({"TOKEN": "old"})
    new_row = MagicMock()
    new_row.values_b64 = _encrypted_user_env_blob({"TOKEN": "new"})

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
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock()
    return prisma


def _encrypted_user_env_blob(values: dict) -> str:
    """Encrypt ``values`` the way the production per-user write does, so tests can
    seed a correctly-encrypted ``values_b64`` blob without a live DB."""
    import json

    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    return encrypt_value_helper(json.dumps(values))


def _transactional_env_vars_prisma(read_delay: float = 0.0):
    """A prisma stand-in backed by an in-memory store that honours
    ``db.tx()`` and the ``pg_advisory_xact_lock`` advisory lock.

    ``read_delay`` inserts an ``await`` point inside ``find_unique`` so two
    concurrent merges interleave between their read and write; the advisory lock
    is what keeps them from clobbering each other. Drop the lock and the second
    write wins, losing the first update.
    """
    import asyncio
    from unittest.mock import MagicMock

    class _Store:
        def __init__(self):
            self.rows = {}
            self.locks = {}

    class _Table:
        def __init__(self, store, delay=0.0):
            self._store = store
            self._delay = delay

        async def find_unique(self, where):
            ident = where["user_id_server_id"]
            key = (ident["user_id"], ident["server_id"])
            blob = self._store.rows.get(key)
            # Yield after capturing the read so an unserialised concurrent merge
            # would race on this stale snapshot.
            if self._delay:
                await asyncio.sleep(self._delay)
            if blob is None:
                return None
            row = MagicMock()
            row.values_b64 = blob
            return row

        async def upsert(self, where, data):
            ident = where["user_id_server_id"]
            key = (ident["user_id"], ident["server_id"])
            self._store.rows[key] = data["update"]["values_b64"]

        async def delete_many(self, where):
            self._store.rows.pop((where["user_id"], where["server_id"]), None)

    class _Tx:
        def __init__(self, store, delay):
            self._store = store
            self._held = None
            self.litellm_mcpuserenvvars = _Table(store, delay=delay)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            if self._held is not None:
                self._held.release()
                self._held = None
            return False

        async def execute_raw(self, query, *args):
            lock_key = args[0]
            lock = self._store.locks.setdefault(lock_key, asyncio.Lock())
            await lock.acquire()
            self._held = lock
            return 1

    class _DB:
        def __init__(self, store, delay):
            self._store = store
            self._delay = delay
            self.litellm_mcpuserenvvars = _Table(store)

        def tx(self):
            return _Tx(self._store, self._delay)

    class _Prisma:
        def __init__(self, delay):
            self.db = _DB(_Store(), delay)

    return _Prisma(read_delay)


@pytest.mark.asyncio
async def test_merge_user_env_vars_does_not_persist_plaintext(env_vars_salt_key):
    """The per-user write path must encrypt values at rest; ``values_b64`` must
    never hold plaintext personal credentials, but must still round-trip."""
    from litellm.proxy._experimental.mcp_server.db import (
        _decode_user_env_vars,
        merge_user_env_vars,
    )

    prisma = _transactional_env_vars_prisma()
    values = {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}
    await merge_user_env_vars(
        prisma, "alice", "srv-1", values, allowed_names=values.keys()
    )

    row = await prisma.db.litellm_mcpuserenvvars.find_unique(
        where={"user_id_server_id": {"user_id": "alice", "server_id": "srv-1"}}
    )
    stored = row.values_b64
    assert "s3cret" not in stored
    assert "alice" not in stored
    assert _decode_user_env_vars(stored) == values


@pytest.mark.asyncio
async def test_get_user_env_vars_round_trip(env_vars_salt_key):
    from unittest.mock import MagicMock

    from litellm.proxy._experimental.mcp_server.db import get_user_env_vars

    payload = {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}
    row = MagicMock()
    row.values_b64 = _encrypted_user_env_blob(payload)
    prisma = _mock_env_vars_prisma(row=row)

    result = await get_user_env_vars(prisma, "alice", "srv-1")
    assert result == payload


@pytest.mark.asyncio
async def test_get_user_env_vars_returns_empty_for_missing_row():
    from litellm.proxy._experimental.mcp_server.db import get_user_env_vars

    prisma = _mock_env_vars_prisma(row=None)
    assert await get_user_env_vars(prisma, "alice", "srv-1") == {}


@pytest.mark.asyncio
async def test_decode_user_env_vars_warns_when_undecryptable(
    env_vars_salt_key, monkeypatch
):
    """A stored blob encrypted under a previous salt key must surface a warning
    (not just a debug line) and decode to ``{}`` so a rotated ``LITELLM_SALT_KEY``
    is diagnosable instead of silently sending the user a misleading "set up your
    credentials" 412 for values they already stored."""
    from unittest.mock import MagicMock

    import litellm.proxy._experimental.mcp_server.db as mcp_db
    from litellm.proxy._experimental.mcp_server.db import _decode_user_env_vars

    blob = _encrypted_user_env_blob({"CORP_PASSWORD": "s3cret"})

    monkeypatch.setenv("LITELLM_SALT_KEY", "a-totally-different-salt-key-0000")
    logger = MagicMock()
    monkeypatch.setattr(mcp_db, "verbose_proxy_logger", logger)

    assert _decode_user_env_vars(blob) == {}
    logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_get_user_env_vars_bulk_distributes_results(env_vars_salt_key):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server.db import get_user_env_vars_bulk

    blob1 = _encrypted_user_env_blob({"A": "1"})
    blob2 = _encrypted_user_env_blob({"B": "2"})

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
async def test_merge_user_env_vars_preserves_existing_and_prunes_disallowed(
    env_vars_salt_key,
):
    """Merging one update keeps the user's other stored values and drops any
    name the admin no longer declares as user-scoped."""
    from litellm.proxy._experimental.mcp_server.db import merge_user_env_vars

    prisma = _transactional_env_vars_prisma()
    await merge_user_env_vars(
        prisma,
        "alice",
        "srv-1",
        {"CORP_USERNAME": "alice", "CORP_PASSWORD": "old", "RETIRED": "x"},
        {"CORP_USERNAME", "CORP_PASSWORD", "RETIRED"},
    )

    merged = await merge_user_env_vars(
        prisma,
        "alice",
        "srv-1",
        {"CORP_PASSWORD": "new"},
        {"CORP_USERNAME", "CORP_PASSWORD"},
    )

    # CORP_USERNAME survives, CORP_PASSWORD updates, RETIRED (no longer declared)
    # is pruned.
    assert merged == {"CORP_USERNAME": "alice", "CORP_PASSWORD": "new"}


@pytest.mark.asyncio
async def test_merge_user_env_vars_serializes_concurrent_writes(env_vars_salt_key):
    """Two simultaneous merges for the same (user, server) must not lose an
    update: the advisory-locked transaction serialises the read-modify-write so
    both distinct values survive."""
    import asyncio

    from litellm.proxy._experimental.mcp_server.db import (
        get_user_env_vars,
        merge_user_env_vars,
    )

    allowed = {"TOKEN_A", "TOKEN_B"}
    prisma = _transactional_env_vars_prisma(read_delay=0.02)

    await asyncio.gather(
        merge_user_env_vars(prisma, "alice", "srv-1", {"TOKEN_A": "a"}, allowed),
        merge_user_env_vars(prisma, "alice", "srv-1", {"TOKEN_B": "b"}, allowed),
    )

    stored = await get_user_env_vars(prisma, "alice", "srv-1")
    assert stored == {"TOKEN_A": "a", "TOKEN_B": "b"}


@pytest.mark.asyncio
async def test_merge_user_env_vars_acquires_lock_without_deserializing_void(
    env_vars_salt_key,
):
    """``pg_advisory_xact_lock`` returns ``void``; running it through ``query_raw``
    makes Prisma try to deserialize that column and raises ``RawQueryError``. The
    lock must be taken via ``execute_raw`` (no result-set deserialization) so the
    merge still completes."""
    from unittest.mock import MagicMock

    from prisma.errors import RawQueryError

    from litellm.proxy._experimental.mcp_server.db import merge_user_env_vars

    class _Tx:
        def __init__(self):
            self.stored = None
            self.litellm_mcpuserenvvars = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def query_raw(self, query, *args):
            raise RawQueryError(
                {
                    "user_facing_error": {
                        "error_code": "P2010",
                        "meta": {
                            "message": "Failed to deserialize column of type 'void'."
                        },
                    }
                }
            )

        async def execute_raw(self, query, *args):
            return 1

        async def find_unique(self, where):
            return None

        async def upsert(self, where, data):
            self.stored = data["create"]["values_b64"]

    tx = _Tx()
    prisma = MagicMock()
    prisma.db.tx = MagicMock(return_value=tx)

    values = {"CORP_TOKEN": "t0ken"}
    merged = await merge_user_env_vars(
        prisma, "alice", "srv-1", values, allowed_names=values.keys()
    )

    assert merged == values
    assert tx.stored is not None


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


@pytest.mark.asyncio
async def test_delete_mcp_server_succeeds_when_orphan_cleanup_fails():
    """The server-row delete is the commit point: a transient failure cleaning
    the FK-less per-user env var rows must not turn a successful delete into a
    caller error, otherwise the caller retries and hits a 404 for a server that
    is already gone."""
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.db import delete_mcp_server

    deleted = object()
    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpservertable.delete = AsyncMock(return_value=deleted)
    prisma.db.litellm_mcpuserenvvars.delete_many = AsyncMock(
        side_effect=Exception("connection pool exhausted")
    )

    result = await delete_mcp_server(prisma, "srv-1")

    assert result is deleted
    prisma.db.litellm_mcpuserenvvars.delete_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_mcp_server_removes_orphaned_user_credentials():
    """Deleting a server must also drop every user's stored BYOK/OAuth credential
    rows for it; there is no FK cascade, so skipping this leaves encrypted secrets
    pointing at a now-missing server."""
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.db import delete_mcp_server

    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpservertable.delete = AsyncMock(return_value=object())

    await delete_mcp_server(prisma, "srv-1")

    prisma.db.litellm_mcpusercredentials.delete_many.assert_awaited_once()
    call = prisma.db.litellm_mcpusercredentials.delete_many.call_args
    assert call.kwargs["where"] == {"server_id": "srv-1"}


@pytest.mark.asyncio
async def test_delete_mcp_server_skips_credential_cleanup_when_server_missing():
    """A no-op delete (server not found) must not touch the credential table."""
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.db import delete_mcp_server

    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpservertable.delete = AsyncMock(return_value=None)

    result = await delete_mcp_server(prisma, "srv-1")

    assert result is None
    prisma.db.litellm_mcpusercredentials.delete_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_mcp_server_credential_cleanup_failure_still_cleans_env_vars():
    """Each per-user table is cleaned independently: a failure dropping credential
    rows must not skip the env var cleanup (or vice versa), and the delete must
    still succeed for the caller."""
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server.db import delete_mcp_server

    deleted = object()
    prisma = _mock_env_vars_prisma()
    prisma.db.litellm_mcpservertable.delete = AsyncMock(return_value=deleted)
    prisma.db.litellm_mcpusercredentials.delete_many = AsyncMock(
        side_effect=Exception("connection pool exhausted")
    )

    result = await delete_mcp_server(prisma, "srv-1")

    assert result is deleted
    prisma.db.litellm_mcpusercredentials.delete_many.assert_awaited_once()
    prisma.db.litellm_mcpuserenvvars.delete_many.assert_awaited_once()


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


def test_prepare_mcp_server_data_skips_unset_env_vars_on_partial_update():
    """On a partial update, env_vars must follow the same exclude_unset filter as
    every other JSON column: if the caller never set env_vars, the field must not
    be written, even when the request object carries a non-None env_vars that was
    never marked as set. Otherwise a partial update could silently overwrite the
    stored values."""
    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._types import MCPEnvVar, UpdateMCPServerRequest

    data = UpdateMCPServerRequest.model_construct(
        _fields_set={"server_id"},
        server_id="srv-1",
        env_vars=[MCPEnvVar(name="DB_PASSWORD", value="s3cr3t", scope="global")],
    )

    prepared = _prepare_mcp_server_data(data, exclude_unset=True)

    assert "env_vars" not in prepared


def test_prepare_mcp_server_data_writes_env_vars_when_set_on_partial_update(
    env_vars_salt_key,
):
    """A partial update that does set env_vars must serialize and encrypt them."""
    import json

    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._types import MCPEnvVar, UpdateMCPServerRequest

    data = UpdateMCPServerRequest(
        server_id="srv-1",
        env_vars=[MCPEnvVar(name="DB_PASSWORD", value="s3cr3t", scope="global")],
    )

    prepared = _prepare_mcp_server_data(data, exclude_unset=True)

    assert "env_vars" in prepared
    entries = json.loads(prepared["env_vars"])
    assert entries[0]["name"] == "DB_PASSWORD"
    assert entries[0]["value"] != "s3cr3t"


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


@pytest.mark.asyncio
async def test_add_server_does_not_double_decrypt_global_env_vars(env_vars_salt_key):
    """The create/fetch endpoints hand ``add_server`` a record whose global env
    var values were already decrypted by the db.py helpers (only ``credentials``
    stays encrypted). Building the registry entry must not decrypt them a second
    time: a second decrypt of an already-plaintext value (e.g. ``postgresql``)
    fails and zeroes it, which would forward the raw ``${NAME}`` placeholder
    upstream instead of the interpolated secret."""
    import json

    from litellm.proxy._experimental.mcp_server.db import (
        _prepare_mcp_server_data,
        decrypt_global_env_var_values,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.proxy._types import LiteLLM_MCPServerTable, MCPEnvVar

    req = _global_env_var_server_request(
        [MCPEnvVar(name="DB_PASSWORD", value="s3cr3t-p@ss", scope="global")]
    )
    env_vars = json.loads(_prepare_mcp_server_data(req)["env_vars"])
    # Mirror what create_mcp_server / get_mcp_server return to add_server.
    decrypt_global_env_var_values(env_vars)
    assert env_vars[0]["value"] == "s3cr3t-p@ss"

    table = LiteLLM_MCPServerTable(
        server_id="srv-add",
        alias="echo",
        url="https://upstream.example.com/mcp",
        transport="http",
        auth_type="none",
        static_headers={"X-Db": "${DB_PASSWORD}"},
        env_vars=env_vars,
        approval_status="active",
    )

    manager = MCPServerManager()
    await manager.add_server(table)

    server = manager.registry["srv-add"]
    headers = await manager._resolve_static_headers_with_env_vars(server, None)
    assert headers == {"X-Db": "s3cr3t-p@ss"}


@pytest.mark.asyncio
async def test_create_mcp_server_decrypts_env_vars_when_prisma_returns_json_string(
    env_vars_salt_key,
):
    """Regression for the reload-reuse path: Prisma can hand back ``env_vars`` on
    a write as the raw JSON string that was persisted, not a parsed list. The
    create/update wrappers must still decrypt globals on the returned row, else
    ``add_server`` (which trusts the caller) seeds the registry with ciphertext
    and the subsequent ``reload_servers_from_database`` reuses that broken entry
    (timestamps match), so headers forward ciphertext upstream."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server.db import (
        _prepare_mcp_server_data,
        create_mcp_server,
        update_mcp_server,
    )
    from litellm.proxy._types import (
        MCPEnvVar,
        NewMCPServerRequest,
        UpdateMCPServerRequest,
    )

    req = _global_env_var_server_request(
        [MCPEnvVar(name="DB_PASSWORD", value="s3cr3t-p@ss", scope="global")]
    )
    encrypted_env_vars_str = _prepare_mcp_server_data(req)["env_vars"]
    assert "s3cr3t-p@ss" not in encrypted_env_vars_str

    def _prisma_row_with_json_string_env_vars():
        row = MagicMock()
        row.env_vars = encrypted_env_vars_str
        return row

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_mcpservertable.create = AsyncMock(
        return_value=_prisma_row_with_json_string_env_vars()
    )

    created = await create_mcp_server(
        mock_prisma,
        NewMCPServerRequest(
            server_id="srv-create",
            url="https://upstream.example.com/mcp",
            transport="http",
        ),
        touched_by="test-user",
    )
    assert isinstance(created.env_vars, list)
    assert created.env_vars[0]["value"] == "s3cr3t-p@ss"

    mock_prisma_upd = MagicMock()
    mock_prisma_upd.db.litellm_mcpservertable.update = AsyncMock(
        return_value=_prisma_row_with_json_string_env_vars()
    )
    updated = await update_mcp_server(
        mock_prisma_upd,
        UpdateMCPServerRequest(server_id="srv-update"),
        touched_by="test-user",
    )
    assert isinstance(updated.env_vars, list)
    assert updated.env_vars[0]["value"] == "s3cr3t-p@ss"


def test_reencrypt_global_env_var_values_handles_json_string(env_vars_salt_key):
    """``rotate_mcp_server_credentials_master_key`` reads ``mcp_server.env_vars``
    straight off the Prisma row, which can be a JSON string. The re-encrypt
    helper must parse it instead of failing on ``dict(v)`` over a string."""
    import json

    from litellm.proxy._experimental.mcp_server.db import (
        _prepare_mcp_server_data,
        _reencrypt_global_env_var_values,
    )
    from litellm.proxy._types import MCPEnvVar

    req = _global_env_var_server_request(
        [MCPEnvVar(name="DB_PASSWORD", value="s3cr3t-p@ss", scope="global")]
    )
    encrypted_env_vars_str = _prepare_mcp_server_data(req)["env_vars"]
    original_ciphertext = json.loads(encrypted_env_vars_str)[0]["value"]

    rebuilt = _reencrypt_global_env_var_values(
        encrypted_env_vars_str, new_encryption_key="rotated-master-key-0000"
    )

    assert rebuilt is not None
    assert rebuilt[0]["name"] == "DB_PASSWORD"
    assert rebuilt[0]["value"] != original_ciphertext
    assert rebuilt[0]["value"] != "s3cr3t-p@ss"


@pytest.mark.asyncio
async def test_rotate_mcp_user_env_vars_logs_rotated_and_skipped_counts(
    env_vars_salt_key, monkeypatch
):
    """Master-key rotation is a rare, high-stakes batch op, so it emits one
    summary line. The counts must track real work: a decryptable row is
    re-encrypted and counted as rotated, while a row that no longer decrypts is
    left untouched and counted as skipped."""
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy._experimental.mcp_server.db as mcp_db
    from litellm.proxy._experimental.mcp_server.db import (
        rotate_mcp_user_env_vars_master_key,
    )
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    def _row(user_id, server_id, blob):
        row = MagicMock()
        row.user_id = user_id
        row.server_id = server_id
        row.values_b64 = blob
        return row

    import json

    # Encrypted under an unrelated key, so it won't decrypt under the active salt
    # key and must be skipped rather than re-encrypted.
    undecryptable = encrypt_value_helper(
        json.dumps({"X": "y"}), new_encryption_key="unrelated-key-9999"
    )
    good_one = _row("alice", "srv-1", _encrypted_user_env_blob({"GH_TOKEN": "tok-1"}))
    good_two = _row("bob", "srv-2", _encrypted_user_env_blob({"GH_TOKEN": "tok-2"}))
    bad = _row("carol", "srv-3", undecryptable)

    prisma = MagicMock()
    prisma.db.litellm_mcpuserenvvars.find_many = AsyncMock(
        return_value=[good_one, good_two, bad]
    )
    prisma.db.litellm_mcpuserenvvars.update = AsyncMock()

    logger = MagicMock()
    monkeypatch.setattr(mcp_db, "verbose_proxy_logger", logger)

    await rotate_mcp_user_env_vars_master_key(prisma, new_master_key="rotated-key-0000")

    update = prisma.db.litellm_mcpuserenvvars.update
    assert update.await_count == 2
    updated_servers = {
        call.kwargs["where"]["user_id_server_id"]["server_id"]
        for call in update.call_args_list
    }
    assert updated_servers == {"srv-1", "srv-2"}  # srv-3 was skipped, not rotated
    for call in update.call_args_list:
        assert call.kwargs["data"]["values_b64"] not in (
            good_one.values_b64,
            good_two.values_b64,
        )

    logger.info.assert_called_once()
    info_args = logger.info.call_args.args
    assert info_args[1] == 2  # rotated
    assert info_args[2] == 1  # skipped


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
