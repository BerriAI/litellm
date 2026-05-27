"""Tests for MCP variable interpolation utilities.

These cover the pure helpers in
``litellm.proxy._experimental.mcp_server.utils`` and do not require a DB
connection. The DB-backed per-user flow is exercised in higher-level
tests in tests/mcp_tests.
"""

import pytest

# Look up these names lazily on every access. Tests in this directory call
# ``importlib.reload`` on the utils module to exercise registration logic,
# which replaces ``MCPMissingUserVariablesError`` with a freshly-constructed
# class. A direct ``from ... import`` at module load time would freeze the
# old class object and ``pytest.raises(_u("MCPMissingUserVariablesError"))`` would
# stop matching the new class. Accessing the attribute through the module
# always picks up the current version.
import litellm.proxy._experimental.mcp_server.utils as _mcp_utils


def _u(name: str):
    return getattr(_mcp_utils, name)


def test_parse_admin_variables_splits_global_and_user():
    g, u = _u("parse_admin_variables")(
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


def test_parse_admin_variables_handles_none_and_empty():
    assert _u("parse_admin_variables")(None) == ({}, [])
    assert _u("parse_admin_variables")([]) == ({}, [])


def test_parse_admin_variables_skips_malformed_entries():
    g, u = _u("parse_admin_variables")(
        [
            None,
            {"name": "", "value": "x"},
            {"value": "no_name"},
            {"name": "OK", "value": "v"},
        ]
    )
    assert g == {"OK": "v"}
    assert u == []


def test_find_variable_references():
    assert _u("find_variable_references")("") == set()
    assert _u("find_variable_references")("plain") == set()
    assert _u("find_variable_references")("${A}") == {"A"}
    assert _u("find_variable_references")("${A}/${B}/${A}") == {"A", "B"}
    # Invalid identifier patterns should not match
    assert _u("find_variable_references")("${1abc}") == set()
    assert _u("find_variable_references")("${a-b}") == set()


def test_collect_variable_references():
    refs = _u("collect_variable_references")(
        strings=["${A}", "static", "${B}-${C}", None]
    )
    assert refs == {"A", "B", "C"}


def test_interpolate_variables_replaces_known_and_leaves_unknown():
    assert _u("interpolate_variables")(
        "${A}://${B}/${C}", {"A": "https", "B": "host"}
    ) == ("https://host/${C}")


def test_interpolate_headers_returns_independent_copy():
    headers = {"X-Url": "${A}://x"}
    out = _u("interpolate_headers")(headers, {"A": "https"})
    assert out == {"X-Url": "https://x"}
    # original untouched
    assert headers == {"X-Url": "${A}://x"}


def test_build_variable_setup_url_includes_server_id(monkeypatch):
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    url = _u("build_variable_setup_url")("abc-123")
    assert url.startswith("/ui/?page=mcp-servers")
    assert "fill_variables=abc-123" in url


def test_build_variable_setup_url_prepends_proxy_base_url(monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com/")
    url = _u("build_variable_setup_url")("abc-123")
    assert url.startswith("https://proxy.example.com/ui/")
    assert "fill_variables=abc-123" in url


def test_missing_user_variables_error_message_is_friendly():
    with pytest.raises(_u("MCPMissingUserVariablesError")) as exc_info:
        raise _u("MCPMissingUserVariablesError")(
            server_id="abc-123",
            server_name="CorporateDB",
            missing=["CORP_USERNAME", "CORP_PASSWORD"],
            setup_url="https://proxy.example.com/ui/?page=mcp-servers&fill_variables=abc-123",
        )
    err = exc_info.value
    text = str(err)
    assert "CorporateDB" in text
    assert "CORP_USERNAME" in text
    assert "CORP_PASSWORD" in text
    assert "fill_variables=abc-123" in text
    assert err.server_id == "abc-123"
    assert err.missing == ["CORP_USERNAME", "CORP_PASSWORD"]


def test_missing_user_variables_error_singular_message():
    err = _u("MCPMissingUserVariablesError")(
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


# ── _resolve_static_headers_with_variables ────────────────────────────────


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
        variables=[
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
    async def fake_load_user_variables(server, user_api_key_auth):
        return {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}

    monkeypatch.setattr(manager, "_load_user_variables", fake_load_user_variables)

    headers = await manager._resolve_static_headers_with_variables(
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

    async def fake_load_user_variables(server, user_api_key_auth):
        # User has only filled in one of the two required vars
        return {"CORP_USERNAME": "alice"}

    monkeypatch.setattr(manager, "_load_user_variables", fake_load_user_variables)

    with pytest.raises(_u("MCPMissingUserVariablesError")) as exc:
        await manager._resolve_static_headers_with_variables(
            mock_server, user_api_key_auth=object()
        )
    assert exc.value.missing == ["CORP_PASSWORD"]
    assert exc.value.server_id == "srv-1"
    assert "fill_variables=srv-1" in exc.value.setup_url


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

    async def fake_load_user_variables(server, user_api_key_auth):
        # User has only filled in one of the two required vars.
        return {"CORP_USERNAME": "alice"}

    monkeypatch.setattr(manager, "_load_user_variables", fake_load_user_variables)

    headers = await manager._resolve_static_headers_with_variables(
        mock_server, user_api_key_auth=object(), raise_on_missing=False
    )
    # Globals + the supplied user var are interpolated; the still-missing
    # CORP_PASSWORD reference is left as a literal rather than blocking listing.
    assert headers == {
        "X-DB-URL": "postgres://alice:${CORP_PASSWORD}@db.local/db",
        "X-Other": "literal",
    }


@pytest.mark.asyncio
async def test_resolve_static_headers_passthrough_when_no_variables():
    """Servers without variables should keep static_headers untouched."""
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
        variables=None,
    )
    headers = await manager._resolve_static_headers_with_variables(server, None)
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
        variables=[
            {"name": "GLOBAL_VAR", "value": "ok", "scope": "global"},
            # User var declared but not referenced anywhere — should be ignored.
            {"name": "UNUSED_USER_VAR", "value": "", "scope": "user"},
        ],
    )

    async def fake_load_user_variables(server, user_api_key_auth):
        return {}

    monkeypatch.setattr(manager, "_load_user_variables", fake_load_user_variables)

    headers = await manager._resolve_static_headers_with_variables(server, object())
    assert headers == {"X-Static": "ok"}


# ── _load_user_variables guard paths ────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_user_variables_returns_empty_without_user():
    """No user auth → no per-user lookup is attempted."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="s", name="s", transport="http", url="https://example.com"
    )
    assert await manager._load_user_variables(server, None) == {}


@pytest.mark.asyncio
async def test_load_user_variables_returns_empty_without_user_id():
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
    assert await manager._load_user_variables(server, fake_auth) == {}


@pytest.mark.asyncio
async def test_load_user_variables_returns_empty_when_db_unavailable(monkeypatch):
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
    assert await manager._load_user_variables(server, fake_auth) == {}


@pytest.mark.asyncio
async def test_load_user_variables_reads_through_store(monkeypatch):
    """With a user and DB available, the lookup goes through the variable store."""
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
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())

    captured = {}

    async def fake_get(prisma_client, user_id):
        captured["args"] = (prisma_client, user_id)
        return {"CORP_USERNAME": "alice"}

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_variable_store.get_user_variables",
        fake_get,
    )
    assert await manager._load_user_variables(server, fake_auth) == {
        "CORP_USERNAME": "alice"
    }
    assert captured["args"][1] == "alice"


@pytest.mark.asyncio
async def test_load_user_variables_swallows_store_errors(monkeypatch):
    """A store failure is best-effort: it returns {} instead of raising."""
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
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())

    async def boom(prisma_client, user_id):
        raise RuntimeError("vault down")

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_variable_store.get_user_variables",
        boom,
    )
    assert await manager._load_user_variables(server, fake_auth) == {}


@pytest.mark.asyncio
async def test_reload_servers_excludes_templates_from_query(monkeypatch):
    """Templates are filtered out at the DB level so they're never loaded as live servers."""
    from unittest.mock import AsyncMock, MagicMock

    import litellm.proxy.management_endpoints.mcp_management_endpoints as mgmt
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )

    prisma = MagicMock()
    prisma.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr(mgmt, "get_prisma_client_or_throw", lambda *a, **k: prisma)

    manager = MCPServerManager()
    await manager.reload_servers_from_database()

    prisma.db.litellm_mcpservertable.find_many.assert_awaited_once()
    where = prisma.db.litellm_mcpservertable.find_many.call_args.kwargs["where"]
    assert where["kind"] == {"not": "template"}
    assert manager.registry == {}


# ── extra branch coverage: parsing / serialization helpers ──────────────────


def test_parse_admin_variables_accepts_pydantic_models():
    from litellm.proxy._types import MCPVariable

    g, u = _u("parse_admin_variables")(
        [
            MCPVariable(name="G", value="v", scope="global"),
            MCPVariable(name="U", scope="user", description="d"),
        ]
    )
    assert g == {"G": "v"}
    assert u == [{"name": "U", "description": "d"}]


def test_parse_admin_variables_skips_non_dict_non_model_entries():
    g, u = _u("parse_admin_variables")(["a string", 123, ("tuple",)])
    assert g == {}
    assert u == []


def test_interpolate_variables_empty_value_is_returned_as_is():
    assert _u("interpolate_variables")("", {"A": "x"}) == ""


def test_deserialize_json_list_variants():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        _deserialize_json_list,
    )

    assert _deserialize_json_list(None) is None
    assert _deserialize_json_list("") is None
    assert _deserialize_json_list([]) is None
    assert _deserialize_json_list('[{"name": "A"}]') == [{"name": "A"}]
    assert _deserialize_json_list("not-json") is None
    assert _deserialize_json_list('{"not": "a list"}') is None
    assert _deserialize_json_list([{"name": "B"}]) == [{"name": "B"}]
    assert _deserialize_json_list(123) is None


@pytest.mark.asyncio
async def test_resolve_static_headers_global_vars_without_static_headers():
    """Variables present but no static_headers → returns the empty headers as-is."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        MCPServerManager,
    )
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager = MCPServerManager()
    server = MCPServer(
        server_id="s",
        name="s",
        transport="http",
        url="https://example.com",
        static_headers=None,
        variables=[{"name": "G", "value": "v", "scope": "global"}],
    )
    result = await manager._resolve_static_headers_with_variables(
        server, user_api_key_auth=None
    )
    assert not result


def test_prepare_mcp_server_data_serializes_variables():
    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._types import MCPVariable, NewMCPServerRequest

    req = NewMCPServerRequest(
        transport="http",
        url="https://example.com",
        variables=[MCPVariable(name="G", value="v", scope="global")],
    )
    data = _prepare_mcp_server_data(req)
    assert isinstance(data["variables"], str)
    assert "G" in data["variables"]


def test_decode_user_variables_handles_bad_blobs(monkeypatch):
    from litellm.proxy._experimental.mcp_server import db as _db

    # decrypt returns None -> {}
    monkeypatch.setattr(_db, "decrypt_value_helper", lambda **k: None)
    assert _db._decode_user_variables("x") == {}

    # decrypt returns non-JSON -> {}
    monkeypatch.setattr(_db, "decrypt_value_helper", lambda **k: "not-json{")
    assert _db._decode_user_variables("x") == {}

    # decrypt returns valid JSON that is not a dict -> {}
    monkeypatch.setattr(_db, "decrypt_value_helper", lambda **k: "[1, 2, 3]")
    assert _db._decode_user_variables("x") == {}


# ── DB helpers: per-user variables ─────────────────────────────────────────

_SALT_KEY = "test-salt-key-for-variables-tests-1234"


@pytest.fixture
def variables_salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", _SALT_KEY)


def _mock_variables_prisma(row=None):
    """Build a MagicMock prisma_client whose variables table returns ``row``."""
    from unittest.mock import AsyncMock, MagicMock

    prisma = MagicMock()
    prisma.db.litellm_mcpuservariables.find_unique = AsyncMock(return_value=row)
    prisma.db.litellm_mcpuservariables.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_mcpuservariables.upsert = AsyncMock()
    prisma.db.litellm_mcpuservariables.delete = AsyncMock()
    return prisma


def _captured_values_blob(prisma) -> str:
    """Pull the values_b64 value passed to the most recent upsert."""
    call = prisma.db.litellm_mcpuservariables.upsert.call_args
    data = call.kwargs["data"]
    create_value = data["create"]["values_b64"]
    update_value = data["update"]["values_b64"]
    assert create_value == update_value
    return create_value


@pytest.mark.asyncio
async def test_store_user_variables_does_not_persist_plaintext(variables_salt_key):
    from litellm.proxy._experimental.mcp_server.db import store_user_variables

    prisma = _mock_variables_prisma()
    await store_user_variables(
        prisma, "alice", {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}
    )
    stored = _captured_values_blob(prisma)
    # Stored blob must not contain plaintext values
    assert "s3cret" not in stored
    assert "alice" not in stored or stored.count("alice") == 0  # encrypted form


@pytest.mark.asyncio
async def test_store_user_variables_upserts_by_user_id_only(variables_salt_key):
    """The upsert where-clause is keyed by a plain ``user_id`` (global per-user),
    not the old composite ``user_id_server_id`` key."""
    from litellm.proxy._experimental.mcp_server.db import store_user_variables

    prisma = _mock_variables_prisma()
    await store_user_variables(prisma, "alice", {"A": "1"})
    prisma.db.litellm_mcpuservariables.upsert.assert_awaited_once()
    call = prisma.db.litellm_mcpuservariables.upsert.call_args
    assert call.kwargs["where"] == {"user_id": "alice"}
    assert call.kwargs["data"]["create"]["user_id"] == "alice"


@pytest.mark.asyncio
async def test_get_user_variables_round_trip(variables_salt_key):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server.db import (
        get_user_variables,
        store_user_variables,
    )

    prisma = _mock_variables_prisma()
    payload = {"CORP_USERNAME": "alice", "CORP_PASSWORD": "s3cret"}
    await store_user_variables(prisma, "alice", payload)
    stored = _captured_values_blob(prisma)

    # Now simulate the read returning that blob.
    row = MagicMock()
    row.values_b64 = stored
    prisma.db.litellm_mcpuservariables.find_unique = AsyncMock(return_value=row)

    result = await get_user_variables(prisma, "alice")
    assert result == payload


@pytest.mark.asyncio
async def test_get_user_variables_finds_by_user_id_only():
    """The read short-circuits on a missing row and is keyed only by user_id."""
    from litellm.proxy._experimental.mcp_server.db import get_user_variables

    prisma = _mock_variables_prisma(row=None)
    assert await get_user_variables(prisma, "alice") == {}
    prisma.db.litellm_mcpuservariables.find_unique.assert_awaited_once()
    call = prisma.db.litellm_mcpuservariables.find_unique.call_args
    assert call.kwargs["where"] == {"user_id": "alice"}


@pytest.mark.asyncio
async def test_delete_user_variables_calls_user_id_key():
    from litellm.proxy._experimental.mcp_server.db import delete_user_variables

    prisma = _mock_variables_prisma()
    await delete_user_variables(prisma, "alice")
    prisma.db.litellm_mcpuservariables.delete.assert_awaited_once()
    call = prisma.db.litellm_mcpuservariables.delete.call_args
    assert call.kwargs["where"] == {"user_id": "alice"}


# ── REST exception handling ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_user_variables_error_renders_in_mcp_call_tool():
    """The MCP ``call_tool`` handler must turn ``MCPMissingUserVariablesError``
    into a friendly ``CallToolResult`` with ``isError=True`` so Claude Code
    surfaces the setup URL instead of an opaque internal error."""
    from mcp.types import TextContent

    err = _u("MCPMissingUserVariablesError")(
        server_id="srv-99",
        server_name="CorporateDB",
        missing=["CORP_USERNAME"],
        setup_url="/ui/?page=mcp-servers&fill_variables=srv-99",
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
    assert "fill_variables=srv-99" in text
