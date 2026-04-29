"""
Tests for the short-ID MCP tool prefix (LITELLM_USE_SHORT_MCP_TOOL_PREFIX).

The short-prefix mode swaps the historical alias/server_name prefix on
tool names for a deterministic three-character base62 ID derived from the
server's ``server_id``. This keeps tool names well below the 60-char
upper bound enforced by some model APIs while remaining stable across
processes/restarts and tolerant of mixed-version clients.
"""

from typing import List

import pytest
from mcp.types import Tool as MCPTool

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._experimental.mcp_server.utils import (
    SHORT_MCP_TOOL_PREFIX_LENGTH,
    add_server_prefix_to_name,
    compute_short_server_prefix,
    get_server_prefix,
    is_short_mcp_tool_prefix_enabled,
    iter_known_server_prefixes,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _make_server(
    *,
    server_id: str = "abcdef-1234",
    server_name: str = "github_onprem",
    alias: str = "github_onprem",
) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=alias or server_name,
        alias=alias,
        server_name=server_name,
        transport="http",
    )


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", raising=False)
    yield


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestShortPrefixHelpers:
    def test_short_prefix_is_three_base62_chars(self):
        prefix = compute_short_server_prefix("any-server-id")
        assert len(prefix) == SHORT_MCP_TOOL_PREFIX_LENGTH
        assert prefix.isalnum() and prefix.isascii()

    def test_short_prefix_first_char_is_alphabetic(self):
        """The first char must be [A-Za-z] so the prefix is a valid identifier
        on every model API (some providers historically required the first
        character of a function name to be alphabetic)."""
        # Sweep many server_ids and rehash attempts to give us coverage of
        # every position the high-order bits can land on.
        for i in range(200):
            for attempt in range(4):
                prefix = compute_short_server_prefix(f"server-{i}", attempt=attempt)
                assert prefix[0].isalpha(), (
                    f"prefix {prefix!r} for server-{i} (attempt={attempt}) "
                    f"starts with a non-alphabetic character"
                )

    def test_short_prefix_is_deterministic(self):
        assert compute_short_server_prefix("abc") == compute_short_server_prefix("abc")
        assert compute_short_server_prefix("abc") != compute_short_server_prefix("abd")

    def test_short_prefix_requires_server_id(self):
        with pytest.raises(ValueError):
            compute_short_server_prefix("")

    def test_flag_defaults_to_false(self):
        assert is_short_mcp_tool_prefix_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On"])
    def test_flag_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", value)
        assert is_short_mcp_tool_prefix_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_flag_falsey_values(self, monkeypatch, value):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", value)
        assert is_short_mcp_tool_prefix_enabled() is False


# ---------------------------------------------------------------------------
# get_server_prefix behaviour
# ---------------------------------------------------------------------------


class TestGetServerPrefix:
    def test_default_mode_uses_alias(self):
        server = _make_server(alias="github_onprem", server_name="github_onprem")
        assert get_server_prefix(server) == "github_onprem"

    def test_short_mode_uses_short_id(self, monkeypatch):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        server = _make_server(server_id="abcdef-1234")
        prefix = get_server_prefix(server)
        assert prefix == compute_short_server_prefix("abcdef-1234")
        assert len(prefix) == SHORT_MCP_TOOL_PREFIX_LENGTH

    def test_short_mode_falls_back_when_no_server_id(self, monkeypatch):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")

        class _Bare:
            alias = "fallback_alias"
            server_name = None
            server_id = None

        assert get_server_prefix(_Bare()) == "fallback_alias"


# ---------------------------------------------------------------------------
# iter_known_server_prefixes — covers reverse-lookup tolerance
# ---------------------------------------------------------------------------


class TestIterKnownServerPrefixes:
    def test_default_mode_includes_short_id_too(self):
        server = _make_server()
        prefixes = list(iter_known_server_prefixes(server))
        # Contains the live prefix and every known form so that mixed-mode
        # clients can be resolved.
        assert "github_onprem" in prefixes
        assert compute_short_server_prefix(server.server_id) in prefixes

    def test_short_mode_still_yields_long_forms(self, monkeypatch):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        server = _make_server()
        prefixes = list(iter_known_server_prefixes(server))
        assert "github_onprem" in prefixes
        assert compute_short_server_prefix(server.server_id) in prefixes


# ---------------------------------------------------------------------------
# Manager-level behaviour: list + reverse-lookup
# ---------------------------------------------------------------------------


def _stub_tools() -> List[MCPTool]:
    return [
        MCPTool(name="get_repo", description="", inputSchema={"type": "object"}),
        MCPTool(name="list_issues", description="", inputSchema={"type": "object"}),
    ]


class TestManagerShortPrefix:
    def test_list_tools_uses_short_prefix_when_flag_on(self, monkeypatch):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        manager = MCPServerManager()
        server = _make_server()

        out = manager._create_prefixed_tools(_stub_tools(), server)

        short = compute_short_server_prefix(server.server_id)
        assert {t.name for t in out} == {f"{short}-get_repo", f"{short}-list_issues"}

    def test_call_tool_lookup_resolves_short_prefix(self, monkeypatch):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        manager = MCPServerManager()
        server = _make_server()
        manager.registry[server.server_id] = server
        manager._create_prefixed_tools(_stub_tools(), server)

        short = compute_short_server_prefix(server.server_id)
        resolved = manager._get_mcp_server_from_tool_name(f"{short}-get_repo")
        assert resolved is server

    def test_call_tool_lookup_resolves_long_prefix_in_short_mode(self, monkeypatch):
        """Old clients that cached the long-prefix name must still route."""
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        manager = MCPServerManager()
        server = _make_server()
        manager.registry[server.server_id] = server
        manager._create_prefixed_tools(_stub_tools(), server)

        resolved = manager._get_mcp_server_from_tool_name("github_onprem-get_repo")
        assert resolved is server

    def test_default_mode_unchanged(self):
        manager = MCPServerManager()
        server = _make_server()

        out = manager._create_prefixed_tools(_stub_tools(), server)

        assert {t.name for t in out} == {
            "github_onprem-get_repo",
            "github_onprem-list_issues",
        }
        assert (
            manager._get_mcp_server_from_tool_name("github_onprem-get_repo") is None
        )  # registry empty
        manager.registry[server.server_id] = server
        assert (
            manager._get_mcp_server_from_tool_name("github_onprem-get_repo") is server
        )

    def test_total_tool_name_length_short_enough(self, monkeypatch):
        """The short prefix keeps tool names under the 60-char limit even
        when the upstream tool name is itself reasonably long."""
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        long_server_name = "a" * 50
        server = _make_server(
            server_id="server-id-1",
            server_name=long_server_name,
            alias=long_server_name,
        )
        prefix = get_server_prefix(server)
        full = add_server_prefix_to_name("get_repo", prefix)
        assert len(full) < 60


# ---------------------------------------------------------------------------
# Collision-resolution at registration time
# ---------------------------------------------------------------------------


class TestShortPrefixCollisionResolution:
    """``_assign_unique_short_prefix`` must rehash on collision.

    The dedup path is exercised by forcing two distinct ``server_id``
    values to both hash to the same natural prefix via a monkeypatched
    ``compute_short_server_prefix``.
    """

    def test_no_op_when_flag_off(self):
        manager = MCPServerManager()
        server = _make_server(server_id="abc")
        manager._assign_unique_short_prefix(server)
        assert server.short_prefix is None

    def test_assigns_natural_hash_when_no_collision(self, monkeypatch):
        from litellm.proxy._experimental.mcp_server import utils as mcp_utils

        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        manager = MCPServerManager()
        server = _make_server(server_id="abc")
        manager._assign_unique_short_prefix(server)

        assert server.short_prefix == mcp_utils.compute_short_server_prefix("abc")

    def test_rehashes_when_natural_hash_collides(self, monkeypatch):
        """Two server_ids that natural-hash to the same prefix get
        deterministic, distinct short prefixes."""
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")

        # Force every attempt=0 hash to "AAA" and attempt=1 to "AAB".
        # That way the second server registered must rehash to "AAB".
        from litellm.proxy._experimental.mcp_server import utils as mcp_utils

        def _fake_hash(server_id: str, attempt: int = 0) -> str:
            return "AAA" if attempt == 0 else f"AA{chr(ord('A') + attempt)}"

        monkeypatch.setattr(mcp_utils, "compute_short_server_prefix", _fake_hash)
        # Also patch the symbol that the manager imported at module load.
        from litellm.proxy._experimental.mcp_server import (
            mcp_server_manager as mgr_module,
        )

        monkeypatch.setattr(mgr_module, "compute_short_server_prefix", _fake_hash)

        manager = MCPServerManager()
        first = _make_server(server_id="server-1", alias="srv1")
        second = _make_server(server_id="server-2", alias="srv2")

        # Pretend both are already in the registry so dedup sees both.
        manager.registry[first.server_id] = first
        manager._assign_unique_short_prefix(first)
        manager.registry[second.server_id] = second
        manager._assign_unique_short_prefix(second)

        assert first.short_prefix == "AAA"
        assert second.short_prefix == "AAB"
        assert first.short_prefix != second.short_prefix

    def test_cached_prefix_is_reused(self, monkeypatch):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        manager = MCPServerManager()
        server = _make_server(server_id="abc")
        server.short_prefix = "ZZZ"  # pretend a previous registration set this

        manager._assign_unique_short_prefix(server)

        assert server.short_prefix == "ZZZ"

    def test_get_server_prefix_prefers_cached(self, monkeypatch):
        monkeypatch.setenv("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "true")
        server = _make_server(server_id="abc")
        server.short_prefix = "Q9q"

        assert get_server_prefix(server) == "Q9q"
