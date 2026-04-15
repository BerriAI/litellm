"""
Tests for is_tool_name_prefixed with known_server_prefixes parameter.

Verifies fix for https://github.com/BerriAI/litellm/issues/25081
"""

import pytest

from litellm.proxy._experimental.mcp_server.utils import is_tool_name_prefixed


# ---------------------------------------------------------------------------
# Legacy behaviour (no known_server_prefixes passed)
# ---------------------------------------------------------------------------


class TestLegacyBehaviour:
    """Without known_server_prefixes the function falls back to heuristic."""

    def test_plain_name_returns_false(self):
        assert is_tool_name_prefixed("get_weather") is False

    def test_hyphenated_name_returns_true_legacy(self):
        """Legacy heuristic: any hyphen → True (the bug this issue reports)."""
        assert is_tool_name_prefixed("text-to-speech") is True

    def test_prefixed_name_returns_true_legacy(self):
        assert is_tool_name_prefixed("myserver-get_weather") is True


# ---------------------------------------------------------------------------
# New behaviour (known_server_prefixes supplied)
# ---------------------------------------------------------------------------


class TestWithKnownPrefixes:
    """When known_server_prefixes is supplied, only real prefixes match."""

    PREFIXES = {"myserver", "weather_api", "code_tools"}

    def test_known_prefix_returns_true(self):
        assert (
            is_tool_name_prefixed(
                "myserver-get_weather", known_server_prefixes=self.PREFIXES
            )
            is True
        )

    def test_hyphenated_non_mcp_tool_returns_false(self):
        """This is the core fix: 'text-to-speech' is NOT an MCP-prefixed tool."""
        assert (
            is_tool_name_prefixed(
                "text-to-speech", known_server_prefixes=self.PREFIXES
            )
            is False
        )

    def test_code_review_not_misclassified(self):
        assert (
            is_tool_name_prefixed(
                "code-review", known_server_prefixes=self.PREFIXES
            )
            is False
        )

    def test_no_separator_returns_false(self):
        assert (
            is_tool_name_prefixed(
                "simple_tool", known_server_prefixes=self.PREFIXES
            )
            is False
        )

    def test_empty_prefixes_set_rejects_all(self):
        """With an empty registry, nothing can be prefixed."""
        assert (
            is_tool_name_prefixed("myserver-get_weather", known_server_prefixes=set())
            is False
        )

    def test_prefix_normalisation(self):
        """Server names with spaces are normalised to underscores."""
        prefixes = {"my_server"}
        # add_server_prefix_to_name normalises spaces → underscores
        assert (
            is_tool_name_prefixed(
                "my_server-list_files", known_server_prefixes=prefixes
            )
            is True
        )
