"""
Unit tests for the parse_search_tools print_initialization_message fix.

Verifies that parse_search_tools does not print the "Proxy initialized with
Search Tools" banner when print_initialization_message=False, which is the
value passed by _update_llm_router() on every periodic router refresh.

Regression: before the fix, print() was called unconditionally, causing the
banner to appear on every request when Brave Search (or any search tool) was
configured.

Fixes: https://github.com/BerriAI/litellm/issues/27645
"""
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))


def _make_proxy_config(search_tool_name="brave_search", search_provider="Brave"):
    """
    Return a minimal config dict that parse_search_tools recognises as having
    one search tool configured.
    """
    return {
        "general_settings": {
            "search_tools": [
                {
                    "search_tool_name": search_tool_name,
                    "litellm_params": {
                        "search_provider": search_provider,
                        "api_key": "fake-key",
                    },
                }
            ]
        }
    }


class TestParseSearchToolsPrintInitializationMessage:
    """parse_search_tools must only print when print_initialization_message=True."""

    def _get_proxy_config_instance(self):
        from litellm.proxy.proxy_server import ProxyConfig
        return ProxyConfig()

    def test_no_print_when_flag_is_false(self, capsys):
        """
        _update_llm_router() passes print_initialization_message=False.
        No output should reach stdout.
        """
        pc = self._get_proxy_config_instance()
        config = _make_proxy_config()
        pc.parse_search_tools(config, print_initialization_message=False)

        captured = capsys.readouterr()
        assert "Search Tools" not in captured.out, (
            f"Banner was printed despite print_initialization_message=False: "
            f"{captured.out!r}"
        )
        assert "brave_search" not in captured.out.lower(), (
            f"Tool name was printed despite print_initialization_message=False: "
            f"{captured.out!r}"
        )

    def test_prints_when_flag_is_true(self, capsys):
        """
        load_config() uses the default (True). The startup banner must appear.
        """
        pc = self._get_proxy_config_instance()
        config = _make_proxy_config()
        pc.parse_search_tools(config, print_initialization_message=True)

        captured = capsys.readouterr()
        assert "Search Tools" in captured.out, (
            f"Expected banner not found in stdout: {captured.out!r}"
        )

    def test_default_is_true(self, capsys):
        """
        The default value for print_initialization_message must be True so
        load_config() callers (which don't pass the argument) still see the
        startup banner.
        """
        pc = self._get_proxy_config_instance()
        config = _make_proxy_config()
        pc.parse_search_tools(config)  # no explicit flag

        captured = capsys.readouterr()
        assert "Search Tools" in captured.out, (
            "Default behaviour (no flag) must print the banner — "
            f"stdout was: {captured.out!r}"
        )

    def test_returns_list_regardless_of_flag(self):
        """
        The return value (list of SearchToolTypedDicts) must be the same
        whether we print or not.
        """
        pc = self._get_proxy_config_instance()
        config = _make_proxy_config("exa_search", "Exa")
        result_silent = pc.parse_search_tools(config, print_initialization_message=False)
        result_verbose = pc.parse_search_tools(config, print_initialization_message=True)

        assert result_silent == result_verbose, (
            "parse_search_tools returned different results based on "
            "print_initialization_message flag"
        )
        assert result_silent is not None and len(result_silent) == 1
