"""
Tests for MCP sampling model resolution (hint matching and fallback chain).

`_resolve_model_from_preferences` first tries to match upstream model hints
against the proxy's available models (direct then substring), then priority
scoring, then the caller default, the first available model, and finally the
configured `default_mcp_sampling_model` before raising.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _resolve_model_from_preferences,
)


def _prefs(*, hints=None, cost=None, speed=None, intelligence=None):
    return SimpleNamespace(
        hints=hints or [],
        costPriority=cost,
        speedPriority=speed,
        intelligencePriority=intelligence,
    )


class TestHintMatching:
    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch("litellm.model_list", [{"model_name": "gpt-4o"}, {"model_name": "claude-3"}])
    def test_should_match_hint_as_substring(self):
        prefs = _prefs(hints=[SimpleNamespace(name="gpt-4")])
        assert _resolve_model_from_preferences(prefs) == "gpt-4o"

    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch("litellm.model_list", ["gpt-4o", "claude-3"])
    def test_should_match_hint_against_string_model_list_entries(self):
        prefs = _prefs(hints=[SimpleNamespace(name="claude-3")])
        assert _resolve_model_from_preferences(prefs) == "claude-3"

    @patch("litellm.model_list", None)
    def test_should_use_router_model_names(self):
        router = MagicMock()
        router.get_model_names.return_value = ["router-gpt", "router-claude"]
        with patch("litellm.proxy.proxy_server.llm_router", router):
            prefs = _prefs(hints=[SimpleNamespace(name="router-claude")])
            assert _resolve_model_from_preferences(prefs) == "router-claude"

    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch("litellm.model_list", [{"model_name": "gpt-4o"}])
    def test_should_skip_hint_without_name(self):
        prefs = _prefs(hints=[SimpleNamespace()])  # hint has no `.name`
        assert (
            _resolve_model_from_preferences(prefs, default_model="gpt-4o") == "gpt-4o"
        )


class TestFallbackChain:
    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch(
        "litellm.model_list", [{"model_name": "first-model"}, {"model_name": "second"}]
    )
    def test_should_fall_back_to_first_available_when_no_default(self):
        prefs = _prefs(hints=[SimpleNamespace(name="no-such")])
        assert _resolve_model_from_preferences(prefs) == "first-model"

    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch("litellm.model_list", [])
    def test_should_use_configured_default_sampling_model(self, monkeypatch):
        import litellm

        monkeypatch.setattr(
            litellm, "default_mcp_sampling_model", "fallback-model", raising=False
        )
        prefs = _prefs()
        assert _resolve_model_from_preferences(prefs) == "fallback-model"

    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch("litellm.model_list", [])
    def test_should_raise_when_nothing_resolvable(self, monkeypatch):
        import litellm

        monkeypatch.setattr(litellm, "default_mcp_sampling_model", None, raising=False)
        prefs = _prefs()
        with pytest.raises(ValueError, match="No model could be resolved"):
            _resolve_model_from_preferences(prefs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
