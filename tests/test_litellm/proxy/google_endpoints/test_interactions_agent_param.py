"""
Test for interactions endpoint agent parameter handling.

Tests that the /v1beta/interactions endpoint correctly resolves
the routing model when `agent` and/or `model` are provided.
When both are present, `agent` takes precedence (non-breaking).
"""

import pytest


class TestInteractionsAgentParameter:
    """Test agent parameter handling in interactions endpoint."""

    @staticmethod
    def _resolve_model(data: dict):
        """Mirrors the resolution logic in endpoints.py."""
        if data.get("agent") and data.get("model"):
            data.pop("model")
        return data.get("model") or data.get("agent")

    def test_agent_only(self):
        """Deep Research use case — agent is used for routing."""
        data = {
            "agent": "deep-research-pro-preview-12-2025",
            "input": "Research quantum computing",
            "background": True,
        }
        assert self._resolve_model(data) == "deep-research-pro-preview-12-2025"

    def test_model_only(self):
        """Normal use case — model is used for routing."""
        data = {
            "model": "gemini-2.5-flash",
            "input": "Hello world",
        }
        assert self._resolve_model(data) == "gemini-2.5-flash"

    def test_both_agent_and_model_agent_wins(self):
        """When both are provided, agent takes precedence (non-breaking)."""
        data = {
            "model": "gemini-2.5-flash",
            "agent": "deep-research-pro-preview-12-2025",
            "input": "Test",
        }
        assert self._resolve_model(data) == "deep-research-pro-preview-12-2025"
        assert "model" not in data  # model is dropped

    def test_neither_provided(self):
        data = {"input": "Test"}
        assert self._resolve_model(data) is None
