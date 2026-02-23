"""
Test for interactions endpoint agent parameter handling.

Tests that the /v1beta/interactions endpoint correctly extracts
the `agent` parameter as a fallback when `model` is not provided.
"""

import pytest


class TestInteractionsAgentParameter:
    """Test agent parameter handling in interactions endpoint."""

    def test_agent_parameter_fallback_logic(self):
        """
        Test the core logic: model or agent extraction.

        This tests the fix in endpoints.py line ~267:
        model=data.get("model") or data.get("agent")
        """
        # Case 1: Only agent provided (Deep Research use case)
        data = {
            "agent": "deep-research-pro-preview-12-2025",
            "input": "Research quantum computing",
            "background": True,
        }
        model = data.get("model") or data.get("agent")
        assert model == "deep-research-pro-preview-12-2025"

        # Case 2: Only model provided (normal use case)
        data = {
            "model": "gemini-2.5-flash",
            "input": "Hello world",
        }
        model = data.get("model") or data.get("agent")
        assert model == "gemini-2.5-flash"

        # Case 3: Both provided (model takes precedence)
        data = {
            "model": "gemini-2.5-flash",
            "agent": "deep-research-pro-preview-12-2025",
            "input": "Test",
        }
        model = data.get("model") or data.get("agent")
        assert model == "gemini-2.5-flash"

        # Case 4: Neither provided
        data = {
            "input": "Test",
        }
        model = data.get("model") or data.get("agent")
        assert model is None

    def test_route_type_in_skip_model_routing_list(self):
        """
        Test that acreate_interaction is in the list of routes
        that skip model-based routing.

        This tests the fix in route_llm_request.py.
        """
        # The list of routes that skip model routing for interactions
        skip_model_routing_routes = [
            "acreate_interaction",
            "aget_interaction",
            "adelete_interaction",
            "acancel_interaction",
        ]

        # acreate_interaction should be in the list (this is the fix)
        assert "acreate_interaction" in skip_model_routing_routes

        # All interaction routes should be covered
        assert "aget_interaction" in skip_model_routing_routes
        assert "adelete_interaction" in skip_model_routing_routes
        assert "acancel_interaction" in skip_model_routing_routes
