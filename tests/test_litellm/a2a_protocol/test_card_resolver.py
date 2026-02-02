"""
Mock tests for LiteLLMA2ACardResolver.

Tests that the card resolver tries both old and new well-known paths.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_card_resolver_fallback_from_new_to_old_path():
    """
    Test that the card resolver tries the new path (/.well-known/agent-card.json) first,
    and falls back to the old path (/.well-known/agent.json) if the new path fails.
    """
    # Mock the AgentCard
    mock_agent_card = MagicMock()
    mock_agent_card.name = "Test Agent"
    mock_agent_card.description = "A test agent"

    # Track which paths were called
    paths_called = []

    # Create a mock base class
    class MockA2ACardResolver:
        def __init__(self, base_url):
            self.base_url = base_url

        async def get_agent_card(self, relative_card_path=None, http_kwargs=None):
            paths_called.append(relative_card_path)
            if relative_card_path == "/.well-known/agent-card.json":
                # First call (new path) fails
                raise Exception("404 Not Found")
            else:
                # Second call (old path) succeeds
                return mock_agent_card

    # Create mock A2A module
    mock_a2a_module = MagicMock()
    mock_a2a_client = MagicMock()
    mock_a2a_constants = MagicMock()
    mock_a2a_constants.AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"
    mock_a2a_constants.PREV_AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent.json"

    with patch.dict(
        sys.modules,
        {
            "a2a": mock_a2a_module,
            "a2a.client": MagicMock(A2ACardResolver=MockA2ACardResolver),
            "a2a.utils.constants": mock_a2a_constants,
        },
    ):
        # Import after patching
        from litellm.a2a_protocol.card_resolver import LiteLLMA2ACardResolver

        resolver = LiteLLMA2ACardResolver(base_url="http://test-agent:8000")
        result = await resolver.get_agent_card()

        # Verify both paths were tried in correct order
        assert len(paths_called) == 2
        assert paths_called[0] == "/.well-known/agent-card.json"  # New path tried first
        assert paths_called[1] == "/.well-known/agent.json"  # Old path tried second

        # Verify the result
        assert result == mock_agent_card
        assert result.name == "Test Agent"
