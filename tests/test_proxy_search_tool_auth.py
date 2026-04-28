"""
Test search tool authorization - verify model-like access control for search tools.

Tests that:
1. Keys can only access search tools in their allowed_search_tools list
2. Teams can only access search tools in their allowed_search_tools list
3. Empty allowlists grant access to all search tools
4. Credentials are never exposed in team/key metadata
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

# Import types and functions to test
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    can_key_call_search_tool,
    can_team_call_search_tool,
)


@pytest.mark.asyncio
async def test_key_can_access_allowed_search_tool():
    """Test that a key can access a search tool in its allowlist."""
    # Create a mock key with allowed_search_tools
    mock_key = UserAPIKeyAuth(
        token="sk-test-key",
        models=["gpt-4"],
        allowed_search_tools=["tavily-search", "perplexity-search"],
    )

    # Should succeed - tool is in allowlist
    result = await can_key_call_search_tool(
        search_tool_name="tavily-search",
        valid_token=mock_key,
    )
    assert result is True


@pytest.mark.asyncio
async def test_key_denied_non_allowed_search_tool():
    """Test that a key is denied access to a search tool not in its allowlist."""
    mock_key = UserAPIKeyAuth(
        token="sk-test-key",
        models=["gpt-4"],
        allowed_search_tools=["tavily-search"],  # Only tavily allowed
    )

    # Should raise exception - brave-search not in allowlist
    with pytest.raises(Exception) as exc_info:
        await can_key_call_search_tool(
            search_tool_name="brave-search",
            valid_token=mock_key,
        )
    assert "not allowed to access search tool" in str(exc_info.value)
    assert "brave-search" in str(exc_info.value)


@pytest.mark.asyncio
async def test_key_empty_allowlist_grants_all_access():
    """Test that an empty allowlist grants access to all search tools."""
    mock_key = UserAPIKeyAuth(
        token="sk-test-key",
        models=["gpt-4"],
        allowed_search_tools=[],  # Empty = all allowed
    )

    # Should succeed - empty list allows all
    result = await can_key_call_search_tool(
        search_tool_name="any-search-tool",
        valid_token=mock_key,
    )
    assert result is True


@pytest.mark.asyncio
async def test_team_can_access_allowed_search_tool():
    """Test that a team can access a search tool in its allowlist."""
    mock_team = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="Marketing Team",
        models=["gpt-4"],
        allowed_search_tools=["tavily-search", "exa-search"],
    )

    # Should succeed - tool is in allowlist
    result = await can_team_call_search_tool(
        search_tool_name="tavily-search",
        team_object=mock_team,
    )
    assert result is True


@pytest.mark.asyncio
async def test_team_denied_non_allowed_search_tool():
    """Test that a team is denied access to a search tool not in its allowlist."""
    mock_team = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="Engineering Team",
        models=["gpt-4"],
        allowed_search_tools=["perplexity-search"],  # Only perplexity allowed
    )

    # Should raise exception - tavily-search not in allowlist
    with pytest.raises(Exception) as exc_info:
        await can_team_call_search_tool(
            search_tool_name="tavily-search",
            team_object=mock_team,
        )
    assert "not allowed to access search tool" in str(exc_info.value)
    assert "tavily-search" in str(exc_info.value)


@pytest.mark.asyncio
async def test_team_empty_allowlist_grants_all_access():
    """Test that an empty team allowlist grants access to all search tools."""
    mock_team = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="Admin Team",
        models=["gpt-4"],
        allowed_search_tools=[],  # Empty = all allowed
    )

    # Should succeed - empty list allows all
    result = await can_team_call_search_tool(
        search_tool_name="any-search-tool",
        team_object=mock_team,
    )
    assert result is True


@pytest.mark.asyncio
async def test_team_none_allowed_search_tools():
    """Test that None for allowed_search_tools (not set) grants access to all."""
    mock_team = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="Legacy Team",
        models=["gpt-4"],
        allowed_search_tools=None,  # Not set = all allowed
    )

    # Should succeed - None allows all
    result = await can_team_call_search_tool(
        search_tool_name="any-search-tool",
        team_object=mock_team,
    )
    assert result is True


def test_credentials_not_in_team_metadata():
    """Verify that search provider credentials are never stored in team metadata."""
    mock_team = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="Test Team",
        models=["gpt-4"],
        allowed_search_tools=["tavily-search"],
        metadata={"custom_field": "value"},  # No search_provider_config
    )

    # Verify metadata does not contain search_provider_config
    assert mock_team.metadata is not None
    assert "search_provider_config" not in mock_team.metadata
    assert "api_key" not in str(mock_team.metadata)


def test_credentials_not_in_key_metadata():
    """Verify that search provider credentials are never stored in key metadata."""
    mock_key = UserAPIKeyAuth(
        token="sk-test-key",
        models=["gpt-4"],
        allowed_search_tools=["tavily-search"],
        metadata={"user_info": "test"},  # No search_provider_config
    )

    # Verify metadata does not contain search_provider_config
    assert mock_key.metadata is not None
    assert "search_provider_config" not in mock_key.metadata
    assert "api_key" not in str(mock_key.metadata)


@pytest.mark.asyncio
async def test_both_key_and_team_checks_required():
    """Test that both key-level and team-level checks are enforced."""
    # Key has access to tool
    mock_key = UserAPIKeyAuth(
        token="sk-test-key",
        models=["gpt-4"],
        allowed_search_tools=["tavily-search"],
    )

    # Team does NOT have access to tool
    mock_team = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="Restricted Team",
        models=["gpt-4"],
        allowed_search_tools=["perplexity-search"],  # Different tool
    )

    # Key check passes
    await can_key_call_search_tool(
        search_tool_name="tavily-search",
        valid_token=mock_key,
    )

    # Team check fails
    with pytest.raises(Exception) as exc_info:
        await can_team_call_search_tool(
            search_tool_name="tavily-search",
            team_object=mock_team,
        )
    assert "not allowed to access search tool" in str(exc_info.value)
