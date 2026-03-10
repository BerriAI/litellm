"""Unit tests for AgentRegistry DB operations."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry


def _sample_agent_card_params() -> dict:
    return {
        "protocolVersion": "1.0",
        "name": "Test Agent",
        "description": "desc",
        "url": "http://localhost",
        "version": "1.0.0",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


@pytest.mark.asyncio
async def test_update_agent_in_db_clears_static_headers_and_extra_headers_when_omitted():
    """
    PUT (full-replace) should clear static_headers and extra_headers when omitted.
    Previously, omitting these fields left stale DB values intact.
    """
    registry = AgentRegistry()
    mock_prisma = MagicMock()

    # Simulate existing agent that had headers set
    updated_agent = MagicMock()
    updated_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Updated Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "static_headers": {},
        "extra_headers": [],
        "object_permission": None,
    }
    updated_agent.object_permission = None

    mock_update = AsyncMock(return_value=updated_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    # Agent config WITHOUT static_headers or extra_headers (omitted)
    agent_config = {
        "agent_name": "Updated Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
    }

    await registry.update_agent_in_db(
        agent_id="agent-123",
        agent=agent_config,
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    mock_update.assert_awaited_once()
    call_kwargs = mock_update.call_args.kwargs
    update_data = call_kwargs["data"]

    # Should include static_headers and extra_headers with empty defaults
    assert "static_headers" in update_data
    assert update_data["static_headers"] == "{}"
    assert "extra_headers" in update_data
    assert update_data["extra_headers"] == []


@pytest.mark.asyncio
async def test_update_agent_in_db_preserves_explicit_static_headers_and_extra_headers():
    """PUT with explicit values should still work correctly."""
    registry = AgentRegistry()
    mock_prisma = MagicMock()

    updated_agent = MagicMock()
    updated_agent.model_dump.return_value = {
        "agent_id": "agent-123",
        "agent_name": "Updated Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "static_headers": {"Authorization": "Bearer xyz"},
        "extra_headers": ["X-Custom-Header"],
        "object_permission": None,
    }
    updated_agent.object_permission = None

    mock_update = AsyncMock(return_value=updated_agent)
    mock_prisma.db.litellm_agentstable.update = mock_update

    agent_config = {
        "agent_name": "Updated Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {},
        "static_headers": {"Authorization": "Bearer xyz"},
        "extra_headers": ["X-Custom-Header"],
    }

    await registry.update_agent_in_db(
        agent_id="agent-123",
        agent=agent_config,
        prisma_client=mock_prisma,
        updated_by="test-user",
    )

    call_kwargs = mock_update.call_args.kwargs
    update_data = call_kwargs["data"]

    assert update_data["static_headers"] == '{"Authorization": "Bearer xyz"}'
    assert update_data["extra_headers"] == ["X-Custom-Header"]
