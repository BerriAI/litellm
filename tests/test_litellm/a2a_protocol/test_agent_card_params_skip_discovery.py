"""
Regression tests for GH #40586: message/send re-discovered the agent card
from well-known paths instead of using the registered agent_card_params.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.a2a_protocol.main import create_a2a_client

MINIMAL_AGENT_CARD_PARAMS = {
    "protocolVersion": "1.0",
    "name": "agent",
    "description": "A test assistant.",
    "url": "https://example.com/agents/a2a",
    "version": "1.0",
    "capabilities": {"streaming": False},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [],
}


@pytest.mark.asyncio
async def test_create_a2a_client_skips_discovery_when_agent_card_params_given():
    """
    When a caller already has a resolved agent card (e.g. one stored via
    POST /v1/agents), create_a2a_client must build the AgentCard directly
    from it instead of probing the upstream server's well-known paths.
    """
    with (
        patch("litellm.a2a_protocol.main.get_async_httpx_client") as mock_get_httpx,
        patch("litellm.a2a_protocol.main.A2ACardResolver") as mock_resolver_cls,
        patch("litellm.a2a_protocol.main.create_client", new_callable=AsyncMock) as mock_create_client,
        patch(
            "litellm.a2a_protocol.main.normalize_agent_card_interfaces",
            side_effect=lambda card: card,
        ),
    ):
        mock_get_httpx.return_value.client = MagicMock()
        mock_create_client.return_value = MagicMock()

        await create_a2a_client(
            base_url="https://example.com/agents/a2a",
            agent_card_params=MINIMAL_AGENT_CARD_PARAMS,
        )

        # The whole point of the fix: the resolver must never be touched
        # when a card was already supplied at registration time.
        mock_resolver_cls.assert_not_called()

        mock_create_client.assert_awaited_once()
        called_card = mock_create_client.await_args.args[0]
        assert called_card.name == "agent"
        assert str(called_card.url) == "https://example.com/agents/a2a"


@pytest.mark.asyncio
async def test_create_a2a_client_falls_back_to_discovery_without_agent_card_params():
    """
    Unchanged behavior: when no agent_card_params is supplied, resolve the
    card via the well-known-path resolver, same as before the fix.
    """
    with (
        patch("litellm.a2a_protocol.main.get_async_httpx_client") as mock_get_httpx,
        patch("litellm.a2a_protocol.main.A2ACardResolver") as mock_resolver_cls,
        patch("litellm.a2a_protocol.main.create_client", new_callable=AsyncMock) as mock_create_client,
        patch(
            "litellm.a2a_protocol.main.normalize_agent_card_interfaces",
            side_effect=lambda card: card,
        ),
    ):
        mock_get_httpx.return_value.client = MagicMock()
        mock_resolver_instance = mock_resolver_cls.return_value
        mock_resolver_instance.get_agent_card = AsyncMock(return_value=MagicMock())
        mock_create_client.return_value = MagicMock()

        await create_a2a_client(base_url="https://example.com/agents/a2a")

        mock_resolver_cls.assert_called_once()
        mock_resolver_instance.get_agent_card.assert_awaited_once()