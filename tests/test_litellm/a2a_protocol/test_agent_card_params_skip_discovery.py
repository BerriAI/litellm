"""
Regression tests for GH #40586: message/send re-discovered the agent card
from well-known paths instead of using the registered agent_card_params.

These tests fake the HTTP boundary with a real httpx.AsyncClient wired to an
httpx.MockTransport, rather than patching SDK internals, so they verify
actual behavior: whether a well-known-path request goes out over the wire,
and what agent card the resulting client ends up holding.
"""

from unittest.mock import patch

import httpx
import pytest

from litellm.a2a_protocol.main import create_a2a_client

BASE_URL = "https://example.com/agents/a2a"
WELL_KNOWN_PATH = "/agents/a2a/.well-known/agent-card.json"

MINIMAL_AGENT_CARD_PARAMS = {
    "protocolVersion": "1.0",
    "name": "pre-registered-agent",
    "description": "A test assistant.",
    "url": BASE_URL,
    "version": "1.0",
    "capabilities": {"streaming": False},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [],
}

DISCOVERED_AGENT_CARD_JSON = {
    "protocolVersion": "1.0",
    "name": "discovered-agent",
    "description": "A test assistant found via well-known discovery.",
    "url": BASE_URL,
    "version": "1.0",
    "capabilities": {"streaming": False},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [],
}


def _mock_client(handler):
    """A real httpx.AsyncClient wired to a local handler instead of the network."""
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    class _FakeAsyncHandler:
        def __init__(self, c):
            self.client = c

    return _FakeAsyncHandler(client)


@pytest.mark.asyncio
async def test_create_a2a_client_skips_discovery_when_agent_card_params_given():
    """
    When a caller already has a resolved agent card (e.g. one stored via
    POST /v1/agents), create_a2a_client must build the client from it
    directly instead of making a well-known-path HTTP request.
    """
    requests_made: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(str(request.url))
        return httpx.Response(404, json={"error": "should not be called"})

    with (
        patch(  # test-quality-ok: injects a real httpx.AsyncClient wired to httpx.MockTransport, not a mock of behavior
            "litellm.a2a_protocol.main.get_async_httpx_client",
            return_value=_mock_client(handler),
        )
    ):
        a2a_client = await create_a2a_client(
            base_url=BASE_URL,
            agent_card_params=MINIMAL_AGENT_CARD_PARAMS,
        )

    # The real, observable proof of the fix: no HTTP request went out at all.
    assert requests_made == []

    resolved_card = a2a_client._litellm_agent_card
    assert resolved_card.name == "pre-registered-agent"
    assert resolved_card.supported_interfaces[0].url == BASE_URL


@pytest.mark.asyncio
async def test_create_a2a_client_falls_back_to_discovery_without_agent_card_params():
    """
    Unchanged behavior: when no agent_card_params is supplied, the client
    resolves the card over HTTP from the well-known path, same as before
    the fix.
    """
    requests_made: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(str(request.url))
        if request.url.path == WELL_KNOWN_PATH:
            return httpx.Response(200, json=DISCOVERED_AGENT_CARD_JSON)
        return httpx.Response(404)

    with (
        patch(  # test-quality-ok: injects a real httpx.AsyncClient wired to httpx.MockTransport, not a mock of behavior
            "litellm.a2a_protocol.main.get_async_httpx_client",
            return_value=_mock_client(handler),
        )
    ):
        a2a_client = await create_a2a_client(base_url=BASE_URL)

    assert any(WELL_KNOWN_PATH in url for url in requests_made)
    resolved_card = a2a_client._litellm_agent_card
    assert resolved_card.name == "discovered-agent"


@pytest.mark.asyncio
async def test_create_a2a_client_falls_back_to_discovery_on_invalid_agent_card_params():
    """
    If the stored agent_card_params doesn't validate as a full AgentCard
    (e.g. a partial card missing required fields), create_a2a_client must
    fall back to well-known-path discovery rather than raising.
    """
    requests_made: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(str(request.url))
        if request.url.path == WELL_KNOWN_PATH:
            return httpx.Response(200, json=DISCOVERED_AGENT_CARD_JSON)
        return httpx.Response(404)

    incomplete_agent_card_params = {"url": BASE_URL}  # missing required fields

    with (
        patch(  # test-quality-ok: injects a real httpx.AsyncClient wired to httpx.MockTransport, not a mock of behavior
            "litellm.a2a_protocol.main.get_async_httpx_client",
            return_value=_mock_client(handler),
        )
    ):
        a2a_client = await create_a2a_client(
            base_url=BASE_URL,
            agent_card_params=incomplete_agent_card_params,
        )

    assert any(WELL_KNOWN_PATH in url for url in requests_made)
    resolved_card = a2a_client._litellm_agent_card
    assert resolved_card.name == "discovered-agent"
