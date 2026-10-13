"""
Mock tests for LiteLLMA2ACardResolver.

Tests that the card resolver tries both old and new well-known paths.
"""

from types import SimpleNamespace
from typing import Any, Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.a2a_protocol.card_resolver import (
    LiteLLMA2ACardResolver,
    fix_agent_card_url,
    is_localhost_or_internal_url,
    normalize_agent_card_interfaces,
    set_agent_card_url,
)
from litellm.a2a_protocol.exceptions import A2AAgentCardDiscoveryError


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

    # Create a mock for the parent's get_agent_card method
    async def mock_parent_get_agent_card(self, relative_card_path=None, http_kwargs=None):
        paths_called.append(relative_card_path)
        if relative_card_path == "/.well-known/agent-card.json":
            # First call (new path) fails
            raise Exception("404 Not Found")
        else:
            # Second call (old path) succeeds
            return mock_agent_card

    # Create a mock httpx client
    mock_httpx_client = MagicMock()

    # Patch the parent class's get_agent_card method
    # We need to patch the actual parent class method that super() calls
    with patch.object(
        LiteLLMA2ACardResolver.__bases__[0],
        "get_agent_card",
        mock_parent_get_agent_card,
    ):
        resolver = LiteLLMA2ACardResolver(httpx_client=mock_httpx_client, base_url="http://test-agent:8000")
        result = await resolver.get_agent_card()

        # Verify both paths were tried in correct order
        assert len(paths_called) == 2
        assert paths_called[0] == "/.well-known/agent-card.json"  # New path tried first
        assert paths_called[1] == "/.well-known/agent.json"  # Old path tried second

        # Verify the result
        assert result == mock_agent_card
        assert result.name == "Test Agent"


def test_is_localhost_or_internal_url():
    """Test that localhost/internal URLs are correctly detected."""
    # Should return True for localhost variants
    assert is_localhost_or_internal_url("http://localhost:8000/") is True
    assert is_localhost_or_internal_url("http://0.0.0.0:8001/") is True

    # Should return False for public URLs
    assert is_localhost_or_internal_url("https://my-agent.example.com/") is False
    assert is_localhost_or_internal_url(None) is False


def test_fix_agent_card_url_replaces_localhost():
    """Test that fix_agent_card_url replaces localhost URLs with base_url."""
    # Create mock agent card with localhost URL
    mock_card = MagicMock()
    mock_card.url = "http://0.0.0.0:8001/"

    # Fix the URL
    result = fix_agent_card_url(mock_card, "https://my-public-agent.example.com")

    # Verify localhost URL was replaced with base_url
    assert result.url == "https://my-public-agent.example.com/"


def test_set_agent_card_url_updates_top_level_and_supported_interface():
    card = SimpleNamespace(
        url="http://localhost:10001/",
        supported_interfaces=[SimpleNamespace(url="http://0.0.0.0:10001/")],
    )

    set_agent_card_url(card, "https://my-public-agent.example.com")

    assert card.url == "https://my-public-agent.example.com/"
    assert card.supported_interfaces[0].url == "https://my-public-agent.example.com/"


def test_fix_agent_card_url_updates_interface_when_top_level_is_localhost():
    card = SimpleNamespace(
        url="http://localhost:10001/",
        supported_interfaces=[SimpleNamespace(url="http://0.0.0.0:10001/")],
    )

    result = fix_agent_card_url(card, "https://my-public-agent.example.com")

    assert result.url == "https://my-public-agent.example.com/"
    assert result.supported_interfaces[0].url == "https://my-public-agent.example.com/"


def test_normalize_agent_card_interfaces_downgrades_miscased_interfaces_to_the_0_3_dialect():
    pb2 = pytest.importorskip("a2a.types.a2a_pb2")

    card = pb2.AgentCard(
        name="langgraph",
        supported_interfaces=[
            pb2.AgentInterface(url="http://a/", protocol_binding="jsonrpc", protocol_version="1.0"),
            pb2.AgentInterface(url="http://b/", protocol_binding="JSONRPC", protocol_version="1.0"),
            pb2.AgentInterface(url="http://c/", protocol_binding="websocket", protocol_version="1.0"),
        ],
    )

    normalized = normalize_agent_card_interfaces(card)

    assert [(i.protocol_binding, i.protocol_version) for i in normalized.supported_interfaces] == [
        ("JSONRPC", "0.3"),
        ("JSONRPC", "1.0"),
        ("websocket", "1.0"),
    ]
    assert card.supported_interfaces[0].protocol_binding == "jsonrpc"
    assert card.supported_interfaces[0].protocol_version == "1.0"


_FOUNDRY_BASE_URL: Final = "https://foundry.example.com/a2a"

_FOUNDRY_CARD_JSON: Final = {
    "name": "Foundry Agent",
    "description": "A test agent",
    "url": "https://foundry.example.com/a2a",
    "version": "1.0",
    "capabilities": {"streaming": True},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [{"id": "chat", "name": "chat", "description": "Chat", "tags": ["chat"]}],
    "protocolVersion": "1.0",
}


class _FakeHttpxClient:
    """Answers GETs from a path -> (status, body) map and records the path of each call."""

    def __init__(self, base_url: str, responses: dict[str, tuple[int, dict[str, Any]]]) -> None:
        self._base_url = base_url.rstrip("/")
        self._responses = responses
        self.calls: list[str] = []

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        path: Final = url.removeprefix(self._base_url)
        self.calls.append(path)
        status_code, body = self._responses[path]
        return httpx.Response(status_code, json=body, request=httpx.Request("GET", url))


@pytest.mark.asyncio
async def test_card_resolver_falls_through_to_the_foundry_card_path():
    httpx_client = _FakeHttpxClient(
        base_url=_FOUNDRY_BASE_URL,
        responses={
            "/.well-known/agent-card.json": (404, {"error": "not found"}),
            "/.well-known/agent.json": (404, {"error": "not found"}),
            "/agentCard/v1.0": (200, dict(_FOUNDRY_CARD_JSON)),
        },
    )

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url=_FOUNDRY_BASE_URL)
    result = await resolver.get_agent_card()

    assert httpx_client.calls == ["/.well-known/agent-card.json", "/.well-known/agent.json", "/agentCard/v1.0"]
    assert result.name == "Foundry Agent"
    assert result.supported_interfaces[0].url == "https://foundry.example.com/a2a"


@pytest.mark.asyncio
async def test_card_resolver_explicit_path_skips_the_probes():
    httpx_client = _FakeHttpxClient(
        base_url=_FOUNDRY_BASE_URL,
        responses={"/agentCard/v1.0": (200, dict(_FOUNDRY_CARD_JSON))},
    )

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url=_FOUNDRY_BASE_URL)
    result = await resolver.get_agent_card(relative_card_path="agentCard/v1.0")

    assert httpx_client.calls == ["/agentCard/v1.0"]
    assert result.name == "Foundry Agent"


@pytest.mark.asyncio
async def test_card_resolver_names_every_probed_path_when_discovery_fails():
    httpx_client = _FakeHttpxClient(
        base_url=_FOUNDRY_BASE_URL,
        responses={
            "/.well-known/agent-card.json": (404, {"error": "not found"}),
            "/.well-known/agent.json": (401, {"error": "unauthorized"}),
            "/agentCard/v1.0": (404, {"error": "not found"}),
        },
    )

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url=_FOUNDRY_BASE_URL)
    with pytest.raises(A2AAgentCardDiscoveryError) as raised:
        await resolver.get_agent_card()

    assert raised.value.status_code == 401
    message = str(raised.value)
    assert _FOUNDRY_BASE_URL in message
    assert "/.well-known/agent-card.json (" in message and "HTTP 404" in message
    assert "/.well-known/agent.json (" in message and "HTTP 401" in message
    assert "/agentCard/v1.0 (" in message


@pytest.mark.asyncio
async def test_card_resolver_discovery_error_is_404_when_every_probe_is_404():
    resolver = LiteLLMA2ACardResolver(
        httpx_client=_FakeHttpxClient(
            base_url=_FOUNDRY_BASE_URL,
            responses={
                "/.well-known/agent-card.json": (404, {"error": "not found"}),
                "/.well-known/agent.json": (404, {"error": "not found"}),
                "/agentCard/v1.0": (404, {"error": "not found"}),
            },
        ),
        base_url=_FOUNDRY_BASE_URL,
    )

    with pytest.raises(A2AAgentCardDiscoveryError) as raised:
        await resolver.get_agent_card()

    assert raised.value.status_code == 404
