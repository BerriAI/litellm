"""
Mock tests for LiteLLMA2ACardResolver.

Tests that the card resolver tries both old and new well-known paths.
"""

from types import SimpleNamespace
from typing import Any, Final
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import httpx
import pytest

from litellm.a2a_protocol.card_resolver import (
    LiteLLMA2ACardResolver,
    fix_agent_card_url,
    get_agent_card_url,
    is_localhost_or_internal_url,
    normalize_agent_card_interfaces,
    resolve_a2a_protocol_version,
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


@pytest.mark.parametrize(
    ("card", "expected"),
    [
        (SimpleNamespace(url="https://agent.example.com/"), "https://agent.example.com/"),
        (
            SimpleNamespace(url="", supported_interfaces=[SimpleNamespace(url="https://interface.example.com/")]),
            "https://interface.example.com/",
        ),
        (
            SimpleNamespace(supported_interfaces=[SimpleNamespace(url="https://interface.example.com/")]),
            "https://interface.example.com/",
        ),
        (SimpleNamespace(url=""), None),
        (SimpleNamespace(url=None, supported_interfaces=[]), None),
        (SimpleNamespace(), None),
    ],
    ids=[
        "top_level_url",
        "empty_url_falls_to_interface",
        "interfaces_only",
        "empty_url_without_interfaces",
        "no_interfaces",
        "bare",
    ],
)
def test_get_agent_card_url_prefers_the_top_level_url_then_the_first_interface(
    card: SimpleNamespace, expected: str | None
):
    assert get_agent_card_url(card) == expected


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


def test_normalize_agent_card_interfaces_applies_override_to_canonical_bindings_without_mutating_input():
    pb2 = pytest.importorskip("a2a.types.a2a_pb2")
    card = pb2.AgentCard(
        name="langgraph",
        supported_interfaces=[
            pb2.AgentInterface(url="http://a/", protocol_binding="jsonrpc", protocol_version="1.0"),
            pb2.AgentInterface(url="http://b/", protocol_binding="JSONRPC", protocol_version="1.0"),
            pb2.AgentInterface(url="http://c/", protocol_binding="HTTP+JSON", protocol_version="1.0"),
            pb2.AgentInterface(url="http://d/", protocol_binding="GRPC", protocol_version="1.0"),
            pb2.AgentInterface(url="http://e/", protocol_binding="websocket", protocol_version="1.0"),
        ],
    )

    normalized = normalize_agent_card_interfaces(card, protocol_version="0.3")

    assert [(item.protocol_binding, item.protocol_version) for item in normalized.supported_interfaces] == [
        ("JSONRPC", "0.3"),
        ("JSONRPC", "0.3"),
        ("HTTP+JSON", "0.3"),
        ("GRPC", "0.3"),
        ("websocket", "1.0"),
    ]
    assert card.supported_interfaces[0].protocol_binding == "jsonrpc"
    assert [(item.protocol_binding, item.protocol_version) for item in card.supported_interfaces] == [
        ("jsonrpc", "1.0"),
        ("JSONRPC", "1.0"),
        ("HTTP+JSON", "1.0"),
        ("GRPC", "1.0"),
        ("websocket", "1.0"),
    ]


def test_normalize_agent_card_interfaces_override_opts_out_of_mis_cased_downgrade():
    pb2 = pytest.importorskip("a2a.types.a2a_pb2")
    card = pb2.AgentCard(
        name="langgraph",
        supported_interfaces=[
            pb2.AgentInterface(url="http://a/", protocol_binding="jsonrpc", protocol_version="1.0"),
        ],
    )

    normalized = normalize_agent_card_interfaces(card, protocol_version="1.0")

    assert normalized.supported_interfaces[0].protocol_binding == "JSONRPC"
    assert normalized.supported_interfaces[0].protocol_version == "1.0"
    assert card.supported_interfaces[0].protocol_binding == "jsonrpc"
    assert card.supported_interfaces[0].protocol_version == "1.0"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.3", "0.3"),
        (" 0.3 ", "0.3"),
        ("0.3.0", "0.3"),
        (0.3, "0.3"),
        (1.0, "1.0"),
        ("1.0.0", "1.0"),
        (None, None),
        (True, None),
        (1, None),
        ("2.0", None),
        ("abc", None),
        ("", None),
        (["0.3"], None),
    ],
)
def test_resolve_a2a_protocol_version(value, expected):
    assert resolve_a2a_protocol_version(value) == expected


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
    """Answers GETs under ``base_url`` from a path -> (status, body) map, and GETs at the URL's
    origin from ``origin_responses``; records the path (or the full origin URL) of each call."""

    def __init__(
        self,
        base_url: str,
        responses: dict[str, tuple[int, dict[str, Any]]],
        origin_responses: dict[str, tuple[int, dict[str, Any]]] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._origin = f"{urlsplit(base_url).scheme}://{urlsplit(base_url).netloc}"
        self._responses = responses
        self._origin_responses = origin_responses or {}
        self.calls: list[str] = []
        self.headers_sent: list[Any] = []

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.headers_sent.append(kwargs.get("headers"))
        if url.startswith(self._base_url + "/"):
            self.calls.append(url.removeprefix(self._base_url))
            status_code, body = self._responses[url.removeprefix(self._base_url)]
        else:
            assert url.startswith(self._origin + "/"), url
            self.calls.append(url)
            status_code, body = self._origin_responses[url.removeprefix(self._origin)]
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
        origin_responses={
            "/.well-known/agent-card.json": (404, {"error": "not found"}),
            "/.well-known/agent.json": (404, {"error": "not found"}),
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
    assert "https://foundry.example.com/.well-known/agent-card.json (" in message


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
            origin_responses={
                "/.well-known/agent-card.json": (404, {"error": "not found"}),
                "/.well-known/agent.json": (404, {"error": "not found"}),
            },
        ),
        base_url=_FOUNDRY_BASE_URL,
    )

    with pytest.raises(A2AAgentCardDiscoveryError) as raised:
        await resolver.get_agent_card()

    assert raised.value.status_code == 404


_SUBPATH_AGENT_URL: Final = "http://rsagent:3000/jsonrpc"

_SUBPATH_CARD_JSON: Final = {
    **_FOUNDRY_CARD_JSON,
    "name": "rspath",
    "url": _SUBPATH_AGENT_URL,
    "supportedInterfaces": [{"url": _SUBPATH_AGENT_URL, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
}


_NO_CARD_UNDER_THE_AGENT_PATH: Final = {
    "/.well-known/agent-card.json": (404, {"error": "not found"}),
    "/.well-known/agent.json": (404, {"error": "not found"}),
    "/agentCard/v1.0": (404, {"error": "not found"}),
}


def _subpath_card_json(endpoint: str) -> dict[str, Any]:
    return {
        **_SUBPATH_CARD_JSON,
        "url": endpoint,
        "supportedInterfaces": [{"url": endpoint, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card_endpoint",
    [_SUBPATH_AGENT_URL, _SUBPATH_AGENT_URL + "/", "http://rsagent:3000", "http://rsagent:3000/"],
    ids=["agent url", "agent url with slash", "origin root", "origin root with slash"],
)
async def test_card_resolver_falls_back_to_the_well_known_card_at_the_agents_origin(card_endpoint):
    """A2A 1.0 places the card at ``{origin}/.well-known/agent-card.json`` while the JSON-RPC
    endpoint is routinely a sub-path, so an agent registered with that endpoint as its URL
    must still be resolvable, from a card that names that endpoint or a path prefix of it."""
    httpx_client = _FakeHttpxClient(
        base_url=_SUBPATH_AGENT_URL,
        responses=dict(_NO_CARD_UNDER_THE_AGENT_PATH),
        origin_responses={"/.well-known/agent-card.json": (200, _subpath_card_json(card_endpoint))},
    )

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url=_SUBPATH_AGENT_URL)
    result = await resolver.get_agent_card()

    assert httpx_client.calls[-1] == "http://rsagent:3000/.well-known/agent-card.json"
    assert result.name == "rspath"
    assert result.supported_interfaces[0].url == card_endpoint


@pytest.mark.asyncio
async def test_card_resolver_keeps_the_agents_headers_off_the_origin_probes():
    """The public card at the origin's well-known URI is unauthenticated by spec, and the origin
    may host other applications, so the agent's own headers only go to the agent's paths."""
    httpx_client = _FakeHttpxClient(
        base_url=_SUBPATH_AGENT_URL,
        responses=dict(_NO_CARD_UNDER_THE_AGENT_PATH),
        origin_responses={"/.well-known/agent-card.json": (200, _subpath_card_json(_SUBPATH_AGENT_URL))},
    )
    agent_headers = {"x-agent-token": "secret-for-this-agent"}

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url=_SUBPATH_AGENT_URL)
    await resolver.get_agent_card(http_kwargs={"headers": agent_headers})

    assert httpx_client.calls[-1] == "http://rsagent:3000/.well-known/agent-card.json"
    assert httpx_client.headers_sent[:-1] == [agent_headers] * 3
    assert httpx_client.headers_sent[-1] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card_json",
    [
        {
            **_subpath_card_json("http://rsagent:3000/grpc"),
            "supportedInterfaces": [
                {"url": "http://rsagent:3000/grpc", "protocolBinding": "GRPC", "protocolVersion": "1.0"},
                {"url": _SUBPATH_AGENT_URL, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
            ],
        },
        {
            **_FOUNDRY_CARD_JSON,
            "name": "rspath",
            "url": "http://rsagent:3000/grpc",
            "preferredTransport": "GRPC",
            "additionalInterfaces": [{"url": _SUBPATH_AGENT_URL, "transport": "JSONRPC"}],
        },
    ],
    ids=["supportedInterfaces", "additionalInterfaces"],
)
async def test_card_resolver_accepts_an_origin_card_that_names_the_agent_url_in_a_later_interface(card_json):
    """The agent URL may be any of the card's interfaces, not only the first one."""
    httpx_client = _FakeHttpxClient(
        base_url=_SUBPATH_AGENT_URL,
        responses=dict(_NO_CARD_UNDER_THE_AGENT_PATH),
        origin_responses={"/.well-known/agent-card.json": (200, card_json)},
    )

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url=_SUBPATH_AGENT_URL)
    result = await resolver.get_agent_card()

    assert httpx_client.calls[-1] == "http://rsagent:3000/.well-known/agent-card.json"
    assert result.name == "rspath"
    assert [interface.url for interface in result.supported_interfaces] == ["http://rsagent:3000/grpc", _SUBPATH_AGENT_URL]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejected_endpoint",
    ["http://rsagent:3000/other", "http://rsagent:3000/json", "http://elsewhere:9/jsonrpc"],
    ids=["sibling path", "string prefix but not a path prefix", "other host"],
)
async def test_card_resolver_rejects_an_origin_card_that_advertises_another_endpoint(rejected_endpoint):
    """On a host serving several agents the origin card can belong to another agent, and its
    endpoint would receive this agent's messages and headers, so it is only taken when one of its
    interfaces names the configured agent URL or a path prefix of it; otherwise the card counts
    as not found."""
    httpx_client = _FakeHttpxClient(
        base_url=_SUBPATH_AGENT_URL,
        responses=dict(_NO_CARD_UNDER_THE_AGENT_PATH),
        origin_responses={
            "/.well-known/agent-card.json": (200, _subpath_card_json(rejected_endpoint)),
            "/.well-known/agent.json": (404, {"error": "not found"}),
        },
    )

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url=_SUBPATH_AGENT_URL)
    with pytest.raises(A2AAgentCardDiscoveryError) as raised:
        await resolver.get_agent_card()

    assert raised.value.status_code == 404
    message = str(raised.value)
    assert "http://rsagent:3000/.well-known/agent-card.json (" in message and repr(rejected_endpoint) in message


@pytest.mark.asyncio
async def test_card_resolver_prefers_the_card_relative_to_the_agent_url():
    """Deployments that serve the card under the agent path keep resolving without an extra
    round trip, and a host that serves a different card at its origin never wins over them."""
    httpx_client = _FakeHttpxClient(
        base_url=_SUBPATH_AGENT_URL,
        responses={"/.well-known/agent-card.json": (200, dict(_SUBPATH_CARD_JSON))},
        origin_responses={"/.well-known/agent-card.json": (200, {**_SUBPATH_CARD_JSON, "name": "other agent"})},
    )

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url=_SUBPATH_AGENT_URL)
    result = await resolver.get_agent_card()

    assert httpx_client.calls == ["/.well-known/agent-card.json"]
    assert result.name == "rspath"


@pytest.mark.asyncio
async def test_card_resolver_probes_only_once_when_the_agent_url_is_the_origin():
    httpx_client = _FakeHttpxClient(
        base_url="http://rsagent:3000",
        responses={
            "/.well-known/agent-card.json": (404, {"error": "not found"}),
            "/.well-known/agent.json": (404, {"error": "not found"}),
            "/agentCard/v1.0": (404, {"error": "not found"}),
        },
    )

    resolver = LiteLLMA2ACardResolver(httpx_client=httpx_client, base_url="http://rsagent:3000/")
    with pytest.raises(A2AAgentCardDiscoveryError):
        await resolver.get_agent_card()

    assert httpx_client.calls == ["/.well-known/agent-card.json", "/.well-known/agent.json", "/agentCard/v1.0"]

