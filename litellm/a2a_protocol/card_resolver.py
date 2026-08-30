"""
Custom A2A Card Resolver for LiteLLM.

Extends the A2A SDK's card resolver to support multiple well-known paths.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_logger
from litellm.constants import LOCALHOST_URL_PATTERNS

if TYPE_CHECKING:
    from a2a.types import AgentCard

# Runtime imports with availability check
_A2ACardResolver: Any = None
AGENT_CARD_WELL_KNOWN_PATH: str = "/.well-known/agent-card.json"
PREV_AGENT_CARD_WELL_KNOWN_PATH: str = "/.well-known/agent.json"

try:
    from a2a.client import A2ACardResolver as _A2ACardResolver
    from a2a.utils.constants import (
        AGENT_CARD_WELL_KNOWN_PATH,
        PREV_AGENT_CARD_WELL_KNOWN_PATH,
    )
except ImportError:
    pass


def is_localhost_or_internal_url(url: str | None) -> bool:
    """
    Check if a URL is a localhost or internal URL.

    This detects common development URLs that are accidentally left in
    agent cards when deploying to production.

    Args:
        url: The URL to check

    Returns:
        True if the URL is localhost/internal
    """
    if not url:
        return False

    url_lower: Final = url.lower()

    return any(pattern in url_lower for pattern in LOCALHOST_URL_PATTERNS)


_CANONICAL_PROTOCOL_BINDINGS: Final = MappingProxyType(
    {
        "jsonrpc": "JSONRPC",
        "http+json": "HTTP+JSON",
        "grpc": "GRPC",
    }
)

_LEGACY_PROTOCOL_VERSION: Final = "0.3"


def normalize_agent_card_interfaces(agent_card: "AgentCard") -> "AgentCard":
    """
    Canonicalize the supported interfaces of spec-adjacent agent cards.

    Some A2A servers (e.g. LangGraph Platform) serve agent cards with lowercase
    bindings like "jsonrpc", but a2a-sdk's ClientFactory matches bindings
    case-sensitively against its uppercase TransportProtocol constants and fails
    with "no compatible transports found." for spec-adjacent casings.

    The same servers also speak the A2A 0.3 JSON dialect ("kind"-discriminated
    payloads) while declaring protocolVersion "1.0", which a2a-sdk's strict v1
    proto parsing rejects. A mis-cased binding fingerprints such a server, so its
    declared version is downgraded to 0.3 to route the SDK's ClientFactory onto
    its v0.3 compat transport, which speaks that dialect.
    """
    normalized: Final = type(agent_card)()
    normalized.CopyFrom(agent_card)
    for interface in normalized.supported_interfaces:
        canonical: str | None = _CANONICAL_PROTOCOL_BINDINGS.get(interface.protocol_binding.lower())
        if canonical is None or canonical == interface.protocol_binding:
            continue
        interface.protocol_binding = canonical
        interface.protocol_version = _LEGACY_PROTOCOL_VERSION
    return normalized


def get_agent_card_url(agent_card: "AgentCard") -> str | None:
    """Return the agent endpoint URL from the resolved SDK card."""
    url: Final = getattr(agent_card, "url", None)
    if url:
        return url

    interfaces: Final = getattr(agent_card, "supported_interfaces", None)
    if interfaces:
        return getattr(interfaces[0], "url", None)
    return None


def set_agent_card_url(agent_card: "AgentCard", url: str) -> None:
    """Set the agent endpoint URL on the resolved SDK card."""
    normalized: Final = url.rstrip("/") + "/"
    if hasattr(agent_card, "url"):
        agent_card.url = normalized

    interfaces: Final = getattr(agent_card, "supported_interfaces", None)
    if interfaces:
        interfaces[0].url = normalized


def fix_agent_card_url(agent_card: "AgentCard", base_url: str) -> "AgentCard":
    """
    Fix the agent card URL if it contains a localhost/internal address.

    Many A2A agents are deployed with agent cards that contain internal URLs
    like "http://0.0.0.0:8001/" or "http://localhost:8000/". This function
    replaces such URLs with the provided base_url.

    Args:
        agent_card: The agent card to fix
        base_url: The base URL to use as replacement

    Returns:
        The agent card with the URL fixed if necessary
    """
    card_url: Final = getattr(agent_card, "url", None)

    if card_url and is_localhost_or_internal_url(card_url):
        # Normalize base_url to ensure it ends with /
        fixed_url: Final = base_url.rstrip("/") + "/"
        agent_card.url = fixed_url

    interfaces: Final = getattr(agent_card, "supported_interfaces", None)
    if interfaces:
        interface_url: Final = getattr(interfaces[0], "url", None)
        if interface_url and is_localhost_or_internal_url(interface_url):
            interfaces[0].url = base_url.rstrip("/") + "/"

    return agent_card


class LiteLLMA2ACardResolver(_A2ACardResolver):
    """
    Custom A2A card resolver that supports multiple well-known paths.

    Extends the base A2ACardResolver to try both:
    - /.well-known/agent-card.json (standard)
    - /.well-known/agent.json (previous/alternative)
    """

    async def get_agent_card(
        self,
        relative_card_path: str | None = None,
        http_kwargs: dict[str, Any] | None = None,
    ) -> "AgentCard":
        """
        Fetch the agent card, trying multiple well-known paths.

        First tries the standard path, then falls back to the previous path.

        Args:
            relative_card_path: Optional path to the agent card endpoint.
                If None, tries both well-known paths.
            http_kwargs: Optional dictionary of keyword arguments to pass to httpx.get

        Returns:
            AgentCard from the A2A agent

        Raises:
            A2AClientHTTPError or A2AClientJSONError if both paths fail
        """
        # If a specific path is provided, use the parent implementation
        if relative_card_path is not None:
            return await super().get_agent_card(
                relative_card_path=relative_card_path,
                http_kwargs=http_kwargs,
            )

        # Try both well-known paths
        paths: Final = [
            AGENT_CARD_WELL_KNOWN_PATH,
            PREV_AGENT_CARD_WELL_KNOWN_PATH,
        ]

        last_error = None
        for path in paths:
            try:
                verbose_logger.debug("Attempting to fetch agent card from %s%s", self.base_url, path)
                return await super().get_agent_card(
                    relative_card_path=path,
                    http_kwargs=http_kwargs,
                )
            except Exception as e:
                verbose_logger.debug("Failed to fetch agent card from %s%s: %s", self.base_url, path, e)
                last_error = e
                continue

        # If we get here, all paths failed - re-raise the last error
        if last_error is not None:
            raise last_error

        # This shouldn't happen, but just in case
        raise Exception(f"Failed to fetch agent card from {self.base_url}. Tried paths: {', '.join(paths)}")
