"""
Custom A2A Card Resolver for LiteLLM.

Extends the A2A SDK's card resolver to support multiple well-known paths
and fixes agent card URL issues where the card contains internal/localhost URLs.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from a2a.types import AgentCard

# Runtime imports with availability check
_A2ACardResolver: Any = None
AGENT_CARD_WELL_KNOWN_PATH: str = "/.well-known/agent-card.json"
PREV_AGENT_CARD_WELL_KNOWN_PATH: str = "/.well-known/agent.json"

try:
    from a2a.client import A2ACardResolver as _A2ACardResolver  # type: ignore[no-redef]
    from a2a.utils.constants import (  # type: ignore[no-redef]
        AGENT_CARD_WELL_KNOWN_PATH,
        PREV_AGENT_CARD_WELL_KNOWN_PATH,
    )
except ImportError:
    pass

# Patterns that indicate a localhost/internal URL that should be replaced
# with the original base_url. This is a common misconfiguration where
# developers deploy agents with development URLs in their agent cards.
LOCALHOST_URL_PATTERNS: List[str] = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "[::1]",  # IPv6 localhost
]


def is_localhost_or_internal_url(url: Optional[str]) -> bool:
    """
    Check if a URL is a localhost or internal URL that should be replaced.

    This detects common development URLs that are accidentally left in
    agent cards when deploying to production.

    Args:
        url: The URL to check

    Returns:
        True if the URL is localhost/internal and should be replaced

    Examples:
        >>> is_localhost_or_internal_url("http://localhost:8000/")
        True
        >>> is_localhost_or_internal_url("http://0.0.0.0:8001/")
        True
        >>> is_localhost_or_internal_url("https://my-agent.example.com/")
        False
        >>> is_localhost_or_internal_url(None)
        False
    """
    if not url:
        return False

    url_lower = url.lower()

    return any(pattern in url_lower for pattern in LOCALHOST_URL_PATTERNS)


class LiteLLMA2ACardResolver(_A2ACardResolver):  # type: ignore[misc]
    """
    Custom A2A card resolver that supports multiple well-known paths.
    
    Extends the base A2ACardResolver to:
    - Try both /.well-known/agent-card.json (standard) and /.well-known/agent.json (previous/alternative)
    - Fix agent card URLs that contain localhost/internal addresses by replacing them with the original base_url
    
    This fixes a common issue where agent cards are deployed with internal URLs
    (e.g., "http://0.0.0.0:8001/") that don't work when accessed from external clients.
    """
    
    def _fix_agent_card_url(self, agent_card: "AgentCard") -> "AgentCard":
        """
        Fix the agent card URL if it contains a localhost/internal address.
        
        Many A2A agents are deployed with agent cards that contain internal URLs
        like "http://0.0.0.0:8001/" or "http://localhost:8000/". This method
        replaces such URLs with the original base_url used to fetch the card.
        
        Args:
            agent_card: The agent card to fix
            
        Returns:
            The agent card with the URL fixed if necessary
        """
        card_url = getattr(agent_card, "url", None)
        
        if card_url and is_localhost_or_internal_url(card_url):
            # Normalize base_url to ensure it ends with /
            fixed_url = self.base_url.rstrip("/") + "/"
            
            verbose_logger.warning(
                f"Agent card contains localhost/internal URL '{card_url}'. "
                f"Replacing with base_url '{fixed_url}'. "
                f"Consider updating the agent's configuration to use the correct public URL."
            )
            
            # Create a new agent card with the fixed URL
            # We need to handle this carefully to preserve all other fields
            agent_card.url = fixed_url
        
        return agent_card
    
    async def get_agent_card(
        self,
        relative_card_path: Optional[str] = None,
        http_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "AgentCard":
        """
        Fetch the agent card, trying multiple well-known paths.
        
        First tries the standard path, then falls back to the previous path.
        Also fixes agent card URLs that contain localhost/internal addresses.
        
        Args:
            relative_card_path: Optional path to the agent card endpoint.
                If None, tries both well-known paths.
            http_kwargs: Optional dictionary of keyword arguments to pass to httpx.get
            
        Returns:
            AgentCard from the A2A agent (with URL fixed if necessary)
            
        Raises:
            A2AClientHTTPError or A2AClientJSONError if both paths fail
        """
        # If a specific path is provided, use the parent implementation
        if relative_card_path is not None:
            agent_card = await super().get_agent_card(
                relative_card_path=relative_card_path,
                http_kwargs=http_kwargs,
            )
            return self._fix_agent_card_url(agent_card)
        
        # Try both well-known paths
        paths = [
            AGENT_CARD_WELL_KNOWN_PATH,
            PREV_AGENT_CARD_WELL_KNOWN_PATH,
        ]
        
        last_error = None
        for path in paths:
            try:
                verbose_logger.debug(
                    f"Attempting to fetch agent card from {self.base_url}{path}"
                )
                agent_card = await super().get_agent_card(
                    relative_card_path=path,
                    http_kwargs=http_kwargs,
                )
                return self._fix_agent_card_url(agent_card)
            except Exception as e:
                verbose_logger.debug(
                    f"Failed to fetch agent card from {self.base_url}{path}: {e}"
                )
                last_error = e
                continue
        
        # If we get here, all paths failed - re-raise the last error
        if last_error is not None:
            raise last_error
        
        # This shouldn't happen, but just in case
        raise Exception(
            f"Failed to fetch agent card from {self.base_url}. "
            f"Tried paths: {', '.join(paths)}"
        )
