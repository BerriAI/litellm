"""
Custom A2A Card Resolver for LiteLLM.

Extends the A2A SDK's card resolver to support multiple well-known paths.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

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


class LiteLLMA2ACardResolver(_A2ACardResolver):  # type: ignore[misc]
    """
    Custom A2A card resolver that supports multiple well-known paths.
    
    Extends the base A2ACardResolver to try both:
    - /.well-known/agent-card.json (standard)
    - /.well-known/agent.json (previous/alternative)
    """
    
    async def get_agent_card(
        self,
        relative_card_path: Optional[str] = None,
        http_kwargs: Optional[Dict[str, Any]] = None,
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
                return await super().get_agent_card(
                    relative_card_path=path,
                    http_kwargs=http_kwargs,
                )
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
