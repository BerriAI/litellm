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


_A2AResolverBase = _A2ACardResolver or object


class LiteLLMA2ACardResolver(_A2AResolverBase):  # type: ignore[misc]
    """Custom A2A card resolver that supports multiple well-known paths.

    The upstream `a2a` SDK is an *optional* dependency. This resolver must not
    break LiteLLM imports when `a2a` is absent.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        if _A2ACardResolver is None:
            raise ImportError(
                "The optional 'a2a' dependency is required for A2A agent card "
                "resolution. Install it to use A2A features."
            )
        super().__init__(*args, **kwargs)

    async def get_agent_card(
        self,
        relative_card_path: Optional[str] = None,
        http_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "AgentCard":
        """Fetch the agent card, trying multiple well-known paths."""
        if _A2ACardResolver is None:
            raise ImportError(
                "The optional 'a2a' dependency is required for A2A agent card "
                "resolution. Install it to use A2A features."
            )

        # If a specific path is provided, use the parent implementation
        if relative_card_path is not None:
            return await super().get_agent_card(
                relative_card_path=relative_card_path,
                http_kwargs=http_kwargs,
            )

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

        if last_error is not None:
            raise last_error

        raise Exception(
            f"Failed to fetch agent card from {self.base_url}. Tried paths: {', '.join(paths)}"
        )
