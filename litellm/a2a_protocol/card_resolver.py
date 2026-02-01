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


_BaseA2ACardResolver: Any = _A2ACardResolver if _A2ACardResolver is not None else object


class LiteLLMA2ACardResolver(_BaseA2ACardResolver):  # type: ignore[misc]
    """Custom A2A card resolver that supports multiple well-known paths.

    If the optional `a2a` SDK is not installed, this class remains import-safe and
    raises an ImportError when instantiated/used.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _A2ACardResolver is None:
            raise ImportError("A2A SDK is not installed. Install `a2a-sdk` to use A2A agent card resolution.")
        super().__init__(*args, **kwargs)

    async def get_agent_card(
        self,
        relative_card_path: Optional[str] = None,
        http_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "AgentCard":
        if _A2ACardResolver is None:
            raise ImportError("A2A SDK is not installed. Install `a2a-sdk` to use A2A agent card resolution.")

        # If a specific path is provided, delegate to the SDK resolver.
        if relative_card_path is not None:
            return await _A2ACardResolver.get_agent_card(  # type: ignore[no-any-return]
                self,
                relative_card_path=relative_card_path,
                http_kwargs=http_kwargs,
            )

        base_url = getattr(self, "base_url", "")

        # Try both well-known paths
        paths = [
            AGENT_CARD_WELL_KNOWN_PATH,
            PREV_AGENT_CARD_WELL_KNOWN_PATH,
        ]

        last_error = None
        for path in paths:
            try:
                verbose_logger.debug(f"Attempting to fetch agent card from {base_url}{path}")
                return await _A2ACardResolver.get_agent_card(  # type: ignore[no-any-return]
                    self,
                    relative_card_path=path,
                    http_kwargs=http_kwargs,
                )
            except Exception as e:
                verbose_logger.debug(f"Failed to fetch agent card from {base_url}{path}: {e}")
                last_error = e
                continue

        # If we get here, all paths failed - re-raise the last error
        if last_error is not None:
            raise last_error

        # This shouldn't happen, but just in case
        raise Exception(f"Failed to fetch agent card from {base_url}. Tried paths: {', '.join(paths)}")
