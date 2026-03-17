"""
A2A Protocol Exception Mapping Utils.

Maps A2A SDK exceptions to LiteLLM A2A exception types.
"""

from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_logger
from litellm.a2a_protocol.card_resolver import (
    fix_agent_card_url,
    is_localhost_or_internal_url,
)
from litellm.a2a_protocol.exceptions import (
    A2AAgentCardError,
    A2AConnectionError,
    A2AError,
    A2ALocalhostURLError,
)
from litellm.constants import CONNECTION_ERROR_PATTERNS

if TYPE_CHECKING:
    from a2a.client import A2AClient as A2AClientType


# Runtime import
A2A_SDK_AVAILABLE = False
try:
    from a2a.client import A2AClient as _A2AClient  # type: ignore[no-redef]

    A2A_SDK_AVAILABLE = True
except ImportError:
    _A2AClient = None  # type: ignore[assignment, misc]


class A2AExceptionCheckers:
    """
    Helper class for checking various A2A error conditions.
    """

    @staticmethod
    def is_connection_error(error_str: str) -> bool:
        """
        Check if an error string indicates a connection error.

        Args:
            error_str: The error string to check

        Returns:
            True if the error indicates a connection issue
        """
        if not isinstance(error_str, str):
            return False

        error_str_lower = error_str.lower()
        return any(pattern in error_str_lower for pattern in CONNECTION_ERROR_PATTERNS)

    @staticmethod
    def is_localhost_url(url: Optional[str]) -> bool:
        """
        Check if a URL is a localhost/internal URL.

        Args:
            url: The URL to check

        Returns:
            True if the URL is localhost/internal
        """
        return is_localhost_or_internal_url(url)

    @staticmethod
    def is_agent_card_error(error_str: str) -> bool:
        """
        Check if an error string indicates an agent card error.

        Args:
            error_str: The error string to check

        Returns:
            True if the error is related to agent card fetching/parsing
        """
        if not isinstance(error_str, str):
            return False

        error_str_lower = error_str.lower()
        agent_card_patterns = [
            "agent card",
            "agent-card",
            ".well-known",
            "card not found",
            "invalid agent",
        ]
        return any(pattern in error_str_lower for pattern in agent_card_patterns)


def map_a2a_exception(
    original_exception: Exception,
    card_url: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> Exception:
    """
    Map an A2A SDK exception to a LiteLLM A2A exception type.

    Args:
        original_exception: The original exception from the A2A SDK
        card_url: The URL from the agent card (if available)
        api_base: The original API base URL
        model: The model/agent name

    Returns:
        A mapped LiteLLM A2A exception

    Raises:
        A2ALocalhostURLError: If the error is a connection error to a localhost URL
        A2AConnectionError: If the error is a general connection error
        A2AAgentCardError: If the error is related to agent card issues
        A2AError: For other A2A-related errors
    """
    error_str = str(original_exception)

    # Check for localhost URL connection error (special case - retryable)
    if (
        card_url
        and api_base
        and A2AExceptionCheckers.is_localhost_url(card_url)
        and A2AExceptionCheckers.is_connection_error(error_str)
    ):
        raise A2ALocalhostURLError(
            localhost_url=card_url,
            base_url=api_base,
            original_error=original_exception,
            model=model,
        )

    # Check for agent card errors
    if A2AExceptionCheckers.is_agent_card_error(error_str):
        raise A2AAgentCardError(
            message=error_str,
            url=api_base,
            model=model,
        )

    # Check for general connection errors
    if A2AExceptionCheckers.is_connection_error(error_str):
        raise A2AConnectionError(
            message=error_str,
            url=card_url or api_base,
            model=model,
        )

    # Default: wrap in generic A2AError
    raise A2AError(
        message=error_str,
        model=model,
    )


def handle_a2a_localhost_retry(
    error: A2ALocalhostURLError,
    agent_card: Any,
    a2a_client: "A2AClientType",
    is_streaming: bool = False,
) -> "A2AClientType":
    """
    Handle A2ALocalhostURLError by fixing the URL and creating a new client.

    This is called when we catch an A2ALocalhostURLError and want to retry
    with the corrected URL.

    Args:
        error: The localhost URL error
        agent_card: The agent card object to fix
        a2a_client: The current A2A client
        is_streaming: Whether this is a streaming request (for logging)

    Returns:
        A new A2A client with the fixed URL

    Raises:
        ImportError: If the A2A SDK is not installed
    """
    if not A2A_SDK_AVAILABLE or _A2AClient is None:
        raise ImportError(
            "A2A SDK is required for localhost retry handling. "
            "Install it with: pip install a2a"
        )

    request_type = "streaming " if is_streaming else ""
    verbose_logger.warning(
        f"A2A {request_type}request to '{error.localhost_url}' failed: {error.original_error}. "
        f"Agent card contains localhost/internal URL. "
        f"Retrying with base_url '{error.base_url}'."
    )

    # Fix the agent card URL
    fix_agent_card_url(agent_card, error.base_url)

    # Create a new client with the fixed agent card (transport caches URL)
    return _A2AClient(
        httpx_client=a2a_client._transport.httpx_client,  # type: ignore[union-attr]
        agent_card=agent_card,
    )
