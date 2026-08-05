"""
Base configuration for A2A protocol providers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class BaseA2AProviderConfig(ABC):
    """
    Base configuration class for A2A protocol providers.

    Each provider should implement this interface to define how to handle
    A2A requests for their specific agent type.
    """

    @abstractmethod
    async def handle_non_streaming(
        self,
        request_id: str,
        params: dict[str, Any],
        api_base: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Handle non-streaming A2A request.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            api_base: Base URL of the agent
            **kwargs: Additional provider-specific parameters

        Returns:
            A2A SendMessageResponse dict
        """

    @abstractmethod
    async def handle_streaming(
        self,
        request_id: str,
        params: dict[str, Any],
        api_base: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Handle streaming A2A request.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            api_base: Base URL of the agent
            **kwargs: Additional provider-specific parameters

        Yields:
            A2A streaming response events
        """
        # This is an abstract method - subclasses must implement
        # The yield is here to make this a generator function
        if False:  # pragma: no cover
            yield {}
