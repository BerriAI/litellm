"""
Base configuration for A2A protocol providers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping


class BaseA2AProviderConfig(ABC):
    """
    Base configuration class for A2A protocol providers.

    Each provider should implement this interface to define how to handle
    A2A requests for their specific agent type.
    """

    def supports_jsonrpc(self) -> bool:
        """Return True when this provider supports arbitrary A2A JSON-RPC methods."""
        return False

    async def handle_jsonrpc(
        self,
        method: str,
        request_id: str | int | None,
        params: Mapping[str, object],
        api_base: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        agent_extra_headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, object]:
        """Handle a non-streaming A2A JSON-RPC request."""
        raise NotImplementedError(f"{type(self).__name__} does not support generic JSON-RPC method {method!r}")

    async def handle_jsonrpc_streaming(
        self,
        method: str,
        request_id: str | int | None,
        params: Mapping[str, object],
        api_base: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        agent_extra_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[Mapping[str, object]]:
        """Handle a streaming A2A JSON-RPC request."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming JSON-RPC method {method!r}")
        if False:  # pragma: no cover
            yield {}

    @abstractmethod
    async def handle_non_streaming(
        self,
        request_id: str,
        params: dict[str, object],
        api_base: str | None = None,
        **kwargs,
    ) -> dict[str, object]:
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
        params: dict[str, object],
        api_base: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, object]]:
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
