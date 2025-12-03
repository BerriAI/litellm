"""
LiteLLM A2A Client class.

Provides a class-based interface for A2A agent invocation.
"""

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

if TYPE_CHECKING:
    from a2a.types import (
        AgentCard,
        SendMessageRequest,
        SendMessageResponse,
        SendStreamingMessageRequest,
        SendStreamingMessageResponse,
    )

# Runtime imports with availability check
A2A_SDK_AVAILABLE = False

try:
    from a2a.client import A2ACardResolver
    from a2a.client import A2AClient as _A2AClient

    A2A_SDK_AVAILABLE = True
except ImportError:
    pass


class A2AClient:
    """
    LiteLLM wrapper for A2A agent invocation.

    Convenience class that wraps the standalone functions.
    For logging/tracking integration, use the standalone functions in main.py.

    Example:
        ```python
        from litellm.a2a import A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        from uuid import uuid4

        client = A2AClient(base_url="http://localhost:10001")

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello!"}],
                    "messageId": uuid4().hex,
                }
            )
        )
        response = await client.send_message(request)
        ```
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        ssl_verify: Optional[bool] = None,
    ):
        """
        Initialize the A2A client.

        Args:
            base_url: The base URL of the A2A agent (e.g., "http://localhost:10001")
            timeout: Request timeout in seconds (default: 60.0)
            extra_headers: Optional additional headers to include in requests
            ssl_verify: SSL verification setting (None uses default, False disables)
        """
        if not A2A_SDK_AVAILABLE:
            raise ImportError(
                "The 'a2a' package is required for A2AClient. "
                "Install it with: pip install a2a"
            )

        self.base_url = base_url
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.ssl_verify = ssl_verify

    async def get_agent_card(self) -> "AgentCard":
        """Fetch the agent card from the server."""
        from litellm.a2a.main import aget_agent_card

        return await aget_agent_card(
            base_url=self.base_url,
            timeout=self.timeout,
            extra_headers=self.extra_headers,
            ssl_verify=self.ssl_verify,
        )

    async def send_message(
        self, request: "SendMessageRequest"
    ) -> "SendMessageResponse":
        """Send a message to the A2A agent."""
        from litellm.a2a.main import asend_message

        return await asend_message(
            base_url=self.base_url,
            request=request,
            timeout=self.timeout,
            extra_headers=self.extra_headers,
            ssl_verify=self.ssl_verify,
        )

    async def send_message_streaming(
        self, request: "SendStreamingMessageRequest"
    ) -> AsyncIterator["SendStreamingMessageResponse"]:
        """Send a streaming message to the A2A agent."""
        from litellm.a2a.main import asend_message_streaming

        async for chunk in asend_message_streaming(
            base_url=self.base_url,
            request=request,
            timeout=self.timeout,
            extra_headers=self.extra_headers,
            ssl_verify=self.ssl_verify,
        ):
            yield chunk
