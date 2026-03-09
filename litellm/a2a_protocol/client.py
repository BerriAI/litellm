"""
LiteLLM A2A Client class.

Provides a class-based interface for A2A agent invocation.
"""

from typing import TYPE_CHECKING, AsyncIterator, Dict, Optional

from litellm.types.agents import LiteLLMSendMessageResponse

if TYPE_CHECKING:
    from a2a.client import A2AClient as A2AClientType
    from a2a.types import (
        AgentCard,
        SendMessageRequest,
        SendStreamingMessageRequest,
        SendStreamingMessageResponse,
    )


class A2AClient:
    """
    LiteLLM wrapper for A2A agent invocation.

    Creates the underlying A2A client once on first use and reuses it.

    Example:
        ```python
        from litellm.a2a_protocol import A2AClient
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
    ):
        """
        Initialize the A2A client wrapper.

        Args:
            base_url: The base URL of the A2A agent (e.g., "http://localhost:10001")
            timeout: Request timeout in seconds (default: 60.0)
            extra_headers: Optional additional headers to include in requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.extra_headers = extra_headers
        self._a2a_client: Optional["A2AClientType"] = None

    async def _get_client(self) -> "A2AClientType":
        """Get or create the underlying A2A client."""
        if self._a2a_client is None:
            from litellm.a2a_protocol.main import create_a2a_client

            self._a2a_client = await create_a2a_client(
                base_url=self.base_url,
                timeout=self.timeout,
                extra_headers=self.extra_headers,
            )
        return self._a2a_client

    async def get_agent_card(self) -> "AgentCard":
        """Fetch the agent card from the server."""
        from litellm.a2a_protocol.main import aget_agent_card

        return await aget_agent_card(
            base_url=self.base_url,
            timeout=self.timeout,
            extra_headers=self.extra_headers,
        )

    async def send_message(
        self, request: "SendMessageRequest"
    ) -> LiteLLMSendMessageResponse:
        """Send a message to the A2A agent."""
        from litellm.a2a_protocol.main import asend_message

        a2a_client = await self._get_client()
        return await asend_message(a2a_client=a2a_client, request=request)

    async def send_message_streaming(
        self, request: "SendStreamingMessageRequest"
    ) -> AsyncIterator["SendStreamingMessageResponse"]:
        """Send a streaming message to the A2A agent."""
        from litellm.a2a_protocol.main import asend_message_streaming

        a2a_client = await self._get_client()
        async for chunk in asend_message_streaming(a2a_client=a2a_client, request=request):
            yield chunk
