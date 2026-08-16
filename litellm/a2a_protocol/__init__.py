"""
LiteLLM A2A - Wrapper for invoking A2A protocol agents.

This module provides a thin wrapper around the official `a2a` SDK that:
- Handles httpx client creation and agent card resolution
- Adds LiteLLM logging via @client decorator
- Matches the A2A SDK interface (SendMessageRequest, SendMessageResponse, etc.)

Example usage (standalone functions with @client decorator):
    ```python
    from litellm.a2a_protocol import asend_message
    from a2a.types import SendMessageRequest, MessageSendParams
    from uuid import uuid4

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
    response = await asend_message(
        base_url="http://localhost:10001",
        request=request,
    )
    print(response.model_dump(mode='json', exclude_none=True))
    ```

Example usage (class-based):
    ```python
    from litellm.a2a_protocol import A2AClient

    client = A2AClient(base_url="http://localhost:10001")
    response = await client.send_message(request)
    ```
"""

from litellm.a2a_protocol.client import A2AClient
from litellm.a2a_protocol.exceptions import (
    A2AAgentCardError,
    A2AConnectionError,
    A2AError,
    A2ALocalhostURLError,
)
from litellm.a2a_protocol.main import (
    aget_agent_card,
    asend_message,
    asend_message_streaming,
    create_a2a_client,
    send_message,
)
from litellm.types.agents import LiteLLMSendMessageResponse

__all__ = [
    # Client
    "A2AClient",
    # Functions
    "asend_message",
    "send_message",
    "asend_message_streaming",
    "aget_agent_card",
    "create_a2a_client",
    # Response types
    "LiteLLMSendMessageResponse",
    # Exceptions
    "A2AError",
    "A2AConnectionError",
    "A2AAgentCardError",
    "A2ALocalhostURLError",
]
