"""
LiteLLM A2A SDK functions.

Provides standalone functions with @client decorator for LiteLLM logging integration.
"""

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Coroutine, Dict, Optional, Union

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.utils import client

if TYPE_CHECKING:
    from a2a.client import A2AClient as A2AClientType
    from a2a.types import (
        AgentCard,
        SendMessageRequest,
        SendMessageResponse,
        SendStreamingMessageRequest,
        SendStreamingMessageResponse,
    )

# Runtime imports with availability check
A2A_SDK_AVAILABLE = False
A2ACardResolver: Any = None
_A2AClient: Any = None

try:
    from a2a.client import A2ACardResolver  # type: ignore[no-redef]
    from a2a.client import A2AClient as _A2AClient  # type: ignore[no-redef]

    A2A_SDK_AVAILABLE = True
except ImportError:
    pass


@client
async def asend_message(
    a2a_client: "A2AClientType",
    request: "SendMessageRequest",
    **kwargs: Any,
) -> "SendMessageResponse":
    """
    Async: Send a message to an A2A agent.

    Uses the @client decorator for LiteLLM logging and tracking.

    Args:
        a2a_client: An initialized a2a.client.A2AClient instance
        request: SendMessageRequest from a2a.types
        **kwargs: Additional arguments passed to the client decorator

    Returns:
        SendMessageResponse from the agent

    Example:
        ```python
        from litellm.a2a import asend_message, create_a2a_client
        from a2a.types import SendMessageRequest, MessageSendParams
        from uuid import uuid4

        # Create client once
        a2a_client = await create_a2a_client(base_url="http://localhost:10001")

        # Use it for multiple requests
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
        response = await asend_message(a2a_client=a2a_client, request=request)
        ```
    """
    verbose_logger.info(f"A2A send_message request_id={request.id}")

    response = await a2a_client.send_message(request)

    verbose_logger.info(f"A2A send_message completed, request_id={request.id}")

    return response


@client
def send_message(
    a2a_client: "A2AClientType",
    request: "SendMessageRequest",
    **kwargs: Any,
) -> Union["SendMessageResponse", Coroutine[Any, Any, "SendMessageResponse"]]:
    """
    Sync: Send a message to an A2A agent.

    Uses the @client decorator for LiteLLM logging and tracking.

    Args:
        a2a_client: An initialized a2a.client.A2AClient instance
        request: SendMessageRequest from a2a.types
        **kwargs: Additional arguments passed to the client decorator

    Returns:
        SendMessageResponse from the agent
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        return asend_message(a2a_client=a2a_client, request=request, **kwargs)
    else:
        return asyncio.run(asend_message(a2a_client=a2a_client, request=request, **kwargs))


async def asend_message_streaming(
    a2a_client: "A2AClientType",
    request: "SendStreamingMessageRequest",
) -> AsyncIterator["SendStreamingMessageResponse"]:
    """
    Async: Send a streaming message to an A2A agent.

    Args:
        a2a_client: An initialized a2a.client.A2AClient instance
        request: SendStreamingMessageRequest from a2a.types

    Yields:
        SendStreamingMessageResponse chunks from the agent
    """
    verbose_logger.info(f"A2A send_message_streaming request_id={request.id}")

    stream = a2a_client.send_message_streaming(request)

    chunk_count = 0
    async for chunk in stream:
        chunk_count += 1
        yield chunk

    verbose_logger.info(
        f"A2A send_message_streaming completed, request_id={request.id}, chunks={chunk_count}"
    )


async def create_a2a_client(
    base_url: str,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> "A2AClientType":
    """
    Create an A2A client for the given agent URL.

    This resolves the agent card and returns a ready-to-use A2A client.
    The client can be reused for multiple requests.

    Args:
        base_url: The base URL of the A2A agent (e.g., "http://localhost:10001")
        timeout: Request timeout in seconds (default: 60.0)
        extra_headers: Optional additional headers to include in requests

    Returns:
        An initialized a2a.client.A2AClient instance

    Example:
        ```python
        from litellm.a2a import create_a2a_client, asend_message

        # Create client once
        client = await create_a2a_client(base_url="http://localhost:10001")

        # Reuse for multiple requests
        response1 = await asend_message(a2a_client=client, request=request1)
        response2 = await asend_message(a2a_client=client, request=request2)
        ```
    """
    if not A2A_SDK_AVAILABLE:
        raise ImportError(
            "The 'a2a' package is required for A2A agent invocation. "
            "Install it with: pip install a2a"
        )

    verbose_logger.info(f"Creating A2A client for {base_url}")

    # Use LiteLLM's cached httpx client
    http_handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.A2A,
        params={"timeout": timeout},
    )
    httpx_client = http_handler.client

    # Resolve agent card
    resolver = A2ACardResolver(
        httpx_client=httpx_client,
        base_url=base_url,
    )
    agent_card = await resolver.get_agent_card()

    verbose_logger.debug(
        f"Resolved agent card: {agent_card.name if hasattr(agent_card, 'name') else 'unknown'}"
    )

    # Create and return A2A client
    a2a_client = _A2AClient(
        httpx_client=httpx_client,
        agent_card=agent_card,
    )

    verbose_logger.info(f"A2A client created for {base_url}")

    return a2a_client


async def aget_agent_card(
    base_url: str,
    timeout: float = 60.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> "AgentCard":
    """
    Fetch the agent card from an A2A agent.

    Args:
        base_url: The base URL of the A2A agent (e.g., "http://localhost:10001")
        timeout: Request timeout in seconds (default: 60.0)
        extra_headers: Optional additional headers to include in requests

    Returns:
        AgentCard from the A2A agent
    """
    if not A2A_SDK_AVAILABLE:
        raise ImportError(
            "The 'a2a' package is required for A2A agent invocation. "
            "Install it with: pip install a2a"
        )

    verbose_logger.info(f"Fetching agent card from {base_url}")

    # Use LiteLLM's cached httpx client
    http_handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.A2A,
        params={"timeout": timeout},
    )
    httpx_client = http_handler.client

    resolver = A2ACardResolver(
        httpx_client=httpx_client,
        base_url=base_url,
    )
    agent_card = await resolver.get_agent_card()

    verbose_logger.info(
        f"Fetched agent card: {agent_card.name if hasattr(agent_card, 'name') else 'unknown'}"
    )
    return agent_card
