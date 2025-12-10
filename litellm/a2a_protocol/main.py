"""
LiteLLM A2A SDK functions.

Provides standalone functions with @client decorator for LiteLLM logging integration.
"""

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Coroutine, Dict, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.a2a_protocol.utils import A2ARequestUtils
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.agents import LiteLLMSendMessageResponse
from litellm.utils import client

if TYPE_CHECKING:
    from a2a.client import A2AClient as A2AClientType
    from a2a.types import (
        AgentCard,
        SendMessageRequest,
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


def _set_usage_on_logging_obj(
    kwargs: Dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """
    Set usage on litellm_logging_obj for standard logging payload.

    Args:
        kwargs: The kwargs dict containing litellm_logging_obj
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
    """
    litellm_logging_obj = kwargs.get("litellm_logging_obj")
    if litellm_logging_obj is not None:
        usage = litellm.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        litellm_logging_obj.model_call_details["usage"] = usage


def _set_agent_id_on_logging_obj(
    kwargs: Dict[str, Any],
    agent_id: Optional[str],
) -> None:
    """
    Set agent_id on litellm_logging_obj metadata for SpendLogs tracking.

    Args:
        kwargs: The kwargs dict containing litellm_logging_obj
        agent_id: The A2A agent ID
    """
    if agent_id is None:
        return

    litellm_logging_obj = kwargs.get("litellm_logging_obj")
    if litellm_logging_obj is not None:
        # Set agent_id in metadata for spend logs
        if "metadata" not in litellm_logging_obj.model_call_details:
            litellm_logging_obj.model_call_details["metadata"] = {}
        litellm_logging_obj.model_call_details["metadata"]["agent_id"] = agent_id


def _get_a2a_model_info(a2a_client: Any, kwargs: Dict[str, Any]) -> str:
    """
    Extract agent info and set model/custom_llm_provider for cost tracking.

    Sets model info on the litellm_logging_obj if available.
    Returns the agent name for logging.
    """
    agent_name = "unknown"

    # Try to get agent card from our stored attribute first, then fallback to SDK attribute
    agent_card = getattr(a2a_client, "_litellm_agent_card", None)
    if agent_card is None:
        agent_card = getattr(a2a_client, "agent_card", None)

    if agent_card is not None:
        agent_name = getattr(agent_card, "name", "unknown") or "unknown"

    # Build model string
    model = f"a2a_agent/{agent_name}"
    custom_llm_provider = "a2a_agent"

    # Set on litellm_logging_obj if available (for standard logging payload)
    litellm_logging_obj = kwargs.get("litellm_logging_obj")
    if litellm_logging_obj is not None:
        litellm_logging_obj.model = model
        litellm_logging_obj.custom_llm_provider = custom_llm_provider
        litellm_logging_obj.model_call_details["model"] = model
        litellm_logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider

    return agent_name


@client
async def asend_message(
    a2a_client: Optional["A2AClientType"] = None,
    request: Optional["SendMessageRequest"] = None,
    api_base: Optional[str] = None,
    litellm_params: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
    **kwargs: Any,
) -> LiteLLMSendMessageResponse:
    """
    Async: Send a message to an A2A agent.

    Uses the @client decorator for LiteLLM logging and tracking.
    If litellm_params contains custom_llm_provider, routes through the completion bridge.

    Args:
        a2a_client: An initialized a2a.client.A2AClient instance (optional if using completion bridge)
        request: SendMessageRequest from a2a.types (optional if using completion bridge with api_base)
        api_base: API base URL (required for completion bridge, optional for standard A2A)
        litellm_params: Optional dict with custom_llm_provider, model, etc. for completion bridge
        agent_id: Optional agent ID for tracking in SpendLogs
        **kwargs: Additional arguments passed to the client decorator

    Returns:
        LiteLLMSendMessageResponse (wraps a2a SendMessageResponse with _hidden_params)

    Example (standard A2A):
        ```python
        from litellm.a2a_protocol import asend_message, create_a2a_client
        from a2a.types import SendMessageRequest, MessageSendParams
        from uuid import uuid4

        a2a_client = await create_a2a_client(base_url="http://localhost:10001")
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={"role": "user", "parts": [{"kind": "text", "text": "Hello!"}], "messageId": uuid4().hex}
            )
        )
        response = await asend_message(a2a_client=a2a_client, request=request)
        ```

    Example (completion bridge with LangGraph):
        ```python
        from litellm.a2a_protocol import asend_message
        from a2a.types import SendMessageRequest, MessageSendParams
        from uuid import uuid4

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={"role": "user", "parts": [{"kind": "text", "text": "Hello!"}], "messageId": uuid4().hex}
            )
        )
        response = await asend_message(
            request=request,
            api_base="http://localhost:2024",
            litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
        )
        ```
    """
    litellm_params = litellm_params or {}
    custom_llm_provider = litellm_params.get("custom_llm_provider")

    # Route through completion bridge if custom_llm_provider is set
    if custom_llm_provider:
        if request is None:
            raise ValueError("request is required for completion bridge")
        if api_base is None:
            raise ValueError("api_base is required for completion bridge")

        verbose_logger.info(
            f"A2A using completion bridge: provider={custom_llm_provider}, api_base={api_base}"
        )

        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2ACompletionBridgeHandler,
        )

        # Extract params from request
        params = request.params.model_dump(mode="json") if hasattr(request.params, "model_dump") else dict(request.params)

        response_dict = await A2ACompletionBridgeHandler.handle_non_streaming(
            request_id=str(request.id),
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
        )

        # Convert to LiteLLMSendMessageResponse
        return LiteLLMSendMessageResponse.from_dict(response_dict)

    # Standard A2A client flow
    if a2a_client is None:
        raise ValueError("a2a_client is required for standard A2A flow")
    if request is None:
        raise ValueError("request is required")

    agent_name = _get_a2a_model_info(a2a_client, kwargs)

    verbose_logger.info(f"A2A send_message request_id={request.id}, agent={agent_name}")

    a2a_response = await a2a_client.send_message(request)

    verbose_logger.info(f"A2A send_message completed, request_id={request.id}")

    # Wrap in LiteLLM response type for _hidden_params support
    response = LiteLLMSendMessageResponse.from_a2a_response(a2a_response)

    # Calculate token usage from request and response
    response_dict = a2a_response.model_dump(mode="json", exclude_none=True)
    prompt_tokens, completion_tokens, _ = A2ARequestUtils.calculate_usage_from_request_response(
        request=request,
        response_dict=response_dict,
    )

    # Set usage on logging obj for standard logging payload
    _set_usage_on_logging_obj(
        kwargs=kwargs,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    # Set agent_id on logging obj for SpendLogs tracking
    _set_agent_id_on_logging_obj(kwargs=kwargs, agent_id=agent_id)

    return response


@client
def send_message(
    a2a_client: "A2AClientType",
    request: "SendMessageRequest",
    **kwargs: Any,
) -> Union[LiteLLMSendMessageResponse, Coroutine[Any, Any, LiteLLMSendMessageResponse]]:
    """
    Sync: Send a message to an A2A agent.

    Uses the @client decorator for LiteLLM logging and tracking.

    Args:
        a2a_client: An initialized a2a.client.A2AClient instance
        request: SendMessageRequest from a2a.types
        **kwargs: Additional arguments passed to the client decorator

    Returns:
        LiteLLMSendMessageResponse (wraps a2a SendMessageResponse with _hidden_params)
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
    a2a_client: Optional["A2AClientType"] = None,
    request: Optional["SendStreamingMessageRequest"] = None,
    api_base: Optional[str] = None,
    litellm_params: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
) -> AsyncIterator[Any]:
    """
    Async: Send a streaming message to an A2A agent.

    If litellm_params contains custom_llm_provider, routes through the completion bridge.

    Args:
        a2a_client: An initialized a2a.client.A2AClient instance (optional if using completion bridge)
        request: SendStreamingMessageRequest from a2a.types
        api_base: API base URL (required for completion bridge)
        litellm_params: Optional dict with custom_llm_provider, model, etc. for completion bridge
        agent_id: Optional agent ID for tracking in SpendLogs (currently unused in streaming)

    Yields:
        SendStreamingMessageResponse chunks from the agent

    Example (completion bridge with LangGraph):
        ```python
        from litellm.a2a_protocol import asend_message_streaming
        from a2a.types import SendStreamingMessageRequest, MessageSendParams
        from uuid import uuid4

        request = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={"role": "user", "parts": [{"kind": "text", "text": "Hello!"}], "messageId": uuid4().hex}
            )
        )
        async for chunk in asend_message_streaming(
            request=request,
            api_base="http://localhost:2024",
            litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
        ):
            print(chunk)
        ```
    """
    litellm_params = litellm_params or {}
    custom_llm_provider = litellm_params.get("custom_llm_provider")

    # Route through completion bridge if custom_llm_provider is set
    if custom_llm_provider:
        if request is None:
            raise ValueError("request is required for completion bridge")
        if api_base is None:
            raise ValueError("api_base is required for completion bridge")

        verbose_logger.info(
            f"A2A streaming using completion bridge: provider={custom_llm_provider}"
        )

        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2ACompletionBridgeHandler,
        )

        # Extract params from request
        params = request.params.model_dump(mode="json") if hasattr(request.params, "model_dump") else dict(request.params)

        async for chunk in A2ACompletionBridgeHandler.handle_streaming(
            request_id=str(request.id),
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
        ):
            yield chunk
        return

    # Standard A2A client flow
    if a2a_client is None:
        raise ValueError("a2a_client is required for standard A2A flow")
    if request is None:
        raise ValueError("request is required")

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
        from litellm.a2a_protocol import create_a2a_client, asend_message

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

    # Create A2A client
    a2a_client = _A2AClient(
        httpx_client=httpx_client,
        agent_card=agent_card,
    )

    # Store agent_card on client for later retrieval (SDK doesn't expose it)
    a2a_client._litellm_agent_card = agent_card  # type: ignore[attr-defined]

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


