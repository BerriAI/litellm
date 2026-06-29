# pyright: reportUnknownArgumentType=false
# a2a-sdk (and its protobuf-generated compat conversions) ships no usable types for
# the call surface used here, so SDK calls take Unknown-typed arguments. This module
# is dedicated to the A2A SDK boundary; the rule is off file-wide instead of
# scattering per-line ignores across every SDK call.
"""
LiteLLM A2A SDK functions.

Provides standalone functions with @client decorator for LiteLLM logging integration.
"""

import asyncio
import datetime
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Optional,
    Union,
    cast,
)

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.a2a_protocol.streaming_iterator import A2AStreamingIterator
from litellm.a2a_protocol.utils import A2ARequestUtils
from litellm.constants import DEFAULT_A2A_AGENT_TIMEOUT
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.agents import LiteLLMSendMessageResponse
from litellm.utils import client

if TYPE_CHECKING:
    from a2a.client import Client as A2AClientType
    from a2a.compat.v0_3.types import (
        AgentCard,
        Message,
        SendMessageRequest,
        SendMessageResponse,
        SendStreamingMessageRequest,
        SendStreamingMessageResponse,
        Task,
    )

# Runtime imports — requires a2a-sdk>=1.1.0
A2A_SDK_AVAILABLE = False
_a2a_conversions: Any = None

try:
    from a2a.client import Client, ClientConfig, create_client
    from a2a.compat.v0_3 import conversions as _a2a_conversions
    from a2a.compat.v0_3.types import (
        Message,
        SendMessageRequest,
        SendMessageResponse,
        SendMessageSuccessResponse,
        SendStreamingMessageRequest,
        SendStreamingMessageResponse,
        Task,
    )

    A2A_SDK_AVAILABLE = True
except ImportError:
    Client = None  # type: ignore[misc, assignment]
    ClientConfig = None  # type: ignore[misc, assignment]
    create_client = None  # type: ignore[misc, assignment]

# Import our custom card resolver that supports multiple well-known paths
from litellm.a2a_protocol.card_resolver import (
    LiteLLMA2ACardResolver,
    get_agent_card_url,
)
from litellm.a2a_protocol.exception_mapping_utils import (
    handle_a2a_localhost_retry,
    map_a2a_exception,
)
from litellm.a2a_protocol.exceptions import A2ALocalhostURLError

# Use our custom resolver instead of the default A2A SDK resolver
A2ACardResolver = LiteLLMA2ACardResolver


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
    agent_id: str | None,
) -> None:
    """
    Set agent_id on litellm_logging_obj for SpendLogs tracking.

    Args:
        kwargs: The kwargs dict containing litellm_logging_obj
        agent_id: The A2A agent ID
    """
    if agent_id is None:
        return

    litellm_logging_obj = kwargs.get("litellm_logging_obj")
    if litellm_logging_obj is not None:
        # Set agent_id directly on model_call_details (same pattern as custom_llm_provider)
        litellm_logging_obj.model_call_details["agent_id"] = agent_id


def _get_a2a_model_info(a2a_client: Any, kwargs: Dict[str, Any]) -> str:
    """
    Extract agent info and set model/custom_llm_provider for cost tracking.

    Sets model info on the litellm_logging_obj if available.
    Returns the agent name for logging.
    """
    agent_name = "unknown"

    agent_card = _get_a2a_client_agent_card(a2a_client)

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


def _get_a2a_client_agent_card(a2a_client: Any) -> Optional["AgentCard"]:
    agent_card = cast(Optional["AgentCard"], getattr(a2a_client, "_litellm_agent_card", None))
    if agent_card is not None:
        return agent_card
    agent_card = cast(Optional["AgentCard"], getattr(a2a_client, "agent_card", None))
    if agent_card is not None:
        return agent_card
    return cast(Optional["AgentCard"], getattr(a2a_client, "_card", None))


async def _send_message_via_completion_bridge(
    request: "SendMessageRequest",
    custom_llm_provider: str,
    api_base: str | None,
    litellm_params: Dict[str, Any],
    agent_extra_headers: Dict[str, str] | None = None,
) -> LiteLLMSendMessageResponse:
    """
    Route a send_message through the LiteLLM completion bridge (e.g. LangGraph, Bedrock AgentCore).

    Requires request; api_base is optional for providers that derive endpoint from model.
    """
    verbose_logger.info(f"A2A using completion bridge: provider={custom_llm_provider}, api_base={api_base}")

    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    params = request.params.model_dump(mode="json") if hasattr(request.params, "model_dump") else dict(request.params)

    response_dict = await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id=str(request.id),
        params=params,
        litellm_params=litellm_params,
        api_base=api_base,
        agent_extra_headers=agent_extra_headers,
    )

    return LiteLLMSendMessageResponse.from_dict(response_dict, request_id=str(request.id))


async def _send_message(a2a_client: "A2AClientType", request: "SendMessageRequest") -> "SendMessageResponse":
    """Send a non-streaming message via a2a-sdk 1.x and return JSON-RPC response."""
    if _a2a_conversions is None:
        raise ImportError(
            "The 'a2a' package is required for A2A agent invocation. Install it with: pip install a2a-sdk"
        )

    pb_request = _a2a_conversions.to_core_send_message_request(request)
    last_event = None
    async for event in a2a_client.send_message(pb_request):
        last_event = event
    if last_event is None:
        raise RuntimeError("A2A send_message failed: no response received from agent.")

    stream_compat = _a2a_conversions.to_compat_stream_response(
        last_event,
        request_id=request.id,
    )
    result = stream_compat.result
    if not isinstance(result, (Message, Task)):
        raise RuntimeError(
            "A2A send_message failed: non-streaming message/send expects the "
            "agent's final event to be a Message or Task result."
        )
    return SendMessageResponse(
        root=SendMessageSuccessResponse(
            id=request.id,
            result=result,
        )
    )


async def _execute_a2a_send_with_retry(
    a2a_client: "A2AClientType",
    request: "SendMessageRequest",
    agent_card: Optional["AgentCard"],
    card_url: str | None,
    api_base: str | None,
    agent_name: str | None,
) -> "SendMessageResponse":
    """Send an A2A message with retry logic for localhost URL errors."""
    a2a_response = None
    for _ in range(2):  # max 2 attempts: original + 1 retry
        try:
            a2a_response = await _send_message(a2a_client, request)
            break  # success, exit retry loop
        except A2ALocalhostURLError as e:
            a2a_client = await handle_a2a_localhost_retry(
                error=e,
                agent_card=agent_card,
                a2a_client=a2a_client,
                is_streaming=False,
            )
            card_url = get_agent_card_url(agent_card) if agent_card else None
        except Exception as e:
            try:
                map_a2a_exception(e, card_url, api_base, model=agent_name)
            except A2ALocalhostURLError as localhost_err:
                a2a_client = await handle_a2a_localhost_retry(
                    error=localhost_err,
                    agent_card=agent_card,
                    a2a_client=a2a_client,
                    is_streaming=False,
                )
                card_url = get_agent_card_url(agent_card) if agent_card else None
                continue
            except Exception:
                raise
    if a2a_response is None:
        raise RuntimeError("A2A send_message failed: no response received after retry attempts.")
    return a2a_response


async def _stream_messages(
    a2a_client: "A2AClientType", request: "SendStreamingMessageRequest"
) -> AsyncIterator["SendStreamingMessageResponse"]:
    """Stream message events via a2a-sdk 1.x and yield JSON-RPC chunks."""
    if _a2a_conversions is None:
        raise ImportError(
            "The 'a2a' package is required for A2A agent invocation. Install it with: pip install a2a-sdk"
        )

    pb_request = _a2a_conversions.to_core_send_message_request(request)
    async for event in a2a_client.send_message(pb_request):
        compat_chunk = _a2a_conversions.to_compat_stream_response(
            event,
            request_id=request.id,
        )
        yield SendStreamingMessageResponse(root=compat_chunk)


async def _execute_a2a_stream_with_retry(
    a2a_client: "A2AClientType",
    request: "SendStreamingMessageRequest",
    agent_card: Optional["AgentCard"],
    card_url: str | None,
    api_base: str | None,
    agent_name: str | None,
) -> AsyncIterator["SendStreamingMessageResponse"]:
    """Stream an A2A message with retry logic for localhost URL errors."""
    response_started = False
    stream_succeeded = False
    for _ in range(2):  # max 2 attempts: original + 1 retry
        try:
            async for chunk in _stream_messages(a2a_client, request):
                response_started = True
                yield chunk
            stream_succeeded = True
            return
        except A2ALocalhostURLError as e:
            if response_started:
                raise
            a2a_client = await handle_a2a_localhost_retry(
                error=e,
                agent_card=agent_card,
                a2a_client=a2a_client,
                is_streaming=True,
            )
            card_url = get_agent_card_url(agent_card) if agent_card else None
            continue
        except Exception as e:
            if response_started:
                raise
            try:
                map_a2a_exception(e, card_url, api_base, model=agent_name)
            except A2ALocalhostURLError as localhost_err:
                a2a_client = await handle_a2a_localhost_retry(
                    error=localhost_err,
                    agent_card=agent_card,
                    a2a_client=a2a_client,
                    is_streaming=True,
                )
                card_url = get_agent_card_url(agent_card) if agent_card else None
                continue
            raise
    if not stream_succeeded:
        raise RuntimeError("A2A send_message_streaming failed: no response received after retry attempts.")


@client
async def asend_message(
    a2a_client: Optional["A2AClientType"] = None,
    request: Optional["SendMessageRequest"] = None,
    api_base: str | None = None,
    litellm_params: Dict[str, Any] | None = None,
    agent_id: str | None = None,
    agent_extra_headers: Dict[str, str] | None = None,
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
    logging_obj = kwargs.get("litellm_logging_obj")
    trace_id = getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None
    custom_llm_provider = litellm_params.get("custom_llm_provider")

    # Route through completion bridge if custom_llm_provider is set
    if custom_llm_provider:
        if request is None:
            raise ValueError("request is required for completion bridge")
        return await _send_message_via_completion_bridge(
            request=request,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            litellm_params=litellm_params,
            agent_extra_headers=agent_extra_headers,
        )

    # Standard A2A client flow
    if request is None:
        raise ValueError("request is required")

    # Create A2A client if not provided but api_base is available
    if a2a_client is None:
        if api_base is None:
            raise ValueError("Either a2a_client or api_base is required for standard A2A flow")
        trace_id = trace_id or str(uuid.uuid4())
        extra_headers: Dict[str, str] = {"X-LiteLLM-Trace-Id": trace_id}
        if agent_id:
            extra_headers["X-LiteLLM-Agent-Id"] = agent_id
        # Overlay agent-level headers (agent headers take precedence over LiteLLM internal ones)
        if agent_extra_headers:
            extra_headers.update(agent_extra_headers)
        a2a_client = await create_a2a_client(base_url=api_base, extra_headers=extra_headers)

    # Type assertion: a2a_client is guaranteed to be non-None here
    assert a2a_client is not None

    agent_name = _get_a2a_model_info(a2a_client, kwargs)

    verbose_logger.info(f"A2A send_message request_id={request.id}, agent={agent_name}")

    # Get agent card URL for localhost retry logic
    agent_card = _get_a2a_client_agent_card(a2a_client)
    card_url = get_agent_card_url(agent_card) if agent_card else None

    a2a_response = await _execute_a2a_send_with_retry(
        a2a_client=a2a_client,
        request=request,
        agent_card=agent_card,
        card_url=card_url,
        api_base=api_base,
        agent_name=agent_name,
    )

    verbose_logger.info(f"A2A send_message completed, request_id={request.id}")

    # Wrap in LiteLLM response type for _hidden_params support
    response = LiteLLMSendMessageResponse.from_a2a_response(a2a_response, request_id=str(request.id))

    # Calculate token usage from request and response
    response_dict = a2a_response.model_dump(mode="json", exclude_none=True)
    (
        prompt_tokens,
        completion_tokens,
        _,
    ) = A2ARequestUtils.calculate_usage_from_request_response(
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


def _build_streaming_logging_obj(
    request: "SendStreamingMessageRequest",
    agent_name: str,
    agent_id: str | None,
    litellm_params: Dict[str, Any] | None,
    metadata: Dict[str, Any] | None,
    proxy_server_request: Dict[str, Any] | None,
) -> Logging:
    """Build logging object for streaming A2A requests."""
    start_time = datetime.datetime.now()
    model = f"a2a_agent/{agent_name}"

    logging_obj = Logging(
        model=model,
        messages=[{"role": "user", "content": "streaming-request"}],
        stream=False,
        call_type="asend_message_streaming",
        start_time=start_time,
        litellm_call_id=str(request.id),
        function_id=str(request.id),
    )
    logging_obj.model = model
    logging_obj.custom_llm_provider = "a2a_agent"
    logging_obj.model_call_details["model"] = model
    logging_obj.model_call_details["custom_llm_provider"] = "a2a_agent"
    if agent_id:
        logging_obj.model_call_details["agent_id"] = agent_id

    _litellm_params = litellm_params.copy() if litellm_params else {}
    if metadata:
        _litellm_params["metadata"] = metadata
    if proxy_server_request:
        _litellm_params["proxy_server_request"] = proxy_server_request

    logging_obj.litellm_params = _litellm_params
    logging_obj.optional_params = _litellm_params
    logging_obj.model_call_details["litellm_params"] = _litellm_params
    logging_obj.model_call_details["metadata"] = metadata or {}

    return logging_obj


async def asend_message_streaming(
    a2a_client: Optional["A2AClientType"] = None,
    request: Optional["SendStreamingMessageRequest"] = None,
    api_base: str | None = None,
    litellm_params: Dict[str, Any] | None = None,
    agent_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    proxy_server_request: Dict[str, Any] | None = None,
    agent_extra_headers: Dict[str, str] | None = None,
    **kwargs: object,
) -> AsyncIterator[Any]:
    """
    Async: Send a streaming message to an A2A agent.

    If litellm_params contains custom_llm_provider, routes through the completion bridge.

    Args:
        a2a_client: An initialized a2a.client.A2AClient instance (optional if using completion bridge)
        request: SendStreamingMessageRequest from a2a.types
        api_base: API base URL (required for completion bridge)
        litellm_params: Optional dict with custom_llm_provider, model, etc. for completion bridge
        agent_id: Optional agent ID for tracking in SpendLogs
        metadata: Optional metadata dict (contains user_api_key, user_id, team_id, etc.)
        proxy_server_request: Optional proxy server request data

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
        # api_base is optional for providers that derive endpoint from model (e.g., bedrock/agentcore)

        verbose_logger.info(f"A2A streaming using completion bridge: provider={custom_llm_provider}")

        from litellm.a2a_protocol.litellm_completion_bridge.handler import (
            A2ACompletionBridgeHandler,
        )

        # Extract params from request
        params = (
            request.params.model_dump(mode="json") if hasattr(request.params, "model_dump") else dict(request.params)
        )

        async for chunk in A2ACompletionBridgeHandler.handle_streaming(
            request_id=str(request.id),
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
            agent_extra_headers=agent_extra_headers,
        ):
            yield chunk
        return

    if request is None:
        raise ValueError("request is required")

    _raw_logging_obj = kwargs.get("litellm_logging_obj")
    logging_obj: Logging | None = _raw_logging_obj if isinstance(_raw_logging_obj, Logging) else None

    if a2a_client is None:
        if api_base is None:
            raise ValueError("Either a2a_client or api_base is required for standard A2A flow")
        logging_trace_id = getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None
        trace_id = logging_trace_id or (str(request.id) if request.id else str(uuid.uuid4()))
        extra_headers: dict[str, str] = {"X-LiteLLM-Trace-Id": trace_id}
        if agent_id:
            extra_headers["X-LiteLLM-Agent-Id"] = agent_id
        if agent_extra_headers:
            extra_headers.update(agent_extra_headers)
        a2a_client = await create_a2a_client(
            base_url=api_base,
            extra_headers=extra_headers,
            streaming=True,
        )

    assert a2a_client is not None

    agent_name = _get_a2a_model_info(a2a_client, kwargs)

    if logging_obj is None:
        logging_obj = _build_streaming_logging_obj(
            request=request,
            agent_name=agent_name,
            agent_id=agent_id,
            litellm_params=litellm_params,
            metadata=metadata,
            proxy_server_request=proxy_server_request,
        )

    verbose_logger.info(f"A2A send_message_streaming request_id={request.id}, agent={agent_name}")

    agent_card = _get_a2a_client_agent_card(a2a_client)
    card_url = get_agent_card_url(agent_card) if agent_card else None

    stream = _execute_a2a_stream_with_retry(
        a2a_client=a2a_client,
        request=request,
        agent_card=agent_card,
        card_url=card_url,
        api_base=api_base,
        agent_name=agent_name,
    )

    _set_agent_id_on_logging_obj(kwargs=kwargs, agent_id=agent_id)

    async for chunk in A2AStreamingIterator(
        stream=stream,
        request=request,
        logging_obj=logging_obj,
        agent_name=agent_name,
    ):
        yield chunk


async def create_a2a_client(
    base_url: str,
    timeout: float = DEFAULT_A2A_AGENT_TIMEOUT,
    extra_headers: Dict[str, str] | None = None,
    streaming: bool = False,
) -> "A2AClientType":
    """
    Create an A2A client for the given agent URL.

    This resolves the agent card and returns a ready-to-use A2A client.
    The client can be reused for multiple requests.

    Args:
        base_url: The base URL of the A2A agent (e.g., "http://localhost:10001")
        timeout: Request timeout in seconds (default: ``DEFAULT_A2A_AGENT_TIMEOUT`` / env ``DEFAULT_A2A_AGENT_TIMEOUT``)
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
            "The 'a2a' package is required for A2A agent invocation. Install it with: pip install a2a-sdk"
        )

    verbose_logger.info(f"Creating A2A client for {base_url}")

    # Use get_async_httpx_client with per-agent params so that different agents
    # (with different extra_headers) get separate cached clients.  The params
    # dict is hashed into the cache key, keeping agent auth isolated while
    # still reusing connections within the same agent.
    #
    # Only pass params that AsyncHTTPHandler.__init__ accepts (e.g. timeout).
    # Use "disable_aiohttp_transport" key for cache-key-only data (it's
    # filtered out before reaching the constructor).
    _client_params: dict = {"timeout": timeout}
    if extra_headers:
        # Encode headers into a cache-key-only param so each unique header
        # set produces a distinct cache key.
        _client_params["disable_aiohttp_transport"] = str(sorted(extra_headers.items()))
    _async_handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.A2AProvider,
        params=_client_params,
    )
    httpx_client = _async_handler.client
    if extra_headers:
        httpx_client.headers.update(extra_headers)
        verbose_proxy_logger.debug(f"A2A client created with extra_headers={list(extra_headers.keys())}")

    a2a_client = await create_client(  # pyright: ignore[reportOptionalCall]
        base_url,
        client_config=ClientConfig(  # pyright: ignore[reportOptionalCall]
            httpx_client=httpx_client,
            streaming=streaming,
        ),
    )
    # Stash LiteLLM-owned handles on the client so the localhost-retry path can reuse
    # the configured httpx client (with this agent's trace-id/auth headers) without
    # excavating a2a-sdk private internals.
    a2a_client._litellm_httpx_client = httpx_client  # type: ignore[attr-defined]
    agent_card = getattr(a2a_client, "_card", None)
    if agent_card is not None:
        a2a_client._litellm_agent_card = agent_card  # type: ignore[attr-defined]

    verbose_logger.info(f"A2A client created for {base_url}")

    return a2a_client


async def aget_agent_card(
    base_url: str,
    timeout: float = DEFAULT_A2A_AGENT_TIMEOUT,
    extra_headers: Dict[str, str] | None = None,
) -> "AgentCard":
    """
    Fetch the agent card from an A2A agent.

    Args:
        base_url: The base URL of the A2A agent (e.g., "http://localhost:10001")
        timeout: Request timeout in seconds (default: ``DEFAULT_A2A_AGENT_TIMEOUT`` / env ``DEFAULT_A2A_AGENT_TIMEOUT``)
        extra_headers: Optional additional headers to include in requests

    Returns:
        AgentCard from the A2A agent
    """
    if not A2A_SDK_AVAILABLE:
        raise ImportError(
            "The 'a2a' package is required for A2A agent invocation. Install it with: pip install a2a-sdk"
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

    verbose_logger.info(f"Fetched agent card: {agent_card.name if hasattr(agent_card, 'name') else 'unknown'}")
    return agent_card
