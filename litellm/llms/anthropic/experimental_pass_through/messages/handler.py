"""
- call /messages on Anthropic API
- Make streaming + non-streaming request - just pass it through direct to Anthropic. No need to do anything special here
- Ensure requests are logged in the DB - stream + non-stream

"""

import asyncio
import contextvars
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.common_utils import (
    sanitize_tool_use_ids_in_anthropic_messages,
    strip_empty_text_blocks_from_anthropic_messages,
)
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.llms.anthropic_messages.anthropic_request import AnthropicMetadata
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

from ..utils import is_reasoning_auto_summary_enabled

from ..adapters.handler import LiteLLMMessagesToCompletionTransformationHandler
from ..responses_adapters.handler import LiteLLMMessagesToResponsesAPIHandler
from .interceptors import get_messages_interceptors
from .utils import AnthropicMessagesRequestUtils, mock_response

# Providers that are routed directly to the OpenAI Responses API instead of
# going through chat/completions.
_RESPONSES_API_PROVIDERS = frozenset({"openai"})


def _should_route_to_responses_api(custom_llm_provider: Optional[str]) -> bool:
    """Return True when the provider should use the Responses API path.

    Set ``litellm.use_chat_completions_url_for_anthropic_messages = True`` to
    opt out and route OpenAI/Azure requests through chat/completions instead.
    """
    if litellm.use_chat_completions_url_for_anthropic_messages:
        return False
    return custom_llm_provider in _RESPONSES_API_PROVIDERS


def _deployment_passes_through_anthropic_messages(model_info: object) -> bool:
    """Whether the deployment opted into forwarding /v1/messages untranslated.

    The opt-in is ``model_info.supported_endpoints`` containing ``"/v1/messages"``,
    declared per deployment in config.yaml and plumbed here as ``kwargs["model_info"]``
    by the router.
    """
    if not isinstance(model_info, dict):
        return False
    supported_endpoints = model_info.get("supported_endpoints")
    return isinstance(supported_endpoints, (list, tuple)) and "/v1/messages" in supported_endpoints


####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


async def _execute_pre_request_hooks(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    stream: Optional[bool],
    custom_llm_provider: Optional[str],
    **kwargs,
) -> Dict:
    """
    Execute pre-request hooks from CustomLogger callbacks.

    Allows CustomLoggers to modify request parameters before the API call.
    Used for WebSearch tool conversion, stream modification, etc.

    Args:
        model: Model name
        messages: List of messages
        tools: Optional tools list
        stream: Optional stream flag
        custom_llm_provider: Provider name (if not set, will be extracted from model)
        **kwargs: Additional request parameters

    Returns:
        Dict containing all (potentially modified) request parameters including tools, stream
    """
    # If custom_llm_provider not provided, extract from model
    if not custom_llm_provider:
        try:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
        except Exception:
            # If extraction fails, continue without provider
            pass

    # Build complete request kwargs dict
    request_kwargs = {
        "tools": tools,
        "stream": stream,
        "litellm_params": {
            "custom_llm_provider": custom_llm_provider,
        },
        **kwargs,
    }

    if not litellm.callbacks:
        return request_kwargs

    from litellm.integrations.custom_logger import CustomLogger as _CustomLogger

    for callback in litellm.callbacks:
        if not isinstance(callback, _CustomLogger):
            continue

        # Call the pre-request hook
        modified_kwargs = await callback.async_pre_request_hook(model, messages, request_kwargs)

        # If hook returned modified kwargs, use them
        if modified_kwargs is not None:
            request_kwargs = modified_kwargs

    return request_kwargs


async def _try_websearch_short_circuit(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    custom_llm_provider: Optional[str],
    stream: Optional[bool],
    kwargs: Optional[dict] = None,
) -> Optional[Union[AnthropicMessagesResponse, AsyncIterator]]:
    """
    Attempt to short-circuit a web-search-only request.

    Claude Code sends web search as a separate, standalone /v1/messages
    request. For providers that don't natively support web search (e.g.
    github_copilot), we detect this pattern, execute the search via
    Tavily/Perplexity, and return a synthetic Anthropic response — bypassing
    the backend LLM entirely.

    Returns the synthetic response if short-circuited, or None to continue
    normal processing.
    """
    if not litellm.callbacks:
        return None

    from litellm.integrations.websearch_interception.handler import (
        WebSearchInterceptionLogger,
    )

    for callback in litellm.callbacks:
        if not isinstance(callback, WebSearchInterceptionLogger):
            continue

        response = await callback.try_short_circuit_search(
            model=model,
            messages=messages,
            tools=tools,
            custom_llm_provider=custom_llm_provider,
            kwargs=kwargs,
        )
        if response is not None:
            anthropic_response = cast(AnthropicMessagesResponse, response)
            if stream:
                from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
                    FakeAnthropicMessagesStreamIterator,
                )

                return FakeAnthropicMessagesStreamIterator(anthropic_response)
            return anthropic_response

    return None


@client
async def anthropic_messages(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[Union[str, list]] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client: Optional[AsyncHTTPHandler] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[AnthropicMessagesResponse, Iterator[bytes], AsyncIterator[Any]]:
    """
    Async: Make llm api request in Anthropic /messages API spec.

    Runs the empty-text-block sanitizer before any backend dispatch.
    """
    # Anthropic's API rejects requests containing empty / whitespace-only
    # text content blocks with "messages: text content blocks must be
    # non-empty".  Multi-turn tool-use clients (e.g. Claude Code) routinely
    # loop assistant responses that contain {"type": "text", "text": ""}
    # alongside tool_use blocks back as conversation history, which then
    # causes the next /v1/messages call to 400.  /v1/chat/completions
    # already handles this in anthropic_messages_pt; sanitize the native
    # Anthropic Messages path here for the same guarantee.  See #22930.
    messages = strip_empty_text_blocks_from_anthropic_messages(messages)
    # Replay of cross-provider tool history (e.g. kimi -> Anthropic) may carry
    # ids like ``functions.Bash:0`` that violate Anthropic's id pattern.
    messages = sanitize_tool_use_ids_in_anthropic_messages(messages)

    from litellm.integrations.anthropic_cache_control_hook import (
        AnthropicCacheControlHook,
    )

    messages, system = AnthropicCacheControlHook.maybe_inject_cache_control(messages, system, kwargs)

    original_stream = stream or kwargs.get("_websearch_interception_converted_stream", False)

    # Execute pre-request hooks to allow CustomLoggers to modify request.
    # tool_choice is forwarded explicitly (it is a named param, not in kwargs)
    # so hooks that rename tools — e.g. websearch_interception converting
    # web_search -> litellm_web_search — can keep a forced tool_choice in sync.
    request_kwargs = await _execute_pre_request_hooks(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        custom_llm_provider=custom_llm_provider,
        tool_choice=tool_choice,
        **kwargs,
    )

    # Extract modified parameters. Pop every named param of `anthropic_messages`
    # that we may forward explicitly downstream, so we (a) honor pre-request hook
    # overrides and (b) avoid duplicate-keyword conflicts when splatting `kwargs`
    # into call sites that already pass these as named arguments.
    tools = request_kwargs.pop("tools", tools)
    stream = request_kwargs.pop("stream", stream)
    metadata = request_kwargs.pop("metadata", metadata)
    stop_sequences = request_kwargs.pop("stop_sequences", stop_sequences)
    system = request_kwargs.pop("system", system)
    temperature = request_kwargs.pop("temperature", temperature)
    thinking = request_kwargs.pop("thinking", thinking)
    tool_choice = request_kwargs.pop("tool_choice", tool_choice)
    top_k = request_kwargs.pop("top_k", top_k)
    top_p = request_kwargs.pop("top_p", top_p)
    # Propagate the provider derived inside pre-request hooks, if not already set.
    # The litellm_params dict may have been overwritten by **kwargs in
    # _execute_pre_request_hooks, so fall back to get_llm_provider() if needed.
    if not custom_llm_provider:
        custom_llm_provider = request_kwargs.get("litellm_params", {}).get("custom_llm_provider")
        if not custom_llm_provider:
            try:
                _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
            except Exception:
                pass
    # Remove litellm_params from kwargs (only needed for hooks)
    request_kwargs.pop("litellm_params", None)
    # Merge back any other modifications
    kwargs.update(request_kwargs)

    # Short-circuit web-search-only requests: detect the pattern, execute
    # search directly via Tavily/Perplexity, and return a synthetic response
    # without ever touching the backend LLM or the adapter path.
    # Use original_stream (not the hook-converted stream) so streaming
    # callers get SSE events instead of a plain dict.
    short_circuit_response = await _try_websearch_short_circuit(
        model=model,
        messages=messages,
        tools=tools,
        custom_llm_provider=custom_llm_provider,
        stream=original_stream,
        kwargs={**kwargs, "metadata": metadata},
    )
    if short_circuit_response is not None:
        return short_circuit_response

    # Run registered MessagesInterceptors (e.g. advisor orchestration loop).
    # Named params on `anthropic_messages` are bound to locals, not `**kwargs`,
    # so forward them explicitly — otherwise interceptor sub-calls drop them.
    for interceptor in get_messages_interceptors():
        if interceptor.can_handle(tools, custom_llm_provider):
            return await interceptor.handle(
                model=model,
                messages=messages,
                tools=tools,
                stream=original_stream,
                max_tokens=max_tokens,
                custom_llm_provider=custom_llm_provider,
                api_key=api_key,
                api_base=api_base,
                metadata=metadata,
                stop_sequences=stop_sequences,
                system=system,
                temperature=temperature,
                thinking=thinking,
                tool_choice=tool_choice,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )

    loop = asyncio.get_event_loop()
    kwargs["is_async"] = True

    func = partial(
        anthropic_messages_handler,
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        metadata=metadata,
        stop_sequences=stop_sequences,
        stream=stream,
        system=system,
        temperature=temperature,
        thinking=thinking,
        tool_choice=tool_choice,
        tools=tools,
        top_k=top_k,
        top_p=top_p,
        api_key=api_key,
        api_base=api_base,
        client=client,
        custom_llm_provider=custom_llm_provider,
        # messages were already empty-text-block sanitized at the top of this
        # function and are NOT reassigned before this dispatch, so the handler
        # can skip its (otherwise redundant) second full-messages scan. Passed
        # explicitly (not via **kwargs) so it only affects this direct
        # dispatch -- interceptor / sync entry points still sanitize.
        _litellm_messages_presanitized=True,
        **kwargs,
    )
    ctx = contextvars.copy_context()
    func_with_context = partial(ctx.run, func)
    init_response = await loop.run_in_executor(None, func_with_context)

    if asyncio.iscoroutine(init_response):
        response = await init_response
    else:
        response = init_response
    return response


def validate_anthropic_api_metadata(metadata: Optional[Dict] = None) -> Optional[Dict]:
    """
    Validate Anthropic API metadata - This is done to ensure only allowed `metadata` fields are passed to Anthropic API

    If there are any litellm specific metadata fields, use `litellm_metadata` key to pass them.
    """
    if metadata is None:
        return None
    anthropic_metadata_obj = AnthropicMetadata(**metadata)
    return anthropic_metadata_obj.model_dump(exclude_none=True)


def anthropic_messages_handler(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[Union[str, list]] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict] = None,
    tool_choice: Optional[Dict] = None,
    tools: Optional[List[Dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    container: Optional[Dict] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client: Optional[AsyncHTTPHandler] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[
    AnthropicMessagesResponse,
    Iterator[bytes],
    AsyncIterator[Any],
    Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator[Any], Iterator[bytes]]],
]:
    """
    Makes Anthropic `/v1/messages` API calls In the Anthropic API Spec

    Args:
        container: Container config with skills for code execution
    """
    from litellm.types.utils import LlmProviders

    # Sanitize empty text blocks so the sync entry point
    # (litellm.messages.create -> anthropic_messages_handler) gets the same
    # protection as the async wrapper. The async wrapper already sanitized and
    # does not reassign messages before dispatch, so it sets
    # ``_litellm_messages_presanitized`` to skip this redundant second
    # full-messages scan. Pop it so it never leaks into provider params.
    if not kwargs.pop("_litellm_messages_presanitized", False):
        messages = strip_empty_text_blocks_from_anthropic_messages(messages)
        messages = sanitize_tool_use_ids_in_anthropic_messages(messages)

    from litellm.integrations.anthropic_cache_control_hook import (
        AnthropicCacheControlHook,
    )

    messages, system = AnthropicCacheControlHook.maybe_inject_cache_control(messages, system, kwargs)

    metadata = validate_anthropic_api_metadata(metadata)

    local_vars = locals()
    is_async = kwargs.pop("is_async", False)
    # Use provided client or create a new one
    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore

    # Store original model name before get_llm_provider strips the provider prefix
    # This is needed by agentic hooks (e.g., websearch_interception) to make follow-up requests
    original_model = model

    litellm_params = GenericLiteLLMParams(
        **kwargs,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
    )
    (
        model,
        custom_llm_provider,
        dynamic_api_key,
        dynamic_api_base,
    ) = litellm.get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
    )

    # Store agentic loop params in logging object for agentic hooks
    # This provides original request context needed for follow-up calls
    if litellm_logging_obj is not None:
        litellm_logging_obj.model_call_details["agentic_loop_params"] = {
            "model": original_model,
            "custom_llm_provider": custom_llm_provider,
        }

        # Check if stream was converted for WebSearch interception
        # This is set in the async wrapper above when stream=True is converted to stream=False
        if kwargs.get("_websearch_interception_converted_stream", False):
            litellm_logging_obj.model_call_details["websearch_interception_converted_stream"] = True

    if litellm_params.mock_response and isinstance(litellm_params.mock_response, str):
        return mock_response(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            mock_response=litellm_params.mock_response,
        )

    anthropic_messages_provider_config: Optional[BaseAnthropicMessagesConfig] = None

    if custom_llm_provider is not None and custom_llm_provider in [provider.value for provider in LlmProviders]:
        anthropic_messages_provider_config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model,
            provider=litellm.LlmProviders(custom_llm_provider),
        )
    if anthropic_messages_provider_config is None and _deployment_passes_through_anthropic_messages(
        kwargs.get("model_info")
    ):
        from litellm.llms.openai_like.messages.transformation import (
            OpenAILikeAnthropicMessagesConfig,
        )

        anthropic_messages_provider_config = OpenAILikeAnthropicMessagesConfig()
    if anthropic_messages_provider_config is None:
        # Route to Responses API for OpenAI / Azure, chat/completions for everything else.
        _shared_kwargs = dict(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            _is_async=is_async,
            api_key=api_key,
            api_base=api_base,
            client=client,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        if _should_route_to_responses_api(custom_llm_provider):
            return LiteLLMMessagesToResponsesAPIHandler.anthropic_messages_handler(**_shared_kwargs)

        # The in-gateway context_management polyfill runs inside
        # ``async_anthropic_messages_handler`` so it can ``await`` the
        # summarization model for ``compact_20260112``. ``context_management``
        # is passed through as a regular kwarg.
        return LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
            **_shared_kwargs,
        )

    if custom_llm_provider is None:
        raise ValueError(
            f"custom_llm_provider is required for Anthropic messages, passed in model={model}, custom_llm_provider={custom_llm_provider}"
        )

    local_vars.update(kwargs)
    anthropic_messages_optional_request_params = (
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params=local_vars,
            model=model,
            drop_params=litellm_params.get("drop_params") is True,
            custom_llm_provider=custom_llm_provider,
        )
    )
    if is_reasoning_auto_summary_enabled():
        thinking_param = anthropic_messages_optional_request_params.get("thinking")
        if isinstance(thinking_param, dict) and thinking_param.get("type") != "disabled":
            anthropic_messages_optional_request_params["thinking"] = {
                **thinking_param,
                "display": "summarized",
            }

    return base_llm_http_handler.anthropic_messages_handler(
        model=model,
        messages=messages,
        anthropic_messages_provider_config=anthropic_messages_provider_config,
        anthropic_messages_optional_request_params=dict(anthropic_messages_optional_request_params),
        _is_async=is_async,
        client=client,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=litellm_logging_obj,
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        kwargs=kwargs,
    )
