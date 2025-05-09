"""
- call /messages on Anthropic API
- Make streaming + non-streaming request - just pass it through direct to Anthropic. No need to do anything special here
- Ensure requests are logged in the DB - stream + non-stream

"""

import asyncio
import contextvars
from functools import partial
from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

from .utils import AnthropicMessagesRequestUtils

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


class AnthropicMessagesHandler:
    @staticmethod
    async def _handle_anthropic_streaming(
        response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        """Helper function to handle Anthropic streaming responses using the existing logging handlers"""
        from datetime import datetime

        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging,
        )
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )

        # Create success handler object
        passthrough_success_handler_obj = PassThroughEndpointLogging()

        # Use the existing streaming handler for Anthropic
        start_time = datetime.now()
        return PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=start_time,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            url_route="/v1/messages",
        )


@client
async def anthropic_messages(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
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
) -> Union[AnthropicMessagesResponse, AsyncIterator]:
    """
    Async: Make llm api request in Anthropic /messages API spec
    """
    local_vars = locals()
    loop = asyncio.get_event_loop()
    kwargs["anthropic_messages"] = True

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


def anthropic_messages_handler(
    max_tokens: int,
    messages: List[Dict],
    model: str,
    metadata: Optional[Dict] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    system: Optional[str] = None,
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
) -> Union[
    AnthropicMessagesResponse,
    Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator]],
]:
    """
    Makes Anthropic `/v1/messages` API calls In the Anthropic API Spec
    """
    local_vars = locals()
    # Use provided client or create a new one
    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
    litellm_params = GenericLiteLLMParams(**kwargs)
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
    anthropic_messages_provider_config: Optional[
        BaseAnthropicMessagesConfig
    ] = ProviderConfigManager.get_provider_anthropic_messages_config(
        model=model,
        provider=litellm.LlmProviders(custom_llm_provider),
    )
    if anthropic_messages_provider_config is None:
        raise ValueError(
            f"Anthropic messages provider config not found for model: {model}"
        )
    if custom_llm_provider is None:
        raise ValueError(
            f"custom_llm_provider is required for Anthropic messages, passed in model={model}, custom_llm_provider={custom_llm_provider}"
        )

    local_vars.update(kwargs)
    anthropic_messages_optional_request_params = (
        AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
            params=local_vars
        )
    )
    return base_llm_http_handler.anthropic_messages_handler(
        model=model,
        messages=messages,
        anthropic_messages_provider_config=anthropic_messages_provider_config,
        anthropic_messages_optional_request_params=dict(
            anthropic_messages_optional_request_params
        ),
        _is_async=True,
        client=client,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=litellm_logging_obj,
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        kwargs=kwargs,
    )
