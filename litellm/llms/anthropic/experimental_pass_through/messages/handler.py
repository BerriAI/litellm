"""
- call /messages on Anthropic API
- Make streaming + non-streaming request - just pass it through direct to Anthropic. No need to do anything special here 
- Ensure requests are logged in the DB - stream + non-stream

"""

import asyncio
import contextvars
import json
from functools import partial
from typing import Any, AsyncIterator, Coroutine, Dict, List, Optional, Union, cast

import httpx

import litellm
from litellm.constants import request_timeout
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
from litellm.types.utils import ProviderSpecificHeader
from litellm.utils import ProviderConfigManager, client

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
        from litellm.proxy.pass_through_endpoints.types import EndpointType

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
async def aanthropic_messages(
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
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[AnthropicMessagesResponse, AsyncIterator]:
    """
    Async: Handles responses API requests by reusing the synchronous function
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aanthropic_messages"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=local_vars.get("base_url", None)
            )

        func = partial(
            anthropic_messages,
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
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def anthropic_messages(
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
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[AnthropicMessagesResponse, Coroutine[Any, Any, AnthropicMessagesResponse]]:
    """
    Makes Anthropic `/v1/messages` API calls In the Anthropic API Spec
    """
    local_vars = locals()
    try:
        # Use provided client or create a new one
        optional_params = GenericLiteLLMParams(**kwargs)
        _is_async = kwargs.pop("aanthropic_messages", False) is True
        (
            model,
            custom_llm_provider,
            dynamic_api_key,
            dynamic_api_base,
        ) = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=optional_params.api_base,
            api_key=optional_params.api_key,
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

        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
            "litellm_logging_obj", None
        )
        if litellm_logging_obj is None:
            raise ValueError(
                f"litellm_logging_obj is required, got litellm_logging_obj={litellm_logging_obj}"
            )

        # Prepare headers
        provider_specific_header = cast(
            Optional[ProviderSpecificHeader],
            kwargs.get("provider_specific_header", None),
        )
        extra_headers = (
            provider_specific_header.get("extra_headers", {})
            if provider_specific_header
            else {}
        )
        headers = anthropic_messages_provider_config.validate_environment(
            headers=extra_headers or {},
            model=model,
            api_key=api_key,
        )

        litellm_logging_obj.update_environment_variables(
            model=model,
            optional_params=dict(optional_params),
            litellm_params={
                "metadata": kwargs.get("metadata", {}),
                "preset_cache_key": None,
                "stream_response": {},
                **optional_params.model_dump(exclude_unset=True),
            },
            custom_llm_provider=custom_llm_provider,
        )
        # Prepare request body
        request_body = locals().copy()
        request_body = {
            k: v
            for k, v in request_body.items()
            if k
            in anthropic_messages_provider_config.get_supported_anthropic_messages_params(
                model=model
            )
            and v is not None
        }
        request_body["stream"] = stream
        request_body["model"] = model
        litellm_logging_obj.stream = stream
        litellm_logging_obj.model_call_details.update(request_body)

        # Make the request
        request_url = anthropic_messages_provider_config.get_complete_url(
            api_base=api_base, model=model
        )

        litellm_logging_obj.pre_call(
            input=[{"role": "user", "content": json.dumps(request_body)}],
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": str(request_url),
                "headers": headers,
            },
        )

        response = base_llm_http_handler.anthropic_messages_handler(
            model=model,
            messages=messages,
            anthropic_messages_provider_config=anthropic_messages_provider_config,
            anthropic_messages_optional_params=request_body,
            custom_llm_provider=custom_llm_provider,
            litellm_params=optional_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=request_body,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
            fake_stream=anthropic_messages_provider_config.should_fake_stream(
                model=model, stream=stream, custom_llm_provider=custom_llm_provider
            ),
        )
        return response

    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
