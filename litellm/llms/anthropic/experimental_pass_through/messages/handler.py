"""
- call /messages on Anthropic API
- Make streaming + non-streaming request - just pass it through direct to Anthropic. No need to do anything special here 
- Ensure requests are logged in the DB - stream + non-stream

"""

import json
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client


class AnthropicMessagesHandler:

    @staticmethod
    async def _handle_anthropic_streaming(
        response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        """Helper function to handle Anthropic streaming responses using the existing logging handlers"""
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
async def anthropic_messages(
    api_key: str,
    model: str,
    stream: bool = False,
    api_base: Optional[str] = None,
    client: Optional[AsyncHTTPHandler] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[Dict[str, Any], AsyncIterator]:
    """
    Makes Anthropic `/v1/messages` API calls In the Anthropic API Spec
    """
    # Use provided client or create a new one
    optional_params = GenericLiteLLMParams(**kwargs)
    model, _custom_llm_provider, dynamic_api_key, dynamic_api_base = (
        litellm.get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=optional_params.api_base,
            api_key=optional_params.api_key,
        )
    )
    anthropic_messages_provider_config: Optional[BaseAnthropicMessagesConfig] = (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model,
            provider=litellm.LlmProviders(_custom_llm_provider),
        )
    )
    if anthropic_messages_provider_config is None:
        raise ValueError(
            f"Anthropic messages provider config not found for model: {model}"
        )

    if client is None:
        async_httpx_client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.ANTHROPIC
        )
    else:
        async_httpx_client = client

    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj", None)

    # Prepare headers
    headers = anthropic_messages_provider_config.validate_environment(
        headers={},
        model=model,
        api_key=api_key,
    )
    # Prepare request body
    request_body = kwargs.copy()
    request_body = {
        k: v
        for k, v in request_body.items()
        if k
        in anthropic_messages_provider_config.get_supported_anthropic_messages_params(
            model=model
        )
    }
    request_body["stream"] = stream
    request_body["model"] = model
    litellm_logging_obj.stream = stream
    litellm_logging_obj.model_call_details.update(request_body)

    verbose_logger.debug(
        "request_body= %s", json.dumps(request_body, indent=4, default=str)
    )

    # Make the request
    response = await async_httpx_client.post(
        url=anthropic_messages_provider_config.get_complete_url(
            api_base=api_base, model=model
        ),
        headers=headers,
        data=json.dumps(request_body),
        timeout=600,
        stream=stream,
    )
    response.raise_for_status()

    # used for logging + cost tracking
    litellm_logging_obj.model_call_details["httpx_response"] = response

    if stream:
        return await AnthropicMessagesHandler._handle_anthropic_streaming(
            response=response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
        )
    else:
        return response.json()
