"""
This module is used to pass through requests to the LLM APIs.
"""

import asyncio
import contextvars
from functools import partial
from typing import TYPE_CHECKING, Any, Coroutine, Optional, Union, cast

import httpx
from httpx._types import CookieTypes, QueryParamTypes, RequestFiles

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.utils import client

base_llm_http_handler = BaseLLMHTTPHandler()
from .utils import BasePassthroughUtils

if TYPE_CHECKING:
    from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig


@client
async def allm_passthrough_route(
    *,
    method: str,
    endpoint: str,
    model: str,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    request_query_params: Optional[dict] = None,
    request_headers: Optional[dict] = None,
    content: Optional[Any] = None,
    data: Optional[dict] = None,
    files: Optional[RequestFiles] = None,
    json: Optional[Any] = None,
    params: Optional[QueryParamTypes] = None,
    cookies: Optional[CookieTypes] = None,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    **kwargs,
) -> Union[httpx.Response, Coroutine[Any, Any, httpx.Response]]:
    """
    Async: Reranks a list of documents based on their relevance to the query
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["allm_passthrough_route"] = True

        func = partial(
            llm_passthrough_route,
            method=method,
            endpoint=endpoint,
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
            request_query_params=request_query_params,
            request_headers=request_headers,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            cookies=cookies,
            client=client,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_text = await e.response.aread()
                error_text_str = error_text.decode("utf-8")
                raise Exception(error_text_str)
        else:
            response = init_response
        return response
    except Exception as e:
        # For passthrough routes, we need to get the provider config to properly handle errors
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager
        
        # Get the provider using the same logic as llm_passthrough_route
        _, resolved_custom_llm_provider, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
        )
        
        # Get provider config if available
        provider_config = None
        if resolved_custom_llm_provider:
            try:
                provider_config = cast(
                    Optional["BasePassthroughConfig"], kwargs.get("provider_config")
                ) or ProviderConfigManager.get_provider_passthrough_config(
                    provider=LlmProviders(resolved_custom_llm_provider),
                    model=model,
                )
            except Exception:
                # If we can't get provider config, pass None
                pass
        
        if provider_config is None:
            # If no provider config available, raise the original exception
            raise e
        
        raise base_llm_http_handler._handle_error(
            e=e,
            provider_config=provider_config,
        )


@client
def llm_passthrough_route(
    *,
    method: str,
    endpoint: str,
    model: str,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    request_query_params: Optional[dict] = None,
    request_headers: Optional[dict] = None,
    allm_passthrough_route: bool = False,
    content: Optional[Any] = None,
    data: Optional[dict] = None,
    files: Optional[RequestFiles] = None,
    json: Optional[Any] = None,
    params: Optional[QueryParamTypes] = None,
    cookies: Optional[CookieTypes] = None,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    **kwargs,
) -> Union[httpx.Response, Coroutine[Any, Any, httpx.Response]]:
    """
    Pass through requests to the LLM APIs.

    Step 1. Build the request
    Step 2. Send the request
    Step 3. Return the response

    [TODO] Refactor this into a provider-config pattern, once we expand this to non-vllm providers.
    """
    if client is None:
        if allm_passthrough_route:
            client = litellm.module_level_aclient
        else:
            client = litellm.module_level_client

    model, custom_llm_provider, api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )

    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    litellm_params_dict = get_litellm_params(**kwargs)

    provider_config = cast(
        Optional["BasePassthroughConfig"], kwargs.get("provider_config")
    ) or ProviderConfigManager.get_provider_passthrough_config(
        provider=LlmProviders(custom_llm_provider),
        model=model,
    )
    if provider_config is None:
        raise Exception(f"Provider {custom_llm_provider} not found")

    updated_url, base_target_url = provider_config.get_complete_url(
        api_base=api_base,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        request_query_params=request_query_params,
        litellm_params=litellm_params_dict,
    )
    # Add or update query parameters
    provider_api_key = provider_config.get_api_key(api_key)

    auth_headers = provider_config.validate_environment(
        headers={},
        model=model,
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=provider_api_key,
        api_base=base_target_url,
    )

    headers = BasePassthroughUtils.forward_headers_from_request(
        request_headers=request_headers or {},
        headers=auth_headers,
        forward_headers=False,
    )

    headers, signed_json_body = provider_config.sign_request(
        headers=headers,
        litellm_params=litellm_params_dict,
        request_data=data if data else json,
        api_base=str(updated_url),
        model=model,
    )

    ## SWAP MODEL IN JSON BODY
    if json and isinstance(json, dict) and "model" in json:
        json["model"] = model

    request = client.client.build_request(
        method=method,
        url=updated_url,
        content=signed_json_body,
        data=data if signed_json_body is None else None,
        files=files,
        json=json if signed_json_body is None else None,
        params=params,
        headers=headers,
        cookies=cookies,
    )

    ## IS STREAMING REQUEST
    is_streaming_request = provider_config.is_streaming_request(
        endpoint=endpoint,
        request_data=data or json or {},
    )

    try:
        response = client.client.send(request=request, stream=is_streaming_request)
        if asyncio.iscoroutine(response):
            return response
        response.raise_for_status()
        return response
    except Exception as e:
        if provider_config is None:
            raise e
        raise base_llm_http_handler._handle_error(
            e=e,
            provider_config=provider_config,
        )
