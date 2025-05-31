"""
This module is used to pass through requests to the LLM APIs.
"""

import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Optional, Union
from urllib.parse import urlencode

import httpx
from httpx._types import CookieTypes, QueryParamTypes, RequestFiles

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import client

from .utils import BasePassthroughUtils


@client
async def allm_passthrough_route(
    *,
    method: str,
    endpoint: str,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    request_query_params: Optional[dict] = None,
    request_headers: Optional[dict] = None,
    stream: bool = False,
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
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
            request_query_params=request_query_params,
            request_headers=request_headers,
            stream=stream,
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
        else:
            response = init_response
        return response
    except Exception as e:
        raise e


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
    stream: bool = False,
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

    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    provider_config = ProviderConfigManager.get_provider_model_info(
        provider=LlmProviders(custom_llm_provider),
        model=model,
    )
    if provider_config is None:
        raise Exception(f"Provider {custom_llm_provider} not found")

    base_target_url = provider_config.get_api_base(api_base)

    if base_target_url is None:
        raise Exception(f"Provider {custom_llm_provider} api base not found")

    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    if request_query_params:
        # Create a new URL with the merged query params
        updated_url = updated_url.copy_with(
            query=urlencode(request_query_params).encode("ascii")
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

    ## SWAP MODEL IN JSON BODY
    if json and isinstance(json, dict) and "model" in json:
        json["model"] = model

    request = client.client.build_request(
        method=method,
        url=updated_url,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
    )

    response = client.client.send(request=request, stream=stream)
    return response
