"""
This module is used to pass through requests to the LLM APIs.
"""

import asyncio
import contextvars
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Coroutine,
    Generator,
    List,
    Optional,
    Union,
    cast,
)

import httpx
from httpx._types import CookieTypes, QueryParamTypes, RequestFiles

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.passthrough.utils import CommonUtils
from litellm.utils import client

base_llm_http_handler = BaseLLMHTTPHandler()
from .utils import BasePassthroughUtils

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
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
) -> Union[
    httpx.Response,
    Coroutine[Any, Any, httpx.Response],
    Generator[Any, Any, Any],
    AsyncGenerator[Any, Any],
]:
    """
    Async: Reranks a list of documents based on their relevance to the query
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["allm_passthrough_route"] = True

        model, custom_llm_provider, api_key, api_base = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
        )

        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        provider_config = cast(
            Optional["BasePassthroughConfig"], kwargs.get("provider_config")
        ) or ProviderConfigManager.get_provider_passthrough_config(
            provider=LlmProviders(custom_llm_provider),
            model=model,
        )

        if provider_config is None:
            raise Exception(f"Provider {custom_llm_provider} not found")

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
) -> Union[
    httpx.Response,
    Coroutine[Any, Any, httpx.Response],
    Generator[Any, Any, Any],
    AsyncGenerator[Any, Any],
]:
    """
    Pass through requests to the LLM APIs.

    Step 1. Build the request
    Step 2. Send the request
    Step 3. Return the response
    """
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    if client is None:
        if allm_passthrough_route:
            client = litellm.module_level_aclient
        else:
            client = litellm.module_level_client

    litellm_logging_obj = cast("LiteLLMLoggingObj", kwargs.get("litellm_logging_obj"))

    model, custom_llm_provider, api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )

    litellm_params_dict = get_litellm_params(**kwargs)
    litellm_logging_obj.update_environment_variables(
        model=model,
        litellm_params=litellm_params_dict,
        optional_params={},
        endpoint=endpoint,
        custom_llm_provider=custom_llm_provider,
        request_data=data if data else json,
    )

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
    
    # need to encode the id of application-inference-profile for bedrock
    if custom_llm_provider == "bedrock" and "application-inference-profile" in endpoint:
        encoded_url_str = CommonUtils.encode_bedrock_runtime_modelid_arn(str(updated_url))
        updated_url = httpx.URL(encoded_url_str)
    
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

    ## SWAP MODEL IN JSON BODY [TODO: REFACTOR TO A provider_config.transform_request method]
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

    # Update logging object with streaming status
    litellm_logging_obj.stream = is_streaming_request

    try:
        response = client.client.send(request=request, stream=is_streaming_request)
        if asyncio.iscoroutine(response):
            if is_streaming_request:
                return _async_streaming(response, litellm_logging_obj, provider_config)
            else:
                return response
        response.raise_for_status()

        if (
            hasattr(response, "iter_bytes") and is_streaming_request
        ):  # yield the chunk, so we can store it in the logging object

            return _sync_streaming(response, litellm_logging_obj, provider_config)
        else:

            # For non-streaming responses, yield the entire response
            return response
    except Exception as e:
        if provider_config is None:
            raise e
        raise base_llm_http_handler._handle_error(
            e=e,
            provider_config=provider_config,
        )


def _sync_streaming(
    response: httpx.Response,
    litellm_logging_obj: "LiteLLMLoggingObj",
    provider_config: "BasePassthroughConfig",
):
    from litellm.utils import executor

    try:
        raw_bytes: List[bytes] = []
        for chunk in response.iter_bytes():  # type: ignore
            raw_bytes.append(chunk)
            yield chunk

        executor.submit(
            litellm_logging_obj.flush_passthrough_collected_chunks,
            raw_bytes=raw_bytes,
            provider_config=provider_config,
        )
    except Exception as e:
        raise e


async def _async_streaming(
    response: Coroutine[Any, Any, httpx.Response],
    litellm_logging_obj: "LiteLLMLoggingObj",
    provider_config: "BasePassthroughConfig",
):
    try:
        iter_response = await response
        raw_bytes: List[bytes] = []

        async for chunk in iter_response.aiter_bytes():  # type: ignore

            raw_bytes.append(chunk)
            yield chunk

        asyncio.create_task(
            litellm_logging_obj.async_flush_passthrough_collected_chunks(
                raw_bytes=raw_bytes,
                provider_config=provider_config,
            )
        )
    except Exception as e:
        raise e
