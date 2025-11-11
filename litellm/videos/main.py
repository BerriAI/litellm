import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Literal, Optional, Union, overload, Dict, List

import json
import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.videos.main import (
    VideoCreateOptionalRequestParams,
    VideoObject,
)
from litellm.videos.utils import VideoGenerationRequestUtils
from litellm.constants import DEFAULT_VIDEO_ENDPOINT_MODEL, request_timeout as DEFAULT_REQUEST_TIMEOUT
from litellm.main import base_llm_http_handler
from litellm.utils import client, ProviderConfigManager
from litellm.types.utils import FileTypes, CallTypes
from litellm.types.router import GenericLiteLLMParams
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.videos.utils import decode_video_id_with_provider

#################### Initialize provider clients ####################
llm_http_handler: BaseLLMHTTPHandler = BaseLLMHTTPHandler()

##### Video Generation #######################
@client
async def avideo_generation(
    prompt: str,
    model: Optional[str] = None,
    input_reference: Optional[FileTypes] = None,
    seconds: Optional[str] = None,
    size: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    custom_llm_provider=None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> VideoObject:
    """
    Asynchronously calls the `video_generation` function with the given arguments and keyword arguments.

    Parameters:
    - `prompt` (str): Text prompt that describes the video to generate
    - `model` (Optional[str]): The video generation model to use
    - `input_reference` (Optional[FileTypes]): Optional image reference that guides generation
    - `seconds` (Optional[str]): Clip duration in seconds
    - `size` (Optional[str]): Output resolution formatted as width x height
    - `user` (Optional[str]): A unique identifier representing your end-user
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Optional[str]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (VideoResponse): The response returned by the `video_generation` function.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model or DEFAULT_VIDEO_ENDPOINT_MODEL, api_base=local_vars.get("api_base", None)
            )

        func = partial(
            video_generation,
            prompt=prompt,
            model=model,
            input_reference=input_reference,
            seconds=seconds,
            size=size,
            user=user,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
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


# fmt: off

# Overload for when avideo_generation=True (returns Coroutine)
@overload
def video_generation(
    prompt: str,
    model: Optional[str] = None,
    input_reference: Optional[str] = None,
    size: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    avideo_generation: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, VideoObject]:
    ...


@overload
def video_generation(
    prompt: str,
    model: Optional[str] = None,
    input_reference: Optional[str] = None,
    seconds: Optional[str] = None,
    size: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    avideo_generation: Literal[False] = False,
    **kwargs,
) -> VideoObject:
    ...

# fmt: on


@client
def video_generation(  # noqa: PLR0915
    prompt: str,
    model: Optional[str] = None,
    input_reference: Optional[FileTypes] = None,
    seconds: Optional[str] = None,
    size: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    custom_llm_provider=None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    VideoObject,
    Coroutine[Any, Any, VideoObject],
]:
    """
    Maps the https://api.openai.com/v1/videos endpoint.

    Currently supports OpenAI
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response", None)
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)

            response = VideoObject(**mock_response)
            return response

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        model, custom_llm_provider, _, _ = get_llm_provider(
            model=model or DEFAULT_VIDEO_ENDPOINT_MODEL,
            custom_llm_provider=custom_llm_provider,
        )

        # get provider config
        video_generation_provider_config: Optional[BaseVideoConfig] = (
            ProviderConfigManager.get_provider_video_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if video_generation_provider_config is None:
            raise ValueError(f"video generation is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # Get VideoGenerationOptionalRequestParams with only valid parameters
        video_generation_optional_params: VideoCreateOptionalRequestParams = (
            VideoGenerationRequestUtils.get_requested_video_generation_optional_param(local_vars)
        )

        # Get optional parameters for the video generation API
        video_generation_request_params: Dict = (
            VideoGenerationRequestUtils.get_optional_params_video_generation(
                model=model,
                video_generation_provider_config=video_generation_provider_config,
                video_generation_optional_params=video_generation_optional_params,
            )
        )

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=model,
            user=user,
            optional_params=dict(video_generation_request_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **video_generation_request_params,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type for video generation
        litellm_logging_obj.call_type = CallTypes.create_video.value

        # Call the handler with _is_async flag instead of directly calling the async handler
        return base_llm_http_handler.video_generation_handler(
            model=model,
            prompt=prompt,
            video_generation_provider_config=video_generation_provider_config,
            video_generation_optional_request_params=video_generation_request_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
            client=kwargs.get("client"),
        )

    except Exception as e:
        raise litellm.exception_type(
            model=model or DEFAULT_VIDEO_ENDPOINT_MODEL,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def video_content(
    video_id: str,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    bytes,
    Coroutine[Any, Any, bytes],
]:
    """
    Download video content from OpenAI's video API.

    Args:
        video_id (str): The identifier of the video whose content to download.
        api_key (Optional[str]): The API key to use for authentication.
        api_base (Optional[str]): The base URL for the API.
        timeout (Optional[float]): The timeout for the request in seconds.
        custom_llm_provider (Optional[str]): The LLM provider to use. If not provided, will be auto-detected.
        variant (Optional[str]): Which downloadable asset to return. Defaults to the MP4 video.
        extra_headers (Optional[Dict[str, Any]]): Additional headers to include in the request.
        extra_query (Optional[Dict[str, Any]]): Additional query parameters.
        extra_body (Optional[Dict[str, Any]]): Additional body parameters.

    Returns:
        bytes: The raw video content as bytes.

    Example:
        ```python
        import litellm

        video_bytes = litellm.video_content(
            video_id="video_123"
        )

        with open("video.mp4", "wb") as f:
            f.write(video_bytes)
        ```
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("async_call", False) is True

        # Try to decode provider from video_id if not explicitly provided
        if custom_llm_provider is None:
            decoded = decode_video_id_with_provider(video_id)
            custom_llm_provider = decoded.get("custom_llm_provider") or "openai"

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        # get provider config
        video_provider_config: Optional[BaseVideoConfig] = (
            ProviderConfigManager.get_provider_video_config(
                model=None,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if video_provider_config is None:
            raise ValueError(f"video support download is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # For video content download, we don't need complex optional parameter handling
        # Just pass the basic parameters that are relevant for content download
        video_content_request_params: Dict = {
            "video_id": video_id,
        }

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            user=kwargs.get("user"),
            optional_params=dict(video_content_request_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **video_content_request_params,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler with _is_async flag instead of directly calling the async handler
        return base_llm_http_handler.video_content_handler(
            video_id=video_id,
            video_content_provider_config=video_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            extra_headers=extra_headers,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Video Content Download #######################
@client
async def avideo_content(
    video_id: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> bytes:
    """
    Asynchronously download video content.

    Parameters:
    - `video_id` (str): The identifier of the video whose content to download
    - `api_key` (Optional[str]): The API key to use for authentication
    - `api_base` (Optional[str]): The base URL for the API
    - `timeout` (Optional[float]): The timeout for the request in seconds
    - `custom_llm_provider` (Optional[str]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `bytes`: The raw video content as bytes
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        # Ensure custom_llm_provider is not None - default to openai if not provided
        # Video content endpoints don't require a model parameter
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        func = partial(
            video_content,
            video_id=video_id,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
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
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )

##### Video Remix #######################
@client
async def avideo_remix(
    video_id: str,
    prompt: str,
    timeout=600,  # default to 10 minutes
    custom_llm_provider=None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> VideoObject:
    """
    Asynchronously calls the `video_remix` function with the given arguments and keyword arguments.

    Parameters:
    - `video_id` (str): The identifier of the completed video to remix
    - `prompt` (str): Updated text prompt that directs the remix generation
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Optional[str]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (VideoObject): The response returned by the `video_remix` function.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        func = partial(
            video_remix,
            video_id=video_id,
            prompt=prompt,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
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
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# fmt: off

# Overload for when avideo_remix=True (returns Coroutine)
@overload
def video_remix(
    video_id: str,
    prompt: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    avideo_remix: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, VideoObject]:
    ...


@overload
def video_remix(
    video_id: str,
    prompt: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    avideo_remix: Literal[False] = False,
    **kwargs,
) -> VideoObject:
    ...

# fmt: on


@client
def video_remix(  # noqa: PLR0915
    video_id: str,
    prompt: str,
    timeout=600,  # default to 10 minutes
    custom_llm_provider=None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    VideoObject,
    Coroutine[Any, Any, VideoObject],
]:
    """
    Maps the https://api.openai.com/v1/videos/{video_id}/remix endpoint.

    Currently supports OpenAI
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response", None)
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)

            response = VideoObject(**mock_response)
            return response

        # Try to decode provider from video_id if not explicitly provided
        if custom_llm_provider is None:
            decoded = decode_video_id_with_provider(video_id)
            custom_llm_provider = decoded.get("custom_llm_provider") or "openai"

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        # get provider config
        video_remix_provider_config: Optional[BaseVideoConfig] = (
            ProviderConfigManager.get_provider_video_config(
                model=None,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if video_remix_provider_config is None:
            raise ValueError(f"video remix is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # For video remix, we need the video_id and prompt
        video_remix_request_params: Dict = {
            "video_id": video_id,
            "prompt": prompt,
        }

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            user=kwargs.get("user"),
            optional_params=dict(video_remix_request_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **video_remix_request_params,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type for video remix
        litellm_logging_obj.call_type = CallTypes.video_remix.value

        # Call the handler with _is_async flag instead of directly calling the async handler
        return base_llm_http_handler.video_remix_handler(
            video_id=video_id,
            prompt=prompt,
            video_remix_provider_config=video_remix_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
            client=kwargs.get("client"),
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Video List #######################
@client
async def avideo_list(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    custom_llm_provider=None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[VideoObject]:
    """
    Asynchronously calls the `video_list` function with the given arguments and keyword arguments.

    Parameters:
    - `after` (Optional[str]): Identifier for the last item from the previous pagination request
    - `limit` (Optional[int]): Number of items to retrieve
    - `order` (Optional[str]): Sort order of results by timestamp. Use asc for ascending order or desc for descending order
    - `api_key` (Optional[str]): The API key to use for authentication
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Optional[str]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (Dict[str, Any]): The response returned by the `video_list` function.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model="", api_base=local_vars.get("api_base", None)
            )

        func = partial(
            video_list,
            after=after,
            limit=limit,
            order=order,
            api_key=api_key,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
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
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# fmt: off

# Overload for when avideo_list=True (returns Coroutine)
@overload
def video_list(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    avideo_list: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, List[VideoObject]]:
    ...


@overload
def video_list(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    avideo_list: Literal[False] = False,
    **kwargs,
) -> List[VideoObject]:
    ...

# fmt: on


@client
def video_list(  # noqa: PLR0915
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    custom_llm_provider=None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    List[VideoObject],
    Coroutine[Any, Any, List[VideoObject]],
]:
    """
    Maps the https://api.openai.com/v1/videos endpoint.

    Currently supports OpenAI
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response", None)
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)
            return [VideoObject(**item) for item in mock_response]

        # Ensure custom_llm_provider is not None - default to openai if not provided
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        # get provider config
        video_list_provider_config: Optional[BaseVideoConfig] = (
            ProviderConfigManager.get_provider_video_config(
                model=None,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if video_list_provider_config is None:
            raise ValueError(f"video list is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # For video list, we need the query parameters
        video_list_request_params: Dict = {
            "after": after,
            "limit": limit,
            "order": order,
        }

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            user=kwargs.get("user"),
            optional_params=dict(video_list_request_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **video_list_request_params,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type for video list
        litellm_logging_obj.call_type = CallTypes.video_list.value

        # Call the handler with _is_async flag instead of directly calling the async handler
        return base_llm_http_handler.video_list_handler(
            after=after,
            limit=limit,
            order=order,
            video_list_provider_config=video_list_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
            client=kwargs.get("client"),
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Video Status/Retrieve #######################
@client
async def avideo_status(
    video_id: str,
    timeout=600,  # default to 10 minutes
    custom_llm_provider=None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> VideoObject:
    """
    Asynchronously retrieve video status from OpenAI's video API.

    Parameters:
    - `video_id` (str): The identifier of the video whose status to retrieve
    - `model` (Optional[str]): The model to use. If not provided, will be auto-detected
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Optional[str]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (VideoObject): The response returned by the `video_status` function.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True


        func = partial(
            video_status,
            video_id=video_id,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
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
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# fmt: off

# Overload for when avideo_status=True (returns Coroutine)
@overload
def video_status(
    video_id: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    avideo_status: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, VideoObject]:
    ...

# Overload for when avideo_status=False (returns VideoObject)
@overload
def video_status(
    video_id: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    avideo_status: Literal[False] = False,
    **kwargs,
) -> VideoObject:
    ...

# fmt: on


@client
def video_status(  # noqa: PLR0915
    video_id: str,
    timeout=600,  # default to 10 minutes
    custom_llm_provider=None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    VideoObject,
    Coroutine[Any, Any, VideoObject],
]:
    """
    Retrieve video status from OpenAI's video API.

    Args:
        video_id (str): The identifier of the video whose status to retrieve.
        timeout (int): The timeout for the request in seconds.
        custom_llm_provider (Optional[str]): The LLM provider to use. If not provided, will be auto-detected.
        extra_headers (Optional[Dict[str, Any]]): Additional headers to include in the request.
        extra_query (Optional[Dict[str, Any]]): Additional query parameters.
        extra_body (Optional[Dict[str, Any]]): Additional body parameters.

    Returns:
        VideoObject: The video status information.

    Example:
        ```python
        import litellm

        # Get video status
        video_status = litellm.video_status(
            video_id="video_123"
        )

        print(f"Video status: {video_status.status}")
        print(f"Progress: {video_status.progress}%")
        ```
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response", None)
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)

            response = VideoObject(**mock_response)
            return response

        # Try to decode provider from video_id if not explicitly provided
        if custom_llm_provider is None:
            decoded = decode_video_id_with_provider(video_id)
            custom_llm_provider = decoded.get("custom_llm_provider") or "openai"

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        # get provider config
        video_status_provider_config: Optional[BaseVideoConfig] = (
            ProviderConfigManager.get_provider_video_config(
                model=None,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if video_status_provider_config is None:
            raise ValueError(f"video status is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # For video status, we need the video_id
        video_status_request_params: Dict = {
            "video_id": video_id,
        }

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            user=kwargs.get("user"),
            optional_params=dict(video_status_request_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **video_status_request_params,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type for video status
        litellm_logging_obj.call_type = CallTypes.video_retrieve.value

        # Call the handler with _is_async flag instead of directly calling the async handler
        return base_llm_http_handler.video_status_handler(
            video_id=video_id,
            video_status_provider_config=video_status_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
            client=kwargs.get("client"),
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
