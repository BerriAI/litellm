import asyncio
import contextvars
import time
from functools import partial
from typing import Any, Coroutine, Literal, Optional, Union, overload, Dict

import litellm
import orjson
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.videos.main import (
    VideoCreateOptionalRequestParams,
    VideoObject,
    VideoResponse,
    VideoUsage,
)
from litellm.videos.utils import VideoGenerationRequestUtils
from litellm.constants import DEFAULT_VIDEO_ENDPOINT_MODEL, request_timeout as DEFAULT_REQUEST_TIMEOUT
from litellm.main import base_llm_http_handler
from litellm.types.utils import all_litellm_params, LlmProviders, CallTypes
from litellm.utils import client, exception_type, ProviderConfigManager
from litellm.types.utils import FileTypes
from litellm.types.router import GenericLiteLLMParams
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.videos_generation.transformation import BaseVideoGenerationConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.custom_llm import CustomLLM

#################### Initialize provider clients ####################
llm_http_handler: BaseLLMHTTPHandler = BaseLLMHTTPHandler()

##### Video Generation #######################
@client
async def avideo_generation(*args, **kwargs) -> VideoResponse:
    """
    Asynchronously calls the `video_generation` function with the given arguments and keyword arguments.

    Parameters:
    - `args` (tuple): Positional arguments to be passed to the `video_generation` function.
    - `kwargs` (dict): Keyword arguments to be passed to the `video_generation` function.

    Returns:
    - `response` (VideoResponse): The response returned by the `video_generation` function.
    """
    loop = asyncio.get_event_loop()
    model = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Video Generation ###
    kwargs["avideo_generation"] = True
    custom_llm_provider = None
    try:
        # Check for mock response first
        mock_response = kwargs.get("mock_response", None)
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = orjson.loads(mock_response)
            
            response = VideoResponse(
                data=[VideoObject(**video_data) for video_data in mock_response.get("data", [])],
                usage=VideoUsage(**mock_response.get("usage", {})),
                hidden_params=kwargs.get("hidden_params", {}),
            )
            return response

        # Use a partial function to pass your keyword arguments
        func = partial(video_generation, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=kwargs.get("api_base", None)
        )

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)

        response: Optional[VideoResponse] = None
        if isinstance(init_response, dict):
            response = VideoResponse(**init_response)
        elif isinstance(init_response, VideoResponse):  ## CACHING SCENARIO
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response  # type: ignore

        if response is None:
            raise ValueError(
                "Unable to get Video Response. Please pass a valid llm_provider."
            )

        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
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
) -> Coroutine[Any, Any, VideoResponse]: 
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
) -> VideoResponse: 
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
    VideoResponse,
    Coroutine[Any, Any, VideoResponse],
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
                mock_response = orjson.loads(mock_response)
            
            response = VideoResponse(
                data=[VideoObject(**video_data) for video_data in mock_response.get("data", [])],
                usage=VideoUsage(**mock_response.get("usage", {})),
                hidden_params=kwargs.get("hidden_params", {}),
            )
            return response

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        model, custom_llm_provider, _, _ = get_llm_provider(
            model=model or DEFAULT_VIDEO_ENDPOINT_MODEL,
            custom_llm_provider=custom_llm_provider,
        )

        # get provider config
        video_generation_provider_config: Optional[BaseVideoGenerationConfig] = (
            ProviderConfigManager.get_provider_video_generation_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if video_generation_provider_config is None:
            raise ValueError(f"image edit is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # Get ImageEditOptionalRequestParams with only valid parameters
        video_generation_optional_params: VideoCreateOptionalRequestParams = (
            VideoGenerationRequestUtils.get_requested_video_generation_optional_param(local_vars)
        )

        # Get optional parameters for the responses API
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
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Video Retrieval #######################
@client
async def avideo_retrieve(*args, **kwargs) -> VideoResponse:
    """
    Asynchronously retrieve a video generation job status.
    """
    loop = asyncio.get_event_loop()
    video_id = args[0] if len(args) > 0 else kwargs["video_id"]
    
    kwargs["avideo_retrieve"] = True
    custom_llm_provider = None
    try:
        # Check for mock response first
        mock_response = kwargs.get("mock_response", None)
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = orjson.loads(mock_response)
            
            response = VideoResponse(
                data=[VideoObject(**video_data) for video_data in mock_response.get("data", [])],
                usage=VideoUsage(**mock_response.get("usage", {})),
                hidden_params=kwargs.get("hidden_params", {}),
            )
            return response

        # Use a partial function to pass your keyword arguments
        func = partial(video_retrieve, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=kwargs.get("model", "sora-2"), api_base=kwargs.get("api_base", None)
        )

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)

        response: Optional[VideoResponse] = None
        if isinstance(init_response, dict):
            response = VideoResponse(**init_response)
        elif isinstance(init_response, VideoResponse):  ## CACHING SCENARIO
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response  # type: ignore

        if response is None:
            raise ValueError(
                "Unable to get Video Response. Please pass a valid llm_provider."
            )

        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=kwargs.get("model", "sora-2"),
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def video_retrieve(
    video_id: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VideoResponse:
    """
    Retrieve a video generation job status.
    """
    try:
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # For now, return a placeholder response since video retrieval is not yet implemented
        verbose_logger.warning(
            f"Video retrieval for provider {custom_llm_provider} is not yet implemented."
        )
        
        model_response = VideoResponse(
            data=[
                VideoObject(
                    id=video_id,
                    object="video",
                    status="completed",
                    created_at=int(time.time()),
                    completed_at=int(time.time()),
                    model=model or "sora-2"
                )
            ],
            usage=VideoUsage(),
            hidden_params=kwargs.get("hidden_params", {}),
        )

        return model_response

    except Exception as e:
        verbose_logger.error(f"Error in video_retrieve: {e}")
        raise e


##### Video Deletion #######################
@client
async def avideo_delete(*args, **kwargs) -> bool:
    """
    Asynchronously delete a video generation job.
    """
    loop = asyncio.get_event_loop()
    video_id = args[0] if len(args) > 0 else kwargs["video_id"]
    
    kwargs["avideo_delete"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(video_delete, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=kwargs.get("model", "sora-2"), api_base=kwargs.get("api_base", None)
        )

        # Await normally
        result = await loop.run_in_executor(None, func_with_context)
        return result
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=kwargs.get("model", "sora-2"),
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def video_delete(
    video_id: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> bool:
    """
    Delete a video generation job.
    """
    try:
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # For now, return a placeholder response since video deletion is not yet implemented
        verbose_logger.warning(
            f"Video deletion for provider {custom_llm_provider} is not yet implemented."
        )
        result = True

        return result

    except Exception as e:
        verbose_logger.error(f"Error in video_delete: {e}")
        raise e


##### Video Content Download #######################
@client
async def avideo_content(*args, **kwargs) -> bytes:
    """
    Asynchronously download video content.
    """
    loop = asyncio.get_event_loop()
    video_id = args[0] if len(args) > 0 else kwargs["video_id"]
    
    kwargs["avideo_content"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(video_content, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=kwargs.get("model", "sora-2"), api_base=kwargs.get("api_base", None)
        )

        # Await normally
        result = await loop.run_in_executor(None, func_with_context)
        return result
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=kwargs.get("model", "sora-2"),
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def video_content(
    video_id: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> bytes:
    """
    Download video content.
    """
    try:
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # For now, return placeholder content
        verbose_logger.warning(
            f"Video content download for provider {custom_llm_provider} is not yet implemented."
        )
        
        # Return placeholder video content
        return b"placeholder video content"

    except Exception as e:
        verbose_logger.error(f"Error in video_content: {e}")
        raise e


##### Video Listing #######################
@client
async def avideo_list(*args, **kwargs) -> VideoResponse:
    """
    Asynchronously list video generation jobs.
    """
    loop = asyncio.get_event_loop()
    
    kwargs["avideo_list"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(video_list, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=kwargs.get("model", "sora-2"), api_base=kwargs.get("api_base", None)
        )

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)

        response: Optional[VideoResponse] = None
        if isinstance(init_response, dict):
            response = VideoResponse(**init_response)
        elif isinstance(init_response, VideoResponse):  ## CACHING SCENARIO
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response  # type: ignore

        if response is None:
            raise ValueError(
                "Unable to get Video Response. Please pass a valid llm_provider."
            )

        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=kwargs.get("model", "sora-2"),
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def video_list(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> VideoResponse:
    """
    List video generation jobs.
    """
    try:
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # For now, return placeholder response
        verbose_logger.warning(
            f"Video listing for provider {custom_llm_provider} is not yet implemented."
        )
        
        model_response = VideoResponse(
            data=[],
            usage=VideoUsage(),
            hidden_params=kwargs.get("hidden_params", {}),
        )

        return model_response

    except Exception as e:
        verbose_logger.error(f"Error in video_list: {e}")
        raise e


# Convenience functions with better names
create_video = video_generation
acreate_video = avideo_generation


def video_content(
    video_id: str,
    custom_llm_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    litellm_params: Optional[Dict[str, Any]] = None,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    variant: Optional[str] = None,
) -> bytes:
    """
    Download video content from OpenAI's video API.

    Args:
        video_id (str): The identifier of the video whose content to download.
        custom_llm_provider (Optional[str]): The LLM provider to use. If not provided, will be auto-detected.
        api_key (Optional[str]): The API key to use for authentication.
        api_base (Optional[str]): The base URL for the API.
        timeout (Optional[float]): The timeout for the request in seconds.
        extra_headers (Optional[Dict[str, Any]]): Additional headers to include in the request.
        litellm_params (Optional[Dict[str, Any]]): Additional LiteLLM parameters.
        client (Optional[Union[HTTPHandler, AsyncHTTPHandler]]): The HTTP client to use.
        variant (Optional[str]): Which downloadable asset to return. Defaults to the MP4 video.

    Returns:
        bytes: The raw video content as bytes.

    Example:
        ```python
        import litellm

        # Download video content
        video_bytes = litellm.video_content(
            video_id="video_123",
            custom_llm_provider="openai"
        )

        # Save to file
        with open("video.mp4", "wb") as f:
            f.write(video_bytes)
        ```
    """
    if litellm_params is None:
        litellm_params = {}

    # Set default timeout if not provided
    if timeout is None:
        timeout = DEFAULT_REQUEST_TIMEOUT

    # Get the LLM provider if not provided
    if custom_llm_provider is None:
        custom_llm_provider = "openai"  # Default to OpenAI for video content

    # Get the provider config
    from litellm.llms.openai.video_retrieval.transformation import OpenAIVideoRetrievalConfig
    video_content_provider_config = OpenAIVideoRetrievalConfig()

    # Create logging object
    logging_obj = LiteLLMLoggingObj(
        model="",  # No model needed for content download
        messages=[],  # No messages for content download
        stream=False,
        call_type=CallTypes.video_content.value,
        start_time=time.time(),
        litellm_call_id="",
        function_id="video_content",
    )

    # Get the HTTP handler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    base_llm_http_handler = BaseLLMHTTPHandler()

    # Call the video content handler
    return base_llm_http_handler.video_content_handler(
        video_id=video_id,
        video_content_provider_config=video_content_provider_config,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        timeout=timeout,
        extra_headers=extra_headers,
        api_key=api_key,
        client=client,
        variant=variant,
    )


async def async_video_content(
    video_id: str,
    custom_llm_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    litellm_params: Optional[Dict[str, Any]] = None,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    variant: Optional[str] = None,
) -> bytes:
    """
    Async version of video content download.

    Args:
        video_id (str): The identifier of the video whose content to download.
        custom_llm_provider (Optional[str]): The LLM provider to use. If not provided, will be auto-detected.
        api_key (Optional[str]): The API key to use for authentication.
        api_base (Optional[str]): The base URL for the API.
        timeout (Optional[float]): The timeout for the request in seconds.
        extra_headers (Optional[Dict[str, Any]]): Additional headers to include in the request.
        litellm_params (Optional[Dict[str, Any]]): Additional LiteLLM parameters.
        client (Optional[Union[HTTPHandler, AsyncHTTPHandler]]): The HTTP client to use.
        variant (Optional[str]): Which downloadable asset to return. Defaults to the MP4 video.

    Returns:
        bytes: The raw video content as bytes.

    Example:
        ```python
        import litellm

        # Download video content asynchronously
        video_bytes = await litellm.async_video_content(
            video_id="video_123",
            custom_llm_provider="openai"
        )

        # Save to file
        with open("video.mp4", "wb") as f:
            f.write(video_bytes)
        ```
    """
    if litellm_params is None:
        litellm_params = {}

    # Set default timeout if not provided
    if timeout is None:
        timeout = DEFAULT_REQUEST_TIMEOUT

    # Get the LLM provider if not provided
    if custom_llm_provider is None:
        custom_llm_provider = "openai"  # Default to OpenAI for video content

    # Get the provider config
    from litellm.llms.openai.video_retrieval.transformation import OpenAIVideoRetrievalConfig
    video_content_provider_config = OpenAIVideoRetrievalConfig()

    # Create logging object
    logging_obj = LiteLLMLoggingObj(
        model="",  # No model needed for content download
        messages=[],  # No messages for content download
        stream=False,
        call_type=CallTypes.video_content.value,
        start_time=time.time(),
        litellm_call_id="",
        function_id="async_video_content",
    )

    # Get the HTTP handler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    base_llm_http_handler = BaseLLMHTTPHandler()

    # Call the async video content handler
    return await base_llm_http_handler.async_video_content_handler(
        video_id=video_id,
        video_content_provider_config=video_content_provider_config,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        timeout=timeout,
        extra_headers=extra_headers,
        api_key=api_key,
        client=client,
        variant=variant,
    )