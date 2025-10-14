import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Literal, Optional, Union, overload, Dict

import json
import litellm
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
from litellm.utils import client, ProviderConfigManager
from litellm.types.utils import FileTypes
from litellm.types.router import GenericLiteLLMParams
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.videos_generation.transformation import BaseVideoGenerationConfig
from litellm.llms.base_llm.video_retrieval.transformation import BaseVideoRetrievalConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

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
) -> VideoResponse:
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
                mock_response = json.loads(mock_response)
            
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
        # Get VideoGenerationOptionalRequestParams with only valid parameters
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


@client
def video_content(
    video_id: str,
    model: Optional[str] = None,
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
        model (Optional[str]): The model to use. If not provided, will be auto-detected.
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
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("async_call", False) is True

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        model, custom_llm_provider, _, _ = get_llm_provider(
            model=model or "sora-2",  # Default model for video content
            custom_llm_provider=custom_llm_provider,
        )

        # get provider config
        video_content_provider_config: Optional[BaseVideoRetrievalConfig] = (
            ProviderConfigManager.get_provider_video_content_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if video_content_provider_config is None:
            raise ValueError(f"video content download is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # For video content download, we don't need complex optional parameter handling
        # Just pass the basic parameters that are relevant for content download
        video_content_request_params: Dict = {
            "video_id": video_id,
        }

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=model,
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
            model=model,
            video_content_provider_config=video_content_provider_config,
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
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Video Content Download #######################
@client
async def avideo_content(
    video_id: str,
    model: Optional[str] = None,
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
    - `model` (Optional[str]): The model to use. If not provided, will be auto-detected
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

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model or DEFAULT_VIDEO_ENDPOINT_MODEL, api_base=api_base
            )

        func = partial(
            video_content,
            video_id=video_id,
            model=model,
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
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
