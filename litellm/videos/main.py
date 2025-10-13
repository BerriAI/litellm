"""
Main File for Videos API implementation

https://platform.openai.com/docs/api-reference/videos

"""

import asyncio
import contextvars
import os
from functools import partial
from typing import Any, Coroutine, Dict, Literal, Optional, Union, cast

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.openai import OpenAIVideosAPI
from litellm.types.llms.openai import (
    CreateVideoRequest,
    HttpxBinaryResponseContent,
    OpenAIVideoObject,
)
from litellm.types.router import *
from litellm.utils import (
    client,
    supports_httpx_timeout,
)

base_llm_http_handler = BaseLLMHTTPHandler()

####### ENVIRONMENT VARIABLES ###################
openai_videos_instance = OpenAIVideosAPI()
#################################################


@client
async def acreate_video(
    prompt: str,
    custom_llm_provider: Literal["openai"] = "openai",
    model: Optional[str] = None,
    seconds: Optional[str] = None,
    size: Optional[str] = None,
    input_reference: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> OpenAIVideoObject:
    """
    Async: Create a video generation job

    LiteLLM Equivalent of POST: POST https://api.openai.com/v1/videos
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_video"] = True

        call_args = {
            "prompt": prompt,
            "model": model,
            "seconds": seconds,
            "size": size,
            "custom_llm_provider": custom_llm_provider,
            "extra_headers": extra_headers,
            "extra_body": extra_body,
            **kwargs,
        }

        # Use a partial function to pass your keyword arguments
        func = partial(create_video, **call_args)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response  # type: ignore

        return response
    except Exception as e:
        raise e


@client
def create_video(
    prompt: str,
    custom_llm_provider: Optional[Literal["openai"]] = None,
    model: Optional[str] = None,
    seconds: Optional[str] = None,
    size: Optional[str] = None,
    input_reference: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[OpenAIVideoObject, Coroutine[Any, Any, OpenAIVideoObject]]:
    """
    Create a video generation job

    LiteLLM Equivalent of POST: POST https://api.openai.com/v1/videos
    """
    try:
        _is_async = kwargs.pop("acreate_video", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)
        logging_obj = cast(
            Optional[LiteLLMLoggingObj], kwargs.get("litellm_logging_obj")
        )
        if logging_obj is None:
            raise ValueError("logging_obj is required")

        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(cast(str, custom_llm_provider)) is False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout  # default 10 min timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        # Build request with only non-None values as it throws an error if any of the values are None
        _create_video_request = CreateVideoRequest(
            prompt=prompt,
        )
        if model is not None:
            _create_video_request["model"] = model
        if seconds is not None:
            _create_video_request["seconds"] = seconds
        if size is not None:
            _create_video_request["size"] = size
        if input_reference is not None:
            _create_video_request["input_reference"] = input_reference
        if extra_headers is not None:
            _create_video_request["extra_headers"] = extra_headers
        if extra_body is not None:
            _create_video_request["extra_body"] = extra_body

        if custom_llm_provider == "openai" or custom_llm_provider is None:
            # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None
            )
            # set API KEY
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            response = openai_videos_instance.create_video(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
                create_video_data=dict(_create_video_request),
            )
        else:
            raise ValueError(
                f"LiteLLM doesn't support {custom_llm_provider} for video generation. Only 'openai' is supported."
            )

        if response is None or isinstance(response, BaseException):
            raise Exception(f"API call failed: {str(response)}")
        elif asyncio.iscoroutine(response):
            if _is_async:
                return response
            else:
                return asyncio.run(response)
        
        # Add usage information for cost calculation
        if hasattr(response, 'seconds') and response.seconds is not None:
            try:
                duration_seconds = int(response.seconds)
                response.usage = {
                    "video_duration_seconds": duration_seconds,
                    "model": response.model or model or "sora-2",
                    "size": response.size or size or "720x1280"
                }
            except (ValueError, TypeError):
                # If seconds is not a valid integer, use default
                response.usage = {
                    "video_duration_seconds": 4,
                    "model": response.model or model or "sora-2", 
                    "size": response.size or size or "720x1280"
                }
        else:
            # Default usage information
            response.usage = {
                "video_duration_seconds": 4,
                "model": response.model or model or "sora-2",
                "size": response.size or size or "720x1280"
            }
        
        return response
    except Exception as e:
        raise e


@client
async def avideo_retrieve(
    video_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    **kwargs,
) -> OpenAIVideoObject:
    """
    Async: Retrieve information about a video generation job

    LiteLLM Equivalent of GET: GET https://api.openai.com/v1/videos/{video_id}
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["avideo_retrieve"] = True

        call_args = {
            "video_id": video_id,
            "custom_llm_provider": custom_llm_provider,
            **kwargs,
        }

        # Use a partial function to pass your keyword arguments
        func = partial(video_retrieve, **call_args)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response  # type: ignore

        return response
    except Exception as e:
        raise e


@client
def video_retrieve(
    video_id: str,
    custom_llm_provider: Optional[Literal["openai"]] = None,
    **kwargs,
) -> Union[OpenAIVideoObject, Coroutine[Any, Any, OpenAIVideoObject]]:
    """
    Retrieve information about a video generation job

    LiteLLM Equivalent of GET: GET https://api.openai.com/v1/videos/{video_id}
    """
    try:
        _is_async = kwargs.pop("avideo_retrieve", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)
        
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(cast(str, custom_llm_provider)) is False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        if custom_llm_provider == "openai" or custom_llm_provider is None:
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            response = openai_videos_instance.retrieve_video(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
                video_id=video_id,
            )
        else:
            raise ValueError(
                f"LiteLLM doesn't support {custom_llm_provider} for video retrieval. Only 'openai' is supported."
            )

        if response is None or isinstance(response, BaseException):
            raise Exception(f"API call failed: {str(response)}")
        elif asyncio.iscoroutine(response):
            if _is_async:
                return response
            else:
                return asyncio.run(response)
        return response
    except Exception as e:
        raise e


@client
async def avideo_delete(
    video_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    **kwargs,
) -> Dict:
    """
    Async: Cancel a video generation job

    LiteLLM Equivalent of DELETE: DELETE https://api.openai.com/v1/videos/{video_id}
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["avideo_delete"] = True

        call_args = {
            "video_id": video_id,
            "custom_llm_provider": custom_llm_provider,
            **kwargs,
        }

        # Use a partial function to pass your keyword arguments
        func = partial(video_delete, **call_args)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response  # type: ignore

        return response
    except Exception as e:
        raise e


@client
def video_delete(
    video_id: str,
    custom_llm_provider: Optional[Literal["openai"]] = None,
    **kwargs,
) -> Union[Dict, Coroutine[Any, Any, Dict]]:
    """
    Cancel a video generation job

    LiteLLM Equivalent of DELETE: DELETE https://api.openai.com/v1/videos/{video_id}
    """
    try:
        _is_async = kwargs.pop("avideo_delete", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)
        
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(cast(str, custom_llm_provider)) is False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        if custom_llm_provider == "openai" or custom_llm_provider is None:
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            response = openai_videos_instance.delete_video(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
                video_id=video_id,
            )
        else:
            raise ValueError(
                f"LiteLLM doesn't support {custom_llm_provider} for video deletion. Only 'openai' is supported."
            )

        if response is None or isinstance(response, BaseException):
            raise Exception(f"API call failed: {str(response)}")
        elif asyncio.iscoroutine(response):
            if _is_async:
                return response
            else:
                return asyncio.run(response)
        return response
    except Exception as e:
        raise e


@client
async def avideo_content(
    video_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    **kwargs,
) -> HttpxBinaryResponseContent:
    """
    Async: Download the generated video content

    LiteLLM Equivalent of GET: GET https://api.openai.com/v1/videos/{video_id}/content
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["avideo_content"] = True

        call_args = {
            "video_id": video_id,
            "custom_llm_provider": custom_llm_provider,
            **kwargs,
        }

        # Use a partial function to pass your keyword arguments
        func = partial(video_content, **call_args)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response  # type: ignore

        return response
    except Exception as e:
        raise e


@client
def video_content(
    video_id: str,
    custom_llm_provider: Optional[Literal["openai"]] = None,
    **kwargs,
) -> Union[HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]]:
    """
    Download the generated video content

    LiteLLM Equivalent of GET: GET https://api.openai.com/v1/videos/{video_id}/content
    """
    try:
        _is_async = kwargs.pop("avideo_content", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)
        
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(cast(str, custom_llm_provider)) is False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        if custom_llm_provider == "openai" or custom_llm_provider is None:
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            response = openai_videos_instance.video_content(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
                video_id=video_id,
            )
        else:
            raise ValueError(
                f"LiteLLM doesn't support {custom_llm_provider} for video content. Only 'openai' is supported."
            )

        if response is None or isinstance(response, BaseException):
            raise Exception(f"API call failed: {str(response)}")
        elif asyncio.iscoroutine(response):
            if _is_async:
                return response
            else:
                return asyncio.run(response)
        return response
    except Exception as e:
        raise e


@client
async def avideo_list(
    custom_llm_provider: Literal["openai"] = "openai",
    limit: Optional[int] = None,
    after: Optional[str] = None,
    **kwargs,
) -> Dict:
    """
    Async: List video generation jobs

    LiteLLM Equivalent of GET: GET https://api.openai.com/v1/videos
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["avideo_list"] = True

        call_args = {
            "custom_llm_provider": custom_llm_provider,
            "limit": limit,
            "after": after,
            **kwargs,
        }

        # Use a partial function to pass your keyword arguments
        func = partial(video_list, **call_args)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response  # type: ignore

        return response
    except Exception as e:
        raise e


@client
def video_list(
    custom_llm_provider: Optional[Literal["openai"]] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    **kwargs,
) -> Union[Dict, Coroutine[Any, Any, Dict]]:
    """
    List video generation jobs

    LiteLLM Equivalent of GET: GET https://api.openai.com/v1/videos
    """
    try:
        _is_async = kwargs.pop("avideo_list", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)
        
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(cast(str, custom_llm_provider)) is False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        if custom_llm_provider == "openai" or custom_llm_provider is None:
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            response = openai_videos_instance.list_videos(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
                limit=limit,
                after=after,
            )
        else:
            raise ValueError(
                f"LiteLLM doesn't support {custom_llm_provider} for video listing. Only 'openai' is supported."
            )

        if response is None or isinstance(response, BaseException):
            raise Exception(f"API call failed: {str(response)}")
        elif asyncio.iscoroutine(response):
            if _is_async:
                return response
            else:
                return asyncio.run(response)
        return response
    except Exception as e:
        raise e

