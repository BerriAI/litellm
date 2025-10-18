######################################################################
#
#                          /v1/videos Endpoints
#
# Equivalent of https://platform.openai.com/docs/api-reference/videos
######################################################################

import asyncio
import traceback
from typing import Optional

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

import litellm
from litellm import CreateVideoRequest, get_secret_str
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_body,
    get_custom_llm_provider_from_request_query,
)
from litellm.types.llms.openai import OpenAIVideoObject

router = APIRouter()

videos_config = None


def set_videos_config(config):
    global videos_config
    if config is None:
        return

    if not isinstance(config, list):
        raise ValueError("invalid videos config, expected a list is not a list")

    for element in config:
        if isinstance(element, dict):
            for key, value in element.items():
                if isinstance(value, str) and value.startswith("os.environ/"):
                    element[key] = get_secret_str(value)

    videos_config = config


def get_videos_provider_config(
    custom_llm_provider: str,
):
    global videos_config
    if custom_llm_provider == "vertex_ai":
        return None
    if videos_config is None:
        raise ValueError(
            "videos_settings is not set, set it on your config.yaml file."
        )
    for setting in videos_config:
        if setting.get("custom_llm_provider") == custom_llm_provider:
            return setting
    return None


async def route_create_video(
    _create_video_request: CreateVideoRequest,
    custom_llm_provider: str,
) -> OpenAIVideoObject:
    # get configs for custom_llm_provider
    llm_provider_config = get_videos_provider_config(
        custom_llm_provider=custom_llm_provider
    )
    if llm_provider_config is not None:
        # add llm_provider_config to data
        _create_video_request.update(llm_provider_config)
    _create_video_request.pop("custom_llm_provider", None)  # type: ignore
    response = await litellm.acreate_video(**_create_video_request, custom_llm_provider=custom_llm_provider)  # type: ignore

    return response


@router.post(
    "/{provider}/v1/videos",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.post(
    "/v1/videos",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.post(
    "/videos",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
async def create_video(
    request: Request,
    fastapi_response: Response,
    prompt: str = Form(...),
    provider: Optional[str] = None,
    custom_llm_provider: str = Form(default="openai"),
    model: Optional[str] = Form(default=None),
    seconds: Optional[str] = Form(default=None),
    size: Optional[str] = Form(default=None),
    input_reference: Optional[UploadFile] = File(default=None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a video generation job
    This is the equivalent of POST https://api.openai.com/v1/videos

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/videos/create

    Example Curl
    ```
    curl http://localhost:4000/v1/videos \
        -H "Authorization: Bearer sk-1234" \
        -F prompt="A cat playing with yarn" \
        -F model="sora-2" \
        -F seconds="4" \
        -F size="720x1280" \
        -F input_reference="@start_frame.jpg;type=image/jpeg"

    ```
    """
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )

        data = {}

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # Build request with only non-None values as it throws an error if any of the values are None
        _create_video_request = CreateVideoRequest(
            prompt=prompt,
            **data,
        )
        if model is not None:
            _create_video_request["model"] = model
        if seconds is not None:
            _create_video_request["seconds"] = seconds
        if size is not None:
            _create_video_request["size"] = size
        if input_reference is not None:
            # Read the file content for input_reference
            file_content = await input_reference.read()
            file_data = (
                input_reference.filename or "input_reference",
                file_content,
                input_reference.content_type or "application/octet-stream"
            )
            _create_video_request["input_reference"] = file_data

        response = await route_create_video(
            _create_video_request=_create_video_request,
            custom_llm_provider=custom_llm_provider,
        )

        if response is None:
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to create video. Please try again."},
            )
        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ## POST CALL HOOKS ###
        _response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response  # type: ignore
        )
        if _response is not None and isinstance(_response, OpenAIVideoObject):
            response = _response

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.create_video(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.get(
    "/{provider}/v1/videos/{video_id:path}/content",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.get(
    "/v1/videos/{video_id:path}/content",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.get(
    "/videos/{video_id:path}/content",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
async def get_video_content(
    request: Request,
    fastapi_response: Response,
    video_id: str,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Download the generated video content
    This is the equivalent of GET https://api.openai.com/v1/videos/{video_id}/content

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/videos/content

    Example Curl
    ```
    curl http://localhost:4000/v1/videos/video-abc123/content \
        -H "Authorization: Bearer sk-1234" \
        --output video.mp4

    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {"video_id": video_id}
    try:
        # Include original request and headers in the data
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
        (
            data,
            litellm_logging_obj,
        ) = await base_llm_response_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type="avideo_content",  # type: ignore
        )

        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )

        response = await litellm.avideo_content(
            **{
                "custom_llm_provider": custom_llm_provider,
                "video_id": video_id,
                **data,
            }  # type: ignore
        )

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        httpx_response: Optional[httpx.Response] = getattr(response, "response", None)
        if httpx_response is None:
            raise ValueError(
                f"Invalid response - response.response is None - got {response}"
            )

        return Response(
            content=httpx_response.content,
            status_code=httpx_response.status_code,
            headers=dict(httpx_response.headers),
        )

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.get_video_content(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.get(
    "/{provider}/v1/videos/{video_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.get(
    "/v1/videos/{video_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.get(
    "/videos/{video_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
async def get_video(
    request: Request,
    fastapi_response: Response,
    video_id: str,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Returns information about a specific video generation job
    This is the equivalent of GET https://api.openai.com/v1/videos/{video_id}

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/videos/retrieve

    Example Curl
    ```
    curl http://localhost:4000/v1/videos/video-abc123 \
        -H "Authorization: Bearer sk-1234"

    ```
    """
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )
        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        response = await litellm.avideo_retrieve(
            custom_llm_provider=custom_llm_provider, video_id=video_id, **data  # type: ignore
        )

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ## POST CALL HOOKS ###
        _response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response  # type: ignore
        )
        if _response is not None and isinstance(_response, OpenAIVideoObject):
            response = _response

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        return response

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.get_video(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.delete(
    "/{provider}/v1/videos/{video_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.delete(
    "/v1/videos/{video_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.delete(
    "/videos/{video_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
async def delete_video(
    request: Request,
    fastapi_response: Response,
    video_id: str,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel a video generation job
    This is the equivalent of DELETE https://api.openai.com/v1/videos/{video_id}

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/videos/delete

    Example Curl
    ```
    curl http://localhost:4000/v1/videos/video-abc123 \
    -X DELETE \
    -H "Authorization: Bearer sk-1234"

    ```
    """
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )
        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        response = await litellm.avideo_delete(
            custom_llm_provider=custom_llm_provider, video_id=video_id, **data  # type: ignore
        )

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        return response

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.delete_video(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.get(
    "/{provider}/v1/videos",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.get(
    "/v1/videos",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
@router.get(
    "/videos",
    dependencies=[Depends(user_api_key_auth)],
    tags=["videos"],
)
async def list_videos(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    provider: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
):
    """
    Returns a list of video generation jobs
    This is the equivalent of GET https://api.openai.com/v1/videos

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/videos/list

    Example Curl
    ```
    curl http://localhost:4000/v1/videos \
        -H "Authorization: Bearer sk-1234"

    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        # Include original request and headers in the data
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
        (
            data,
            litellm_logging_obj,
        ) = await base_llm_response_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type=CallTypes.avideo_list.value,  # type: ignore
        )

        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )

        if limit is not None:
            data["limit"] = limit
        if after is not None:
            data["after"] = after

        response = await litellm.avideo_list(
            custom_llm_provider=custom_llm_provider, **data  # type: ignore
        )

        ## POST CALL HOOKS ###
        _response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response  # type: ignore
        )
        if _response is not None and isinstance(_response, OpenAIVideoObject):
            response = _response

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        return response

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.list_videos(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )
