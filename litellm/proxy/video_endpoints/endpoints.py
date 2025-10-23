#### Video Endpoints #####

import orjson
from fastapi import APIRouter, Depends, Request, Response, UploadFile, File
from fastapi.responses import ORJSONResponse
from typing import Optional

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.image_endpoints.endpoints import batch_to_bytesio
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

router = APIRouter()


@router.post(
    "/v1/videos",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.post(
    "/videos",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_generation(
    request: Request,
    fastapi_response: Response,
    input_reference: Optional[UploadFile] = File(None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video generation endpoint for creating videos from text prompts.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/videos" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "sora-2",
            "prompt": "A beautiful sunset over the ocean"
        }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body
    data = await _read_request_body(request=request)
    if input_reference is not None:
        input_reference_file = await batch_to_bytesio([input_reference])
        if input_reference_file:
            data["input_reference"] = input_reference_file[0]

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_generation",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/videos",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.get(
    "/videos",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_list(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video list endpoint for retrieving a list of videos.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/videos" \
        -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read query parameters
    query_params = dict(request.query_params)
    data = {"query_params": query_params}

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_list",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/videos/{video_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.get(
    "/videos/{video_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_status(
    video_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video status endpoint for retrieving video status and metadata.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/videos/video_123" \
        -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Create data with video_id
    data = {"video_id": video_id}

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_status",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/videos/{video_id}/content",
    dependencies=[Depends(user_api_key_auth)],
    response_class=Response,
    tags=["videos"],
)
@router.get(
    "/videos/{video_id}/content",
    dependencies=[Depends(user_api_key_auth)],
    response_class=Response,
    tags=["videos"],
)
async def video_content(
    video_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video content endpoint for downloading video content.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/videos/video_123/content" \
        -H "Authorization: Bearer sk-1234" \
        --output video.mp4
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Create data with video_id
    data = {"video_id": video_id}

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        # Call the video content function directly to get raw bytes
        video_bytes = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_content",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        
        # Return raw video bytes with proper content type
        return Response(
            content=video_bytes,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=video_{video_id}.mp4"
            }
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/videos/{video_id}/remix",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.post(
    "/videos/{video_id}/remix",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_remix(
    video_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video remix endpoint for remixing existing videos with new prompts.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/videos/video_123/remix" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "A new version with different colors"
        }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body
    body = await request.body()
    data = orjson.loads(body)
    data["video_id"] = video_id

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_remix",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
