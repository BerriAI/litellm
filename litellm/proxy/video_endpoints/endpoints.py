#### Video Endpoints #####

from typing import Any, Dict, Optional

import orjson
from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_body,
    get_custom_llm_provider_from_request_headers,
    get_custom_llm_provider_from_request_query,
)
from litellm.proxy.image_endpoints.endpoints import batch_to_bytesio
from litellm.proxy.video_endpoints.utils import (
    encode_character_id_in_response,
    extract_model_from_target_model_names,
    get_custom_provider_from_data,
)
from litellm.types.videos.utils import (
    decode_character_id_with_provider,
    decode_video_id_with_provider,
)

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
    data: Dict[str, Any] = {"query_params": query_params}

    # Extract custom_llm_provider from headers, query params, or body
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or await get_custom_llm_provider_from_request_body(request=request)
    )
    if custom_llm_provider:
        data["custom_llm_provider"] = custom_llm_provider
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
    data: Dict[str, Any] = {"video_id": video_id}

    decoded = decode_video_id_with_provider(video_id)
    provider_from_id = decoded.get("custom_llm_provider")
    model_id_from_decoded = decoded.get("model_id")

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or await get_custom_llm_provider_from_request_body(request=request)
        or provider_from_id
        or "openai"
    )
    if custom_llm_provider:
        data["custom_llm_provider"] = custom_llm_provider

    # Resolve model_name from model_id if available
    # This allows the router to automatically inject litellm_params from the model config
    if model_id_from_decoded and llm_router:
        resolved_model = llm_router.resolve_model_name_from_model_id(
            model_id_from_decoded
        )
        if resolved_model:
            data["model"] = resolved_model

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
    curl -X GET "http://localhost:4000/v1/videos/{video_id}/content" \
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
    data: Dict[str, Any] = {"video_id": video_id}

    decoded = decode_video_id_with_provider(video_id)
    provider_from_id = decoded.get("custom_llm_provider")
    model_id_from_decoded = decoded.get("model_id")

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or await get_custom_llm_provider_from_request_body(request=request)
        or provider_from_id
    )
    if custom_llm_provider:
        data["custom_llm_provider"] = custom_llm_provider

    # Resolve model_name from model_id if available
    # This allows the router to automatically inject litellm_params from the model config
    if model_id_from_decoded and llm_router:
        resolved_model = llm_router.resolve_model_name_from_model_id(
            model_id_from_decoded
        )
        if resolved_model:
            data["model"] = resolved_model
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
            },
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

    decoded = decode_video_id_with_provider(video_id)
    provider_from_id = decoded.get("custom_llm_provider")
    model_id_from_decoded = decoded.get("model_id")

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or data.get("custom_llm_provider")
        or provider_from_id
    )
    if custom_llm_provider:
        data["custom_llm_provider"] = custom_llm_provider

    # Resolve model_name from model_id if available
    # This allows the router to automatically inject litellm_params from the model config
    if model_id_from_decoded and llm_router:
        resolved_model = llm_router.resolve_model_name_from_model_id(
            model_id_from_decoded
        )
        if resolved_model:
            data["model"] = resolved_model

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


@router.post(
    "/v1/videos/characters",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.post(
    "/videos/characters",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_create_character(
    request: Request,
    fastapi_response: Response,
    video: UploadFile = File(...),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a character from an uploaded video file.

    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos/create-character

    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/videos/characters" \
        -H "Authorization: Bearer sk-1234" \
        -F "video=@character_video.mp4" \
        -F "name=my_character"
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

    data = await _read_request_body(request=request)
    video_file = await batch_to_bytesio([video])
    if video_file:
        data["video"] = video_file[0]

    target_model_name = extract_model_from_target_model_names(
        data.get("target_model_names")
    )
    if target_model_name and not data.get("model"):
        data["model"] = target_model_name

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or get_custom_provider_from_data(data=data)
        or "openai"
    )
    data["custom_llm_provider"] = custom_llm_provider

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        response = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_create_character",
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
        if target_model_name:
            hidden_params = getattr(response, "_hidden_params", {}) or {}
            provider_for_encoding = (
                hidden_params.get("custom_llm_provider")
                or custom_llm_provider
                or "openai"
            )
            model_id_for_encoding = hidden_params.get("model_id") or data.get("model")
            response = encode_character_id_in_response(
                response=response,
                custom_llm_provider=provider_for_encoding,
                model_id=model_id_for_encoding,
            )
        return response
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/videos/characters/{character_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.get(
    "/videos/characters/{character_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_get_character(
    character_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Retrieve a character by ID.

    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos/get-character

    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/videos/characters/char_123" \
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

    original_requested_character_id = character_id
    data: Dict[str, Any] = {"character_id": character_id}

    decoded = decode_character_id_with_provider(character_id)
    provider_from_id = decoded.get("custom_llm_provider")
    model_id_from_decoded = decoded.get("model_id")
    decoded_character_id = decoded.get("character_id")
    if decoded_character_id:
        data["character_id"] = decoded_character_id

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or await get_custom_llm_provider_from_request_body(request=request)
        or provider_from_id
        or "openai"
    )
    data["custom_llm_provider"] = custom_llm_provider

    if model_id_from_decoded and llm_router:
        resolved_model = llm_router.resolve_model_name_from_model_id(
            model_id_from_decoded
        )
        if resolved_model:
            data["model"] = resolved_model

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        response = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_get_character",
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
        if original_requested_character_id.startswith("character_"):
            provider_for_encoding = provider_from_id or custom_llm_provider or "openai"
            model_id_for_encoding = model_id_from_decoded
            response = encode_character_id_in_response(
                response=response,
                custom_llm_provider=provider_for_encoding,
                model_id=model_id_for_encoding,
            )
        return response
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/videos/edits",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.post(
    "/videos/edits",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_edit(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a video edit job.

    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos/create-edit

    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/videos/edits" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Make it brighter", "video": {"id": "video_123"}}'
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

    body = await request.body()
    data = orjson.loads(body)

    # Extract video_id from nested video object
    video_ref = data.pop("video", {})
    video_id = video_ref.get("id", "") if isinstance(video_ref, dict) else ""
    data["video_id"] = video_id

    decoded = decode_video_id_with_provider(video_id)
    provider_from_id = decoded.get("custom_llm_provider")
    model_id_from_decoded = decoded.get("model_id")

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or get_custom_provider_from_data(data=data)
        or provider_from_id
        or "openai"
    )
    data["custom_llm_provider"] = custom_llm_provider

    if model_id_from_decoded and llm_router:
        resolved_model = llm_router.resolve_model_name_from_model_id(
            model_id_from_decoded
        )
        if resolved_model:
            data["model"] = resolved_model

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_edit",
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


@router.post(
    "/v1/videos/extensions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.post(
    "/videos/extensions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_extension(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a video extension.

    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos/create-extension

    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/videos/extensions" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Continue the scene", "seconds": "5", "video": {"id": "video_123"}}'
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

    body = await request.body()
    data = orjson.loads(body)

    # Extract video_id from nested video object
    video_ref = data.pop("video", {})
    video_id = video_ref.get("id", "") if isinstance(video_ref, dict) else ""
    data["video_id"] = video_id

    decoded = decode_video_id_with_provider(video_id)
    provider_from_id = decoded.get("custom_llm_provider")
    model_id_from_decoded = decoded.get("model_id")

    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or get_custom_provider_from_data(data=data)
        or provider_from_id
        or "openai"
    )
    data["custom_llm_provider"] = custom_llm_provider

    if model_id_from_decoded and llm_router:
        resolved_model = llm_router.resolve_model_name_from_model_id(
            model_id_from_decoded
        )
        if resolved_model:
            data["model"] = resolved_model

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_extension",
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
