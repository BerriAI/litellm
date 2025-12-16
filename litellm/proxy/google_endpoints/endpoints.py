from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import ORJSONResponse, StreamingResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse

router = APIRouter(
    tags=["google genai endpoints"],
)


@router.post(
    "/v1beta/models/{model_name:path}:generateContent",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/models/{model_name:path}:generateContent", dependencies=[Depends(user_api_key_auth)]
)
async def google_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        version,
    )

    data = await _read_request_body(request=request)
    if "model" not in data:
        data["model"] = model_name
    
    # Add user authentication metadata for cost tracking
    data = await add_litellm_data_to_request(
        data=data,
        request=request,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
        general_settings=general_settings,
        version=version,
    )
    
    # call router
    if llm_router is None:
        raise HTTPException(status_code=500, detail="Router not initialized")
    response = await llm_router.agenerate_content(**data)
    return response


@router.post(
    "/v1beta/models/{model_name:path}:streamGenerateContent",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/models/{model_name:path}:streamGenerateContent",
    dependencies=[Depends(user_api_key_auth)],
)
async def google_stream_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        version,
    )

    data = await _read_request_body(request=request)

    if "model" not in data:
        data["model"] = model_name

    data["stream"] = True  # enforce streaming for this endpoint

    # Add user authentication metadata for cost tracking
    data = await add_litellm_data_to_request(
        data=data,
        request=request,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
        general_settings=general_settings,
        version=version,
    )

    # call router
    if llm_router is None:
        raise HTTPException(status_code=500, detail="Router not initialized")
    response = await llm_router.agenerate_content_stream(**data)

    # Check if response is an async iterator (streaming response)
    if response is not None and hasattr(response, "__aiter__"):
        return StreamingResponse(content=response, media_type="text/event-stream")
    return response


@router.post(
    "/v1beta/models/{model_name:path}:countTokens",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountDetailsResponse,
)
@router.post(
    "/models/{model_name:path}:countTokens",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountDetailsResponse,
)
async def google_count_tokens(request: Request, model_name: str):
    """
    ```json
    return {
        "totalTokens": 31,
        "totalBillableCharacters": 96,
        "promptTokensDetails": [
            {
            "modality": "TEXT",
            "tokenCount": 31
            }
        ]
    }
    ```
    """
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
    from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
    from litellm.proxy.proxy_server import token_counter as internal_token_counter

    data = await _read_request_body(request=request)
    contents = data.get("contents", [])
    # Create TokenCountRequest for the internal endpoint
    from litellm.proxy._types import TokenCountRequest

    # Translate contents to openai format messages using the adapter
    messages = (
        GoogleGenAIAdapter()
        .translate_generate_content_to_completion(model_name, contents)
        .get("messages", [])
    )

    token_request = TokenCountRequest(
        model=model_name,
        contents=contents,
        messages=messages,  # compatibility when use openai-like endpoint
    )

    # Call the internal token counter function with direct request flag set to False
    token_response = await internal_token_counter(
        request=token_request,
        call_endpoint=True,
    )
    if token_response is not None:
        # cast the response to the well known format
        original_response: dict = token_response.original_response or {}
        if original_response:
            return TokenCountDetailsResponse(
                totalTokens=original_response.get("totalTokens", 0),
                promptTokensDetails=original_response.get("promptTokensDetails", []),
            )
        else:
            return TokenCountDetailsResponse(
                totalTokens=token_response.total_tokens or 0,
                promptTokensDetails=[],
            )

    #########################################################
    # Return the response in the well known format
    #########################################################
    return TokenCountDetailsResponse(
        totalTokens=0,
        promptTokensDetails=[],
    )


# ============================================================
# Google Interactions API Endpoints
# Per OpenAPI spec: https://ai.google.dev/static/api/interactions.openapi.json
# ============================================================


@router.post(
    "/v1beta/interactions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.post(
    "/interactions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def create_interaction(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new interaction using Google's Interactions API.
    
    Per OpenAPI spec: POST /{api_version}/interactions
    
    Supports both model interactions and agent interactions:
    - Model: Provide `model` parameter (e.g., "gemini-2.5-flash")
    - Agent: Provide `agent` parameter (e.g., "deep-research-pro-preview-12-2025")
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1beta/interactions" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gemini/gemini-2.5-flash",
            "input": "Hello, how are you?"
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

    data = await _read_request_body(request=request)
    
    # Default to gemini provider for interactions
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = "gemini"
    
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
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
    "/v1beta/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.get(
    "/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def get_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get an interaction by ID.
    
    Per OpenAPI spec: GET /{api_version}/interactions/{interaction_id}
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

    data = {"interaction_id": interaction_id, "custom_llm_provider": "gemini"}
    
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_interaction",
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


@router.delete(
    "/v1beta/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.delete(
    "/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def delete_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete an interaction by ID.
    
    Per OpenAPI spec: DELETE /{api_version}/interactions/{interaction_id}
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

    data = {"interaction_id": interaction_id, "custom_llm_provider": "gemini"}
    
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_interaction",
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
    "/v1beta/interactions/{interaction_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.post(
    "/interactions/{interaction_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def cancel_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel an interaction by ID.
    
    Per OpenAPI spec: POST /{api_version}/interactions/{interaction_id}:cancel
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

    data = {"interaction_id": interaction_id, "custom_llm_provider": "gemini"}
    
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acancel_interaction",
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
