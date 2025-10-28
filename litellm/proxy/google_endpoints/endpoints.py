from fastapi import APIRouter, Depends, Request, Response, HTTPException
from fastapi.responses import StreamingResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth

from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse

router = APIRouter(
    tags=["google genai endpoints"],
)


@router.post(
    "/v1beta/models/{model_name}:generateContent",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/models/{model_name}:generateContent", dependencies=[Depends(user_api_key_auth)]
)
async def google_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import llm_router, general_settings, proxy_config, version
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

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
    "/v1beta/models/{model_name}:streamGenerateContent",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/models/{model_name}:streamGenerateContent",
    dependencies=[Depends(user_api_key_auth)],
)
async def google_stream_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import llm_router, general_settings, proxy_config, version
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

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
    if hasattr(response, "__aiter__"):
        return StreamingResponse(response, media_type="text/event-stream")
    return response


@router.post(
    "/v1beta/models/{model_name}:countTokens",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountDetailsResponse,
)
@router.post(
    "/models/{model_name}:countTokens",
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
