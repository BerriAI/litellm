from fastapi import APIRouter, Depends, Request, Response

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router = APIRouter(
    tags=["google genai endpoints"],
)


@router.post("/v1beta/models/{model_name}:generateContent", dependencies=[Depends(user_api_key_auth)])
@router.post("/models/{model_name}:generateContent", dependencies=[Depends(user_api_key_auth)])
async def google_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Not Implemented, this is a placeholder for the google genai generateContent endpoint.
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
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
    if "model" not in data:
        data["model"] = model_name
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="agenerate_content",
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


class GoogleAIStudioDataGenerator:
    """
    Ensures SSE data generator is used for Google AI Studio streaming responses

    Thin wrapper around ProxyBaseLLMRequestProcessing.async_sse_data_generator
    """
    @staticmethod
    def _select_data_generator(response, user_api_key_dict, request_data):
        from litellm.proxy.proxy_server import proxy_logging_obj
        return ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
            proxy_logging_obj=proxy_logging_obj,
        )

@router.post("/v1beta/models/{model_name}:streamGenerateContent", dependencies=[Depends(user_api_key_auth)])
@router.post("/models/{model_name}:streamGenerateContent", dependencies=[Depends(user_api_key_auth)])
async def google_stream_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Not Implemented, this is a placeholder for the google genai streamGenerateContent endpoint.
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    if "model" not in data:
        data["model"] = model_name


    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="agenerate_content_stream",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=GoogleAIStudioDataGenerator._select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
            is_streaming_request=True,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )




@router.post("/v1beta/models/{model_name}:countTokens", dependencies=[Depends(user_api_key_auth)])
@router.post("/models/{model_name}:countTokens", dependencies=[Depends(user_api_key_auth)])
async def google_count_tokens(request: Request, model_name: str):
    """
    Not Implemented, this is a placeholder for the google genai countTokens endpoint.
    """
    return {}
