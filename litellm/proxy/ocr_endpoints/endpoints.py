#### OCR Endpoints #####

import orjson
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router = APIRouter()


@router.post(
    "/v1/ocr",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["ocr"],
)
@router.post(
    "/ocr",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["ocr"],
)
async def ocr(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    OCR endpoint for extracting text from documents and images.
    
    Follows the Mistral OCR API spec:
    https://docs.mistral.ai/capabilities/vision/#optical-character-recognition-ocr
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/ocr" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mistral/mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234"
            }
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

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aocr",
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

