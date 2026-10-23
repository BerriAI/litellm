from typing import Final

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import read_request_body

router: Final = APIRouter()


@router.post(
    "/v1/tokenize",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["tokenize"],
)
@router.post(
    "/tokenize",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["tokenize"],
)
async def tokenize(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> object:
    """Tokenize a string with a configured model, returning its token count and available tokens."""
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

    data: dict = {}
    try:
        data = await read_request_body(request)
        processor = ProxyBaseLLMRequestProcessing(data=data)
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="atokenize",
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
    except Exception as error:
        processor = ProxyBaseLLMRequestProcessing(data=data)
        raise await processor.handle_llm_api_exception(
            e=error,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
