#### Rerank Endpoints #####

import asyncio
from typing import Final

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    log_llm_api_exception,
    resolve_litellm_call_id,
)
from litellm.proxy.common_utils.openai_error_payload import (
    error_status_code,
    litellm_call_id_headers,
    openai_error_param,
    openai_error_type,
)

router: Final = APIRouter()


@router.post(
    "/v2/rerank",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rerank"],
)
@router.post(
    "/v1/rerank",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rerank"],
)
@router.post(
    "/rerank",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rerank"],
)
async def rerank(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        route_request,
        user_model,
        version,
    )

    litellm_call_id: Final = resolve_litellm_call_id(request.headers.get("x-litellm-call-id"))
    data = {"litellm_call_id": litellm_call_id}
    try:
        body: Final = await request.body()
        data = orjson.loads(body) | data

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(user_api_key_dict=user_api_key_dict, data=data, call_type="rerank")

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call: Final = await route_request(
            data=data,
            route_type="arerank",
            llm_router=llm_router,
            user_model=user_model,
        )
        response: Final = await llm_call

        ### ALERTING ###
        asyncio.create_task(proxy_logging_obj.update_request_status(litellm_call_id=litellm_call_id, status="success"))

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""
        additional_headers: Final = hidden_params.get("additional_headers", None) or {}
        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=hidden_params.get("litellm_call_id", None) or litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=hidden_params.get("response_cost", None),
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
                **additional_headers,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        log_llm_api_exception(e, litellm_call_id)
        if isinstance(e, ProxyException):
            raise
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=openai_error_type(e, error_status_code(e, status.HTTP_400_BAD_REQUEST)),
                param=openai_error_param(e),
                headers=litellm_call_id_headers(litellm_call_id),
                code=error_status_code(e, status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=openai_error_type(e, error_status_code(e, 500)),
                param=openai_error_param(e),
                headers=litellm_call_id_headers(litellm_call_id),
                code=error_status_code(e, 500),
            )
