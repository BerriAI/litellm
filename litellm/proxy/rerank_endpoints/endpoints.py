#### Rerank Endpoints #####
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
import orjson
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()
import asyncio


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
        get_custom_headers,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        route_request,
        user_model,
        version,
    )

    data = {}
    try:
        body = await request.body()
        data = orjson.loads(body)

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
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="rerank"
        )

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call = await route_request(
            data=data,
            route_type="arerank",
            llm_router=llm_router,
            user_model=user_model,
        )
        response = await llm_call

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
        additional_headers = hidden_params.get("additional_headers", None) or {}
        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                **additional_headers,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.rerank(): Exception occured - {}".format(str(e))
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
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
