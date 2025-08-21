######################################################################

#                          /v1/batches Endpoints


######################################################################
import asyncio
from typing import Dict, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.main import CancelBatchRequest, RetrieveBatchRequest
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_body,
)
from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
    get_models_from_unified_file_id,
)
from litellm.proxy.utils import handle_exception_on_proxy, is_known_model
from litellm.types.llms.openai import LiteLLMBatchCreateRequest

router = APIRouter()


@router.post(
    "/{provider}/v1/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.post(
    "/v1/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.post(
    "/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
async def create_batch(
    request: Request,
    fastapi_response: Response,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create large batches of API requests for asynchronous processing.
    This is the equivalent of POST https://api.openai.com/v1/batch
    Supports Identical Params as: https://platform.openai.com/docs/api-reference/batch

    Example Curl
    ```
    curl http://localhost:4000/v1/batches \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "input_file_id": "file-abc123",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        data = await _read_request_body(request=request)
        verbose_proxy_logger.debug(
            "Request received by LiteLLM:\n{}".format(json.dumps(data, indent=4)),
        )
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
            route_type="acreate_batch",
        )

        ## check if model is a loadbalanced model
        router_model: Optional[str] = None
        is_router_model = False
        if litellm.enable_loadbalancing_on_batch_endpoints is True:
            router_model = data.get("model", None)
            is_router_model = is_known_model(model=router_model, llm_router=llm_router)

        custom_llm_provider = (
            provider or data.pop("custom_llm_provider", None) or "openai"
        )
        _create_batch_data = LiteLLMBatchCreateRequest(**data)
        input_file_id = _create_batch_data.get("input_file_id", None)
        unified_file_id: Union[str, Literal[False]] = False
        if input_file_id:
            unified_file_id = _is_base64_encoded_unified_file_id(input_file_id)
        if (
            litellm.enable_loadbalancing_on_batch_endpoints is True
            and is_router_model
            and router_model is not None
        ):
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "LLM Router not initialized. Ensure models added to proxy."
                    },
                )

            response = await llm_router.acreate_batch(**_create_batch_data)  # type: ignore
        elif (
            unified_file_id and input_file_id
        ):  # litellm_proxy:application/octet-stream;unified_id,c4843482-b176-4901-8292-7523fd0f2c6e;target_model_names,gpt-4o-mini
            target_model_names = get_models_from_unified_file_id(unified_file_id)
            ## EXPECTS 1 MODEL
            if len(target_model_names) != 1:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Expected 1 model, got {}".format(
                            len(target_model_names)
                        )
                    },
                )
            model = target_model_names[0]
            _create_batch_data["model"] = model
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "LLM Router not initialized. Ensure models added to proxy."
                    },
                )

            response = await llm_router.acreate_batch(**_create_batch_data)
            response.input_file_id = input_file_id
            response._hidden_params["unified_file_id"] = unified_file_id
        else:
            response = await litellm.acreate_batch(
                custom_llm_provider=custom_llm_provider, **_create_batch_data  # type: ignore
            )

        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
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
                request_data=data,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.create_batch(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.get(
    "/{provider}/v1/batches/{batch_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/v1/batches/{batch_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/batches/{batch_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
async def retrieve_batch(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    provider: Optional[str] = None,
    batch_id: str = Path(
        title="Batch ID to retrieve", description="The ID of the batch to retrieve"
    ),
):
    """
    Retrieves a batch.
    This is the equivalent of GET https://api.openai.com/v1/batches/{batch_id}
    Supports Identical Params as: https://platform.openai.com/docs/api-reference/batch/retrieve

    Example Curl
    ```
    curl http://localhost:4000/v1/batches/batch_abc123 \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \

    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        ## check if model is a loadbalanced model
        _retrieve_batch_request = RetrieveBatchRequest(
            batch_id=batch_id,
        )

        data = cast(dict, _retrieve_batch_request)
        unified_batch_id = _is_base64_encoded_unified_file_id(batch_id)

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
            route_type="aretrieve_batch",
        )

        if litellm.enable_loadbalancing_on_batch_endpoints is True or unified_batch_id:
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "LLM Router not initialized. Ensure models added to proxy."
                    },
                )

            response = await llm_router.aretrieve_batch(**data)  # type: ignore
            response._hidden_params["unified_batch_id"] = unified_batch_id
        else:
            custom_llm_provider = (
                provider
                or await get_custom_llm_provider_from_request_body(request=request)
                or "openai"
            )
            response = await litellm.aretrieve_batch(
                custom_llm_provider=custom_llm_provider, **data  # type: ignore
            )

        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
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
                request_data=data,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.retrieve_batch(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.get(
    "/{provider}/v1/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/v1/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
async def list_batches(
    request: Request,
    fastapi_response: Response,
    provider: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    target_model_names: Optional[str] = None,
):
    """
    Lists 
    This is the equivalent of GET https://api.openai.com/v1/batches/
    Supports Identical Params as: https://platform.openai.com/docs/api-reference/batch/list

    Example Curl
    ```
    curl http://localhost:4000/v1/batches?limit=2 \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \

    ```
    """
    from litellm.proxy.proxy_server import llm_router, proxy_logging_obj, version

    verbose_proxy_logger.debug("GET /v1/batches after={} limit={}".format(after, limit))
    try:
        if llm_router is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.no_llm_router.value},
            )

        ## check for target model names
        data = await _read_request_body(request=request)
        target_model_names = target_model_names or data.get("target_model_names", None)
        if target_model_names:
            model = target_model_names.split(",")[0]
            response = await llm_router.alist_batches(
                model=model,
                after=after,
                limit=limit,
            )
        else:
            custom_llm_provider = (
                provider
                or await get_custom_llm_provider_from_request_body(request=request)
                or "openai"
            )
            response = await litellm.alist_batches(
                custom_llm_provider=custom_llm_provider,  # type: ignore
                after=after,
                limit=limit,
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
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data={"after": after, "limit": limit},
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.retrieve_batch(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.post(
    "/{provider}/v1/batches/{batch_id:path}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.post(
    "/v1/batches/{batch_id:path}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.post(
    "/batches/{batch_id:path}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
async def cancel_batch(
    request: Request,
    batch_id: str,
    fastapi_response: Response,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel a batch.
    This is the equivalent of POST https://api.openai.com/v1/batches/{batch_id}/cancel

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/batch/cancel

    Example Curl
    ```
    curl http://localhost:4000/v1/batches/batch_abc123/cancel \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -X POST

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
        data = await _read_request_body(request=request)
        verbose_proxy_logger.debug(
            "Request received by LiteLLM:\n{}".format(json.dumps(data, indent=4)),
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

        custom_llm_provider = (
            provider or data.pop("custom_llm_provider", None) or "openai"
        )
        _cancel_batch_data = CancelBatchRequest(batch_id=batch_id, **data)
        response = await litellm.acancel_batch(
            custom_llm_provider=custom_llm_provider,  # type: ignore
            **_cancel_batch_data,
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
                request_data=data,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.create_batch(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


######################################################################

#            END OF  /v1/batches Endpoints Implementation

######################################################################
