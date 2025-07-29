#########################################################################

#                          /v1/fine_tuning Endpoints

# Equivalent of https://platform.openai.com/docs/api-reference/fine-tuning
##########################################################################

import asyncio
from typing import Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
)
from litellm.proxy.utils import handle_exception_on_proxy
from litellm.types.utils import LiteLLMFineTuningJob

router = APIRouter()

from litellm.types.llms.openai import LiteLLMFineTuningJobCreate

fine_tuning_config = None


def set_fine_tuning_config(config):
    if config is None:
        return

    global fine_tuning_config
    if not isinstance(config, list):
        raise ValueError("invalid fine_tuning config, expected a list is not a list")

    for element in config:
        if isinstance(element, dict):
            for key, value in element.items():
                if isinstance(value, str) and value.startswith("os.environ/"):
                    element[key] = litellm.get_secret(value)

    fine_tuning_config = config


# Function to search for specific custom_llm_provider and return its configuration
def get_fine_tuning_provider_config(
    custom_llm_provider: str,
):
    global fine_tuning_config
    if fine_tuning_config is None:
        raise ValueError(
            "fine_tuning_config is not set, set it on your config.yaml file."
        )
    for setting in fine_tuning_config:
        if setting.get("custom_llm_provider") == custom_llm_provider:
            return setting
    return None


@router.post(
    "/v1/fine_tuning/jobs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["fine-tuning"],
    summary="✨ (Enterprise) Create Fine-Tuning Job",
)
@router.post(
    "/fine_tuning/jobs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["fine-tuning"],
    summary="✨ (Enterprise) Create Fine-Tuning Job",
)
async def create_fine_tuning_job(
    request: Request,
    fastapi_response: Response,
    fine_tuning_request: LiteLLMFineTuningJobCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Creates a fine-tuning job which begins the process of creating a new model from a given dataset.
    This is the equivalent of POST https://api.openai.com/v1/fine_tuning/jobs

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/fine-tuning/create

    Example Curl:
    ```
    curl http://localhost:4000/v1/fine_tuning/jobs \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer sk-1234" \
      -d '{
        "model": "gpt-3.5-turbo",
        "training_file": "file-abc123",
        "hyperparameters": {
          "n_epochs": 4
        }
      }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        premium_user,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data = fine_tuning_request.model_dump(exclude_none=True)
    try:
        if premium_user is not True:
            raise ValueError(
                f"Only premium users can use this endpoint + {CommonProxyErrors.not_premium_user.value}"
            )
        # Convert Pydantic model to dict

        verbose_proxy_logger.debug(
            "Request received by LiteLLM:\n{}".format(json.dumps(data, indent=4)),
        )

        # Include original request and headers in the data
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
            route_type="acreate_fine_tuning_job",
        )

        ## CHECK IF MANAGED FILE ID
        unified_file_id: Union[str, Literal[False]] = False
        training_file = fine_tuning_request.training_file
        response: Optional[LiteLLMFineTuningJob] = None
        if training_file:
            unified_file_id = _is_base64_encoded_unified_file_id(training_file)
        ## IF SO, Route based on that
        if unified_file_id:
            """ """
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "LLM Router not initialized. Ensure models added to proxy."
                    },
                )

            response = cast(
                LiteLLMFineTuningJob, await llm_router.acreate_fine_tuning_job(**data)
            )
            response.training_file = unified_file_id
            response._hidden_params["unified_file_id"] = unified_file_id
        ## ELSE, Route based on custom_llm_provider
        elif fine_tuning_request.custom_llm_provider:
            # get configs for custom_llm_provider
            llm_provider_config = get_fine_tuning_provider_config(
                custom_llm_provider=fine_tuning_request.custom_llm_provider,
            )
            # add llm_provider_config to data
            if llm_provider_config is not None:
                data.update(llm_provider_config)

            response = await litellm.acreate_fine_tuning_job(**data)

        if response is None:
            raise ValueError(
                "Invalid request, No litellm managed file id or custom_llm_provider provided."
            )

        ### CALL HOOKS ### - modify outgoing data
        _response = await proxy_logging_obj.post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,
        )
        if _response is not None and isinstance(_response, LiteLLMFineTuningJob):
            response = _response

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
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.create_fine_tuning_job(): Exception occurred - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.get(
    "/v1/fine_tuning/jobs/{fine_tuning_job_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["fine-tuning"],
    summary="✨ (Enterprise) Retrieve Fine-Tuning Job",
)
@router.get(
    "/fine_tuning/jobs/{fine_tuning_job_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["fine-tuning"],
    summary="✨ (Enterprise) Retrieve Fine-Tuning Job",
)
async def retrieve_fine_tuning_job(
    request: Request,
    fastapi_response: Response,
    fine_tuning_job_id: str,
    custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Retrieves a fine-tuning job.
    This is the equivalent of GET https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}

    Supported Query Params:
    - `custom_llm_provider`: Name of the LiteLLM provider
    - `fine_tuning_job_id`: The ID of the fine-tuning job to retrieve.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        premium_user,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: dict = {"fine_tuning_job_id": fine_tuning_job_id}
    try:
        if premium_user is not True:
            raise ValueError(
                f"Only premium users can use this endpoint + {CommonProxyErrors.not_premium_user.value}"
            )
        # Include original request and headers in the data
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
            route_type=CallTypes.aretrieve_fine_tuning_job.value,
        )

        try:
            request_body = await request.json()
        except Exception:
            request_body = {}

        custom_llm_provider = request_body.get("custom_llm_provider", None)

        ## CHECK IF MANAGED FILE ID
        unified_finetuning_job_id: Union[str, Literal[False]] = False
        response: Optional[LiteLLMFineTuningJob] = None
        if fine_tuning_job_id:
            unified_finetuning_job_id = _is_base64_encoded_unified_file_id(
                fine_tuning_job_id
            )
        if unified_finetuning_job_id:
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "LLM Router not initialized. Ensure models added to proxy."
                    },
                )
            response = cast(
                LiteLLMFineTuningJob,
                await llm_router.aretrieve_fine_tuning_job(
                    **data,
                ),
            )
            response._hidden_params[
                "unified_finetuning_job_id"
            ] = unified_finetuning_job_id
        elif custom_llm_provider:
            # get configs for custom_llm_provider
            llm_provider_config = get_fine_tuning_provider_config(
                custom_llm_provider=custom_llm_provider
            )

            if llm_provider_config is not None:
                data.update(llm_provider_config)

            response = await litellm.aretrieve_fine_tuning_job(
                **data,
            )

        if response is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid request, No litellm managed file id or custom_llm_provider provided.",
            )

        ### CALL HOOKS ### - modify outgoing data
        _response = await proxy_logging_obj.post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,
        )
        if _response is not None and isinstance(_response, LiteLLMFineTuningJob):
            response = _response

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
            )
        )

        return response

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.retrieve_fine_tuning_job(): Exception occurred - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.get(
    "/v1/fine_tuning/jobs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["fine-tuning"],
    summary="✨ (Enterprise) List Fine-Tuning Jobs",
)
@router.get(
    "/fine_tuning/jobs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["fine-tuning"],
    summary="✨ (Enterprise) List Fine-Tuning Jobs",
)
async def list_fine_tuning_jobs(
    request: Request,
    fastapi_response: Response,
    custom_llm_provider: Optional[Literal["openai", "azure"]] = None,
    target_model_names: Optional[str] = Query(
        default=None,
        description="Comma separated list of model names to filter by. Example: 'gpt-4o,gpt-4o-mini'",
    ),
    after: Optional[str] = None,
    limit: Optional[int] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Lists fine-tuning jobs for the organization.
    This is the equivalent of GET https://api.openai.com/v1/fine_tuning/jobs

    Supported Query Params:
    - `custom_llm_provider`: Name of the LiteLLM provider
    - `after`: Identifier for the last job from the previous pagination request.
    - `limit`: Number of fine-tuning jobs to retrieve (default is 20).
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        premium_user,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: dict = {}
    try:
        if premium_user is not True:
            raise ValueError(
                f"Only premium users can use this endpoint + {CommonProxyErrors.not_premium_user.value}"
            )
        # Include original request and headers in the data
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
            route_type=CallTypes.alist_fine_tuning_jobs.value,
        )

        response: Optional[Any] = None
        if target_model_names and isinstance(target_model_names, str):
            target_model_names_list = target_model_names.split(",")
            if len(target_model_names_list) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="target_model_names on list fine-tuning jobs must be a list of one model name. Example: ['gpt-4o']",
                )
            ## Use router to list fine-tuning jobs for that model
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail="LLM Router not initialized. Ensure models added to proxy.",
                )
            data["model"] = target_model_names_list[0]
            response = await llm_router.alist_fine_tuning_jobs(
                **data,
                after=after,
                limit=limit,
            )
            return response
        elif custom_llm_provider:
            # get configs for custom_llm_provider
            llm_provider_config = get_fine_tuning_provider_config(
                custom_llm_provider=custom_llm_provider
            )

            if llm_provider_config is not None:
                data.update(llm_provider_config)

            response = await litellm.alist_fine_tuning_jobs(
                **data,
                after=after,
                limit=limit,
            )
        if response is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid request, No litellm managed file id or custom_llm_provider provided.",
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
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.list_fine_tuning_jobs(): Exception occurred - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.post(
    "/v1/fine_tuning/jobs/{fine_tuning_job_id:path}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["fine-tuning"],
    summary="✨ (Enterprise) Cancel Fine-Tuning Jobs",
)
@router.post(
    "/fine_tuning/jobs/{fine_tuning_job_id:path}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["fine-tuning"],
    summary="✨ (Enterprise) Cancel Fine-Tuning Jobs",
)
async def cancel_fine_tuning_job(
    request: Request,
    fastapi_response: Response,
    fine_tuning_job_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel a fine-tuning job.

    This is the equivalent of POST https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel

    Supported Query Params:
    - `custom_llm_provider`: Name of the LiteLLM provider
    - `fine_tuning_job_id`: The ID of the fine-tuning job to cancel.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        premium_user,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: dict = {"fine_tuning_job_id": fine_tuning_job_id}
    try:
        if premium_user is not True:
            raise ValueError(
                f"Only premium users can use this endpoint + {CommonProxyErrors.not_premium_user.value}"
            )
        # Include original request and headers in the data
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
            route_type=CallTypes.acancel_fine_tuning_job.value,
        )

        try:
            request_body = await request.json()
        except Exception:
            request_body = {}

        custom_llm_provider = request_body.get("custom_llm_provider", None)

        ## CHECK IF MANAGED FILE ID
        unified_finetuning_job_id: Union[str, Literal[False]] = False
        response: Optional[LiteLLMFineTuningJob] = None
        if fine_tuning_job_id:
            unified_finetuning_job_id = _is_base64_encoded_unified_file_id(
                fine_tuning_job_id
            )
        if unified_finetuning_job_id:
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "LLM Router not initialized. Ensure models added to proxy."
                    },
                )
            response = cast(
                LiteLLMFineTuningJob,
                await llm_router.acancel_fine_tuning_job(
                    **data,
                ),
            )
            response._hidden_params[
                "unified_finetuning_job_id"
            ] = unified_finetuning_job_id
        else:
            # get configs for custom_llm_provider
            llm_provider_config = get_fine_tuning_provider_config(
                custom_llm_provider=custom_llm_provider
            )

            if llm_provider_config is not None:
                data.update(llm_provider_config)

            response = await litellm.acancel_fine_tuning_job(
                **data,
            )

        if response is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid request, No litellm managed file id or custom_llm_provider provided.",
            )

        ### CALL HOOKS ### - modify outgoing data
        _response = await proxy_logging_obj.post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,
        )
        if _response is not None and isinstance(_response, LiteLLMFineTuningJob):
            response = _response

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
            )
        )

        return response

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.cancel_fine_tuning_job(): Exception occurred - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)
