#########################################################################

#                          /v1/fine_tuning Endpoints

# Equivalent of https://platform.openai.com/docs/api-reference/fine-tuning
##########################################################################

import asyncio
import traceback
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import handle_exception_on_proxy

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
        add_litellm_data_to_request,
        general_settings,
        get_custom_headers,
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
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # get configs for custom_llm_provider
        llm_provider_config = get_fine_tuning_provider_config(
            custom_llm_provider=fine_tuning_request.custom_llm_provider,
        )

        # add llm_provider_config to data
        if llm_provider_config is not None:
            data.update(llm_provider_config)

        response = await litellm.acreate_fine_tuning_job(**data)

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
            get_custom_headers(
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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.create_fine_tuning_job(): Exception occurred - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
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
    custom_llm_provider: Literal["openai", "azure"],
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
        add_litellm_data_to_request,
        general_settings,
        get_custom_headers,
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
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # get configs for custom_llm_provider
        llm_provider_config = get_fine_tuning_provider_config(
            custom_llm_provider=custom_llm_provider
        )

        if llm_provider_config is not None:
            data.update(llm_provider_config)

        response = await litellm.aretrieve_fine_tuning_job(
            **data,
            fine_tuning_job_id=fine_tuning_job_id,
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            get_custom_headers(
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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.list_fine_tuning_jobs(): Exception occurred - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
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
    custom_llm_provider: Literal["openai", "azure"],
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
        add_litellm_data_to_request,
        general_settings,
        get_custom_headers,
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
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

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

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            get_custom_headers(
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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.list_fine_tuning_jobs(): Exception occurred - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
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
        add_litellm_data_to_request,
        general_settings,
        get_custom_headers,
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
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        request_body = await request.json()

        custom_llm_provider = request_body.get("custom_llm_provider", None)

        # get configs for custom_llm_provider
        llm_provider_config = get_fine_tuning_provider_config(
            custom_llm_provider=custom_llm_provider
        )

        if llm_provider_config is not None:
            data.update(llm_provider_config)

        response = await litellm.acancel_fine_tuning_job(
            **data,
            fine_tuning_job_id=fine_tuning_job_id,
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            get_custom_headers(
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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.list_fine_tuning_jobs(): Exception occurred - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        raise handle_exception_on_proxy(e)
