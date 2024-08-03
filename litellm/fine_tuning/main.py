"""
Main File for Fine Tuning API implementation

https://platform.openai.com/docs/api-reference/fine-tuning

- fine_tuning.jobs.create()
- fine_tuning.jobs.list()
- client.fine_tuning.jobs.list_events()
"""

import asyncio
import contextvars
import os
from functools import partial
from typing import Any, Coroutine, Dict, Literal, Optional, Union

import httpx

import litellm
from litellm import get_secret
from litellm._logging import verbose_logger
from litellm.llms.fine_tuning_apis.azure import AzureOpenAIFineTuningAPI
from litellm.llms.fine_tuning_apis.openai import (
    FineTuningJob,
    FineTuningJobCreate,
    OpenAIFineTuningAPI,
)
from litellm.llms.fine_tuning_apis.vertex_ai import VertexFineTuningAPI
from litellm.types.llms.openai import Hyperparameters
from litellm.types.router import *
from litellm.utils import supports_httpx_timeout

####### ENVIRONMENT VARIABLES ###################
openai_fine_tuning_apis_instance = OpenAIFineTuningAPI()
azure_fine_tuning_apis_instance = AzureOpenAIFineTuningAPI()
vertex_fine_tuning_apis_instance = VertexFineTuningAPI()
#################################################


async def acreate_fine_tuning_job(
    model: str,
    training_file: str,
    hyperparameters: Optional[Hyperparameters] = {},  # type: ignore
    suffix: Optional[str] = None,
    validation_file: Optional[str] = None,
    integrations: Optional[List[str]] = None,
    seed: Optional[int] = None,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> FineTuningJob:
    """
    Async: Creates and executes a batch from an uploaded file of request

    """
    verbose_logger.debug(
        "inside acreate_fine_tuning_job model=%s and kwargs=%s", model, kwargs
    )
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_fine_tuning_job"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            create_fine_tuning_job,
            model,
            training_file,
            hyperparameters,
            suffix,
            validation_file,
            integrations,
            seed,
            custom_llm_provider,
            extra_headers,
            extra_body,
            **kwargs,
        )

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response  # type: ignore
        return response
    except Exception as e:
        raise e


def create_fine_tuning_job(
    model: str,
    training_file: str,
    hyperparameters: Optional[Hyperparameters] = {},  # type: ignore
    suffix: Optional[str] = None,
    validation_file: Optional[str] = None,
    integrations: Optional[List[str]] = None,
    seed: Optional[int] = None,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[FineTuningJob, Coroutine[Any, Any, FineTuningJob]]:
    """
    Creates a fine-tuning job which begins the process of creating a new model from a given dataset.

    Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete

    """
    try:
        _is_async = kwargs.pop("acreate_fine_tuning_job", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(custom_llm_provider) == False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout  # default 10 min timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        # OpenAI
        if custom_llm_provider == "openai":

            # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = (
                optional_params.api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale we check in get_llm_provider and pass in the api key from there
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            create_fine_tuning_job_data = FineTuningJobCreate(
                model=model,
                training_file=training_file,
                hyperparameters=hyperparameters,
                suffix=suffix,
                validation_file=validation_file,
                integrations=integrations,
                seed=seed,
            )

            create_fine_tuning_job_data_dict = create_fine_tuning_job_data.model_dump(
                exclude_none=True
            )

            response = openai_fine_tuning_apis_instance.create_fine_tuning_job(
                api_base=api_base,
                api_key=api_key,
                organization=organization,
                create_fine_tuning_job_data=create_fine_tuning_job_data_dict,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
            )
        # Azure OpenAI
        elif custom_llm_provider == "azure":
            api_base = optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")  # type: ignore

            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret("AZURE_API_VERSION")
            )  # type: ignore

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )  # type: ignore

            extra_body = optional_params.get("extra_body", {})
            azure_ad_token: Optional[str] = None
            if extra_body is not None:
                azure_ad_token = extra_body.pop("azure_ad_token", None)
            else:
                azure_ad_token = get_secret("AZURE_AD_TOKEN")  # type: ignore

            create_fine_tuning_job_data = FineTuningJobCreate(
                model=model,
                training_file=training_file,
                hyperparameters=hyperparameters,
                suffix=suffix,
                validation_file=validation_file,
                integrations=integrations,
                seed=seed,
            )

            create_fine_tuning_job_data_dict = create_fine_tuning_job_data.model_dump(
                exclude_none=True
            )

            response = azure_fine_tuning_apis_instance.create_fine_tuning_job(
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                create_fine_tuning_job_data=create_fine_tuning_job_data_dict,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
            )
        elif custom_llm_provider == "vertex_ai":
            api_base = optional_params.api_base or ""
            vertex_ai_project = (
                optional_params.vertex_project
                or litellm.vertex_project
                or get_secret("VERTEXAI_PROJECT")
            )
            vertex_ai_location = (
                optional_params.vertex_location
                or litellm.vertex_location
                or get_secret("VERTEXAI_LOCATION")
            )
            vertex_credentials = optional_params.vertex_credentials or get_secret(
                "VERTEXAI_CREDENTIALS"
            )
            create_fine_tuning_job_data = FineTuningJobCreate(
                model=model,
                training_file=training_file,
                hyperparameters=hyperparameters,
                suffix=suffix,
                validation_file=validation_file,
                integrations=integrations,
                seed=seed,
            )
            response = vertex_fine_tuning_apis_instance.create_fine_tuning_job(
                _is_async=_is_async,
                create_fine_tuning_job_data=create_fine_tuning_job_data,
                vertex_credentials=vertex_credentials,
                vertex_project=vertex_ai_project,
                vertex_location=vertex_ai_location,
                timeout=timeout,
                api_base=api_base,
                kwargs=kwargs,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'create_batch'. Only 'openai' is supported.".format(
                    custom_llm_provider
                ),
                model="n/a",
                llm_provider=custom_llm_provider,
                response=httpx.Response(
                    status_code=400,
                    content="Unsupported provider",
                    request=httpx.Request(method="create_thread", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )
        return response
    except Exception as e:
        verbose_logger.error("got exception in create_fine_tuning_job=%s", str(e))
        raise e


async def acancel_fine_tuning_job(
    fine_tuning_job_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> FineTuningJob:
    """
    Async: Immediately cancel a fine-tune job.
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["acancel_fine_tuning_job"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            cancel_fine_tuning_job,
            fine_tuning_job_id,
            custom_llm_provider,
            extra_headers,
            extra_body,
            **kwargs,
        )

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response  # type: ignore
        return response
    except Exception as e:
        raise e


def cancel_fine_tuning_job(
    fine_tuning_job_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[FineTuningJob, Coroutine[Any, Any, FineTuningJob]]:
    """
    Immediately cancel a fine-tune job.

    Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete

    """
    try:
        optional_params = GenericLiteLLMParams(**kwargs)
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(custom_llm_provider) == False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout  # default 10 min timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        _is_async = kwargs.pop("acancel_fine_tuning_job", False) is True

        # OpenAI
        if custom_llm_provider == "openai":

            # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = (
                optional_params.api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale we check in get_llm_provider and pass in the api key from there
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            response = openai_fine_tuning_apis_instance.cancel_fine_tuning_job(
                api_base=api_base,
                api_key=api_key,
                organization=organization,
                fine_tuning_job_id=fine_tuning_job_id,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
            )
        # Azure OpenAI
        elif custom_llm_provider == "azure":
            api_base = optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")  # type: ignore

            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret("AZURE_API_VERSION")
            )  # type: ignore

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )  # type: ignore

            extra_body = optional_params.get("extra_body", {})
            azure_ad_token: Optional[str] = None
            if extra_body is not None:
                azure_ad_token = extra_body.pop("azure_ad_token", None)
            else:
                azure_ad_token = get_secret("AZURE_AD_TOKEN")  # type: ignore

            response = azure_fine_tuning_apis_instance.cancel_fine_tuning_job(
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                fine_tuning_job_id=fine_tuning_job_id,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'create_batch'. Only 'openai' is supported.".format(
                    custom_llm_provider
                ),
                model="n/a",
                llm_provider=custom_llm_provider,
                response=httpx.Response(
                    status_code=400,
                    content="Unsupported provider",
                    request=httpx.Request(method="create_thread", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )
        return response
    except Exception as e:
        raise e


async def alist_fine_tuning_jobs(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> FineTuningJob:
    """
    Async: List your organization's fine-tuning jobs
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["alist_fine_tuning_jobs"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            list_fine_tuning_jobs,
            after,
            limit,
            custom_llm_provider,
            extra_headers,
            extra_body,
            **kwargs,
        )

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response  # type: ignore
        return response
    except Exception as e:
        raise e


def list_fine_tuning_jobs(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
):
    """
    List your organization's fine-tuning jobs

    Params:

    - after: Optional[str] = None, Identifier for the last job from the previous pagination request.
    - limit: Optional[int] = None, Number of fine-tuning jobs to retrieve. Defaults to 20
    """
    try:
        optional_params = GenericLiteLLMParams(**kwargs)
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(custom_llm_provider) == False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout  # default 10 min timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        _is_async = kwargs.pop("alist_fine_tuning_jobs", False) is True

        # OpenAI
        if custom_llm_provider == "openai":

            # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = (
                optional_params.api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale we check in get_llm_provider and pass in the api key from there
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            response = openai_fine_tuning_apis_instance.list_fine_tuning_jobs(
                api_base=api_base,
                api_key=api_key,
                organization=organization,
                after=after,
                limit=limit,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
            )
        # Azure OpenAI
        elif custom_llm_provider == "azure":
            api_base = optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")  # type: ignore

            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret("AZURE_API_VERSION")
            )  # type: ignore

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )  # type: ignore

            extra_body = optional_params.get("extra_body", {})
            azure_ad_token: Optional[str] = None
            if extra_body is not None:
                azure_ad_token = extra_body.pop("azure_ad_token", None)
            else:
                azure_ad_token = get_secret("AZURE_AD_TOKEN")  # type: ignore

            response = azure_fine_tuning_apis_instance.list_fine_tuning_jobs(
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                after=after,
                limit=limit,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'create_batch'. Only 'openai' is supported.".format(
                    custom_llm_provider
                ),
                model="n/a",
                llm_provider=custom_llm_provider,
                response=httpx.Response(
                    status_code=400,
                    content="Unsupported provider",
                    request=httpx.Request(method="create_thread", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )
        return response
    except Exception as e:
        raise e
