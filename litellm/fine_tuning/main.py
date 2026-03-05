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
from litellm._logging import verbose_logger
from litellm.llms.azure.fine_tuning.handler import AzureOpenAIFineTuningAPI
from litellm.llms.openai.fine_tuning.handler import OpenAIFineTuningAPI
from litellm.llms.vertex_ai.fine_tuning.handler import VertexFineTuningAPI
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import FineTuningJobCreate, Hyperparameters
from litellm.types.router import *
from litellm.types.utils import LiteLLMFineTuningJob
from litellm.utils import client, supports_httpx_timeout

####### ENVIRONMENT VARIABLES ###################
openai_fine_tuning_apis_instance = OpenAIFineTuningAPI()
azure_fine_tuning_apis_instance = AzureOpenAIFineTuningAPI()
vertex_fine_tuning_apis_instance = VertexFineTuningAPI()
#################################################


def _prepare_azure_extra_body(
    extra_body: Optional[Dict[str, Any]],
    kwargs: Dict[str, Any],
    azure_specific_hyperparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prepare extra_body for Azure fine-tuning API by combining Azure-specific parameters.
    
    Azure fine-tuning API accepts additional parameters beyond the standard OpenAI spec:
    - trainingType: Type of training (e.g., 1 for supervised fine-tuning)
    - prompt_loss_weight: Weight for prompt loss in training
    
    These parameters must be passed in the extra_body field when calling the Azure OpenAI SDK.
    
    Args:
        extra_body: Optional existing extra_body dict
        kwargs: Request kwargs that may contain Azure-specific parameters
        azure_specific_hyperparams: Dict of Azure-specific hyperparameters already extracted
        
    Returns:
        Dict containing all Azure-specific parameters to be passed in extra_body
    """
    if extra_body is None:
        extra_body = {}
    
    # Azure-specific root-level parameters
    azure_specific_params = ["trainingType"]
    for param in azure_specific_params:
        if param in kwargs:
            extra_body[param] = kwargs[param]
    
    # Add Azure-specific hyperparameters
    if azure_specific_hyperparams:
        extra_body.update(azure_specific_hyperparams)
    
    return extra_body


@client
async def acreate_fine_tuning_job(
    model: str,
    training_file: str,
    hyperparameters: Optional[dict] = {},
    suffix: Optional[str] = None,
    validation_file: Optional[str] = None,
    integrations: Optional[List[str]] = None,
    seed: Optional[int] = None,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> LiteLLMFineTuningJob:
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


def _resolve_fine_tuning_timeout(
    timeout: Any,
    custom_llm_provider: str,
) -> Union[float, httpx.Timeout]:
    """Normalise a raw timeout value to a float (seconds) for fine-tuning calls."""
    timeout = timeout or 600.0
    if isinstance(timeout, httpx.Timeout):
        if not supports_httpx_timeout(custom_llm_provider):
            return float(timeout.read or 600)
        return timeout
    return float(timeout)


@client
def create_fine_tuning_job(
    model: str,
    training_file: str,
    hyperparameters: Optional[dict] = {},
    suffix: Optional[str] = None,
    validation_file: Optional[str] = None,
    integrations: Optional[List[str]] = None,
    seed: Optional[int] = None,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[LiteLLMFineTuningJob, Coroutine[Any, Any, LiteLLMFineTuningJob]]:
    """
    Creates a fine-tuning job which begins the process of creating a new model from a given dataset.

    Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete

    """
    try:
        _is_async = kwargs.pop("acreate_fine_tuning_job", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)

        # handle hyperparameters
        hyperparameters = hyperparameters or {}  # original hyperparameters
        
        # For Azure, extract Azure-specific hyperparameters before creating OpenAI-spec hyperparameters
        azure_specific_hyperparams = {}
        if custom_llm_provider == "azure":
            azure_hyperparameter_keys = ["prompt_loss_weight"]
            for key in azure_hyperparameter_keys:
                if key in hyperparameters:
                    azure_specific_hyperparams[key] = hyperparameters.pop(key)
        
        _oai_hyperparameters: Hyperparameters = Hyperparameters(
            **hyperparameters
        )  # Typed Hyperparameters for OpenAI Spec
        timeout = _resolve_fine_tuning_timeout(
            optional_params.timeout or kwargs.get("request_timeout", 600),
            custom_llm_provider,
        )

        # OpenAI
        if custom_llm_provider == "openai":
            # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_BASE_URL")
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
                hyperparameters=_oai_hyperparameters,
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
                api_version=optional_params.api_version,
                organization=organization,
                create_fine_tuning_job_data=create_fine_tuning_job_data_dict,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
                client=kwargs.get(
                    "client", None
                ),  # note, when we add this to `GenericLiteLLMParams` it impacts a lot of other tests + linting
            )
        # Azure OpenAI
        elif custom_llm_provider == "azure":
            api_base = optional_params.api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")  # type: ignore

            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
            )  # type: ignore

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_OPENAI_API_KEY")
                or get_secret_str("AZURE_API_KEY")
            )  # type: ignore

            extra_body = optional_params.get("extra_body", {})
            if extra_body is not None:
                extra_body.pop("azure_ad_token", None)
            else:
                get_secret_str("AZURE_AD_TOKEN")  # type: ignore
            
            # Prepare Azure-specific parameters for extra_body
            extra_body = _prepare_azure_extra_body(extra_body, kwargs, azure_specific_hyperparams)
            
            create_fine_tuning_job_data = FineTuningJobCreate(
                model=model,
                training_file=training_file,
                hyperparameters=_oai_hyperparameters,
                suffix=suffix,
                validation_file=validation_file,
                integrations=integrations,
                seed=seed,
            )

            create_fine_tuning_job_data_dict = create_fine_tuning_job_data.model_dump(
                exclude_none=True
            )
            
            # Add extra_body if it has Azure-specific parameters
            if extra_body:
                create_fine_tuning_job_data_dict["extra_body"] = extra_body

            response = azure_fine_tuning_apis_instance.create_fine_tuning_job(
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                create_fine_tuning_job_data=create_fine_tuning_job_data_dict,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
                organization=optional_params.organization,
            )
        elif custom_llm_provider == "vertex_ai":
            api_base = optional_params.api_base or ""
            vertex_ai_project = (
                optional_params.vertex_project
                or litellm.vertex_project
                or get_secret_str("VERTEXAI_PROJECT")
            )
            vertex_ai_location = (
                optional_params.vertex_location
                or litellm.vertex_location
                or get_secret_str("VERTEXAI_LOCATION")
            )
            vertex_credentials = optional_params.vertex_credentials or get_secret_str(
                "VERTEXAI_CREDENTIALS"
            )
            create_fine_tuning_job_data = FineTuningJobCreate(
                model=model,
                training_file=training_file,
                hyperparameters=_oai_hyperparameters,
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
                original_hyperparameters=hyperparameters,
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


@client
async def acancel_fine_tuning_job(
    fine_tuning_job_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> LiteLLMFineTuningJob:
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


@client
def cancel_fine_tuning_job(
    fine_tuning_job_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[LiteLLMFineTuningJob, Coroutine[Any, Any, LiteLLMFineTuningJob]]:
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
            and supports_httpx_timeout(custom_llm_provider) is False
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
                or os.getenv("OPENAI_BASE_URL")
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
                api_version=optional_params.api_version,
                organization=organization,
                fine_tuning_job_id=fine_tuning_job_id,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
                client=kwargs.get("client", None),
            )
        # Azure OpenAI
        elif custom_llm_provider == "azure":
            api_base = optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")  # type: ignore

            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
            )  # type: ignore

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_OPENAI_API_KEY")
                or get_secret_str("AZURE_API_KEY")
            )  # type: ignore

            extra_body = optional_params.get("extra_body", {})
            if extra_body is not None:
                extra_body.pop("azure_ad_token", None)
            else:
                get_secret_str("AZURE_AD_TOKEN")  # type: ignore

            response = azure_fine_tuning_apis_instance.cancel_fine_tuning_job(
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                fine_tuning_job_id=fine_tuning_job_id,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
                organization=optional_params.organization,
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
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
):
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
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
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
            and supports_httpx_timeout(custom_llm_provider) is False
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
                or os.getenv("OPENAI_BASE_URL")
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
                api_version=optional_params.api_version,
                organization=organization,
                after=after,
                limit=limit,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
                client=kwargs.get("client", None),
            )
        # Azure OpenAI
        elif custom_llm_provider == "azure":
            api_base = optional_params.api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")  # type: ignore

            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
            )  # type: ignore

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_OPENAI_API_KEY")
                or get_secret_str("AZURE_API_KEY")
            )  # type: ignore

            extra_body = optional_params.get("extra_body", {})
            if extra_body is not None:
                extra_body.pop("azure_ad_token", None)
            else:
                get_secret("AZURE_AD_TOKEN")  # type: ignore

            response = azure_fine_tuning_apis_instance.list_fine_tuning_jobs(
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                after=after,
                limit=limit,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
                organization=optional_params.organization,
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


@client
async def aretrieve_fine_tuning_job(
    fine_tuning_job_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> LiteLLMFineTuningJob:
    """
    Async: Get info about a fine-tuning job.
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["aretrieve_fine_tuning_job"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            retrieve_fine_tuning_job,
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


@client
def retrieve_fine_tuning_job(
    fine_tuning_job_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[LiteLLMFineTuningJob, Coroutine[Any, Any, LiteLLMFineTuningJob]]:
    """
    Get info about a fine-tuning job.
    """
    try:
        optional_params = GenericLiteLLMParams(**kwargs)
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(custom_llm_provider) is False
        ):
            read_timeout = timeout.read or 600
            timeout = read_timeout  # default 10 min timeout
        elif timeout is not None and not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore
        elif timeout is None:
            timeout = 600.0

        _is_async = kwargs.pop("aretrieve_fine_tuning_job", False) is True

        # OpenAI
        if custom_llm_provider == "openai":
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            organization = (
                optional_params.organization
                or litellm.organization
                or os.getenv("OPENAI_ORGANIZATION", None)
                or None
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.openai_key
                or os.getenv("OPENAI_API_KEY")
            )

            response = openai_fine_tuning_apis_instance.retrieve_fine_tuning_job(
                api_base=api_base,
                api_key=api_key,
                api_version=optional_params.api_version,
                organization=organization,
                fine_tuning_job_id=fine_tuning_job_id,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
                client=kwargs.get("client", None),
            )
        # Azure OpenAI
        elif custom_llm_provider == "azure":
            api_base = optional_params.api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")  # type: ignore

            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
            )  # type: ignore

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_OPENAI_API_KEY")
                or get_secret_str("AZURE_API_KEY")
            )  # type: ignore

            extra_body = optional_params.get("extra_body", {})
            if extra_body is not None:
                extra_body.pop("azure_ad_token", None)
            else:
                get_secret_str("AZURE_AD_TOKEN")  # type: ignore

            response = azure_fine_tuning_apis_instance.retrieve_fine_tuning_job(
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                fine_tuning_job_id=fine_tuning_job_id,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
                organization=optional_params.organization,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'retrieve_fine_tuning_job'. Only 'openai' and 'azure' are supported.".format(
                    custom_llm_provider
                ),
                model="n/a",
                llm_provider=custom_llm_provider,
                response=httpx.Response(
                    status_code=400,
                    content="Unsupported provider",
                    request=httpx.Request(method="retrieve_fine_tuning_job", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )
        return response
    except Exception as e:
        raise e
