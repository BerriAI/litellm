"""
Main File for Batches API implementation

https://platform.openai.com/docs/api-reference/batch

- create_batch()
- retrieve_batch()
- cancel_batch()
- list_batch()

"""

import asyncio
import contextvars
import os
from functools import partial
from typing import Any, Coroutine, Dict, Literal, Optional, Union, cast

import httpx
from openai.types.batch import BatchRequestCounts

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.batches.handler import AnthropicBatchesHandler
from litellm.llms.azure.batches.handler import AzureBatchesAPI
from litellm.llms.bedrock.batches.handler import BedrockBatchesHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.openai import OpenAIBatchesAPI
from litellm.llms.vertex_ai.batches.handler import VertexAIBatchPrediction
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    Batch,
    CancelBatchRequest,
    CreateBatchRequest,
    RetrieveBatchRequest,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS,
    LiteLLMBatch,
    LlmProviders,
)
from litellm.utils import (
    ProviderConfigManager,
    client,
    get_litellm_params,
    get_llm_provider,
    supports_httpx_timeout,
)

####### ENVIRONMENT VARIABLES ###################
openai_batches_instance = OpenAIBatchesAPI()
azure_batches_instance = AzureBatchesAPI()
vertex_ai_batches_instance = VertexAIBatchPrediction(gcs_bucket_name="")
anthropic_batches_instance = AnthropicBatchesHandler()
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


def _resolve_timeout(
    optional_params: GenericLiteLLMParams,
    kwargs: Dict[str, Any],
    custom_llm_provider: str,
    default_timeout: float = 600.0,
) -> float:
    """
    Resolve timeout value from various sources and handle httpx.Timeout objects.

    Args:
        optional_params: GenericLiteLLMParams object containing timeout
        kwargs: Additional kwargs that may contain request_timeout
        custom_llm_provider: Provider name for httpx timeout support check
        default_timeout: Default timeout value to use

    Returns:
        Resolved timeout as float
    """
    timeout = (
        optional_params.timeout
        or kwargs.get("request_timeout", default_timeout)
        or default_timeout
    )

    # Handle httpx.Timeout objects
    if isinstance(timeout, httpx.Timeout):
        if supports_httpx_timeout(custom_llm_provider) is False:
            # Extract read timeout for providers that don't support httpx.Timeout
            read_timeout = timeout.read or default_timeout
            return float(read_timeout)
        else:
            # For providers that support httpx.Timeout, we still need to return a float
            # This case might need to be handled differently based on the actual use case
            return float(timeout.read or default_timeout)

    # Handle None case
    if timeout is None:
        return float(default_timeout)

    # Handle numeric values (int, float, string representations)
    return float(timeout)


@client
async def acreate_batch(
    completion_window: Literal["24h"],
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"],
    input_file_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "bedrock", "hosted_vllm"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> LiteLLMBatch:
    """
    Async: Creates and executes a batch from an uploaded file of request

    LiteLLM Equivalent of POST: https://api.openai.com/v1/batches
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_batch"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            create_batch,
            completion_window,
            endpoint,
            input_file_id,
            custom_llm_provider,
            metadata,
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
            response = init_response

        return response
    except Exception as e:
        raise e


@client
def create_batch(
    completion_window: Literal["24h"],
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"],
    input_file_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "bedrock", "hosted_vllm"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
    """
    Creates and executes a batch from an uploaded file of request

    LiteLLM Equivalent of POST: https://api.openai.com/v1/batches
    """
    try:
        optional_params = GenericLiteLLMParams(**kwargs)
        litellm_call_id = kwargs.get("litellm_call_id", None)
        proxy_server_request = kwargs.get("proxy_server_request", None)
        model_info = kwargs.get("model_info", None)
        model: Optional[str] = kwargs.get("model", None)
        try:
            if model is not None:
                model, _, _, _ = get_llm_provider(
                    model=model,
                    custom_llm_provider=None,
                )
        except Exception as e:
            verbose_logger.exception(
                f"litellm.batches.main.py::create_batch() - Error inferring custom_llm_provider - {str(e)}"
            )

        _is_async = kwargs.pop("acreate_batch", False) is True
        litellm_params = dict(GenericLiteLLMParams(**kwargs))
        litellm_logging_obj: LiteLLMLoggingObj = cast(
            LiteLLMLoggingObj, kwargs.get("litellm_logging_obj", None)
        )
        ### TIMEOUT LOGIC ###
        timeout = _resolve_timeout(optional_params, kwargs, custom_llm_provider)
        litellm_logging_obj.update_environment_variables(
            model=model,
            user=None,
            optional_params=optional_params.model_dump(),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                "proxy_server_request": proxy_server_request,
                "model_info": model_info,
                "metadata": metadata,
                "preset_cache_key": None,
                "stream_response": {},
                **optional_params.model_dump(exclude_unset=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        _create_batch_request = CreateBatchRequest(
            completion_window=completion_window,
            endpoint=endpoint,
            input_file_id=input_file_id,
            metadata=metadata,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        if model is not None:
            provider_config = ProviderConfigManager.get_provider_batches_config(
                model=model,
                provider=LlmProviders(custom_llm_provider),
            )
        else:
            provider_config = None
        if provider_config is not None:
            response = base_llm_http_handler.create_batch(
                provider_config=provider_config,
                litellm_params=litellm_params,
                create_batch_data=_create_batch_request,
                headers=extra_headers or {},
                api_base=optional_params.api_base,
                api_key=optional_params.api_key,
                logging_obj=litellm_logging_obj,
                _is_async=_is_async,
                client=(
                    client
                    if client is not None
                    and isinstance(client, (HTTPHandler, AsyncHTTPHandler))
                    else None
                ),
                timeout=timeout,
                model=model,
            )
            return response
        api_base: Optional[str] = None
        if custom_llm_provider in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS:
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

            response = openai_batches_instance.create_batch(
                api_base=api_base,
                api_key=api_key,
                organization=organization,
                create_batch_data=_create_batch_request,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                _is_async=_is_async,
            )
        elif custom_llm_provider == "azure":
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or get_secret_str("AZURE_API_BASE")
            )
            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
            )

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_OPENAI_API_KEY")
                or get_secret_str("AZURE_API_KEY")
            )

            extra_body = optional_params.get("extra_body", {})
            if extra_body is not None:
                extra_body.pop("azure_ad_token", None)
            else:
                get_secret_str("AZURE_AD_TOKEN")  # type: ignore

            response = azure_batches_instance.create_batch(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                create_batch_data=_create_batch_request,
                litellm_params=litellm_params,
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

            response = vertex_ai_batches_instance.create_batch(
                _is_async=_is_async,
                api_base=api_base,
                vertex_project=vertex_ai_project,
                vertex_location=vertex_ai_location,
                vertex_credentials=vertex_credentials,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                create_batch_data=_create_batch_request,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support custom_llm_provider={} for 'create_batch'".format(
                    custom_llm_provider
                ),
                model="n/a",
                llm_provider=custom_llm_provider,
                response=httpx.Response(
                    status_code=400,
                    content="Unsupported provider",
                    request=httpx.Request(method="create_batch", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )
        return response
    except Exception as e:
        raise e


@client
async def aretrieve_batch(
    batch_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "bedrock", "hosted_vllm", "anthropic"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> LiteLLMBatch:
    """
    Async: Retrieves a batch.

    LiteLLM Equivalent of GET https://api.openai.com/v1/batches/{batch_id}
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["aretrieve_batch"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            retrieve_batch,
            batch_id,
            custom_llm_provider,
            metadata,
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


def _handle_retrieve_batch_providers_without_provider_config(
    batch_id: str,
    optional_params: GenericLiteLLMParams,
    timeout: Union[float, httpx.Timeout],
    litellm_params: dict,
    _retrieve_batch_request: RetrieveBatchRequest,
    _is_async: bool,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "bedrock", "hosted_vllm", "anthropic"] = "openai",
):
    api_base: Optional[str] = None
    if custom_llm_provider in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS:
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

        response = openai_batches_instance.retrieve_batch(
            _is_async=_is_async,
            retrieve_batch_data=_retrieve_batch_request,
            api_base=api_base,
            api_key=api_key,
            organization=organization,
            timeout=timeout,
            max_retries=optional_params.max_retries,
        )
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base
            or litellm.api_base
            or get_secret_str("AZURE_API_BASE")
        )
        api_version = (
            optional_params.api_version
            or litellm.api_version
            or get_secret_str("AZURE_API_VERSION")
        )

        api_key = (
            optional_params.api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret_str("AZURE_OPENAI_API_KEY")
            or get_secret_str("AZURE_API_KEY")
        )

        extra_body = optional_params.get("extra_body", {})
        if extra_body is not None:
            extra_body.pop("azure_ad_token", None)
        else:
            get_secret_str("AZURE_AD_TOKEN")  # type: ignore

        response = azure_batches_instance.retrieve_batch(
            _is_async=_is_async,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            retrieve_batch_data=_retrieve_batch_request,
            litellm_params=litellm_params,
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

        response = vertex_ai_batches_instance.retrieve_batch(
            _is_async=_is_async,
            batch_id=batch_id,
            api_base=api_base,
            vertex_project=vertex_ai_project,
            vertex_location=vertex_ai_location,
            vertex_credentials=vertex_credentials,
            timeout=timeout,
            max_retries=optional_params.max_retries,
        )
    elif custom_llm_provider == "anthropic":
        api_base = (
            optional_params.api_base
            or litellm.api_base
            or get_secret_str("ANTHROPIC_API_BASE")
        )
        api_key = (
            optional_params.api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret_str("ANTHROPIC_API_KEY")
        )

        response = anthropic_batches_instance.retrieve_batch(
            _is_async=_is_async,
            batch_id=batch_id,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
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


@client
def retrieve_batch(
    batch_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "bedrock", "hosted_vllm", "anthropic"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
    """
    Retrieves a batch.

    LiteLLM Equivalent of GET https://api.openai.com/v1/batches/{batch_id}
    """
    try:
        optional_params = GenericLiteLLMParams(**kwargs)
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
            "litellm_logging_obj", None
        )
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        litellm_params = get_litellm_params(
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        if litellm_logging_obj is not None:
            litellm_logging_obj.update_environment_variables(
                model=None,
                user=None,
                optional_params=optional_params.model_dump(),
                litellm_params=litellm_params,
                custom_llm_provider=custom_llm_provider,
            )

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

        _retrieve_batch_request = RetrieveBatchRequest(
            batch_id=batch_id,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

        _is_async = kwargs.pop("aretrieve_batch", False) is True
        client = kwargs.get("client", None)

        # Check if this is an async invoke ARN (different from regular batch ARN)
        # Async invoke ARNs have format: arn:aws(-[^:]+)?:bedrock:[a-z0-9-]{1,20}:[0-9]{12}:async-invoke/[a-z0-9]{12}
        if (
            batch_id.startswith("arn:aws")
            and ":bedrock:" in batch_id
            and ":async-invoke/" in batch_id
        ):
            # Handle async invoke status check
            # Remove aws_region_name from kwargs to avoid duplicate parameter
            async_kwargs = kwargs.copy()
            async_kwargs.pop("aws_region_name", None)

            return BedrockBatchesHandler._handle_async_invoke_status(
                batch_id=batch_id,
                aws_region_name=kwargs.get("aws_region_name", "us-east-1"),
                logging_obj=litellm_logging_obj,
                **async_kwargs,
            )

        # Try to use provider config first (for providers like bedrock)
        model: Optional[str] = kwargs.get("model", None)
        if model is not None:
            provider_config = ProviderConfigManager.get_provider_batches_config(
                model=model,
                provider=LlmProviders(custom_llm_provider),
            )
        else:
            provider_config = None

        if provider_config is not None:
            response = base_llm_http_handler.retrieve_batch(
                batch_id=batch_id,
                provider_config=provider_config,
                litellm_params=litellm_params,
                headers=extra_headers or {},
                api_base=optional_params.api_base,
                api_key=optional_params.api_key,
                logging_obj=litellm_logging_obj
                or LiteLLMLoggingObj(
                    model=model or f"{custom_llm_provider}/unknown",
                    messages=[],
                    stream=False,
                    call_type="batch_retrieve",
                    start_time=None,
                    litellm_call_id="batch_retrieve_" + batch_id,
                    function_id="batch_retrieve",
                ),
                _is_async=_is_async,
                client=(
                    client
                    if client is not None
                    and isinstance(client, (HTTPHandler, AsyncHTTPHandler))
                    else None
                ),
                timeout=timeout,
                model=model,
            )
            return response

        #########################################################
        # Handle providers without provider config
        #########################################################
        return _handle_retrieve_batch_providers_without_provider_config(
            batch_id=batch_id,
            custom_llm_provider=custom_llm_provider,
            optional_params=optional_params,
            litellm_params=litellm_params,
            _retrieve_batch_request=_retrieve_batch_request,
            _is_async=_is_async,
            timeout=timeout,
        )

    except Exception as e:
        raise e


@client
async def alist_batches(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    custom_llm_provider: Literal["openai", "azure", "hosted_vllm", "vertex_ai"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
):
    """
    Async: List your organization's batches.
    """

    try:
        loop = asyncio.get_event_loop()
        kwargs["alist_batches"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            list_batches,
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


@client
def list_batches(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    custom_llm_provider: Literal["openai", "azure", "hosted_vllm", "vertex_ai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
):
    """
    Lists batches

    List your organization's batches.
    """
    try:
        # set API KEY
        optional_params = GenericLiteLLMParams(**kwargs)
        litellm_params = get_litellm_params(
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        api_key = (
            optional_params.api_key
            or litellm.api_key  # for deepinfra/perplexity/anyscale we check in get_llm_provider and pass in the api key from there
            or litellm.openai_key
            or os.getenv("OPENAI_API_KEY")
        )
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

        _is_async = kwargs.pop("alist_batches", False) is True
        if custom_llm_provider in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS:
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

            response = openai_batches_instance.list_batches(
                _is_async=_is_async,
                after=after,
                limit=limit,
                api_base=api_base,
                api_key=api_key,
                organization=organization,
                timeout=timeout,
                max_retries=optional_params.max_retries,
            )
        elif custom_llm_provider == "azure":
            api_base = optional_params.api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")  # type: ignore
            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
            )

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_OPENAI_API_KEY")
                or get_secret_str("AZURE_API_KEY")
            )

            extra_body = optional_params.get("extra_body", {})
            if extra_body is not None:
                extra_body.pop("azure_ad_token", None)
            else:
                get_secret_str("AZURE_AD_TOKEN")  # type: ignore

            response = azure_batches_instance.list_batches(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                litellm_params=litellm_params,
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

            response = vertex_ai_batches_instance.list_batches(
                _is_async=_is_async,
                after=after,
                limit=limit,
                api_base=api_base,
                vertex_project=vertex_ai_project,
                vertex_location=vertex_ai_location,
                vertex_credentials=vertex_credentials,
                timeout=timeout,
                max_retries=optional_params.max_retries,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'list_batch'. Supported providers: openai, azure, vertex_ai.".format(
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


async def acancel_batch(
    batch_id: str,
    model: Optional[str] = None,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Batch:
    """
    Async: Cancels a batch.

    LiteLLM Equivalent of POST https://api.openai.com/v1/batches/{batch_id}/cancel
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["acancel_batch"] = True
        model = kwargs.pop("model", None)

        # Use a partial function to pass your keyword arguments
        func = partial(
            cancel_batch,
            batch_id,
            model,
            custom_llm_provider,
            metadata,
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
            response = init_response

        return response
    except Exception as e:
        raise e


def cancel_batch(
    batch_id: str,
    model: Optional[str] = None,
    custom_llm_provider: Union[Literal["openai", "azure"], str] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[Batch, Coroutine[Any, Any, Batch]]:
    """
    Cancels a batch.

    LiteLLM Equivalent of POST https://api.openai.com/v1/batches/{batch_id}/cancel
    """
    try:

        try:
            if model is not None:
                _, custom_llm_provider, _, _ = get_llm_provider(
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
        except Exception as e:
            verbose_logger.exception(
                f"litellm.batches.main.py::cancel_batch() - Error inferring custom_llm_provider - {str(e)}"
            )
        optional_params = GenericLiteLLMParams(**kwargs)
        litellm_params = get_litellm_params(
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
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

        _cancel_batch_request = CancelBatchRequest(
            batch_id=batch_id,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

        _is_async = kwargs.pop("acancel_batch", False) is True
        api_base: Optional[str] = None
        if custom_llm_provider in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS:
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

            response = openai_batches_instance.cancel_batch(
                _is_async=_is_async,
                cancel_batch_data=_cancel_batch_request,
                api_base=api_base,
                api_key=api_key,
                organization=organization,
                timeout=timeout,
                max_retries=optional_params.max_retries,
            )
        elif custom_llm_provider == "azure":
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or get_secret_str("AZURE_API_BASE")
            )
            api_version = (
                optional_params.api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
            )

            api_key = (
                optional_params.api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_OPENAI_API_KEY")
                or get_secret_str("AZURE_API_KEY")
            )

            extra_body = optional_params.get("extra_body", {})
            if extra_body is not None:
                extra_body.pop("azure_ad_token", None)
            else:
                get_secret_str("AZURE_AD_TOKEN")  # type: ignore

            response = azure_batches_instance.cancel_batch(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                cancel_batch_data=_cancel_batch_request,
                litellm_params=litellm_params,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'cancel_batch'. Only 'openai' and 'azure' are supported.".format(
                    custom_llm_provider
                ),
                model="n/a",
                llm_provider=custom_llm_provider,
                response=httpx.Response(
                    status_code=400,
                    content="Unsupported provider",
                    request=httpx.Request(method="cancel_batch", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )
        return response
    except Exception as e:
        raise e


def _handle_async_invoke_status(
    batch_id: str, aws_region_name: str, logging_obj=None, **kwargs
) -> "LiteLLMBatch":
    """
    Handle async invoke status check for AWS Bedrock.

    Args:
        batch_id: The async invoke ARN
        aws_region_name: AWS region name
        **kwargs: Additional parameters

    Returns:
        dict: Status information including status, output_file_id (S3 URL), etc.
    """
    import asyncio

    from litellm.llms.bedrock.embed.embedding import BedrockEmbedding

    async def _async_get_status():
        # Create embedding handler instance
        embedding_handler = BedrockEmbedding()

        # Get the status of the async invoke job
        status_response = await embedding_handler._get_async_invoke_status(
            invocation_arn=batch_id,
            aws_region_name=aws_region_name,
            logging_obj=logging_obj,
            **kwargs,
        )

        # Transform response to a LiteLLMBatch object
        from litellm.types.llms.openai import BatchJobStatus
        from litellm.types.utils import LiteLLMBatch

        # Normalize status to lowercase (AWS returns 'Completed', 'Failed', etc.)
        aws_status_raw = status_response.get("status", "")
        aws_status_lower = aws_status_raw.lower()
        # Map AWS status values to LiteLLM expected values
        status_mapping: dict[str, BatchJobStatus] = {
            "completed": "completed",
            "failed": "failed",
            "inprogress": "in_progress",
            "in_progress": "in_progress",
        }
        normalized_status: BatchJobStatus = status_mapping.get(aws_status_lower, "failed")  # Default to "failed" if unknown status

        # Get output S3 URI safely
        output_s3_uri = ""
        try:
            output_s3_uri = status_response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        except (KeyError, TypeError):
            pass
        
        # Use BedrockBatchesConfig's timestamp parsing method (expects raw AWS status string)
        import time

        from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig
        created_at, in_progress_at, completed_at, failed_at, _, _ = BedrockBatchesConfig()._parse_timestamps_and_status(status_response, aws_status_raw)
        result = LiteLLMBatch(
            id=status_response["invocationArn"],
            object="batch",
            status=normalized_status,
            created_at=created_at or int(time.time()),  # Provide default timestamp if None
            in_progress_at=in_progress_at,
            completed_at=completed_at,
            failed_at=failed_at,
            request_counts=BatchRequestCounts(
                total=1,
                completed=1 if normalized_status == "completed" else 0,
                failed=1 if normalized_status == "failed" else 0,
            ),
            metadata=dict(
                **{
                    "output_file_id": output_s3_uri,
                    "failure_message": status_response.get("failureMessage") or "",
                    "model_arn": status_response["modelArn"],
                }
            ),
            completion_window="24h",
            endpoint="/v1/embeddings",
            input_file_id="",
        )

        return result

    # Since this function is called from within an async context via run_in_executor,
    # we need to create a new event loop in a thread to avoid conflicts
    import concurrent.futures

    def run_in_thread():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(_async_get_status())
        finally:
            new_loop.close()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()
