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
from typing import Any, Coroutine, Dict, Literal, Optional, Union

import httpx

import litellm
from litellm import client
from litellm.llms.azure import AzureBatchesAPI
from litellm.llms.openai import OpenAIBatchesAPI
from litellm.types.llms.openai import (
    Batch,
    CancelBatchRequest,
    CreateBatchRequest,
    CreateFileRequest,
    FileContentRequest,
    FileObject,
    FileTypes,
    HttpxBinaryResponseContent,
    RetrieveBatchRequest,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import get_secret, supports_httpx_timeout

####### ENVIRONMENT VARIABLES ###################
openai_batches_instance = OpenAIBatchesAPI()
azure_batches_instance = AzureBatchesAPI()
#################################################


async def acreate_batch(
    completion_window: Literal["24h"],
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"],
    input_file_id: str,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Batch:
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
            response = init_response  # type: ignore

        return response
    except Exception as e:
        raise e


def create_batch(
    completion_window: Literal["24h"],
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"],
    input_file_id: str,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[Batch, Coroutine[Any, Any, Batch]]:
    """
    Creates and executes a batch from an uploaded file of request

    LiteLLM Equivalent of POST: https://api.openai.com/v1/batches
    """
    try:
        optional_params = GenericLiteLLMParams(**kwargs)
        _is_async = kwargs.pop("acreate_batch", False) is True
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

        _create_batch_request = CreateBatchRequest(
            completion_window=completion_window,
            endpoint=endpoint,
            input_file_id=input_file_id,
            metadata=metadata,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

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

            response = azure_batches_instance.create_batch(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                create_batch_data=_create_batch_request,
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


async def aretrieve_batch(
    batch_id: str,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Batch:
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


def retrieve_batch(
    batch_id: str,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[Batch, Coroutine[Any, Any, Batch]]:
    """
    Retrieves a batch.

    LiteLLM Equivalent of GET https://api.openai.com/v1/batches/{batch_id}
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

        _retrieve_batch_request = RetrieveBatchRequest(
            batch_id=batch_id,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

        _is_async = kwargs.pop("aretrieve_batch", False) is True

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

            response = azure_batches_instance.retrieve_batch(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                retrieve_batch_data=_retrieve_batch_request,
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


async def alist_batches(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Batch:
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


def list_batches(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
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

            response = azure_batches_instance.list_batches(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'list_batch'. Only 'openai' is supported.".format(
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
    pass


def cancel_batch():
    pass


async def acancel_batch():
    pass
