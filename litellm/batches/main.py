"""
Main File for Batches API implementation

https://platform.openai.com/docs/api-reference/batch

- create_batch()
- retrieve_batch()
- cancel_batch()
- list_batch()

"""

import os
import asyncio
from functools import partial
import contextvars
from typing import Literal, Optional, Dict, Coroutine, Any, Union
import httpx

import litellm
from litellm import client
from litellm.utils import supports_httpx_timeout
from ..types.router import *
from ..llms.openai import OpenAIBatchesAPI, OpenAIFilesAPI
from ..types.llms.openai import (
    CreateBatchRequest,
    RetrieveBatchRequest,
    CancelBatchRequest,
    CreateFileRequest,
    FileTypes,
    FileObject,
    Batch,
    FileContentRequest,
    HttpxBinaryResponseContent,
)

####### ENVIRONMENT VARIABLES ###################
openai_batches_instance = OpenAIBatchesAPI()
openai_files_instance = OpenAIFilesAPI()
#################################################


async def acreate_file(
    file: FileTypes,
    purpose: Literal["assistants", "batch", "fine-tune"],
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Coroutine[Any, Any, FileObject]:
    """
    Async: Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.

    LiteLLM Equivalent of POST: POST https://api.openai.com/v1/files
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_file"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            create_file,
            file,
            purpose,
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


def create_file(
    file: FileTypes,
    purpose: Literal["assistants", "batch", "fine-tune"],
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[FileObject, Coroutine[Any, Any, FileObject]]:
    """
    Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.

    LiteLLM Equivalent of POST: POST https://api.openai.com/v1/files
    """
    try:
        optional_params = GenericLiteLLMParams(**kwargs)
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
            ### TIMEOUT LOGIC ###
            timeout = (
                optional_params.timeout or kwargs.get("request_timeout", 600) or 600
            )
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

            _create_file_request = CreateFileRequest(
                file=file,
                purpose=purpose,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )

            _is_async = kwargs.pop("acreate_file", False) is True

            response = openai_files_instance.create_file(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
                create_file_data=_create_file_request,
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


async def afile_content(
    file_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Coroutine[Any, Any, HttpxBinaryResponseContent]:
    """
    Async: Get file contents

    LiteLLM Equivalent of GET https://api.openai.com/v1/files
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["afile_content"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            file_content,
            file_id,
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


def file_content(
    file_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]]:
    """
    Returns the contents of the specified file.

    LiteLLM Equivalent of POST: POST https://api.openai.com/v1/files
    """
    try:
        optional_params = GenericLiteLLMParams(**kwargs)
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
            ### TIMEOUT LOGIC ###
            timeout = (
                optional_params.timeout or kwargs.get("request_timeout", 600) or 600
            )
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

            _file_content_request = FileContentRequest(
                file_id=file_id,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )

            _is_async = kwargs.pop("afile_content", False) is True

            response = openai_files_instance.file_content(
                _is_async=_is_async,
                file_content_request=_file_content_request,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
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


async def acreate_batch(
    completion_window: Literal["24h"],
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"],
    input_file_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Coroutine[Any, Any, Batch]:
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
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings"],
    input_file_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
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
            ### TIMEOUT LOGIC ###
            timeout = (
                optional_params.timeout or kwargs.get("request_timeout", 600) or 600
            )
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

            _is_async = kwargs.pop("acreate_batch", False) is True

            _create_batch_request = CreateBatchRequest(
                completion_window=completion_window,
                endpoint=endpoint,
                input_file_id=input_file_id,
                metadata=metadata,
                extra_headers=extra_headers,
                extra_body=extra_body,
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
    custom_llm_provider: Literal["openai"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Coroutine[Any, Any, Batch]:
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
    custom_llm_provider: Literal["openai"] = "openai",
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
            ### TIMEOUT LOGIC ###
            timeout = (
                optional_params.timeout or kwargs.get("request_timeout", 600) or 600
            )
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

            _retrieve_batch_request = RetrieveBatchRequest(
                batch_id=batch_id,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )

            _is_async = kwargs.pop("aretrieve_batch", False) is True

            response = openai_batches_instance.retrieve_batch(
                _is_async=_is_async,
                retrieve_batch_data=_retrieve_batch_request,
                api_base=api_base,
                api_key=api_key,
                organization=organization,
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
    except Exception as e:
        raise e


def cancel_batch():
    pass


def list_batch():
    pass


async def acancel_batch():
    pass


async def alist_batch():
    pass
