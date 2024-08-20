"""
Main File for Files API implementation

https://platform.openai.com/docs/api-reference/files

"""

import asyncio
import contextvars
import os
from functools import partial
from typing import Any, Coroutine, Dict, Literal, Optional, Union

import httpx

import litellm
from litellm import client, get_secret
from litellm.llms.files_apis.azure import AzureOpenAIFilesAPI
from litellm.llms.openai import FileDeleted, FileObject, OpenAIFilesAPI
from litellm.types.llms.openai import (
    Batch,
    CreateFileRequest,
    FileContentRequest,
    FileTypes,
    HttpxBinaryResponseContent,
)
from litellm.types.router import *
from litellm.utils import supports_httpx_timeout

####### ENVIRONMENT VARIABLES ###################
openai_files_instance = OpenAIFilesAPI()
azure_files_instance = AzureOpenAIFilesAPI()
#################################################


async def afile_retrieve(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Coroutine[Any, Any, FileObject]:
    """
    Async: Get file contents

    LiteLLM Equivalent of GET https://api.openai.com/v1/files
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["is_async"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            file_retrieve,
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


def file_retrieve(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> FileObject:
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

            _is_async = kwargs.pop("is_async", False) is True

            response = openai_files_instance.retrieve_file(
                file_id=file_id,
                _is_async=_is_async,
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


# Delete file
async def afile_delete(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Coroutine[Any, Any, FileObject]:
    """
    Async: Delete file

    LiteLLM Equivalent of DELETE https://api.openai.com/v1/files
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["is_async"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            file_delete,
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


def file_delete(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> FileDeleted:
    """
    Delete file

    LiteLLM Equivalent of DELETE https://api.openai.com/v1/files
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
        _is_async = kwargs.pop("is_async", False) is True
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
            response = openai_files_instance.delete_file(
                file_id=file_id,
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
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

            response = azure_files_instance.delete_file(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                file_id=file_id,
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


# List files
async def afile_list(
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    purpose: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
):
    """
    Async: List files

    LiteLLM Equivalent of GET https://api.openai.com/v1/files
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["is_async"] = True

        # Use a partial function to pass your keyword arguments
        func = partial(
            file_list,
            custom_llm_provider,
            purpose,
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


def file_list(
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    purpose: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
):
    """
    List files

    LiteLLM Equivalent of GET https://api.openai.com/v1/files
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

            _is_async = kwargs.pop("is_async", False) is True

            response = openai_files_instance.list_files(
                purpose=purpose,
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'file_list'. Only 'openai' is supported.".format(
                    custom_llm_provider
                ),
                model="n/a",
                llm_provider=custom_llm_provider,
                response=httpx.Response(
                    status_code=400,
                    content="Unsupported provider",
                    request=httpx.Request(method="file_list", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )
        return response
    except Exception as e:
        raise e


async def acreate_file(
    file: FileTypes,
    purpose: Literal["assistants", "batch", "fine-tune"],
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> FileObject:
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
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[FileObject, Coroutine[Any, Any, FileObject]]:
    """
    Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.

    LiteLLM Equivalent of POST: POST https://api.openai.com/v1/files
    """
    try:
        _is_async = kwargs.pop("acreate_file", False) is True
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

        _create_file_request = CreateFileRequest(
            file=file,
            purpose=purpose,
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

            response = openai_files_instance.create_file(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
                create_file_data=_create_file_request,
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

            response = azure_files_instance.create_file(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
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
    custom_llm_provider: Literal["openai", "azure"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> HttpxBinaryResponseContent:
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
    custom_llm_provider: Literal["openai", "azure"] = "openai",
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
