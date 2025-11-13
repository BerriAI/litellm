"""
Main File for Files API implementation

https://platform.openai.com/docs/api-reference/files

"""

import asyncio
import contextvars
import os
from functools import partial
from typing import Any, Coroutine, Dict, Literal, Optional, Union, cast

import httpx

import litellm
from litellm import get_secret_str
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.azure.files.handler import AzureOpenAIFilesAPI
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.openai import FileDeleted, FileObject, OpenAIFilesAPI
from litellm.llms.vertex_ai.files.handler import VertexAIFilesHandler
from litellm.types.llms.openai import (
    CreateFileRequest,
    FileContentRequest,
    FileTypes,
    HttpxBinaryResponseContent,
    OpenAIFileObject,
)
from litellm.types.router import *
from litellm.types.utils import LlmProviders
from litellm.utils import (
    ProviderConfigManager,
    client,
    get_litellm_params,
    supports_httpx_timeout,
)

base_llm_http_handler = BaseLLMHTTPHandler()

####### ENVIRONMENT VARIABLES ###################
openai_files_instance = OpenAIFilesAPI()
azure_files_instance = AzureOpenAIFilesAPI()
vertex_ai_files_instance = VertexAIFilesHandler()
#################################################


# Helper function for Anthropic Files API
def _transform_anthropic_file_response(http_response: httpx.Response) -> FileObject:
    """Transform Anthropic file response to OpenAI FileObject format"""
    import time
    response_json = http_response.json()

    # Parse created_at
    try:
        from dateutil import parser
        created_at = int(parser.parse(response_json.get("created_at", "")).timestamp())
    except Exception:
        created_at = int(time.time())

    return FileObject(
        id=response_json.get("id", ""),
        object="file",
        bytes=response_json.get("size_bytes", 0),
        created_at=created_at,
        filename=response_json.get("filename", ""),
        purpose="assistants",
        status="processed",
    )


@client
async def acreate_file(
    file: FileTypes,
    purpose: Literal["assistants", "batch", "fine-tune"],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "bedrock", "anthropic"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> OpenAIFileObject:
    """
    Async: Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.

    LiteLLM Equivalent of POST: POST https://api.openai.com/v1/files
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_file"] = True

        call_args = {
            "file": file,
            "purpose": purpose,
            "custom_llm_provider": custom_llm_provider,
            "extra_headers": extra_headers,
            "extra_body": extra_body,
            **kwargs,
        }

        # Use a partial function to pass your keyword arguments
        func = partial(create_file, **call_args)

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
def create_file(
    file: FileTypes,
    purpose: Literal["assistants", "batch", "fine-tune"],
    custom_llm_provider: Optional[Literal["openai", "azure", "vertex_ai", "bedrock", "anthropic"]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
    """
    Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and Batch API.

    LiteLLM Equivalent of POST: POST https://api.openai.com/v1/files

    Specify either provider_list or custom_llm_provider.
    """
    try:
        _is_async = kwargs.pop("acreate_file", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)
        litellm_params_dict = dict(**kwargs)
        logging_obj = cast(
            Optional[LiteLLMLoggingObj], kwargs.get("litellm_logging_obj")
        )
        if logging_obj is None:
            raise ValueError("logging_obj is required")
        client = kwargs.get("client")

        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(cast(str, custom_llm_provider)) is False
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

        provider_config = ProviderConfigManager.get_provider_files_config(
            model="",
            provider=LlmProviders(custom_llm_provider),
        )
        if provider_config is not None:
            response = base_llm_http_handler.create_file(
                provider_config=provider_config,
                litellm_params=litellm_params_dict,
                create_file_data=_create_file_request,
                headers=extra_headers or {},
                api_base=optional_params.api_base,
                api_key=optional_params.api_key,
                logging_obj=logging_obj,
                _is_async=_is_async,
                client=client
                if client is not None
                and isinstance(client, (HTTPHandler, AsyncHTTPHandler))
                else None,
                timeout=timeout,
            )
        elif custom_llm_provider == "openai":
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

            response = azure_files_instance.create_file(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                create_file_data=_create_file_request,
                litellm_params=litellm_params_dict,
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

            response = vertex_ai_files_instance.create_file(
                _is_async=_is_async,
                api_base=api_base,
                vertex_project=vertex_ai_project,
                vertex_location=vertex_ai_location,
                vertex_credentials=vertex_credentials,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                create_file_data=_create_file_request,
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'create_file'. Only ['openai', 'azure', 'vertex_ai', 'anthropic'] are supported.".format(
                    custom_llm_provider
                ),
                model="n/a",
                llm_provider=custom_llm_provider,
                response=httpx.Response(
                    status_code=400,
                    content="Unsupported provider",
                    request=httpx.Request(method="create_file", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )
        return response
    except Exception as e:
        raise e


@client
async def afile_retrieve(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure", "anthropic"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> OpenAIFileObject:
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
            response = init_response

        return OpenAIFileObject(**response.model_dump())
    except Exception as e:
        raise e


@client
def file_retrieve(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure", "anthropic"] = "openai",
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

        _is_async = kwargs.pop("is_async", False) is True

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

            response = openai_files_instance.retrieve_file(
                file_id=file_id,
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
            )
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

            response = azure_files_instance.retrieve_file(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                file_id=file_id,
            )
        elif custom_llm_provider == "anthropic":
            # Direct HTTP implementation for Anthropic Files API
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or get_secret_str("ANTHROPIC_API_BASE")
                or "https://api.anthropic.com"
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or get_secret_str("ANTHROPIC_API_KEY")
            )

            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY is required")

            # Prepare headers
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "files-api-2025-04-14",
            }

            url = f"{api_base}/v1/files/{file_id}"

            # Make HTTP request
            if _is_async:
                async def _retrieve():
                    if client is None or not isinstance(client, AsyncHTTPHandler):
                        async_client = AsyncHTTPHandler(timeout=timeout)
                    else:
                        async_client = client

                    http_response = await async_client.get(url=url, headers=headers)
                    return _transform_anthropic_file_response(http_response)

                response = _retrieve()
            else:
                if client is None or not isinstance(client, HTTPHandler):
                    sync_client = HTTPHandler(timeout=timeout)
                else:
                    sync_client = client

                http_response = sync_client.get(url=url, headers=headers)
                response = _transform_anthropic_file_response(http_response)
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'file_retrieve'. Only 'openai', 'azure', and 'anthropic' are supported.".format(
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

        return cast(FileObject, response)
    except Exception as e:
        raise e


# Delete file
@client
async def afile_delete(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure", "anthropic"] = "openai",
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

        return cast(FileDeleted, response)  # type: ignore
    except Exception as e:
        raise e


@client
def file_delete(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure", "anthropic"] = "openai",
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
        litellm_params_dict = get_litellm_params(**kwargs)
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default
        client = kwargs.get("client")

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
        _is_async = kwargs.pop("is_async", False) is True
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

            response = azure_files_instance.delete_file(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                file_id=file_id,
                client=client,
                litellm_params=litellm_params_dict,
            )
        elif custom_llm_provider == "anthropic":
            # Direct HTTP implementation for Anthropic Files API
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or get_secret_str("ANTHROPIC_API_BASE")
                or "https://api.anthropic.com"
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or get_secret_str("ANTHROPIC_API_KEY")
            )

            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY is required")

            # Prepare headers
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "files-api-2025-04-14",
            }

            # Make HTTP DELETE request
            url = f"{api_base}/v1/files/{file_id}"

            if _is_async:
                async def _delete():
                    if client is None or not isinstance(client, AsyncHTTPHandler):
                        async_client = AsyncHTTPHandler(timeout=timeout)
                    else:
                        async_client = client

                    await async_client.delete(url=url, headers=headers)
                    return FileDeleted(id=file_id, deleted=True, object="file")

                response = _delete()
            else:
                if client is None or not isinstance(client, HTTPHandler):
                    sync_client = HTTPHandler(timeout=timeout)
                else:
                    sync_client = client

                sync_client.delete(url=url, headers=headers)
                # Anthropic returns empty response on successful delete
                response = FileDeleted(id=file_id, deleted=True, object="file")
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'file_delete'. Only 'openai', 'azure', and 'anthropic' are supported.".format(
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
        return cast(FileDeleted, response)
    except Exception as e:
        raise e


# List files
@client
async def afile_list(
    custom_llm_provider: Literal["openai", "azure", "anthropic"] = "openai",
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


@client
def file_list(
    custom_llm_provider: Literal["openai", "azure", "anthropic"] = "openai",
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

        _is_async = kwargs.pop("is_async", False) is True
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

            response = openai_files_instance.list_files(
                purpose=purpose,
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
            )
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

            response = azure_files_instance.list_files(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                purpose=purpose,
            )
        elif custom_llm_provider == "anthropic":
            # Direct HTTP implementation for Anthropic Files API
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or get_secret_str("ANTHROPIC_API_BASE")
                or "https://api.anthropic.com"
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or get_secret_str("ANTHROPIC_API_KEY")
            )

            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY is required")

            # Prepare headers
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "files-api-2025-04-14",
            }

            # Make HTTP GET request
            url = f"{api_base}/v1/files"

            if _is_async:
                async def _list():
                    if client is None or not isinstance(client, AsyncHTTPHandler):
                        async_client = AsyncHTTPHandler(timeout=timeout)
                    else:
                        async_client = client

                    http_response = await async_client.get(url=url, headers=headers)

                    # Transform Anthropic response to OpenAI format
                    response_json = http_response.json()
                    files = []

                    for file_data in response_json.get("data", []):
                        # Parse created_at
                        import time
                        try:
                            from dateutil import parser
                            created_at = int(parser.parse(file_data.get("created_at", "")).timestamp())
                        except Exception:
                            created_at = int(time.time())

                        files.append(
                            FileObject(
                                id=file_data.get("id", ""),
                                object="file",
                                bytes=file_data.get("size_bytes", 0),
                                created_at=created_at,
                                filename=file_data.get("filename", ""),
                                purpose="assistants",
                                status="processed",
                            )
                        )

                    return files

                response = _list()
            else:
                if client is None or not isinstance(client, HTTPHandler):
                    sync_client = HTTPHandler(timeout=timeout)
                else:
                    sync_client = client

                http_response = sync_client.get(url=url, headers=headers)

                # Transform Anthropic response to OpenAI format
                response_json = http_response.json()
                files = []

                for file_data in response_json.get("data", []):
                    # Parse created_at
                    import time
                    try:
                        from dateutil import parser
                        created_at = int(parser.parse(file_data.get("created_at", "")).timestamp())
                    except Exception:
                        created_at = int(time.time())

                    files.append(
                        FileObject(
                            id=file_data.get("id", ""),
                            object="file",
                            bytes=file_data.get("size_bytes", 0),
                            created_at=created_at,
                            filename=file_data.get("filename", ""),
                            purpose="assistants",
                            status="processed",
                        )
                    )

                response = files
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'file_list'. Only 'openai', 'azure', and 'anthropic' are supported.".format(
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


@client
async def afile_content(
    file_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "anthropic"] = "openai",
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
        model = kwargs.pop("model", None)

        # Use a partial function to pass your keyword arguments
        func = partial(
            file_content,
            file_id,
            model,
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
def file_content(
    file_id: str,
    model: Optional[str] = None,
    custom_llm_provider: Optional[
        Union[Literal["openai", "azure", "vertex_ai", "anthropic"], str]
    ] = None,
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
        litellm_params_dict = get_litellm_params(**kwargs)
        ### TIMEOUT LOGIC ###
        timeout = optional_params.timeout or kwargs.get("request_timeout", 600) or 600
        client = kwargs.get("client")
        # set timeout for 10 minutes by default

        try:
            if model is not None:
                _, custom_llm_provider, _, _ = get_llm_provider(
                    model, custom_llm_provider
                )
        except Exception:
            pass

        if (
            timeout is not None
            and isinstance(timeout, httpx.Timeout)
            and supports_httpx_timeout(cast(str, custom_llm_provider)) is False
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

            response = openai_files_instance.file_content(
                _is_async=_is_async,
                file_content_request=_file_content_request,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                organization=organization,
            )
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

            response = azure_files_instance.file_content(
                _is_async=_is_async,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                timeout=timeout,
                max_retries=optional_params.max_retries,
                file_content_request=_file_content_request,
                client=client,
                litellm_params=litellm_params_dict,
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

            response = vertex_ai_files_instance.file_content(
                _is_async=_is_async,
                file_content_request=_file_content_request,
                api_base=api_base,
                vertex_credentials=vertex_credentials,
                vertex_project=vertex_ai_project,
                vertex_location=vertex_ai_location,
                timeout=timeout,
                max_retries=optional_params.max_retries,
            )
        elif custom_llm_provider == "anthropic":
            # Direct HTTP implementation for Anthropic Files API
            api_base = (
                optional_params.api_base
                or litellm.api_base
                or get_secret_str("ANTHROPIC_API_BASE")
                or "https://api.anthropic.com"
            )
            api_key = (
                optional_params.api_key
                or litellm.api_key
                or get_secret_str("ANTHROPIC_API_KEY")
            )

            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY is required")

            # Prepare headers
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "files-api-2025-04-14",
            }

            # Make HTTP GET request to download file content
            url = f"{api_base}/v1/files/{_file_content_request['file_id']}/content"

            if _is_async:
                async def _content():
                    if client is None or not isinstance(client, AsyncHTTPHandler):
                        async_client = AsyncHTTPHandler(timeout=timeout)
                    else:
                        async_client = client

                    http_response = await async_client.get(url=url, headers=headers)
                    return HttpxBinaryResponseContent(http_response.content)

                response = _content()
            else:
                if client is None or not isinstance(client, HTTPHandler):
                    sync_client = HTTPHandler(timeout=timeout)
                else:
                    sync_client = client

                http_response = sync_client.get(url=url, headers=headers)
                response = HttpxBinaryResponseContent(http_response.content)
        else:
            raise litellm.exceptions.BadRequestError(
                message="LiteLLM doesn't support {} for 'file_content'. Supported providers are 'openai', 'azure', 'vertex_ai', 'anthropic'.".format(
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
