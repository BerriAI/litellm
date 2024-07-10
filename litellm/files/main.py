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
from openai.types.file_object import FileObject

import litellm
from litellm import client
from litellm.batches.main import openai_files_instance
from litellm.types.llms.openai import (
    Batch,
    CreateFileRequest,
    FileContentRequest,
    FileTypes,
    HttpxBinaryResponseContent,
)
from litellm.types.router import *
from litellm.utils import supports_httpx_timeout


async def afile_retrieve(
    file_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
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
    custom_llm_provider: Literal["openai"] = "openai",
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
