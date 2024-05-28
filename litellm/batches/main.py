"""
Main File for Batches API implementation

https://platform.openai.com/docs/api-reference/batch

- create_batch()
- retrieve_batch()
- cancel_batch()
- list_batch()

"""

from typing import Iterable
import os
import litellm
from openai import OpenAI
import httpx
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
)

from typing import Literal, Optional, Dict

####### ENVIRONMENT VARIABLES ###################
openai_batches_instance = OpenAIBatchesAPI()
openai_files_instance = OpenAIFilesAPI()
#################################################


def create_file(
    file: FileTypes,
    purpose: Literal["assistants", "batch", "fine-tune"],
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
) -> FileObject:
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

            response = openai_files_instance.create_file(
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


def create_batch(
    completion_window: Literal["24h"],
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings", "/v1/completions"],
    input_file_id: str,
    custom_llm_provider: Literal["openai"] = "openai",
    metadata: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, str]] = None,
    **kwargs,
):
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


def retrieve_batch():
    pass


def cancel_batch():
    pass


def list_batch():
    pass


# Async Functions
async def acreate_batch():
    pass


async def aretrieve_batch():
    pass


async def acancel_batch():
    pass


async def alist_batch():
    pass
