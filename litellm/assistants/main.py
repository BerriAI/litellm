# What is this?
## Main file for assistants API logic
import asyncio
import contextvars
import os
from functools import partial
from typing import Any, Coroutine, Dict, Iterable, List, Literal, Optional, Union

import httpx
from openai import AsyncOpenAI, OpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_deleted import AssistantDeleted

import litellm
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import (
    exception_type,
    get_litellm_params,
    get_llm_provider,
    get_secret,
    supports_httpx_timeout,
)

from ..llms.azure.assistants import AzureAssistantsAPI
from ..llms.openai.openai import OpenAIAssistantsAPI
from ..types.llms.openai import *
from ..types.router import *
from .utils import get_optional_params_add_message

####### ENVIRONMENT VARIABLES ###################
openai_assistants_api = OpenAIAssistantsAPI()
azure_assistants_api = AzureAssistantsAPI()

### ASSISTANTS ###


async def aget_assistants(
    custom_llm_provider: Literal["openai", "azure"],
    client: Optional[AsyncOpenAI] = None,
    **kwargs,
) -> AsyncCursorPage[Assistant]:
    loop = asyncio.get_event_loop()
    ### PASS ARGS TO GET ASSISTANTS ###
    kwargs["aget_assistants"] = True
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(get_assistants, custom_llm_provider, client, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
            model="", custom_llm_provider=custom_llm_provider
        )  # type: ignore

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response  # type: ignore
    except Exception as e:
        raise exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={},
            extra_kwargs=kwargs,
        )


def get_assistants(
    custom_llm_provider: Literal["openai", "azure"],
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    **kwargs,
) -> SyncCursorPage[Assistant]:
    aget_assistants: Optional[bool] = kwargs.pop("aget_assistants", None)
    if aget_assistants is not None and not isinstance(aget_assistants, bool):
        raise Exception(
            "Invalid value passed in for aget_assistants. Only bool or None allowed"
        )
    optional_params = GenericLiteLLMParams(
        api_key=api_key, api_base=api_base, api_version=api_version, **kwargs
    )
    litellm_params_dict = get_litellm_params(**kwargs)

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

    response: Optional[SyncCursorPage[Assistant]] = None
    if custom_llm_provider == "openai":
        api_base = (
            optional_params.api_base  # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
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

        response = openai_assistants_api.get_assistants(
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            organization=organization,
            client=client,
            aget_assistants=aget_assistants,  # type: ignore
        )  # type: ignore
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        )  # type: ignore

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

        response = azure_assistants_api.get_assistants(
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            client=client,
            aget_assistants=aget_assistants,  # type: ignore
            litellm_params=litellm_params_dict,
        )
    else:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'get_assistants'. Only 'openai' is supported.".format(
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

    if response is None:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'get_assistants'. Only 'openai' is supported.".format(
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


async def acreate_assistants(
    custom_llm_provider: Literal["openai", "azure"],
    client: Optional[AsyncOpenAI] = None,
    **kwargs,
) -> Assistant:
    loop = asyncio.get_event_loop()
    ### PASS ARGS TO GET ASSISTANTS ###
    kwargs["async_create_assistants"] = True
    model = kwargs.pop("model", None)
    try:
        kwargs["client"] = client
        # Use a partial function to pass your keyword arguments
        func = partial(create_assistants, custom_llm_provider, model, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
            model=model, custom_llm_provider=custom_llm_provider
        )  # type: ignore

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response  # type: ignore
    except Exception as e:
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={},
            extra_kwargs=kwargs,
        )


def create_assistants(
    custom_llm_provider: Literal["openai", "azure"],
    model: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_resources: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, str]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    response_format: Optional[Union[str, Dict[str, str]]] = None,
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    **kwargs,
) -> Union[Assistant, Coroutine[Any, Any, Assistant]]:
    async_create_assistants: Optional[bool] = kwargs.pop(
        "async_create_assistants", None
    )
    if async_create_assistants is not None and not isinstance(
        async_create_assistants, bool
    ):
        raise ValueError(
            "Invalid value passed in for async_create_assistants. Only bool or None allowed"
        )
    optional_params = GenericLiteLLMParams(
        api_key=api_key, api_base=api_base, api_version=api_version, **kwargs
    )
    litellm_params_dict = get_litellm_params(**kwargs)

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

    create_assistant_data = {
        "model": model,
        "name": name,
        "description": description,
        "instructions": instructions,
        "tools": tools,
        "tool_resources": tool_resources,
        "metadata": metadata,
        "temperature": temperature,
        "top_p": top_p,
        "response_format": response_format,
    }

    # only send params that are not None
    create_assistant_data = {
        k: v for k, v in create_assistant_data.items() if v is not None
    }

    response: Optional[Union[Coroutine[Any, Any, Assistant], Assistant]] = None
    if custom_llm_provider == "openai":
        api_base = (
            optional_params.api_base  # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
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

        response = openai_assistants_api.create_assistants(
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            organization=organization,
            create_assistant_data=create_assistant_data,
            client=client,
            async_create_assistants=async_create_assistants,  # type: ignore
        )  # type: ignore
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        )  # type: ignore

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

        if isinstance(client, OpenAI):
            client = None  # only pass client if it's AzureOpenAI

        response = azure_assistants_api.create_assistants(
            api_base=api_base,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            api_version=api_version,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            client=client,
            async_create_assistants=async_create_assistants,
            create_assistant_data=create_assistant_data,
            litellm_params=litellm_params_dict,
        )
    else:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'create_assistants'. Only 'openai' is supported.".format(
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
    if response is None:
        raise litellm.exceptions.InternalServerError(
            message="No response returned from 'create_assistants'",
            model=model,
            llm_provider=custom_llm_provider,
        )
    return response


async def adelete_assistant(
    custom_llm_provider: Literal["openai", "azure"],
    client: Optional[AsyncOpenAI] = None,
    **kwargs,
) -> AssistantDeleted:
    loop = asyncio.get_event_loop()
    ### PASS ARGS TO GET ASSISTANTS ###
    kwargs["async_delete_assistants"] = True
    try:
        kwargs["client"] = client
        # Use a partial function to pass your keyword arguments
        func = partial(delete_assistant, custom_llm_provider, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
            model="", custom_llm_provider=custom_llm_provider
        )  # type: ignore

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response  # type: ignore
    except Exception as e:
        raise exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={},
            extra_kwargs=kwargs,
        )


def delete_assistant(
    custom_llm_provider: Literal["openai", "azure"],
    assistant_id: str,
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    **kwargs,
) -> Union[AssistantDeleted, Coroutine[Any, Any, AssistantDeleted]]:
    optional_params = GenericLiteLLMParams(
        api_key=api_key, api_base=api_base, api_version=api_version, **kwargs
    )

    litellm_params_dict = get_litellm_params(**kwargs)

    async_delete_assistants: Optional[bool] = kwargs.pop(
        "async_delete_assistants", None
    )
    if async_delete_assistants is not None and not isinstance(
        async_delete_assistants, bool
    ):
        raise ValueError(
            "Invalid value passed in for async_delete_assistants. Only bool or None allowed"
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

    response: Optional[
        Union[AssistantDeleted, Coroutine[Any, Any, AssistantDeleted]]
    ] = None
    if custom_llm_provider == "openai":
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
            or None
        )
        # set API KEY
        api_key = (
            optional_params.api_key
            or litellm.api_key
            or litellm.openai_key
            or os.getenv("OPENAI_API_KEY")
        )

        response = openai_assistants_api.delete_assistant(
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            organization=organization,
            assistant_id=assistant_id,
            client=client,
            async_delete_assistants=async_delete_assistants,
        )
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        )  # type: ignore

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

        if isinstance(client, OpenAI):
            client = None  # only pass client if it's AzureOpenAI

        response = azure_assistants_api.delete_assistant(
            assistant_id=assistant_id,
            api_base=api_base,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            api_version=api_version,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            client=client,
            async_delete_assistants=async_delete_assistants,
            litellm_params=litellm_params_dict,
        )
    else:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'delete_assistant'. Only 'openai' is supported.".format(
                custom_llm_provider
            ),
            model="n/a",
            llm_provider=custom_llm_provider,
            response=httpx.Response(
                status_code=400,
                content="Unsupported provider",
                request=httpx.Request(
                    method="delete_assistant", url="https://github.com/BerriAI/litellm"
                ),
            ),
        )
    if response is None:
        raise litellm.exceptions.InternalServerError(
            message="No response returned from 'delete_assistant'",
            model="n/a",
            llm_provider=custom_llm_provider,
        )
    return response


### THREADS ###


async def acreate_thread(
    custom_llm_provider: Literal["openai", "azure"], **kwargs
) -> Thread:
    loop = asyncio.get_event_loop()
    ### PASS ARGS TO GET ASSISTANTS ###
    kwargs["acreate_thread"] = True
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(create_thread, custom_llm_provider, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
            model="", custom_llm_provider=custom_llm_provider
        )  # type: ignore

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response  # type: ignore
    except Exception as e:
        raise exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={},
            extra_kwargs=kwargs,
        )


def create_thread(
    custom_llm_provider: Literal["openai", "azure"],
    messages: Optional[Iterable[OpenAICreateThreadParamsMessage]] = None,
    metadata: Optional[dict] = None,
    tool_resources: Optional[OpenAICreateThreadParamsToolResources] = None,
    client: Optional[OpenAI] = None,
    **kwargs,
) -> Thread:
    """
    - get the llm provider
    - if openai - route it there
    - pass through relevant params

    ```
    from litellm import create_thread

    create_thread(
        custom_llm_provider="openai",
        ### OPTIONAL ###
        messages =  {
            "role": "user",
            "content": "Hello, what is AI?"
            },
            {
            "role": "user",
            "content": "How does AI work? Explain it in simple terms."
        }]
    )
    ```
    """
    acreate_thread = kwargs.get("acreate_thread", None)
    optional_params = GenericLiteLLMParams(**kwargs)
    litellm_params_dict = get_litellm_params(**kwargs)

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

    api_base: Optional[str] = None
    api_key: Optional[str] = None

    response: Optional[Thread] = None
    if custom_llm_provider == "openai":
        api_base = (
            optional_params.api_base  # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
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
        response = openai_assistants_api.create_thread(
            messages=messages,
            metadata=metadata,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            organization=organization,
            client=client,
            acreate_thread=acreate_thread,
        )
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        )  # type: ignore

        api_key = (
            optional_params.api_key
            or litellm.api_key
            or litellm.azure_key
            or get_secret("AZURE_OPENAI_API_KEY")
            or get_secret("AZURE_API_KEY")
        )  # type: ignore

        api_version: Optional[str] = (
            optional_params.api_version
            or litellm.api_version
            or get_secret("AZURE_API_VERSION")
        )  # type: ignore

        extra_body = optional_params.get("extra_body", {})
        azure_ad_token: Optional[str] = None
        if extra_body is not None:
            azure_ad_token = extra_body.pop("azure_ad_token", None)
        else:
            azure_ad_token = get_secret("AZURE_AD_TOKEN")  # type: ignore

        if isinstance(client, OpenAI):
            client = None  # only pass client if it's AzureOpenAI

        response = azure_assistants_api.create_thread(
            messages=messages,
            metadata=metadata,
            api_base=api_base,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            api_version=api_version,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            client=client,
            acreate_thread=acreate_thread,
            litellm_params=litellm_params_dict,
        )
    else:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'create_thread'. Only 'openai' is supported.".format(
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
    return response  # type: ignore


async def aget_thread(
    custom_llm_provider: Literal["openai", "azure"],
    thread_id: str,
    client: Optional[AsyncOpenAI] = None,
    **kwargs,
) -> Thread:
    loop = asyncio.get_event_loop()
    ### PASS ARGS TO GET ASSISTANTS ###
    kwargs["aget_thread"] = True
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(get_thread, custom_llm_provider, thread_id, client, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
            model="", custom_llm_provider=custom_llm_provider
        )  # type: ignore

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response  # type: ignore
    except Exception as e:
        raise exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={},
            extra_kwargs=kwargs,
        )


def get_thread(
    custom_llm_provider: Literal["openai", "azure"],
    thread_id: str,
    client=None,
    **kwargs,
) -> Thread:
    """Get the thread object, given a thread_id"""
    aget_thread = kwargs.pop("aget_thread", None)
    optional_params = GenericLiteLLMParams(**kwargs)
    litellm_params_dict = get_litellm_params(**kwargs)
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
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    response: Optional[Thread] = None
    if custom_llm_provider == "openai":
        api_base = (
            optional_params.api_base  # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
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

        response = openai_assistants_api.get_thread(
            thread_id=thread_id,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            organization=organization,
            client=client,
            aget_thread=aget_thread,
        )
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        )  # type: ignore

        api_version: Optional[str] = (
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

        if isinstance(client, OpenAI):
            client = None  # only pass client if it's AzureOpenAI

        response = azure_assistants_api.get_thread(
            thread_id=thread_id,
            api_base=api_base,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            api_version=api_version,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            client=client,
            aget_thread=aget_thread,
            litellm_params=litellm_params_dict,
        )
    else:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'get_thread'. Only 'openai' is supported.".format(
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
    return response  # type: ignore


### MESSAGES ###


async def a_add_message(
    custom_llm_provider: Literal["openai", "azure"],
    thread_id: str,
    role: Literal["user", "assistant"],
    content: str,
    attachments: Optional[List[Attachment]] = None,
    metadata: Optional[dict] = None,
    client=None,
    **kwargs,
) -> OpenAIMessage:
    loop = asyncio.get_event_loop()
    ### PASS ARGS TO GET ASSISTANTS ###
    kwargs["a_add_message"] = True
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(
            add_message,
            custom_llm_provider,
            thread_id,
            role,
            content,
            attachments,
            metadata,
            client,
            **kwargs,
        )

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
            model="", custom_llm_provider=custom_llm_provider
        )  # type: ignore

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            # Call the synchronous function using run_in_executor
            response = init_response
        return response  # type: ignore
    except Exception as e:
        raise exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={},
            extra_kwargs=kwargs,
        )


def add_message(
    custom_llm_provider: Literal["openai", "azure"],
    thread_id: str,
    role: Literal["user", "assistant"],
    content: str,
    attachments: Optional[List[Attachment]] = None,
    metadata: Optional[dict] = None,
    client=None,
    **kwargs,
) -> OpenAIMessage:
    ### COMMON OBJECTS ###
    a_add_message = kwargs.pop("a_add_message", None)
    _message_data = MessageData(
        role=role, content=content, attachments=attachments, metadata=metadata
    )
    litellm_params_dict = get_litellm_params(**kwargs)
    optional_params = GenericLiteLLMParams(**kwargs)

    message_data = get_optional_params_add_message(
        role=_message_data["role"],
        content=_message_data["content"],
        attachments=_message_data["attachments"],
        metadata=_message_data["metadata"],
        custom_llm_provider=custom_llm_provider,
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
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    response: Optional[OpenAIMessage] = None
    if custom_llm_provider == "openai":
        api_base = (
            optional_params.api_base  # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
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
        response = openai_assistants_api.add_message(
            thread_id=thread_id,
            message_data=message_data,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            organization=organization,
            client=client,
            a_add_message=a_add_message,
        )
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        )  # type: ignore

        api_version: Optional[str] = (
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

        response = azure_assistants_api.add_message(
            thread_id=thread_id,
            message_data=message_data,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            client=client,
            a_add_message=a_add_message,
            litellm_params=litellm_params_dict,
        )
    else:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'create_thread'. Only 'openai' is supported.".format(
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

    return response  # type: ignore


async def aget_messages(
    custom_llm_provider: Literal["openai", "azure"],
    thread_id: str,
    client: Optional[AsyncOpenAI] = None,
    **kwargs,
) -> AsyncCursorPage[OpenAIMessage]:
    loop = asyncio.get_event_loop()
    ### PASS ARGS TO GET ASSISTANTS ###
    kwargs["aget_messages"] = True
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(
            get_messages,
            custom_llm_provider,
            thread_id,
            client,
            **kwargs,
        )

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
            model="", custom_llm_provider=custom_llm_provider
        )  # type: ignore

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            # Call the synchronous function using run_in_executor
            response = init_response
        return response  # type: ignore
    except Exception as e:
        raise exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={},
            extra_kwargs=kwargs,
        )


def get_messages(
    custom_llm_provider: Literal["openai", "azure"],
    thread_id: str,
    client: Optional[Any] = None,
    **kwargs,
) -> SyncCursorPage[OpenAIMessage]:
    aget_messages = kwargs.pop("aget_messages", None)
    optional_params = GenericLiteLLMParams(**kwargs)
    litellm_params_dict = get_litellm_params(**kwargs)

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

    response: Optional[SyncCursorPage[OpenAIMessage]] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    if custom_llm_provider == "openai":
        api_base = (
            optional_params.api_base  # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
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
        response = openai_assistants_api.get_messages(
            thread_id=thread_id,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            organization=organization,
            client=client,
            aget_messages=aget_messages,
        )
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        )  # type: ignore

        api_version: Optional[str] = (
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

        response = azure_assistants_api.get_messages(
            thread_id=thread_id,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            client=client,
            aget_messages=aget_messages,
            litellm_params=litellm_params_dict,
        )
    else:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'get_messages'. Only 'openai' is supported.".format(
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

    return response  # type: ignore


### RUNS ###
def arun_thread_stream(
    *,
    event_handler: Optional[AssistantEventHandler] = None,
    **kwargs,
) -> AsyncAssistantStreamManager[AsyncAssistantEventHandler]:
    kwargs["arun_thread"] = True
    return run_thread(stream=True, event_handler=event_handler, **kwargs)  # type: ignore


async def arun_thread(
    custom_llm_provider: Literal["openai", "azure"],
    thread_id: str,
    assistant_id: str,
    additional_instructions: Optional[str] = None,
    instructions: Optional[str] = None,
    metadata: Optional[dict] = None,
    model: Optional[str] = None,
    stream: Optional[bool] = None,
    tools: Optional[Iterable[AssistantToolParam]] = None,
    client: Optional[Any] = None,
    **kwargs,
) -> Run:
    loop = asyncio.get_event_loop()
    ### PASS ARGS TO GET ASSISTANTS ###
    kwargs["arun_thread"] = True
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(
            run_thread,
            custom_llm_provider,
            thread_id,
            assistant_id,
            additional_instructions,
            instructions,
            metadata,
            model,
            stream,
            tools,
            client,
            **kwargs,
        )

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(  # type: ignore
            model="", custom_llm_provider=custom_llm_provider
        )  # type: ignore

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            # Call the synchronous function using run_in_executor
            response = init_response
        return response  # type: ignore
    except Exception as e:
        raise exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs={},
            extra_kwargs=kwargs,
        )


def run_thread_stream(
    *,
    event_handler: Optional[AssistantEventHandler] = None,
    **kwargs,
) -> AssistantStreamManager[AssistantEventHandler]:
    return run_thread(stream=True, event_handler=event_handler, **kwargs)  # type: ignore


def run_thread(
    custom_llm_provider: Literal["openai", "azure"],
    thread_id: str,
    assistant_id: str,
    additional_instructions: Optional[str] = None,
    instructions: Optional[str] = None,
    metadata: Optional[dict] = None,
    model: Optional[str] = None,
    stream: Optional[bool] = None,
    tools: Optional[Iterable[AssistantToolParam]] = None,
    client: Optional[Any] = None,
    event_handler: Optional[AssistantEventHandler] = None,  # for stream=True calls
    **kwargs,
) -> Run:
    """Run a given thread + assistant."""
    arun_thread = kwargs.pop("arun_thread", None)
    optional_params = GenericLiteLLMParams(**kwargs)
    litellm_params_dict = get_litellm_params(**kwargs)

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

    response: Optional[Run] = None
    if custom_llm_provider == "openai":
        api_base = (
            optional_params.api_base  # for deepinfra/perplexity/anyscale/groq we check in get_llm_provider and pass in the api base from there
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

        response = openai_assistants_api.run_thread(
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            stream=stream,
            tools=tools,
            api_base=api_base,
            api_key=api_key,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            organization=organization,
            client=client,
            arun_thread=arun_thread,
            event_handler=event_handler,
        )
    elif custom_llm_provider == "azure":
        api_base = (
            optional_params.api_base or litellm.api_base or get_secret("AZURE_API_BASE")
        )  # type: ignore

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
        azure_ad_token = None
        if extra_body is not None:
            azure_ad_token = extra_body.pop("azure_ad_token", None)
        else:
            azure_ad_token = get_secret("AZURE_AD_TOKEN")  # type: ignore

        response = azure_assistants_api.run_thread(
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            stream=stream,
            tools=tools,
            api_base=str(api_base) if api_base is not None else None,
            api_key=str(api_key) if api_key is not None else None,
            api_version=str(api_version) if api_version is not None else None,
            azure_ad_token=str(azure_ad_token) if azure_ad_token is not None else None,
            timeout=timeout,
            max_retries=optional_params.max_retries,
            client=client,
            arun_thread=arun_thread,
            litellm_params=litellm_params_dict,
        )  # type: ignore
    else:
        raise litellm.exceptions.BadRequestError(
            message="LiteLLM doesn't support {} for 'run_thread'. Only 'openai' is supported.".format(
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
    return response  # type: ignore
