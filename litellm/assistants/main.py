# What is this?
## Main file for assistants API logic
from typing import Iterable
from functools import partial
import os, asyncio, contextvars
import litellm
from openai import OpenAI, AsyncOpenAI
from litellm import client
from litellm.utils import supports_httpx_timeout, exception_type, get_llm_provider
from ..llms.openai import OpenAIAssistantsAPI
from ..types.llms.openai import *
from ..types.router import *

####### ENVIRONMENT VARIABLES ###################
openai_assistants_api = OpenAIAssistantsAPI()

### ASSISTANTS ###


async def aget_assistants(
    custom_llm_provider: Literal["openai"],
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
    custom_llm_provider: Literal["openai"],
    client: Optional[OpenAI] = None,
    **kwargs,
) -> SyncCursorPage[Assistant]:
    aget_assistants = kwargs.pop("aget_assistants", None)
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
            aget_assistants=aget_assistants,
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
    return response


### THREADS ###


async def acreate_thread(custom_llm_provider: Literal["openai"], **kwargs) -> Thread:
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
    custom_llm_provider: Literal["openai"],
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
    return response


async def aget_thread(
    custom_llm_provider: Literal["openai"],
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
    custom_llm_provider: Literal["openai"],
    thread_id: str,
    client: Optional[OpenAI] = None,
    **kwargs,
) -> Thread:
    """Get the thread object, given a thread_id"""
    aget_thread = kwargs.pop("aget_thread", None)
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
    return response


### MESSAGES ###


async def a_add_message(
    custom_llm_provider: Literal["openai"],
    thread_id: str,
    role: Literal["user", "assistant"],
    content: str,
    attachments: Optional[List[Attachment]] = None,
    metadata: Optional[dict] = None,
    client: Optional[AsyncOpenAI] = None,
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
    custom_llm_provider: Literal["openai"],
    thread_id: str,
    role: Literal["user", "assistant"],
    content: str,
    attachments: Optional[List[Attachment]] = None,
    metadata: Optional[dict] = None,
    client: Optional[OpenAI] = None,
    **kwargs,
) -> OpenAIMessage:
    ### COMMON OBJECTS ###
    a_add_message = kwargs.pop("a_add_message", None)
    message_data = MessageData(
        role=role, content=content, attachments=attachments, metadata=metadata
    )
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

    return response


async def aget_messages(
    custom_llm_provider: Literal["openai"],
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
    custom_llm_provider: Literal["openai"],
    thread_id: str,
    client: Optional[OpenAI] = None,
    **kwargs,
) -> SyncCursorPage[OpenAIMessage]:
    aget_messages = kwargs.pop("aget_messages", None)
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

    response: Optional[SyncCursorPage[OpenAIMessage]] = None
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

    return response


### RUNS ###
async def arun_thread(
    custom_llm_provider: Literal["openai"],
    thread_id: str,
    assistant_id: str,
    additional_instructions: Optional[str] = None,
    instructions: Optional[str] = None,
    metadata: Optional[dict] = None,
    model: Optional[str] = None,
    stream: Optional[bool] = None,
    tools: Optional[Iterable[AssistantToolParam]] = None,
    client: Optional[AsyncOpenAI] = None,
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


def run_thread(
    custom_llm_provider: Literal["openai"],
    thread_id: str,
    assistant_id: str,
    additional_instructions: Optional[str] = None,
    instructions: Optional[str] = None,
    metadata: Optional[dict] = None,
    model: Optional[str] = None,
    stream: Optional[bool] = None,
    tools: Optional[Iterable[AssistantToolParam]] = None,
    client: Optional[OpenAI] = None,
    **kwargs,
) -> Run:
    """Run a given thread + assistant."""
    arun_thread = kwargs.pop("arun_thread", None)
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
        )
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
    return response
