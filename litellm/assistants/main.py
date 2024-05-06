# What is this?
## Main file for assistants API logic
from typing import Iterable
import os
import litellm
from openai import OpenAI
from litellm import client
from litellm.utils import supports_httpx_timeout
from ..llms.openai import OpenAIAssistantsAPI
from ..types.llms.openai import *
from ..types.router import *

####### ENVIRONMENT VARIABLES ###################
openai_assistants_api = OpenAIAssistantsAPI()

### ASSISTANTS ###


def get_assistants(
    custom_llm_provider: Literal["openai"],
    client: Optional[OpenAI] = None,
    **kwargs,
) -> SyncCursorPage[Assistant]:
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


def get_thread(
    custom_llm_provider: Literal["openai"],
    thread_id: str,
    client: Optional[OpenAI] = None,
    **kwargs,
) -> Thread:
    """Get the thread object, given a thread_id"""
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


def get_messages(
    custom_llm_provider: Literal["openai"],
    thread_id: str,
    client: Optional[OpenAI] = None,
    **kwargs,
) -> SyncCursorPage[OpenAIMessage]:
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
