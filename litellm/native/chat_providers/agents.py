from collections.abc import Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # casts moved verbatim from litellm/main.py

from pydantic import TypeAdapter

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.native.chat_providers.handlers import base_llm_http_handler
from litellm.native.chat_providers.support import dispatch_client_http, get_encoding
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.types.llms.openai import AllMessageValues


def complete_ragflow(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base: Final = ctx.api_base
    api_key: Final = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    provider_config: Final = ctx.provider_config
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    try:
        response: Final = base_llm_http_handler.completion(
            model=model,
            messages=messages,
            headers=headers,
            model_response=model_response,
            api_key=api_key,
            api_base=api_base,
            acompletion=acompletion,
            logging_obj=logging,
            optional_params=optional_params,
            litellm_params=litellm_params,
            shared_session=shared_session,
            timeout=timeout,
            client=client,
            custom_llm_provider=custom_llm_provider,
            encoding=get_encoding(),
            stream=stream,
            provider_config=provider_config,
        )
    except Exception as e:
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=str(e),
            additional_args={"headers": headers},
        )
        raise e

    return response


def complete_a2a(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    provider_config: Final = ctx.provider_config
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    (
        api_base,
        api_key,
        headers,
    ) = litellm.A2AConfig.resolve_agent_config_from_registry(
        agent_name=model,
        api_base=api_base,
        api_key=api_key,
        headers=headers,
        optional_params=optional_params,
    )

    # Fall back to environment variables and defaults
    api_base = api_base or litellm.api_base or get_secret_str("A2A_API_BASE")

    if api_base is None:
        raise Exception(
            "api_base is required for A2A provider. "
            "Either provide api_base parameter, set A2A_API_BASE environment variable, "
            "or register the agent in the proxy with model='a2a/<agent-name>'."
        )

    headers = headers or litellm.headers

    return base_llm_http_handler.completion(
        model=model,
        stream=stream,
        messages=messages,
        acompletion=acompletion,
        api_base=api_base,
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        shared_session=shared_session,
        custom_llm_provider=custom_llm_provider,
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,
        client=client,
        provider_config=provider_config,
    )


def complete_langgraph(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    from litellm.llms.langgraph.chat.transformation import LangGraphConfig

    (
        api_base,
        api_key,
    ) = LangGraphConfig().get_openai_compatible_provider_info(
        api_base=api_base or litellm.api_base,
        api_key=api_key or litellm.api_key,
    )

    headers = headers or litellm.headers

    return base_llm_http_handler.completion(
        model=model,
        stream=stream,
        messages=messages,
        acompletion=acompletion,
        api_base=api_base,
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        shared_session=shared_session,
        custom_llm_provider=custom_llm_provider,
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,
        client=client,
    )


def complete_microsoft_365_copilot(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    from litellm.llms.microsoft_365_copilot.chat.handler import acompletion, completion
    from litellm.llms.microsoft_365_copilot.common_utils import as_secret_fields

    messages: Final[Sequence[AllMessageValues]] = cast(  # cast-ok: dispatch messages are prevalidated
        Sequence[AllMessageValues],
        ctx.messages,
    )
    optional_params_value: Final[object] = cast(  # cast-ok: dispatch params are validated below
        object,
        ctx.optional_params,
    )
    optional_params: Final = TypeAdapter(Mapping[str, object]).validate_python(optional_params_value)
    kwargs_value: Final[object] = cast(  # cast-ok: dispatch kwargs are validated below
        object,
        ctx.kwargs,
    )
    kwargs: Final = TypeAdapter(Mapping[str, object]).validate_python(kwargs_value)
    secret_fields: Final = as_secret_fields(kwargs.get("secret_fields"))
    client: Final = dispatch_client_http(ctx)
    if ctx.acompletion:
        async_client: Final = client if isinstance(client, AsyncHTTPHandler) else None
        return acompletion(
            model=ctx.model,
            messages=messages,
            api_key=ctx.api_key,
            litellm_params=ctx.litellm_params,
            optional_params=optional_params,
            secret_fields=secret_fields,
            stream=ctx.stream is True,
            timeout=ctx.timeout,
            logging_obj=ctx.logging,
            client=async_client,
            shared_session=ctx.shared_session,
        )
    sync_client: Final = client if isinstance(client, HTTPHandler) else None
    return completion(
        model=ctx.model,
        messages=messages,
        api_key=ctx.api_key,
        litellm_params=ctx.litellm_params,
        optional_params=optional_params,
        secret_fields=secret_fields,
        stream=ctx.stream is True,
        timeout=ctx.timeout,
        logging_obj=ctx.logging,
        client=sync_client,
    )


def complete_langflow(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    (
        api_base,
        api_key,
    ) = LangFlowConfig().get_openai_compatible_provider_info(
        api_base=api_base or litellm.api_base,
        api_key=api_key or litellm.api_key,
    )

    headers = headers or litellm.headers

    return base_llm_http_handler.completion(
        model=model,
        stream=stream,
        messages=messages,
        acompletion=acompletion,
        api_base=api_base,
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        shared_session=shared_session,
        custom_llm_provider=custom_llm_provider,
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,
        client=client,
    )
