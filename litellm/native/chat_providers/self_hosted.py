import os
from typing import Final

import litellm
from litellm.llms.oobabooga.chat import oobabooga
from litellm.llms.petals.completion import handler as petals_handler
from litellm.llms.vllm.completion import handler as vllm_handler
from litellm.native.chat_providers.handlers import base_llm_http_handler, lemonade_transformation
from litellm.native.chat_providers.support import dispatch_client_http, get_encoding, print_verbose
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import CustomStreamWrapper, get_secret


def complete_hosted_vllm(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
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

    api_base = api_base or litellm.api_base or get_secret_str("HOSTED_VLLM_API_BASE")

    response: Final = base_llm_http_handler.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        model_response=model_response,
        encoding=get_encoding(),
        logging_obj=logging,
        optional_params=optional_params,
        timeout=timeout,
        litellm_params=litellm_params,
        shared_session=shared_session,
        acompletion=acompletion,
        stream=stream,
        api_key=api_key,
        headers=headers,
        client=client,
        provider_config=provider_config,
    )
    logging.post_call(input=messages, api_key=api_key, original_response=response)

    return response


def complete_oobabooga(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    api_base: Final = ctx.api_base
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response = ctx.model_response
    optional_params: Final = ctx.optional_params

    model_response = oobabooga.completion(
        model=model,
        messages=messages,
        model_response=model_response,
        api_base=api_base,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        api_key=None,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        logging_obj=logging,
    )
    if "stream" in optional_params and optional_params["stream"] is True:
        # don't try to access stream object,
        return CustomStreamWrapper(
            model_response,
            model,
            custom_llm_provider="oobabooga",
            logging_obj=logging,
        )
    return model_response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract


def complete_vllm(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    custom_prompt_dict = ctx.custom_prompt_dict
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response = ctx.model_response
    optional_params: Final = ctx.optional_params

    custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
    model_response = vllm_handler.completion(
        model=model,
        messages=messages,
        custom_prompt_dict=custom_prompt_dict,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        logging_obj=logging,
    )

    if "stream" in optional_params and optional_params["stream"] is True:  ## [BETA]
        # don't try to access stream object,
        return CustomStreamWrapper(
            model_response,
            model,
            custom_llm_provider="vllm",
            logging_obj=logging,
        )

    ## RESPONSE OBJECT
    return model_response


def complete_ollama(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_base = litellm.api_base or api_base or get_secret("OLLAMA_API_BASE") or "http://localhost:11434"
    if api_key is not None and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"

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
        custom_llm_provider="ollama",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
        client=client,
    )


def complete_ollama_chat(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_base = litellm.api_base or api_base or get_secret("OLLAMA_API_BASE") or "http://localhost:11434"

    api_key = api_key or litellm.ollama_key or os.environ.get("OLLAMA_API_KEY") or litellm.api_key
    if api_key is not None and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"

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
        custom_llm_provider="ollama_chat",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
        client=client,
    )


def complete_triton(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_base = litellm.api_base or api_base
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
    )


def complete_petals(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    api_base = ctx.api_base
    client: Final = dispatch_client_http(ctx)
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response = ctx.model_response
    optional_params: Final = ctx.optional_params
    stream = ctx.stream

    api_base = api_base or litellm.api_base

    stream = optional_params.pop("stream", False)
    model_response = petals_handler.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        logging_obj=logging,
        client=client,
    )
    if stream is True:  ## [BETA]
        # Fake streaming for petals
        resp_string: Final = model_response["choices"][0]["message"]["content"]
        return CustomStreamWrapper(
            resp_string,
            model,
            custom_llm_provider="petals",
            logging_obj=logging,
        )
    return model_response


def complete_lemonade(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base: Final = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_key = api_key or litellm.lemonade_key or get_secret_str("LEMONADE_API_KEY") or litellm.api_key

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
        timeout=timeout,
        client=client,
        custom_llm_provider=custom_llm_provider,
        encoding=get_encoding(),
        stream=stream,
        provider_config=lemonade_transformation,
    )

    return response
