import os
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # casts moved verbatim from litellm/main.py

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.cohere.common_utils import CohereModelInfo
from litellm.llms.deprecated_providers import aleph_alpha
from litellm.llms.nlp_cloud.chat.handler import completion as nlp_cloud_chat_completion
from litellm.llms.replicate.chat.handler import completion as replicate_chat_completion
from litellm.native.chat_providers.handlers import (
    base_llm_http_handler,
    bytez_transformation,
    codestral_text_completions,
    openai_like_chat_completion,
    predibase_chat_completions,
)
from litellm.native.chat_providers.support import (
    dispatch_client_http,
    get_encoding,
    print_verbose,
    provider_config_items,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import CustomStreamWrapper, get_secret


def complete_deepseek(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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
        ## LOGGING - log the original exception returned
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=str(e),
            additional_args={"headers": headers},
        )
        raise e

    return response


def complete_fireworks_ai(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
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
    provider_config: Final = ctx.provider_config
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout
    optional_params: Final = (
        provider_config.map_extra_body_params(optional_params=ctx.optional_params, model=model)
        if isinstance(provider_config, litellm.FireworksAIConfig)
        else ctx.optional_params
    )

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
        ## LOGGING - log the original exception returned
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=str(e),
            additional_args={"headers": headers},
        )
        raise e

    return response


def complete_together_ai(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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
            additional_args=MappingProxyType({"headers": headers}),
        )
        raise

    return response


def complete_scaledown(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    try:
        return base_llm_http_handler.completion(
            model=ctx.model,
            messages=ctx.messages,
            headers=ctx.headers,
            model_response=ctx.model_response,
            api_key=ctx.api_key,
            api_base=ctx.api_base,
            acompletion=ctx.acompletion,
            logging_obj=ctx.logging,
            optional_params=ctx.optional_params,
            litellm_params=ctx.litellm_params,
            shared_session=ctx.shared_session,
            timeout=cast(  # cast-ok: CompletionTimeout.resolve normalizes this value
                float | httpx.Timeout, ctx.timeout
            ),
            client=dispatch_client_http(ctx),
            custom_llm_provider=ctx.custom_llm_provider,
            encoding=get_encoding(),
            stream=ctx.stream,
            provider_config=ctx.provider_config,
        )
    except Exception as exc:
        ctx.logging.post_call(
            input=ctx.messages,
            api_key=ctx.api_key,
            original_response=str(exc),
            additional_args={"headers": ctx.headers},
        )
        raise


def complete_xai(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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
        ## LOGGING - log the original exception returned
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=str(e),
            additional_args={"headers": headers},
        )
        raise e

    return response


def complete_groq(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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

    api_base = (
        api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
        or litellm.api_base
        or get_secret("GROQ_API_BASE")
        or "https://api.groq.com/openai/v1"
    )

    # set API KEY
    api_key = (
        api_key
        or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
        or litellm.groq_key
        or get_secret("GROQ_API_KEY")
    )

    headers = headers or litellm.headers

    ## LOAD CONFIG - if set
    config: Final = litellm.GroqChatConfig.get_config()
    for k, v in provider_config_items(config):
        if (
            k not in optional_params
        ):  # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

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
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
        client=client,
    )


def complete_minimax(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
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
    provider_config: Final = ctx.provider_config
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_key = api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key

    api_base = api_base or litellm.api_base or get_secret_str("MINIMAX_API_BASE") or "https://api.minimax.io/v1"

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


def complete_mistral(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
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
    provider_config: Final = ctx.provider_config
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_key = api_key or litellm.api_key or get_secret("MISTRAL_API_KEY")
    api_base = api_base or litellm.api_base or get_secret("MISTRAL_API_BASE") or "https://api.mistral.ai/v1"

    return base_llm_http_handler.completion(
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


def complete_replicate(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
    custom_prompt_dict = ctx.custom_prompt_dict
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response = ctx.model_response
    optional_params: Final = ctx.optional_params

    replicate_key: Final = (
        api_key
        or litellm.replicate_key
        or litellm.api_key
        or get_secret("REPLICATE_API_KEY")
        or get_secret("REPLICATE_API_TOKEN")
    )

    api_base = api_base or litellm.api_base or get_secret("REPLICATE_API_BASE") or "https://api.replicate.com/v1"

    custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict

    model_response = replicate_chat_completion(
        model=model,
        messages=messages,
        api_base=api_base,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),  # for calculating input/output tokens
        api_key=replicate_key,
        logging_obj=logging,
        custom_prompt_dict=custom_prompt_dict,
        acompletion=acompletion,
        headers=headers,
    )

    if optional_params.get("stream", False) is True:
        ## LOGGING
        logging.post_call(
            input=messages,
            api_key=replicate_key,
            original_response=model_response,
        )

    return model_response


def complete_nlp_cloud(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params

    nlp_cloud_key: Final = api_key or litellm.nlp_cloud_key or get_secret("NLP_CLOUD_API_KEY") or litellm.api_key

    api_base = api_base or litellm.api_base or get_secret("NLP_CLOUD_API_BASE") or "https://api.nlpcloud.io/v1/gpu/"

    response = nlp_cloud_chat_completion(
        model=model,
        messages=messages,
        api_base=api_base,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        api_key=nlp_cloud_key,
        logging_obj=logging,
    )

    if "stream" in optional_params and optional_params["stream"] is True:
        # don't try to access stream object,
        response = CustomStreamWrapper(
            response,
            model,
            custom_llm_provider="nlp_cloud",
            logging_obj=logging,
        )

    if optional_params.get("stream", False) or acompletion is True:
        ## LOGGING
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=response,
        )

    return response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract


def complete_aleph_alpha(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response = ctx.model_response
    optional_params: Final = ctx.optional_params

    aleph_alpha_key: Final = (
        api_key
        or litellm.aleph_alpha_key
        or get_secret("ALEPH_ALPHA_API_KEY")
        or get_secret("ALEPHALPHA_API_KEY")
        or litellm.api_key
    )

    api_base = (
        api_base or litellm.api_base or get_secret("ALEPH_ALPHA_API_BASE") or "https://api.aleph-alpha.com/complete"
    )

    model_response = aleph_alpha.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        default_max_tokens_to_sample=litellm.max_tokens,
        api_key=aleph_alpha_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
    )

    if "stream" in optional_params and optional_params["stream"] is True:
        # don't try to access stream object,
        return CustomStreamWrapper(
            model_response,
            model,
            custom_llm_provider="aleph_alpha",
            logging_obj=logging,
        )
    return model_response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract


def complete_cohere_chat(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
    extra_headers: Final = ctx.extra_headers
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    provider_config: Final = ctx.provider_config
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    cohere_key: Final = (
        api_key
        or litellm.cohere_key
        or get_secret_str("COHERE_API_KEY")
        or get_secret_str("CO_API_KEY")
        or litellm.api_key
    )

    cohere_route: Final = CohereModelInfo.get_cohere_route(model)
    verbose_logger.debug("Cohere route: %s", cohere_route)
    # Set API base based on route
    if cohere_route == "v2":
        api_base = api_base or litellm.api_base or get_secret_str("COHERE_API_BASE") or "https://api.cohere.com/v2/chat"
        # Remove v2/ prefix from model name for the actual API call
        if "v2/" in model:
            model = model.replace("v2/", "")
    else:
        api_base = api_base or litellm.api_base or get_secret_str("COHERE_API_BASE") or "https://api.cohere.ai/v1/chat"

    headers = headers or litellm.headers or {}
    if headers is None:
        headers = {}

    if extra_headers is not None:
        headers.update(extra_headers)

    verbose_logger.debug("Model: %s, API Base: %s", model, api_base)
    verbose_logger.debug("Provider Config: %s", provider_config)
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
        custom_llm_provider="cohere_chat",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=cohere_key,
        provider_config=provider_config,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
    )


def complete_maritalk(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params

    maritalk_key: Final = api_key or litellm.maritalk_key or get_secret("MARITALK_API_KEY") or litellm.api_key

    api_base = api_base or litellm.api_base or get_secret("MARITALK_API_BASE") or "https://chat.maritaca.ai/api"

    return openai_like_chat_completion.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        api_key=maritalk_key,
        logging_obj=logging,
        custom_llm_provider="maritalk",
        custom_prompt_dict=custom_prompt_dict,
    )


def complete_huggingface(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    huggingface_key: Final = (
        api_key
        or litellm.huggingface_key
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_API_KEY")
        or litellm.api_key
    )
    hf_headers: Final = headers or litellm.headers
    return base_llm_http_handler.completion(
        model=model,
        messages=messages,
        headers=hf_headers,
        model_response=model_response,
        api_key=huggingface_key,
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
    )


def complete_compactifai(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
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
    provider_config: Final = ctx.provider_config
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_key = api_key or get_secret_str("COMPACTIFAI_API_KEY") or litellm.api_key

    api_base = api_base or "https://api.compactif.ai/v1"

    ## COMPLETION CALL
    return base_llm_http_handler.completion(
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
        provider_config=provider_config,
    )


def complete_predibase(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    timeout: Final = ctx.timeout

    tenant_id: Final = (
        optional_params.pop("tenant_id", None)
        or optional_params.pop("predibase_tenant_id", None)
        or litellm.predibase_tenant_id
        or get_secret("PREDIBASE_TENANT_ID")
    )

    if tenant_id is None:
        raise ValueError(
            "Missing Predibase Tenant ID - Required for making the request. Set dynamically (e.g. `completion(..tenant_id=<MY-ID>)`) or in env - `PREDIBASE_TENANT_ID`."
        )

    api_base = (
        api_base
        or optional_params.pop("api_base", None)
        or optional_params.pop("base_url", None)
        or litellm.api_base
        or get_secret("PREDIBASE_API_BASE")
    )

    api_key = api_key or litellm.api_key or litellm.predibase_key or get_secret("PREDIBASE_API_KEY")

    _model_response: Final = predibase_chat_completions.completion(
        model=model,
        messages=messages,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        logging_obj=logging,
        acompletion=acompletion,
        api_base=api_base,
        custom_prompt_dict=custom_prompt_dict,
        api_key=api_key,
        tenant_id=tenant_id,
        timeout=timeout,
    )

    if "stream" in optional_params and optional_params["stream"] is True and acompletion is False:
        return _model_response
    return _model_response


def complete_text_completion_codestral(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    optional_params: Final = ctx.optional_params
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_base = (
        api_base
        or optional_params.pop("api_base", None)
        or optional_params.pop("base_url", None)
        or litellm.api_base
        or "https://codestral.mistral.ai/v1/fim/completions"
    )

    api_key = api_key or litellm.api_key or get_secret("CODESTRAL_API_KEY")

    text_completion_model_response: Final = litellm.TextCompletionResponse(stream=stream)

    _model_response: Final = codestral_text_completions.completion(
        model=model,
        messages=messages,
        model_response=text_completion_model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        logging_obj=logging,
        acompletion=acompletion,
        api_base=api_base,
        custom_prompt_dict=custom_prompt_dict,
        api_key=api_key,
        timeout=timeout,
    )

    if "stream" in optional_params and optional_params["stream"] is True and acompletion is False:
        return _model_response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract
    return _model_response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract


def complete_bytez(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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

    api_key = api_key or litellm.bytez_key or get_secret_str("BYTEZ_API_KEY") or litellm.api_key

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
        provider_config=bytez_transformation,
    )

    return response
