import os
from typing import Final

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.native.chat_providers.handlers import (
    base_llm_http_handler,
    ovhcloud_transformation,
    sap_gen_ai_hub_chat_completions,
    watsonx_chat_completion,
)
from litellm.native.chat_providers.support import (
    dispatch_client_http,
    get_encoding,
    print_verbose,
    provider_config_items,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import get_secret


def complete_heroku(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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


def complete_gigachat(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base: Final = ctx.api_base
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

    api_key = (
        api_key
        or litellm.api_key
        or litellm.gigachat_key
        or get_secret("GIGACHAT_API_KEY")
        or get_secret("GIGACHAT_CREDENTIALS")
    )

    headers = headers or litellm.headers or {}

    ## COMPLETION CALL
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


def complete_sap(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base: Final = ctx.api_base
    api_key: Final = ctx.api_key
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

    headers = headers or litellm.headers
    ## LOAD CONFIG - if set
    config: Final = litellm.GenAIHubOrchestrationConfig.get_config()
    for k, v in provider_config_items(config):
        if (
            k not in optional_params
        ):  # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    return sap_gen_ai_hub_chat_completions.completion(
        model=model,
        messages=messages,
        headers=headers,
        model_response=model_response,
        acompletion=acompletion,
        logging_obj=logging,
        optional_params=optional_params,
        litellm_params=litellm_params,
        timeout=timeout,
        shared_session=shared_session,
        client=client,
        custom_llm_provider=custom_llm_provider,
        encoding=get_encoding(),
        api_key=api_key,
        api_base=api_base,
        stream=stream,
    )


def complete_oci(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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
    )


def complete_databricks(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_base = (
        api_base  # for databricks we check in get_llm_provider and pass in the api base from there
        or litellm.api_base
        or os.getenv("DATABRICKS_API_BASE")
    )

    # set API KEY
    api_key = (
        api_key
        or litellm.api_key  # for databricks we check in get_llm_provider and pass in the api key from there
        or litellm.databricks_key
        or get_secret("DATABRICKS_API_KEY")
    )

    headers = headers or litellm.headers

    ## COMPLETION CALL
    try:
        response: Final = base_llm_http_handler.completion(
            model=model,
            stream=stream,
            messages=messages,
            acompletion=acompletion,
            api_base=api_base,
            model_response=model_response,
            optional_params=optional_params,
            litellm_params=litellm_params,
            custom_llm_provider="databricks",
            timeout=timeout,
            headers=headers,
            encoding=get_encoding(),
            api_key=api_key,
            logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            client=client,
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

    if optional_params.get("stream", False):
        ## LOGGING
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=response,
            additional_args={"headers": headers},
        )

    return response


def complete_datarobot(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

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


def complete_watsonx(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base: Final = ctx.api_base
    api_key: Final = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    timeout: Final = ctx.timeout

    return watsonx_chat_completion.completion(
        model=model,
        messages=messages,
        headers=headers,
        model_response=model_response,
        print_verbose=print_verbose,
        api_key=api_key,
        api_base=api_base,
        acompletion=acompletion,
        logging_obj=logging,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        timeout=timeout,
        custom_prompt_dict=custom_prompt_dict,
        client=client,  # pass AsyncOpenAI, OpenAI client
        encoding=get_encoding(),
        custom_llm_provider="watsonx",
    )


def complete_watsonx_text(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
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

    api_key = (
        api_key
        or optional_params.pop("apikey", None)
        or get_secret_str("WATSONX_APIKEY")
        or get_secret_str("WATSONX_API_KEY")
        or get_secret_str("WX_API_KEY")
    )

    api_base = (
        api_base
        or optional_params.pop(
            "url",
            optional_params.pop("api_base", optional_params.pop("base_url", None)),
        )
        or get_secret_str("WATSONX_API_BASE")
        or get_secret_str("WATSONX_URL")
        or get_secret_str("WX_URL")
        or get_secret_str("WML_URL")
    )

    wx_credentials: Final = optional_params.pop(
        "wx_credentials",
        optional_params.pop("watsonx_credentials", None),  # follow {provider}_credentials, same as vertex ai
    )

    token: str | None = None
    if wx_credentials is not None:
        api_base = wx_credentials.get("url", api_base)
        api_key = wx_credentials.get("apikey", wx_credentials.get("api_key", api_key))
        token = wx_credentials.get(
            "token",
            wx_credentials.get(
                "watsonx_token", None
            ),  # follow format of {provider}_token, same as azure - e.g. 'azure_ad_token=..'
        )

    if token is not None:
        optional_params["token"] = token

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
        custom_llm_provider="watsonx_text",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
        client=client,
    )


def complete_cloudflare(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    custom_prompt_dict = ctx.custom_prompt_dict
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

    api_key = api_key or litellm.cloudflare_api_key or litellm.api_key or get_secret("CLOUDFLARE_API_KEY")
    api_base = api_base or litellm.api_base or get_secret("CLOUDFLARE_API_BASE")

    custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
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
        custom_llm_provider="cloudflare",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
    )


def complete_snowflake(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base: Final = ctx.api_base
    api_key: Final = ctx.api_key
    injected_client: Final = dispatch_client_http(ctx)
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
    client: Final = (
        injected_client if injected_client is not None else (HTTPHandler(timeout=timeout) if stream is False else None)
    )  # Keep this here, otherwise, the httpx.client closes and streaming is impossible

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


def complete_gradient_ai(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
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
        custom_llm_provider="gradient_ai",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,
    )


def complete_ovhcloud(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_key = api_key or litellm.ovhcloud_key or get_secret_str("OVHCLOUD_API_KEY") or litellm.api_key

    api_base = (
        api_base
        or litellm.api_base
        or get_secret_str("OVHCLOUD_API_BASE")
        or "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    )

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
        provider_config=ovhcloud_transformation,
    )

    return response
