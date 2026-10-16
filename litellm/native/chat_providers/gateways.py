from typing import Final

import openai

import litellm
from litellm.constants import NADIR_DEFAULT_API_BASE
from litellm.native.chat_providers.handlers import base_llm_http_handler
from litellm.native.chat_providers.support import dispatch_client_http, get_encoding, provider_config_items
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import get_secret


def complete_cometapi(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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

    api_key = api_key or litellm.cometapi_key or get_secret_str("COMETAPI_KEY") or litellm.api_key

    api_base = api_base or litellm.api_base or get_secret_str("COMETAPI_API_BASE") or "https://api.cometapi.com/v1"

    ## COMPLETION CALL
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

    ## LOGGING
    logging.post_call(input=messages, api_key=api_key, original_response=response)

    return response


def complete_openrouter(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
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
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_base = api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"

    api_key = (
        api_key
        or litellm.api_key
        or litellm.openrouter_key
        or get_secret_str("OPENROUTER_API_KEY")
        or get_secret_str("OR_API_KEY")
    )

    openrouter_site_url: Final = get_secret("OR_SITE_URL") or "https://litellm.ai"
    openrouter_app_name: Final = get_secret("OR_APP_NAME") or "liteLLM"

    openrouter_headers: Final = {
        "HTTP-Referer": openrouter_site_url,
        "X-Title": openrouter_app_name,
    }

    _headers: Final = headers or litellm.headers
    if _headers:
        openrouter_headers.update(_headers)

    headers = openrouter_headers

    ## Load Config
    config: Final = litellm.OpenrouterConfig.get_config()
    for k, v in provider_config_items(config):
        if k == "extra_body":
            # we use openai 'extra_body' to pass openrouter specific params - transforms, route, models
            if "extra_body" in optional_params:
                optional_params[k].update(v)
            else:
                optional_params[k] = v
        elif k not in optional_params:
            optional_params[k] = v

    ## COMPLETION CALL
    response: Final = base_llm_http_handler.completion(
        model=model,
        stream=stream,
        messages=messages,
        acompletion=acompletion,
        api_base=api_base,
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        shared_session=shared_session,
        custom_llm_provider="openrouter",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
        client=client,
    )
    ## LOGGING
    logging.post_call(input=messages, api_key=openai.api_key, original_response=response)

    return response


def complete_nadir(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    api_base: Final = ctx.api_base or litellm.api_base or get_secret_str("NADIR_API_BASE") or NADIR_DEFAULT_API_BASE
    api_key: Final = ctx.api_key

    response: Final = base_llm_http_handler.completion(
        model=ctx.model,
        stream=ctx.stream,
        messages=ctx.messages,
        acompletion=ctx.acompletion,
        api_base=api_base,
        model_response=ctx.model_response,
        optional_params=ctx.optional_params,
        litellm_params=ctx.litellm_params,
        shared_session=ctx.shared_session,
        custom_llm_provider="nadir",
        timeout=ctx.timeout,
        headers=ctx.headers or litellm.headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=ctx.logging,
        client=ctx.client,
    )
    ctx.logging.post_call(input=ctx.messages, api_key=api_key, original_response=response)

    return response


def complete_vercel_ai_gateway(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
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
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_base = (
        api_base
        or litellm.api_base
        or get_secret_str("VERCEL_AI_GATEWAY_API_BASE")
        or "https://ai-gateway.vercel.sh/v1"
    )

    api_key = api_key or litellm.api_key or get_secret("VERCEL_AI_GATEWAY_API_KEY")

    vercel_site_url: Final = get_secret("VERCEL_SITE_URL") or "https://litellm.ai"
    vercel_app_name: Final = get_secret("VERCEL_APP_NAME") or "liteLLM"

    vercel_headers: Final = {
        "http-referer": vercel_site_url,
        "x-title": vercel_app_name,
    }

    _headers: Final = headers or litellm.headers
    if _headers:
        vercel_headers.update(_headers)

    headers = vercel_headers

    ## Load Config
    config: Final = litellm.VercelAIGatewayConfig.get_config()
    for k, v in provider_config_items(config):
        if k == "extra_body":
            # we use openai 'extra_body' to pass vercel specific params - providerOptions
            if "extra_body" in optional_params:
                optional_params[k].update(v)
            else:
                optional_params[k] = v
        elif k not in optional_params:
            optional_params[k] = v

    ## COMPLETION CALL
    response: Final = base_llm_http_handler.completion(
        model=model,
        stream=stream,
        messages=messages,
        acompletion=acompletion,
        api_base=api_base,
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        shared_session=shared_session,
        custom_llm_provider="vercel_ai_gateway",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
        client=client,
    )
    ## LOGGING
    logging.post_call(input=messages, api_key=openai.api_key, original_response=response)

    return response


def complete_edenai(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    api_base: Final = litellm.EdenAIChatConfig.get_api_base(ctx.api_base)
    api_key: Final = litellm.EdenAIChatConfig.get_api_key(ctx.api_key or litellm.api_key)
    response: Final = base_llm_http_handler.completion(
        model=ctx.model,
        messages=ctx.messages,
        api_base=api_base,
        custom_llm_provider="edenai",
        model_response=ctx.model_response,
        encoding=get_encoding(),
        logging_obj=ctx.logging,
        optional_params=ctx.optional_params,
        timeout=ctx.timeout,
        litellm_params=ctx.litellm_params,
        shared_session=ctx.shared_session,
        acompletion=ctx.acompletion,
        stream=ctx.stream,
        api_key=api_key,
        headers=ctx.headers or litellm.headers,
        client=dispatch_client_http(ctx),
        provider_config=ctx.provider_config,
    )
    ctx.logging.post_call(input=ctx.messages, api_key=api_key, original_response=response)
    return response


def complete_fal_ai(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    if ctx.stream:
        raise litellm.FalAIError(
            status_code=400,
            message="fal_ai chat completions do not support streaming",
        )
    api_base: Final = litellm.FalAIChatConfig.get_api_base(ctx.api_base)
    api_key: Final = litellm.FalAIChatConfig.get_api_key(ctx.api_key or litellm.api_key)
    response: Final = base_llm_http_handler.completion(
        model=ctx.model,
        messages=ctx.messages,
        api_base=api_base,
        custom_llm_provider="fal_ai",
        model_response=ctx.model_response,
        encoding=get_encoding(),
        logging_obj=ctx.logging,
        optional_params=ctx.optional_params,
        timeout=ctx.timeout,
        litellm_params=ctx.litellm_params,
        shared_session=ctx.shared_session,
        acompletion=ctx.acompletion,
        stream=ctx.stream,
        api_key=api_key,
        headers=ctx.headers or litellm.headers,
        client=dispatch_client_http(ctx),
        provider_config=ctx.provider_config,
    )
    ctx.logging.post_call(input=ctx.messages, api_key=api_key, original_response=response)
    return response
