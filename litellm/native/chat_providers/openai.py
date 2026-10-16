from typing import Final, cast  # noqa: TID251  # casts moved verbatim from litellm/main.py

import openai

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import with_attribution_headers
from litellm.llms.custom_httpx.http_handler import http2_enabled
from litellm.native.chat_providers.handlers import (
    base_llm_aiohttp_handler,
    base_llm_http_handler,
    openai_chat_completions,
    openai_text_completions,
)
from litellm.native.chat_providers.support import (
    dispatch_client_http,
    dispatch_client_openai,
    dispatch_metadata,
    get_encoding,
    print_verbose,
    provider_config_items,
)
from litellm.responses.bridge_check import resolve_openai_api_base
from litellm.secret_managers.main import get_secret_bool, get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import get_requester_metadata, get_secret


def complete_text_completion_openai(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_openai(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    text_completion: Final = ctx.text_completion
    timeout: Final = ctx.timeout

    openai.api_type = "openai"

    api_base = (
        api_base
        or litellm.api_base
        or get_secret("OPENAI_BASE_URL")
        or get_secret("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )

    openai.api_version = None
    # set API KEY

    api_key = api_key or litellm.api_key or litellm.openai_key or get_secret("OPENAI_API_KEY")

    headers = headers or litellm.headers

    ## LOAD CONFIG - if set
    config: Final = litellm.OpenAITextCompletionConfig.get_config()
    for k, v in provider_config_items(config):
        if (
            k not in optional_params
        ):  # completion(top_k=3) > openai_text_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v
    if litellm.organization:
        openai.organization = litellm.organization

    ## COMPLETION CALL
    _response = openai_text_completions.completion(
        model=model,
        messages=messages,
        headers=headers,
        model_response=model_response,
        print_verbose=print_verbose,
        api_key=api_key,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        acompletion=acompletion,
        client=client,  # pass AsyncOpenAI, OpenAI client
        logging_obj=logging,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        timeout=timeout,
    )

    if optional_params.get("stream", False) is False and acompletion is False and text_completion is False:
        # convert to chat completion response
        _response = litellm.OpenAITextCompletionConfig().convert_to_chat_model_response_object(
            response_object=_response, model_response_object=model_response
        )

    if optional_params.get("stream", False) or acompletion is True:
        ## LOGGING
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=_response,
            additional_args={"headers": headers},
        )
    return _response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract


def complete_aiohttp_openai(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
    if http2_enabled():
        verbose_logger.warning(
            "litellm.http2 is enabled but aiohttp_openai/ always uses aiohttp, which has no HTTP/2 client; this request stays on HTTP/1.1"
        )
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    extra_headers: Final = ctx.extra_headers
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
        api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
        or litellm.api_base
        or get_secret("OPENAI_BASE_URL")
        or get_secret("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    # set API KEY
    api_key = (
        api_key
        or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
        or litellm.openai_key
        or get_secret("OPENAI_API_KEY")
    )

    headers = headers or litellm.headers

    if extra_headers is not None:
        optional_params["extra_headers"] = extra_headers
    return base_llm_aiohttp_handler.completion(
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


def complete_custom_openai(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    extra_headers = ctx.extra_headers
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    metadata: Final = dispatch_metadata(ctx)
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    organization = ctx.organization
    provider_config: Final = ctx.provider_config
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
    api_base = resolve_openai_api_base(api_base)
    organization = (
        organization
        or litellm.organization
        or get_secret("OPENAI_ORGANIZATION")
        or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
    )
    openai.organization = organization
    # set API KEY
    api_key = (
        api_key
        or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
        or litellm.openai_key
        or get_secret("OPENAI_API_KEY")
    )

    headers = headers or litellm.headers
    outbound_headers: Final = (
        headers
        if provider_config is None
        else with_attribution_headers(provider_config.get_attribution_headers(), headers)
    )

    # Add GitHub Copilot headers (same as /responses endpoint does)
    if custom_llm_provider == "github_copilot":
        from litellm.llms.github_copilot.authenticator import Authenticator
        from litellm.llms.github_copilot.common_utils import (
            get_copilot_default_headers,
            pin_session_authorization,
        )
        from litellm.llms.github_copilot.per_user_auth import (
            require_github_copilot_user_session,
        )

        user_session: Final = require_github_copilot_user_session(litellm_params)
        copilot_headers: Final = get_copilot_default_headers(
            user_session.token if user_session is not None else Authenticator().get_api_key()
        )
        if extra_headers:
            copilot_headers.update(
                cast("dict[str, str]", extra_headers)  # cast-ok: extra_headers is a str-valued request dict
            )
        if user_session is not None:
            pin_session_authorization(copilot_headers, user_session.token)
        extra_headers = copilot_headers

    use_base_llm_http_handler: Final = get_secret_bool("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER")

    if extra_headers is not None and not use_base_llm_http_handler:
        optional_params["extra_headers"] = extra_headers

    if litellm.enable_preview_features and metadata is not None:  # [PREVIEW] allow metadata to be passed to OPENAI
        openai_metadata: Final = get_requester_metadata(metadata)
        if openai_metadata is not None:
            optional_params["metadata"] = openai_metadata

    ## LOAD CONFIG - if set
    config: Final = litellm.OpenAIConfig.get_config()
    for k, v in provider_config_items(config):
        if (
            k not in optional_params
        ):  # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    ## COMPLETION CALL
    try:
        if use_base_llm_http_handler:
            response = base_llm_http_handler.completion(
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
                headers=outbound_headers,
                client=client,
                provider_config=provider_config,
            )
        else:
            response = openai_chat_completions.completion(
                model=model,
                messages=messages,
                headers=outbound_headers,
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
                organization=organization,
                custom_llm_provider=custom_llm_provider,
                shared_session=shared_session,
            )
    except Exception as e:
        ## LOGGING - log the original exception returned
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=str(e),
            additional_args={"headers": outbound_headers},
        )
        raise e

    if optional_params.get("stream", False):
        ## LOGGING
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=response,
            additional_args={"headers": outbound_headers},
        )

    return response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract


def complete_text_completion_inception(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_openai(ctx)
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    text_completion: Final = ctx.text_completion
    timeout: Final = ctx.timeout

    passed_api_base: Final = api_base or optional_params.pop("api_base", None) or optional_params.pop("base_url", None)
    api_base = passed_api_base or get_secret_str("INCEPTION_API_BASE") or "https://api.inceptionlabs.ai/v1"
    # FIM is served at `/v1/fim/completions`; the OpenAI client appends
    # `/completions`, so point it at the `/v1/fim` base.
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/fim"):
        api_base += "/fim"

    # Don't forward the server-managed Inception key to a caller-supplied
    # api_base; only resolve it for the default/server base, or when the
    # caller passes their own key.
    if passed_api_base is None or api_key:
        api_key = api_key or litellm.inception_key or get_secret_str("INCEPTION_API_KEY")

    _response = openai_text_completions.completion(
        model=model,
        messages=messages,
        model_response=model_response,
        print_verbose=print_verbose,
        api_key=api_key,
        custom_llm_provider="text-completion-inception",
        api_base=api_base,
        acompletion=acompletion,
        client=client,
        logging_obj=logging,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        timeout=timeout,
    )

    if optional_params.get("stream", False) is False and acompletion is False and text_completion is False:
        _response = litellm.OpenAITextCompletionConfig().convert_to_chat_model_response_object(
            response_object=_response, model_response_object=model_response
        )

    if optional_params.get("stream", False) or acompletion is True:
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=_response,
            additional_args={"headers": headers},
        )
    return _response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract
