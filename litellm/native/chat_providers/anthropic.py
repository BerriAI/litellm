import os
from typing import Final, cast  # noqa: TID251  # casts moved verbatim from litellm/main.py

import litellm
from litellm._logging import verbose_logger
from litellm.native.chat_providers.handlers import anthropic_chat_completions, base_llm_http_handler
from litellm.native.chat_providers.support import dispatch_client_http, get_encoding, print_verbose
from litellm.secret_managers.main import get_secret_bool
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import get_secret


def complete_anthropic_text(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
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

    api_key = api_key or litellm.anthropic_key or litellm.api_key or os.environ.get("ANTHROPIC_API_KEY")
    custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
    api_base = cast(
        str | None,
        api_base
        or litellm.api_base
        or get_secret("ANTHROPIC_API_BASE")
        or get_secret("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com/v1/complete",
    )

    # Check if we should disable automatic URL suffix appending
    disable_url_suffix: Final = get_secret_bool("LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX")
    if api_base is not None and not disable_url_suffix and not api_base.endswith("/v1/complete"):
        api_base += "/v1/complete"
    elif disable_url_suffix:
        verbose_logger.debug("LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX is set, skipping /v1/complete suffix")

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
        custom_llm_provider="anthropic_text",
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
    )


def complete_anthropic(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    custom_prompt_dict = ctx.custom_prompt_dict
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    timeout: Final = ctx.timeout

    api_key = api_key or litellm.anthropic_key or litellm.api_key or os.environ.get("ANTHROPIC_API_KEY")
    custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict
    # call /messages
    # default route for all anthropic models
    api_base = cast(
        str | None,
        api_base
        or litellm.api_base
        or get_secret("ANTHROPIC_API_BASE")
        or get_secret("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com/v1/messages",
    )

    # Check if we should disable automatic URL suffix appending
    disable_url_suffix: Final = get_secret_bool("LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX")
    if api_base is not None and not disable_url_suffix and not api_base.endswith("/v1/messages"):
        api_base += "/v1/messages"
    elif disable_url_suffix:
        verbose_logger.debug("LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX is set, skipping /v1/messages suffix")

    response: Final = anthropic_chat_completions.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        acompletion=acompletion,
        custom_prompt_dict=litellm.custom_prompt_dict,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),  # for calculating input/output tokens
        api_key=api_key,
        logging_obj=logging,
        headers=headers,
        timeout=timeout,
        client=client,
        custom_llm_provider=custom_llm_provider,
    )
    if optional_params.get("stream", False) or acompletion is True:
        ## LOGGING
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=response,
        )
    return response
