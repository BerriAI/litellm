from typing import Final, cast  # noqa: TID251  # casts moved verbatim from litellm/main.py

import openai

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import stringify_json_tool_call_content
from litellm.llms.azure.azure import check_dynamic_azure_params
from litellm.native.chat_providers.handlers import (
    azure_anthropic_chat_completions,
    azure_chat_completions,
    azure_o1_chat_completions,
    azure_text_completions,
    base_llm_http_handler,
)
from litellm.native.chat_providers.support import (
    dispatch_client_azure,
    dispatch_client_http,
    get_encoding,
    print_verbose,
    provider_config_items,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import get_secret


def complete_azure(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    _azure_detection_model: Final = ctx._azure_detection_model
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    api_version = ctx.api_version
    client: Final = dispatch_client_azure(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    extra_headers: Final = ctx.extra_headers
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    max_retries: Final = ctx.max_retries
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    timeout: Final = ctx.timeout

    dynamic_params = False
    if client is not None and (isinstance(client, openai.AzureOpenAI) or isinstance(client, openai.AsyncAzureOpenAI)):
        dynamic_params = check_dynamic_azure_params(
            azure_client_params={"api_version": api_version},
            azure_client=client,
        )

    api_type: Final = get_secret("AZURE_API_TYPE") or "azure"

    api_base = api_base or litellm.api_base or get_secret("AZURE_API_BASE")

    api_version = (
        api_version or litellm.api_version or get_secret_str("AZURE_API_VERSION") or litellm.AZURE_DEFAULT_API_VERSION
    )

    api_key = (
        api_key
        or litellm.api_key
        or litellm.azure_key
        or get_secret_str("AZURE_OPENAI_API_KEY")
        or get_secret_str("AZURE_API_KEY")
    )

    azure_ad_token: Final = optional_params.get("extra_body", {}).pop("azure_ad_token", None) or get_secret_str(
        "AZURE_AD_TOKEN"
    )

    azure_ad_token_provider_value: Final = litellm_params.get("azure_ad_token_provider", None)
    azure_ad_token_provider: Final = azure_ad_token_provider_value if callable(azure_ad_token_provider_value) else None

    headers = headers or litellm.headers

    if extra_headers is not None:
        optional_params["extra_headers"] = extra_headers
    if max_retries is not None:
        optional_params["max_retries"] = max_retries

    if litellm.AzureOpenAIO1Config().is_o_series_model(model=_azure_detection_model):
        ## LOAD CONFIG - if set
        config = litellm.AzureOpenAIO1Config.get_config()
        for k, v in provider_config_items(config):
            if (
                k not in optional_params
            ):  # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        response = azure_o1_chat_completions.completion(
            model=model,
            messages=messages,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            dynamic_params=dynamic_params,
            azure_ad_token=azure_ad_token,
            model_response=model_response,
            print_verbose=print_verbose,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            logging_obj=logging,
            acompletion=acompletion,
            timeout=timeout,
            client=client,  # pass AsyncAzureOpenAI, AzureOpenAI client
            custom_llm_provider=custom_llm_provider,
        )
    else:
        ## LOAD CONFIG - if set
        config = litellm.AzureOpenAIConfig.get_config()
        for k, v in provider_config_items(config):
            if (
                k not in optional_params
            ):  # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        ## COMPLETION CALL
        response = azure_chat_completions.completion(
            model=model,
            messages=messages,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,
            dynamic_params=dynamic_params,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            model_response=model_response,
            print_verbose=print_verbose,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            logging_obj=logging,
            acompletion=acompletion,
            timeout=timeout,
            client=client,  # pass AsyncAzureOpenAI, AzureOpenAI client
        )

    if optional_params.get("stream", False):
        ## LOGGING
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=response,
            additional_args={
                "headers": headers,
                "api_version": api_version,
                "api_base": api_base,
            },
        )

    return response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract


def complete_azure_text(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    api_version = ctx.api_version
    client: Final = dispatch_client_azure(ctx)
    extra_headers: Final = ctx.extra_headers
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    timeout: Final = ctx.timeout

    api_type: Final = get_secret_str("AZURE_API_TYPE") or "azure"

    api_base = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")

    if api_base is None:
        raise ValueError(
            "api_base is required for Azure OpenAI LLM provider. Either set it dynamically or set the AZURE_API_BASE environment variable."
        )

    api_version = api_version or litellm.api_version or get_secret_str("AZURE_API_VERSION")

    api_key = (
        api_key
        or litellm.api_key
        or litellm.azure_key
        or get_secret_str("AZURE_OPENAI_API_KEY")
        or get_secret_str("AZURE_API_KEY")
    )

    azure_ad_token: Final = optional_params.get("extra_body", {}).pop("azure_ad_token", None) or get_secret_str(
        "AZURE_AD_TOKEN"
    )

    azure_ad_token_provider_value: Final = litellm_params.get("azure_ad_token_provider", None)
    azure_ad_token_provider: Final = azure_ad_token_provider_value if callable(azure_ad_token_provider_value) else None

    headers = headers or litellm.headers

    if extra_headers is not None:
        optional_params["extra_headers"] = extra_headers

    ## LOAD CONFIG - if set
    config: Final = litellm.AzureOpenAIConfig.get_config()
    for k, v in provider_config_items(config):
        if (
            k not in optional_params
        ):  # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    ## COMPLETION CALL
    response: Final = azure_text_completions.completion(
        model=model,
        messages=messages,
        headers=headers,
        api_key=api_key,
        api_base=api_base,
        api_version=cast(str, api_version),
        api_type=api_type,
        azure_ad_token=azure_ad_token,
        azure_ad_token_provider=azure_ad_token_provider,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        logging_obj=logging,
        acompletion=acompletion,
        timeout=timeout,
        client=client,  # pass AsyncAzureOpenAI, AzureOpenAI client
    )

    if optional_params.get("stream", False) or acompletion is True:
        ## LOGGING
        logging.post_call(
            input=messages,
            api_key=api_key,
            original_response=response,
            additional_args={
                "headers": headers,
                "api_version": api_version,
                "api_base": api_base,
            },
        )

    return response


def complete_azure_ai(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    extra_headers: Final = ctx.extra_headers
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

    azure_ai_route: Final = AzureFoundryModelInfo.get_azure_ai_route(model)

    # Check if this is an agents route - model format: azure_ai/agents/<agent_id>
    if azure_ai_route == "agents":
        from litellm.llms.azure_ai.agents import AzureAIAgentsConfig

        api_base = AzureFoundryModelInfo.get_api_base(api_base)
        if api_base is None:
            raise ValueError(
                "Azure AI Agents requests require an api_base. Set `api_base` or the AZURE_AI_API_BASE env var."
            )
        api_key = AzureFoundryModelInfo.get_api_key(api_key)

        response = AzureAIAgentsConfig.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            model_response=model_response,
            logging_obj=logging,
            optional_params=optional_params,
            litellm_params=litellm_params,
            timeout=timeout,
            acompletion=acompletion,
            stream=stream,
            headers=headers or litellm.headers,
        )

    # Check if this is a Claude model - route to Azure Anthropic handler
    elif "claude" in model.lower():
        # Use Azure Anthropic handler for Claude models
        api_base = AzureFoundryModelInfo.get_api_base(api_base)
        if api_base is None:
            raise ValueError(
                "Azure Anthropic requests require an api_base. Set `api_base` or the AZURE_AI_API_BASE env var."
            )
        api_key = AzureFoundryModelInfo.get_api_key(api_key)

        # Ensure the URL ends with /v1/messages for Anthropic
        if api_base:
            api_base = api_base.rstrip("/")
            if not api_base.endswith("/v1/messages"):
                if "/anthropic" in api_base:
                    parts: Final = api_base.split("/anthropic", 1)
                    api_base = parts[0] + "/anthropic"
                else:
                    api_base = api_base + "/anthropic"
                api_base = api_base + "/v1/messages"

        response = azure_anthropic_chat_completions.completion(
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
            encoding=get_encoding(),
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
        response = response
    else:
        # Non-Claude models use standard Azure AI flow
        api_base = AzureFoundryModelInfo.get_api_base(api_base)
        # set API KEY
        api_key = AzureFoundryModelInfo.get_api_key(api_key)

        headers = headers or litellm.headers

        if extra_headers is not None:
            optional_params["extra_headers"] = extra_headers

        ## FOR COHERE
        if "command-r" in model:  # make sure tool call in messages are str
            messages = stringify_json_tool_call_content(messages=messages)

        ## COMPLETION CALL
        try:
            response = base_llm_http_handler.completion(
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
                client=client,  # pass AsyncOpenAI, OpenAI client
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

        if optional_params.get("stream", False):
            ## LOGGING
            logging.post_call(
                input=messages,
                api_key=api_key,
                original_response=response,
                additional_args={"headers": headers},
            )

    return response
