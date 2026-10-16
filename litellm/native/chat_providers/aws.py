from collections.abc import Mapping
from functools import partial
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.bedrock.chat.native_output_fallback import NativeOutputAttemptResult, call_with_native_output_fallback
from litellm.llms.bedrock.common_utils import (
    BedrockModelInfo,
    NativeOutputFallback,
    bedrock_route_for_request,
    deployment_model_info,
    without_bedrock_native_output_opt_in,
    without_bedrock_route_prefix,
)
from litellm.llms.bedrock_mantle.chat.claude_transformation import bedrock_mantle_chat_config
from litellm.native.chat_providers.handlers import (
    base_llm_http_handler,
    bedrock_converse_chat_completion,
    openai_like_chat_completion,
    sagemaker_llm,
)
from litellm.native.chat_providers.support import (
    dispatch_client_http,
    get_encoding,
    print_verbose,
    provider_config_items,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_secret

_ADDITIONAL_DROP_PARAMS_ADAPTER: Final = TypeAdapter(list[str])
_OPTIONAL_PARAMS_ADAPTER: Final = TypeAdapter(dict[str, object])


def complete_bedrock_mantle(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
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

    api_base = api_base or litellm.api_base or get_secret("BEDROCK_MANTLE_API_BASE")
    api_key = api_key or litellm.api_key or get_secret("BEDROCK_MANTLE_API_KEY")
    headers = headers or litellm.headers
    config: Final = bedrock_mantle_chat_config(model).get_config()
    for k, v in provider_config_items(config):
        if k not in optional_params:
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
        logging_obj=logging,
        client=client,
    )


def complete_amazon_nova(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    api_base = ctx.api_base
    api_key = ctx.api_key
    custom_llm_provider: Final = ctx.custom_llm_provider
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    timeout: Final = ctx.timeout

    api_key = api_key or litellm.amazon_nova_api_key or get_secret_str("AMAZON_NOVA_API_KEY") or litellm.api_key
    api_base = (
        api_base or litellm.api_base or get_secret_str("AMAZON_NOVA_API_BASE") or "https://api.nova.amazon.com/v1"
    )
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
        api_key=api_key,
        logging_obj=logging,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        custom_prompt_dict=custom_prompt_dict,
    )


def complete_sagemaker_chat(
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
    optional_params: Final = ctx.optional_params
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    return base_llm_http_handler.completion(
        model=model,
        stream=stream,
        messages=messages,
        acompletion=acompletion,
        api_base=api_base,
        model_response=model_response,
        optional_params=optional_params,
        litellm_params=litellm_params,
        custom_llm_provider=custom_llm_provider,
        timeout=timeout,
        headers=headers,
        encoding=get_encoding(),
        api_key=api_key,
        logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
        client=client,
    )


def complete_sagemaker(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    hf_model_name: Final = ctx.hf_model_name
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params

    return sagemaker_llm.completion(
        model=model,
        messages=messages,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=optional_params,
        litellm_params=litellm_params,
        custom_prompt_dict=custom_prompt_dict,
        hf_model_name=hf_model_name,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        logging_obj=logging,
        acompletion=acompletion,
    )


_AWS_BEDROCK_CLIENT_CREDENTIAL_KEYS: Final = (
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
)


def _native_output_fallback(
    ctx: CompletionDispatchContext,
    first_attempt_params: Mapping[str, object],
    first_attempt_litellm_params: Mapping[str, object],
) -> NativeOutputFallback:
    litellm_params: Final = without_bedrock_native_output_opt_in(first_attempt_litellm_params)
    remapped_params: Final = ctx.optional_params_for_model_info(deployment_model_info(litellm_params))
    return NativeOutputFallback(
        optional_params={
            **{key: value for key, value in remapped_params.items() if key != "aws_bedrock_client"},
            **{
                key: first_attempt_params[key]
                for key in _AWS_BEDROCK_CLIENT_CREDENTIAL_KEYS
                if key in first_attempt_params
            },
        },
        litellm_params=litellm_params,
    )


def _converse_native_output_fallback(
    ctx: CompletionDispatchContext,
    first_attempt_params: Mapping[str, object],
    first_attempt_litellm_params: Mapping[str, object],
) -> NativeOutputFallback | None:
    fallback: Final = _native_output_fallback(ctx, first_attempt_params, first_attempt_litellm_params)
    return None if AmazonConverseConfig.sends_native_text_format(fallback.optional_params) else fallback


def _native_output_attempt_timeout(
    ctx: CompletionDispatchContext, fallback: NativeOutputFallback | None
) -> float | str | httpx.Timeout | None:
    if fallback is None:
        return ctx.timeout
    retry_timeout: Final = ctx.timeout_for_optional_params(fallback.optional_params)
    ctx.logging.record_applied_timeout(retry_timeout)
    return retry_timeout


def complete_bedrock(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base: Final = ctx.api_base
    api_key: Final = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_prompt_dict = ctx.custom_prompt_dict
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    provider_config = ctx.provider_config
    shared_session: Final = ctx.shared_session
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    custom_prompt_dict = custom_prompt_dict or litellm.custom_prompt_dict

    if "aws_bedrock_client" in optional_params:
        verbose_logger.warning(
            "'aws_bedrock_client' is a deprecated param. Please move to another auth method - https://docs.litellm.ai/docs/providers/bedrock#boto3---authentication."
        )
        # Extract credentials for legacy boto3 client and pass thru to httpx
        aws_bedrock_client: Final = optional_params.pop("aws_bedrock_client")
        creds: Final = aws_bedrock_client._get_credentials().get_frozen_credentials()

        if creds.access_key:
            optional_params["aws_access_key_id"] = creds.access_key
        if creds.secret_key:
            optional_params["aws_secret_access_key"] = creds.secret_key
        if creds.token:
            optional_params["aws_session_token"] = creds.token
        if "aws_region_name" not in optional_params or optional_params["aws_region_name"] is None:
            optional_params["aws_region_name"] = aws_bedrock_client.meta.region_name

    additional_drop_params: Final = (
        _ADDITIONAL_DROP_PARAMS_ADAPTER.validate_python(ctx.kwargs["additional_drop_params"])
        if ctx.kwargs.get("additional_drop_params") is not None
        else None
    )
    bedrock_route: Final = bedrock_route_for_request(model, ctx.request_params, additional_drop_params)
    if bedrock_route == "claude_platform":
        provider_config = ProviderConfigManager.get_provider_chat_config(
            model=model,
            provider=LlmProviders.BEDROCK,
        )
        model = BedrockModelInfo.get_claude_platform_model(model)
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
            custom_llm_provider="bedrock",
            timeout=timeout,
            headers=headers,
            encoding=get_encoding(),
            api_key=api_key,
            logging_obj=logging,
            client=client,
            provider_config=provider_config,
        )
    elif bedrock_route == "converse":
        model = without_bedrock_route_prefix(model)
        converse_model: Final = model

        def converse_attempt(fallback: NativeOutputFallback | None) -> NativeOutputAttemptResult:
            return bedrock_converse_chat_completion.completion(
                model=converse_model,
                messages=messages,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                optional_params=optional_params if fallback is None else dict(fallback.optional_params),
                litellm_params=litellm_params if fallback is None else dict(fallback.litellm_params),
                logger_fn=logger_fn,
                encoding=get_encoding(),
                logging_obj=logging,
                extra_headers=headers,  # Use merged headers instead of original extra_headers
                timeout=_native_output_attempt_timeout(ctx, fallback),
                acompletion=acompletion,
                client=client,
                api_base=api_base,
                api_key=api_key,
            )

        converse_params: Final = _OPTIONAL_PARAMS_ADAPTER.validate_python(optional_params)
        response = call_with_native_output_fallback(
            converse_attempt,
            (
                partial(_converse_native_output_fallback, ctx, converse_params, dict(litellm_params))
                if AmazonConverseConfig().uses_opt_in_native_output(converse_model, converse_params, litellm_params)
                else None
            ),
            converse_model,
        )
    elif bedrock_route == "converse_like":
        model = model.replace("converse_like/", "")
        response = base_llm_http_handler.completion(
            model=model,
            stream=stream,
            messages=messages,
            acompletion=acompletion,
            api_base=api_base,
            model_response=model_response,
            optional_params=optional_params,
            litellm_params=litellm_params,
            custom_llm_provider="bedrock",
            timeout=timeout,
            headers=headers,
            encoding=get_encoding(),
            api_key=api_key,
            logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            client=client,
        )
    else:

        def invoke_attempt(fallback: NativeOutputFallback | None) -> NativeOutputAttemptResult:
            return base_llm_http_handler.completion(
                model=model,
                stream=stream,
                messages=messages,
                acompletion=acompletion,
                api_base=api_base,
                model_response=model_response,
                optional_params=optional_params if fallback is None else dict(fallback.optional_params),
                litellm_params=litellm_params if fallback is None else dict(fallback.litellm_params),
                custom_llm_provider="bedrock",
                timeout=_native_output_attempt_timeout(ctx, fallback),
                headers=headers,
                encoding=get_encoding(),
                api_key=api_key,
                logging_obj=logging,
                client=client,
            )

        invoke_params: Final = _OPTIONAL_PARAMS_ADAPTER.validate_python(optional_params)
        response = call_with_native_output_fallback(
            invoke_attempt,
            (
                partial(_native_output_fallback, ctx, invoke_params, dict(litellm_params))
                if isinstance(provider_config, AmazonAnthropicClaudeConfig)
                and provider_config.uses_opt_in_native_output(model, invoke_params, litellm_params)
                else None
            ),
            model,
        )

    return response
