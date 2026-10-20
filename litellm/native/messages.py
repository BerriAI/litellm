"""Sends an Anthropic Messages request to a provider that speaks Messages natively."""

from collections.abc import AsyncIterator, Coroutine, Mapping
from typing import Final

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.pass_through.messages.utils import AnthropicMessagesRequestUtils
from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.router_utils.reasoning_effort_capability import (
    RequestedReasoningEffort,
    deployment_reasoning_effort_levels,
    keeps_effort,
    model_group_from_metadata,
    resolve_deployment_reasoning_effort,
    with_adaptive_thinking_as_effort,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.router import GenericLiteLLMParams

__all__ = ("execute_messages",)


def _with_output_config(params: Mapping[str, object], output_config: RequestedReasoningEffort) -> Mapping[str, object]:
    rest: Final = {key: value for key, value in params.items() if key != "output_config"}
    return rest if output_config is None else {**rest, "output_config": output_config}


def _with_deployment_output_effort(
    params: Mapping[str, object],
    *,
    model: str,
    custom_llm_provider: str,
    kwargs: Mapping[str, object],
    drop_params: bool,
) -> Mapping[str, object]:
    deployment_model_info: Final = kwargs.get("model_info")

    def resolve(requested: RequestedReasoningEffort) -> RequestedReasoningEffort:
        return resolve_deployment_reasoning_effort(
            requested,
            model=model,
            custom_llm_provider=custom_llm_provider,
            deployment_model_info=deployment_model_info,
            model_group=model_group_from_metadata(kwargs.get("litellm_metadata"), kwargs.get("metadata")),
            drop_params=drop_params,
        ).value

    def translate(source: Mapping[str, object]) -> Mapping[str, object]:
        return with_adaptive_thinking_as_effort(
            source, model=model, custom_llm_provider=custom_llm_provider, drop_params=drop_params
        )

    requested_output: Final = as_str_mapping(params.get("output_config"))
    caller_output: Final = None if requested_output is None else resolve(requested_output)
    caller_params: Final = params if requested_output is None else _with_output_config(params, caller_output)
    if deployment_reasoning_effort_levels(deployment_model_info) is None:
        return caller_params
    if requested_output is not None and keeps_effort(requested_output, caller_output):
        return _with_output_config(translate(params), caller_output)
    translated: Final = translate(caller_params)
    budget_output: Final = as_str_mapping(translated.get("output_config"))
    return translated if budget_output is None else _with_output_config(translated, resolve(budget_output))


def execute_messages(
    *,
    http_handler: BaseLLMHTTPHandler,
    model: str,
    messages: list[dict[str, object]],
    provider_config: BaseAnthropicMessagesConfig,
    request_params: Mapping[str, object],
    custom_llm_provider: str,
    litellm_params: GenericLiteLLMParams,
    logging_obj: LiteLLMLoggingObj,
    api_key: str | None,
    api_base: str | None,
    dynamic_api_base: str | None,
    client: AsyncHTTPHandler | None,
    is_async: bool,
    stream: bool | None,
    request_kwargs: dict[str, object],
) -> AnthropicMessagesResponse | Coroutine[object, object, AnthropicMessagesResponse | AsyncIterator[bytes]]:
    optional_params: Final = AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param(
        params=request_params,
        model=model,
        drop_params=litellm_params.get("drop_params") is True,
        custom_llm_provider=custom_llm_provider,
    )
    resolved_api_base: Final = (
        dynamic_api_base
        if dynamic_api_base is not None and provider_config.uses_get_llm_provider_api_base()
        else api_base
    )
    return http_handler.anthropic_messages_handler(
        model=model,
        messages=messages,
        anthropic_messages_provider_config=provider_config,
        anthropic_messages_optional_request_params=dict(
            _with_deployment_output_effort(
                optional_params,
                model=model,
                custom_llm_provider=custom_llm_provider,
                kwargs=request_kwargs,
                drop_params=litellm.drop_params is True or litellm_params.get("drop_params") is True,
            )
        ),
        _is_async=is_async,
        client=client,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        api_key=api_key,
        api_base=resolved_api_base,
        stream=stream,
        kwargs=request_kwargs,
    )
