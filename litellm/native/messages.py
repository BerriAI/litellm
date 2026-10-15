"""Sends an Anthropic Messages request to a provider that speaks Messages natively."""

from collections.abc import AsyncIterator, Coroutine, Mapping
from typing import Final

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.pass_through.messages.utils import AnthropicMessagesRequestUtils
from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.router import GenericLiteLLMParams

__all__ = ("execute_messages",)


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
        anthropic_messages_optional_request_params=dict(optional_params),
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
