"""
Support for Anthropic's `/v1/messages` endpoint on Eden AI.

Eden AI serves the Anthropic Messages API at `/v3/v1/messages` for every model in its catalog, so
the Anthropic payload is forwarded untranslated and the answer comes back in Anthropic's shape with
Eden's per-request `cost` beside it. Eden does not report a cost inside a Messages stream yet, so
streams fall back to the price map.

Docs: https://www.edenai.co/docs/api-reference/anthropic-messages/create-anthropic-message
"""

from typing import TYPE_CHECKING, Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai_like.json_loader import SimpleProviderConfig
from litellm.llms.openai_like.messages.transformation import JSONProviderAnthropicMessagesConfig
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.utils import LlmProviders

from ..common_utils import EDENAI_API_BASE, EdenAIException, reported_cost, require_api_key

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_EDENAI_PROVIDER_SPEC: Final[dict[str, str]] = {  # mutable-ok: SimpleProviderConfig takes a plain dict
    "base_url": EDENAI_API_BASE,
    "api_key_env": "EDENAI_API_KEY",
    "api_base_env": "EDENAI_API_BASE",
}
_EDENAI_PROVIDER: Final = SimpleProviderConfig(LlmProviders.EDENAI.value, _EDENAI_PROVIDER_SPEC)


class EdenAIAnthropicMessagesConfig(JSONProviderAnthropicMessagesConfig):
    def __init__(self) -> None:
        super().__init__(_EDENAI_PROVIDER)

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],  # mutable-ok: inherited contract
        model: str,
        messages: list[object],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: inherited contract
        return super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=require_api_key(api_key, model),
            api_base=api_base,
        )

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> AnthropicMessagesResponse:
        response: Final = super().transform_anthropic_messages_response(
            model=model, raw_response=raw_response, logging_obj=logging_obj
        )
        cost: Final = reported_cost(response)
        if cost is not None:
            logging_obj.model_call_details["response_cost"] = cost  # rebind-ok: the per-call record spend logging reads
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return EdenAIException(message=error_message, status_code=status_code, headers=headers)
