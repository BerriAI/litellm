"""
Support for OpenAI's `/v1/responses` endpoint on Eden AI.

Eden AI serves the Responses API at `/v3/responses` in OpenAI's wire format, so the OpenAI config
does the work; this one points it at Eden and authenticates with the Eden key. Eden reports the
per-request cost on `usage.cost` of every body, the final `response.completed` event included, so
the shared usage-cost lift bills both modes.

Docs: https://www.edenai.co/docs/v3/llms/responses
"""

from typing import TYPE_CHECKING, Final

import httpx

from litellm.litellm_core_utils.core_helpers import set_response_cost_in_hidden_params
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import EdenAIException, authorized_headers, resolve_api_base

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class EdenAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.EDENAI

    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: inherited contract
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return authorized_headers(headers, litellm_params.api_key if litellm_params else None, model)

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
    ) -> str:
        return super().get_complete_url(api_base=resolve_api_base(api_base), litellm_params=litellm_params)

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ResponsesAPIResponse:
        response: Final = super().transform_response_api_response(
            model=model, raw_response=raw_response, logging_obj=logging_obj
        )
        set_response_cost_in_hidden_params(response, response.usage.cost if response.usage else None)
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return EdenAIException(message=error_message, status_code=status_code, headers=headers)

    def should_fake_stream(
        self,
        model: str | None,
        stream: bool | None,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """Eden streams every catalog model natively; the base class would fake-stream any model the
        price map does not know, which is all of them."""
        return False

    def supports_native_websocket(self) -> bool:
        return False
