from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx

from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.opencode.common_utils import attach_go_reported_cost, go_api_base, go_headers
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging


class OpenCodeGoResponsesConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENCODE_GO

    def supports_native_websocket(self) -> bool:
        return False

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return ensure_api_route(go_api_base(api_base).rstrip("/").removesuffix("/v1"), "v1/responses")

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, object]:  # mutable-ok: protocol handlers mutate the returned request and headers
        return go_headers(headers, litellm_params, litellm_params.api_key if litellm_params else None, model)

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> ResponsesAPIResponse:
        response: Final = super().transform_response_api_response(
            model=model, raw_response=raw_response, logging_obj=logging_obj
        )
        attach_go_reported_cost(response, response.usage)
        return response

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: Mapping[str, object],
        logging_obj: "Logging",
    ) -> ResponsesAPIStreamingResponse:
        event: Final = super().transform_streaming_response(  # pyright: ignore[reportUnknownMemberType]  # base takes an untyped dict
            model=model, parsed_chunk=dict(parsed_chunk), logging_obj=logging_obj
        )
        if isinstance(event, ResponseCompletedEvent | ResponseFailedEvent | ResponseIncompleteEvent):
            attach_go_reported_cost(event.response, event.response.usage)
        return event
