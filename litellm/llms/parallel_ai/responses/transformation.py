"""
Parallel AI Responses API, an OpenAI Responses-compatible web-research endpoint.

Provider quirks:
- single `parallel` model; the performance tier is selected via `reasoning.effort` (low/medium/high),
  or with the `parallel-low` / `parallel-medium` / `parallel-high` aliases, which pin the tier and
  carry per-tier pricing in the cost map
- no `tools` param; web grounding is built in

Ref: https://docs.parallel.ai/responses-api/responses-quickstart
"""

from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.parallel_ai.common_utils import resolve_parallel_ai_credentials
from litellm.llms.parallel_ai.responses.cost_calculator import (
    PARALLEL_AI_EFFORT_TIER_MODELS as EFFORT_TIER_MODELS,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class ParallelAIResponsesConfig(OpenAIResponsesAPIConfig):
    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseResponsesAPIConfig contract
        """Ref: https://docs.parallel.ai/responses-api/responses-quickstart"""
        return [  # mutable-ok: callers concatenate with other lists
            "stream",
            "reasoning",
            "instructions",
            "text",
            "previous_response_id",
        ]

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.PARALLEL_AI

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseResponsesAPIConfig contract
        """Parallel rejects unknown Responses params (e.g. `tools`), so unsupported keys are filtered out."""
        supported: Final = frozenset(self.get_supported_openai_params(model))
        return {  # mutable-ok: request payload, merged downstream
            key: value for key, value in response_api_optional_params.items() if key in supported
        }

    def validate_environment(  # mutable-ok: BaseResponsesAPIConfig contract
        self,
        headers: dict,  # mutable-ok: BaseResponsesAPIConfig contract
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:  # mutable-ok: BaseResponsesAPIConfig contract
        resolved_params: Final = litellm_params or GenericLiteLLMParams()
        _, api_key = resolve_parallel_ai_credentials(api_base=resolved_params.api_base, api_key=resolved_params.api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"  # rebind-ok: stamping the caller's headers is the contract
        return headers

    def get_complete_url(self, api_base: str | None, litellm_params: dict) -> str:  # mutable-ok: base contract
        resolved_api_base: Final = api_base or get_secret_str("PARALLEL_AI_API_BASE") or "https://api.parallel.ai"
        trimmed: Final = resolved_api_base.rstrip("/")
        if trimmed.endswith("/v1/responses"):
            return trimmed
        return f"{trimmed.removesuffix('/v1')}/v1/responses"

    def transform_responses_api_request(  # mutable-ok: BaseResponsesAPIConfig contract
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,  # mutable-ok: BaseResponsesAPIConfig contract
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseResponsesAPIConfig contract
    ) -> dict:  # mutable-ok: BaseResponsesAPIConfig contract
        """The tier aliases pin `reasoning.effort` so each alias bills at its cost map entry."""
        effort: Final = EFFORT_TIER_MODELS.get(model)
        if effort is None:
            return super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        return super().transform_responses_api_request(
            model="parallel",
            input=input,
            response_api_optional_request_params={  # mutable-ok: request payload
                **response_api_optional_request_params,
                "reasoning": {"effort": effort},  # mutable-ok: request payload
            },
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """The API reports model `parallel` for every tier; cost tracking prices by response model, so the alias is restored."""
        response: Final = super().transform_response_api_response(
            model=model, raw_response=raw_response, logging_obj=logging_obj
        )
        if model in EFFORT_TIER_MODELS:
            response.model = model
        return response

    def supports_native_websocket(self) -> bool:
        """Parallel AI does not support native WebSocket for the Responses API"""
        return False
