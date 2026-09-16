"""
OpenRouter Responses API Configuration.

OpenRouter supports the Responses API at https://openrouter.ai/api/v1/responses
with OpenAI-compatible request/response format, including reasoning with
encrypted_content for multi-turn stateless workflows.

Docs: https://openrouter.ai/docs/api/reference/responses/overview
"""

from typing import TYPE_CHECKING, Final

import httpx

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseAPIUsage, ResponseInputParam, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging


class OpenRouterResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for OpenRouter's Responses API.

    Inherits from OpenAIResponsesAPIConfig since OpenRouter's Responses API
    is compatible with OpenAI's Responses API specification.

    Key difference from direct OpenAI:
    - Uses https://openrouter.ai/api/v1 as the API base
    - Uses OPENROUTER_API_KEY for authentication
    - Requests OpenRouter's `usage.cost` and surfaces it for spend tracking
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENROUTER

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key: Final = (
            litellm_params.api_key
            or litellm.api_key
            or get_secret_str("OPENROUTER_API_KEY")
            or get_secret_str("OR_API_KEY")
        )

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        api_base = (
            api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
        )

        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """OpenRouter does not support native WebSocket for Responses API"""
        return False

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict[str, object],  # mutable-ok: base signature
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: base signature
    ) -> dict[str, object]:  # mutable-ok: request body, the HTTP handler extends it in place
        """Ask OpenRouter to report real spend in `usage.cost`.

        Mirrors the chat path (`OpenrouterConfig.transform_request`). Without
        `usage.include=true` OpenRouter omits `cost` from the response, so the
        Responses API path has nothing to fall back on and every request logs
        `spend = 0` for any model not in litellm's bundled pricing JSON.
        """
        transformed: Final = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        if "usage" not in transformed:
            transformed["usage"] = {"include": True}  # mutable-ok: extends the request body super() just built
        return transformed

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> ResponsesAPIResponse:
        """Carry OpenRouter's returned `usage.cost` into hidden params.

        `get_response_cost_from_hidden_params()` reads
        `additional_headers["llm_provider-x-litellm-response-cost"]` before any
        static price-map lookup, so this keeps cost accounting working for
        OpenRouter models that are absent from litellm's bundled pricing JSON.
        Same mechanism as the chat path (`OpenrouterConfig.transform_response`).
        """
        response: Final = super().transform_response_api_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )
        usage: Final = response.usage
        response_cost: Final = usage.cost if isinstance(usage, ResponseAPIUsage) else None
        if response_cost is not None:
            hidden: Final = response._hidden_params  # pyright: ignore[reportPrivateUsage]  # cost-header write, mirrors chat path
            hidden["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(response_cost)
        return response
