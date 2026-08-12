"""
Parallel AI Responses API, an OpenAI Responses-compatible web-research endpoint.

Provider quirks:
- single `parallel` model; the performance tier is selected via `reasoning.effort` (low/medium/high)
- no `tools` param; web grounding is built in

Ref: https://docs.parallel.ai/responses-api/responses-quickstart
"""

from typing import Final

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class ParallelAIResponsesConfig(OpenAIResponsesAPIConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """Ref: https://docs.parallel.ai/responses-api/responses-quickstart"""
        return [
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
    ) -> dict:
        """Parallel rejects unknown Responses params (e.g. `tools`), so unsupported keys are filtered out."""
        supported: Final = frozenset(self.get_supported_openai_params(model))
        return {key: value for key, value in dict(response_api_optional_params).items() if key in supported}

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        resolved_params: Final = litellm_params or GenericLiteLLMParams()
        api_key: Final = (
            resolved_params.api_key or get_secret_str("PARALLEL_AI_API_KEY") or get_secret_str("PARALLEL_API_KEY")
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(self, api_base: str | None, litellm_params: dict) -> str:
        resolved_api_base: Final = api_base or get_secret_str("PARALLEL_AI_API_BASE") or "https://api.parallel.ai"
        trimmed: Final = resolved_api_base.rstrip("/")
        if trimmed.endswith("/v1/responses"):
            return trimmed
        return f"{trimmed.removesuffix('/v1')}/v1/responses"

    def supports_native_websocket(self) -> bool:
        """Parallel AI does not support native WebSocket for the Responses API"""
        return False
