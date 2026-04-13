"""
OpenAI-like Responses API transformation.

Base class for JSON-declared providers that support the /v1/responses endpoint.
Inherits everything from OpenAIResponsesAPIConfig; subclasses only override
provider-specific resolution (slug, API key env var, base URL).
"""

from typing import Optional, Union

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class OpenAILikeResponsesConfig(OpenAIResponsesAPIConfig):
    """
    Responses API config for OpenAI-compatible providers declared via JSON.

    Concrete per-provider classes are generated dynamically in dynamic_config.py.
    This base provides the three overridable hooks that the dynamic generator
    fills in: custom_llm_provider, validate_environment, get_complete_url.
    """

    @property
    def custom_llm_provider(self) -> Union[str, LlmProviders]:  # type: ignore[override]
        return "openai_like"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or get_secret_str("OPENAI_LIKE_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or get_secret_str("OPENAI_LIKE_API_BASE")
        if not api_base:
            raise ValueError("api_base is required for openai_like provider")
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"
