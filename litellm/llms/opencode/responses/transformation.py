"""
OpenCode routes its OpenAI-family models to a `/responses` endpoint rather than
`/chat/completions`, which answers those models with a 500.

The surface is OpenAI's own Responses API, so only the base URL, the credential lookup and
the required `x-opencode-session` header differ.
"""

from collections.abc import Mapping
from typing import ClassVar, Final

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import resolve_opencode_api_key, with_opencode_session_header


class OpenCodeResponsesAPIConfig(OpenAIResponsesAPIConfig):
    _provider: ClassVar[LlmProviders]
    _default_api_base: ClassVar[str]
    _api_base_env_var: ClassVar[str]

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return self._provider

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        resolved_api_base: Final = (
            api_base or get_secret_str(self._api_base_env_var) or self._default_api_base
        ).rstrip("/")
        return f"{resolved_api_base}/responses"

    def validate_environment(
        self, headers: Mapping[str, object], model: str, litellm_params: GenericLiteLLMParams | None
    ) -> dict:  # mutable-ok: signature is fixed by BaseResponsesAPIConfig.validate_environment
        resolved_params: Final = litellm_params or GenericLiteLLMParams()
        base_headers: Final[Mapping[str, object]] = {
            "Content-Type": "application/json",
            **headers,
            "Authorization": f"Bearer {resolve_opencode_api_key(resolved_params.api_key)}",
        }
        return with_opencode_session_header(base_headers, resolved_params.model_dump())


class OpenCodeZenResponsesAPIConfig(OpenCodeResponsesAPIConfig):
    _provider: ClassVar[LlmProviders] = LlmProviders.OPENCODE
    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_API_BASE"


class OpenCodeGoResponsesAPIConfig(OpenCodeResponsesAPIConfig):
    _provider: ClassVar[LlmProviders] = LlmProviders.OPENCODE_GO
    _default_api_base: ClassVar[str] = "https://opencode.ai/zen/go/v1"
    _api_base_env_var: ClassVar[str] = "OPENCODE_GO_API_BASE"
