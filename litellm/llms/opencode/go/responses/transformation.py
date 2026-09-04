"""
OpenCode Go Responses API Configuration.

Routes models in the Go responses-model set to ``{base}/v1/responses``
with OpenAI-compatible request/response shape.  Uses ``Bearer`` auth.
"""

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # OpenAI Responses API wire format uses Any in param/return shapes
    Final,
)

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

GO_MESSAGES_BASE: Final = "https://opencode.ai/zen/go"

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj: Final = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj: Final = Any  # mutable-ok: fallback placeholder for runtime


class OpenCodeGoResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for OpenCode Go's Responses API.

    Inherits from OpenAIResponsesAPIConfig since Go's Responses API
    is compatible with OpenAI's Responses API specification.

    Key differences from direct OpenAI:
    - Uses ``{base}/v1/responses`` as the API base (Go gateway)
    - Uses ``OPENCODE_GO_API_KEY`` for authentication
    - Returns ``Bearer`` auth header
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENCODE_GO

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature must match OpenAIResponsesAPIConfig
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:  # mutable-ok: signature must match OpenAIResponsesAPIConfig
        litellm_params = litellm_params or GenericLiteLLMParams()  # rebind-ok: default to empty params
        api_key: Final = (
            litellm_params.api_key
            or litellm.opencode_go_api_key
            or get_secret_str("OPENCODE_GO_API_KEY")
            or get_secret_str("OPENCODE_API_KEY")
            or litellm.api_key
        )

        if not api_key:
            raise ValueError(
                "OpenCode Go API key is required. Set OPENCODE_GO_API_KEY environment variable or pass api_key parameter."
            )

        headers["Content-Type"] = "application/json"  # rebind-ok: caller expects auth header injected
        headers["Authorization"] = f"Bearer {api_key}"  # rebind-ok: caller expects auth header injected
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: Mapping[str, Any],
    ) -> str:
        base: Final = (
            api_base
            or litellm.opencode_go_api_base
            or get_secret_str("OPENCODE_GO_API_BASE")
            or litellm.api_base
            or GO_MESSAGES_BASE
        ).rstrip("/")

        if base.endswith("/v1/responses"):
            return base
        if base.endswith("/v1"):
            return f"{base}/responses"
        if base.endswith("/responses"):
            return base
        return f"{base}/v1/responses"

    def supports_native_websocket(self) -> bool:
        """OpenCode Go does not support native WebSocket for Responses API."""
        return False
