"""
Z.AI OpenAI-compatible Responses API transformation config.
"""

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class ZAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Z.AI exposes an OpenAI-compatible Responses API at
    https://api.z.ai/api/v1 (see https://docs.z.ai/guides/llm/glm-5.3).

    The endpoint authenticates with the Z.AI API key sent as an
    ``Authorization: Bearer`` header.
    """

    _ZAI_CHAT_API_BASE_SUFFIXES: Final = ("/api/paas/v4", "/api/coding/paas/v4")

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.ZAI

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("ZAI_API_KEY") or litellm.api_key

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, str]:  # mutable-ok: the responses handler owns and mutates the returned headers dict
        request_api_key: Final = litellm_params.api_key if litellm_params is not None else None
        resolved_params: Final = GenericLiteLLMParams(api_key=self.get_api_key(api_key=request_api_key))

        return super().validate_environment(
            headers=headers,
            model=model,
            litellm_params=resolved_params,
        )

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        # ``litellm_params.api_base`` can carry the Z.AI chat-completions base
        # (``/api/paas/v4``) when the generic provider resolver pre-fills it from
        # the chat config. Z.AI serves Responses on a different base, so ignore
        # the chat-only bases and use the Responses base instead.
        normalized_api_base: Final = (api_base or "").rstrip("/")
        chat_base_passed_in: Final = normalized_api_base.endswith(self._ZAI_CHAT_API_BASE_SUFFIXES)
        base_url: Final = normalized_api_base if api_base and not chat_base_passed_in else "https://api.z.ai/api/v1"

        if base_url.endswith("/responses"):
            return base_url
        return f"{base_url}/responses"
