"""
Translates from OpenAI's `/v1/chat/completions` to Consus Gateway's
`/v1/chat/completions`.

Consus Gateway is OpenAI-compatible in every respect except authentication:
it requires the API key in an `x-api-key` header instead of
`Authorization: Bearer <key>`.
"""

from typing import List, Optional, Tuple

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

CONSUS_API_BASE = "https://api.consus.io/v1"


class ConsusChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "consus"

    @staticmethod
    def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
        return api_key or litellm.consus_key or get_secret_str("CONSUS_API_KEY")

    def get_supported_openai_params(self, model: str) -> list:
        base_params = super().get_supported_openai_params(model)
        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                base_params.append("reasoning_effort")
        except Exception:
            pass
        return base_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("CONSUS_API_BASE") or CONSUS_API_BASE
        dynamic_api_key = ConsusChatConfig._resolve_api_key(api_key)
        return api_base, dynamic_api_key

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        resolved_key = ConsusChatConfig._resolve_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "Missing Consus API key. Set the CONSUS_API_KEY environment "
                "variable, set litellm.consus_key, or pass api_key=... to "
                "completion()."
            )

        headers["x-api-key"] = resolved_key

        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        return headers
