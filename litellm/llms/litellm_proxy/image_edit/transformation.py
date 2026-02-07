from typing import Optional

from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str


class LiteLLMProxyImageEditConfig(OpenAIImageEditConfig):
    """Configuration for image edit requests routed through LiteLLM Proxy."""

    def validate_environment(
        self, headers: dict, model: str, api_key: Optional[str] = None
    ) -> dict:
        api_key = api_key or get_secret_str("LITELLM_PROXY_API_KEY")
        headers.update({"Authorization": f"Bearer {api_key}"})
        return headers

    def get_complete_url(
        self, model: str, api_base: Optional[str], litellm_params: dict
    ) -> str:
        api_base = api_base or get_secret_str("LITELLM_PROXY_API_BASE")
        if api_base is None:
            raise ValueError(
                "api_base not set for LiteLLM Proxy route. Set in env via `LITELLM_PROXY_API_BASE`"
            )
        api_base = api_base.rstrip("/")
        return f"{api_base}/images/edits"
