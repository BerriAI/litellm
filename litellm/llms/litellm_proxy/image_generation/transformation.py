from collections.abc import Mapping

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.openai.image_generation.gpt_transformation import (
    GPTImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class LiteLLMProxyImageGenerationConfig(GPTImageGenerationConfig):
    """Configuration for image generation requests routed through LiteLLM Proxy."""

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        api_key = api_key or get_secret_str("LITELLM_PROXY_API_KEY")
        headers.update({"Authorization": f"Bearer {api_key}"})
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        api_base = api_base or get_secret_str("LITELLM_PROXY_API_BASE")
        if api_base is None:
            raise ValueError("api_base not set for LiteLLM Proxy route. Set in env via `LITELLM_PROXY_API_BASE`")
        return join_api_base(api_base, "images/generations")
