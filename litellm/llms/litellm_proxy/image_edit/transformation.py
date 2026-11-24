from collections.abc import Mapping

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str


class LiteLLMProxyImageEditConfig(OpenAIImageEditConfig):
    """Configuration for image edit requests routed through LiteLLM Proxy."""

    def validate_environment(
        self,
        headers: dict[str, object],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:
        api_key = api_key or get_secret_str("LITELLM_PROXY_API_KEY")
        headers.update({"Authorization": f"Bearer {api_key}"})
        return headers

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        api_base = api_base or get_secret_str("LITELLM_PROXY_API_BASE")
        if api_base is None:
            raise ValueError("api_base not set for LiteLLM Proxy route. Set in env via `LITELLM_PROXY_API_BASE`")
        return join_api_base(api_base, "images/edits")
