from typing import Optional

from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str


class NearAIImageEditConfig(OpenAIImageEditConfig):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        resolved_api_key = api_key or get_secret_str("NEARAI_API_KEY")
        return {**headers, "Authorization": f"Bearer {resolved_api_key}"}

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        resolved_api_base = api_base or get_secret_str("NEARAI_API_BASE") or "https://cloud-api.near.ai/v1"
        return f"{resolved_api_base.rstrip('/')}/images/edits"
