from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.extraction.transformation import BaseExtractionConfig


class TurionExtractionConfig(BaseExtractionConfig):
    api_key_env = ("TURION_API_KEY", "TURION_API_TOKEN")
    api_base_env = ("TURION_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.turion.ai"

    def get_complete_url(self, api_base: str, model: str) -> str:
        return join_api_base(api_base.rstrip("/").removesuffix("/v1"), "/v1/extract")
