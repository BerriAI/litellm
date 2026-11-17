from litellm.llms.base_llm.extraction.transformation import BaseExtractionConfig
from litellm.llms.cloudflare.common_utils import (
    CLOUDFLARE_MISSING_API_BASE_MESSAGE,
    unwrap_workers_ai_result,
    workers_ai_canonical_model,
    workers_ai_default_api_base,
    workers_ai_request_model,
    workers_ai_run_url,
)


class CloudflareExtractionConfig(BaseExtractionConfig):
    api_key_env = ("CLOUDFLARE_API_KEY",)
    api_base_env = ("CLOUDFLARE_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return workers_ai_default_api_base()

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return CLOUDFLARE_MISSING_API_BASE_MESSAGE

    def canonical_model(self, model: str) -> str:
        return workers_ai_canonical_model(model)

    def request_model(self, model: str) -> str:
        return workers_ai_request_model(model)

    def get_complete_url(self, api_base: str, model: str) -> str:
        return workers_ai_run_url(api_base, model)

    def unwrap_response(self, payload: object) -> object:
        return unwrap_workers_ai_result(payload)
