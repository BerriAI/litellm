from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.cloudflare.common_utils import (
    CLOUDFLARE_MISSING_API_BASE_MESSAGE,
    unwrap_workers_ai_result,
    workers_ai_canonical_model,
    workers_ai_default_api_base,
    workers_ai_request_model,
    workers_ai_run_url,
)


class CloudflareDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("CLOUDFLARE_API_KEY",)
    api_base_env = ("CLOUDFLARE_API_BASE",)
    supports_images = True
    max_images = 4
    max_image_bytes = 4 * 1024 * 1024
    max_total_image_bytes = 8 * 1024 * 1024
    image_media_types = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})
    inlines_remote_images = True

    def get_default_api_base(self) -> str | None:
        # Decision models (Clef) require native /ai/run/{model} requests with
        # state and questions; /ai/v1/chat/completions only accepts chat requests.
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
