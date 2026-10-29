from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class PerplexityDecisionsConfig(BaseDecisionsConfig):
    path = "/v1/decisions"
    api_key_env = ("PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY")
    api_base_env = ("PERPLEXITY_API_BASE",)
    supports_images = True
    max_images = 8
    image_media_types = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})
    inlines_remote_images = True
    images_in_state = True

    def get_default_api_base(self) -> str | None:
        return "https://api.perplexity.ai"
