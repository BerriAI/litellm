from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.types.decisions import DecisionsIRResponse

_TYPESAFE_IMAGE_MEDIA_TYPES: Final = frozenset({"image/png", "image/jpeg", "image/jpg"})


class OpenRouterDecisionsConfig(BaseDecisionsConfig):
    path = "/alpha/decisions"
    api_key_env = ("OPENROUTER_API_KEY",)
    api_base_env = ("OPENROUTER_API_BASE",)
    supports_images = True
    image_media_types = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})
    inlines_remote_images = True
    images_in_state = True

    def get_default_api_base(self) -> str | None:
        return "https://openrouter.ai/api"

    def image_media_types_for(self, model: str) -> frozenset[str] | None:
        if model.removeprefix("~").startswith("typesafe/"):
            return _TYPESAFE_IMAGE_MEDIA_TYPES
        return self.image_media_types

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        cost: Final = response.usage.extra.get("cost") if response.usage is not None else None
        if not isinstance(cost, (int, float)):
            return None
        return float(cost)
