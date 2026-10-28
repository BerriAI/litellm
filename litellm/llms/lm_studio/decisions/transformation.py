from typing import Final

from litellm.constants import PLACEHOLDER_API_KEY
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class LMStudioDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("LM_STUDIO_API_KEY",)
    api_base_env = ("LM_STUDIO_API_BASE",)
    api_key_required = False
    supports_images = True
    inlines_remote_images = True

    def resolve_api_key(self, api_key: str | None) -> str | None:
        resolved: Final = super().resolve_api_key(api_key)
        return None if resolved == PLACEHOLDER_API_KEY else resolved
