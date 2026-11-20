from collections.abc import Mapping
from typing import Final

from litellm.constants import PLACEHOLDER_API_KEY
from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    input_images,
    ir_to_systemone_request,
)
from litellm.types.decisions import DecisionsBase64Image, DecisionsIRRequest


class OllamaDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("OLLAMA_API_KEY",)
    api_base_env = ("OLLAMA_API_BASE",)
    api_key_required = False
    supports_images = True
    inlines_remote_images = True

    def get_default_api_base(self) -> str | None:
        return "http://localhost:11434"

    def resolve_api_key(self, api_key: str | None) -> str | None:
        resolved: Final = super().resolve_api_key(api_key)
        return None if resolved == PLACEHOLDER_API_KEY else resolved

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object]:
        body: Final = ir_to_systemone_request(self.request_model(model), request)
        images: Final = input_images(request.input)
        if not images:
            return body
        return {
            **body,
            "images": [
                image.base64 if isinstance(image, DecisionsBase64Image) else image.partition(",")[2]
                for image in images
            ],
        }
