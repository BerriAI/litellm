from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.types.decisions import (
    DecisionsBase64Image,
    DecisionsImage,
    DecisionsIRRequest,
    DecisionsIRState,
    UnsupportedDecisionsRequest,
)


def _data_url(image: DecisionsImage) -> DecisionsImage:
    if isinstance(image, DecisionsBase64Image):
        return f"data:{image.content_type};base64,{image.base64}"
    return image


def _with_images_as_data_urls(request: DecisionsIRRequest) -> DecisionsIRRequest:
    if not isinstance(request.input, DecisionsIRState):
        return request
    return replace(request, input=replace(request.input, images=tuple(map(_data_url, request.input.images))))


class NaceDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("NACE_API_KEY",)
    api_base_env = ("NACE_API_BASE", "NACE_BASE_URL")
    supports_images = True
    max_images = 8
    image_media_types = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})
    inlines_remote_images = True

    def get_default_api_base(self) -> str | None:
        return "https://console.nace.ai"

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        return super().transform_decisions_request(model, _with_images_as_data_urls(request), custom_llm_provider)


NACE_DECISIONS_CONFIG: Final = NaceDecisionsConfig()
