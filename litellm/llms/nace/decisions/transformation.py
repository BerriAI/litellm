import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from litellm._version import version as _litellm_version
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.types.decisions import (
    DecisionsBase64Image,
    DecisionsImage,
    DecisionsIRRequest,
    DecisionsIRState,
    UnsupportedDecisionsRequest,
)

NACE_CLIENT_HEADER: Final = "X-Nace-Client"
_NACE_CLIENT_DISALLOWED: Final = re.compile(r"[^A-Za-z0-9 ._@+-]")
_NACE_CLIENT_MAX_LENGTH: Final = 64


def nace_client_header() -> Mapping[str, str]:
    version: Final = _NACE_CLIENT_DISALLOWED.sub("-", _litellm_version)
    return {NACE_CLIENT_HEADER: f"litellm/{version}"[:_NACE_CLIENT_MAX_LENGTH]}


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

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, str]:
        base: Final = super().validate_environment(headers, model, api_key, litellm_params)
        return {
            **{name: value for name, value in base.items() if name.lower() != NACE_CLIENT_HEADER.lower()},
            **nace_client_header(),
        }

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        return super().transform_decisions_request(model, _with_images_as_data_urls(request), custom_llm_provider)


NACE_DECISIONS_CONFIG: Final = NaceDecisionsConfig()
