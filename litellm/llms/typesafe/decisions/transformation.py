import hashlib
from collections.abc import Mapping
from dataclasses import replace
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal

from litellm.constants import TYPESAFE_DANGER_ZONE_HEADER_PREFIX
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig, image_media_type
from litellm.types.decisions import (
    DecisionsAsset,
    DecisionsIRMessages,
    DecisionsIRRequest,
    DecisionsIRState,
    OpenAIDecisionInputImage,
    OpenAIDecisionInputMessage,
    OpenAIDecisionInputText,
    UnsupportedDecisionsRequest,
)

_IMAGES_HEADER: Final = f"{TYPESAFE_DANGER_ZONE_HEADER_PREFIX}images"
_ASSET_FORMATS: Final[Mapping[str, Literal["png", "jpg"]]] = MappingProxyType(
    {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg"}
)
_URL_SAFE_TO_STANDARD_BASE64: Final = str.maketrans("-_", "+/")


def _image_reference(image_url: str) -> str:
    return f"image-{hashlib.sha256(image_url.encode()).hexdigest()[:32]}"


def _image_asset(image_url: str) -> DecisionsAsset | None:
    asset_format: Final = _ASSET_FORMATS.get(image_media_type(image_url) or "")
    if asset_format is None:
        return None
    return DecisionsAsset(
        type="image",
        format=asset_format,
        data_base64=image_url.partition(",")[2].translate(_URL_SAFE_TO_STANDARD_BASE64),
    )


def _part_text(part: OpenAIDecisionInputText | OpenAIDecisionInputImage) -> str:
    if isinstance(part, OpenAIDecisionInputText):
        return part.text
    return part.image_url if _image_asset(part.image_url) is None else _image_reference(part.image_url)


def _message_texts(message: OpenAIDecisionInputMessage) -> tuple[str, ...]:
    if isinstance(message.content, str):
        return (message.content,)
    return tuple(map(_part_text, message.content))


def _message_assets(message: OpenAIDecisionInputMessage) -> Mapping[str, DecisionsAsset]:
    if isinstance(message.content, str):
        return {}
    return {
        _image_reference(part.image_url): asset
        for part in message.content
        if isinstance(part, OpenAIDecisionInputImage) and (asset := _image_asset(part.image_url)) is not None
    }


def _with_images_as_assets(request: DecisionsIRRequest) -> DecisionsIRRequest:
    if not isinstance(request.input, DecisionsIRMessages):
        return request
    messages: Final = request.input.messages
    return replace(
        request,
        input=DecisionsIRState(state="\n\n".join(chain.from_iterable(map(_message_texts, messages)))),
        assets=MappingProxyType(
            dict(chain(request.assets.items(), *(_message_assets(message).items() for message in messages)))
        ),
    )


class TypeSafeDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("TYPESAFE_API_KEY",)
    api_base_env = ("TYPESAFE_API_BASE",)
    image_media_types = frozenset(_ASSET_FORMATS)
    inlines_remote_images = True

    def get_default_api_base(self) -> str | None:
        return "https://api.typesafe.ai"

    def supports_images_in(
        self, decision_input: DecisionsIRState | DecisionsIRMessages, headers: Mapping[str, str]
    ) -> bool:
        return isinstance(decision_input, DecisionsIRMessages) and self.supports_assets(headers)

    def supports_assets(self, headers: Mapping[str, str]) -> bool:
        return any(
            name.lower() == _IMAGES_HEADER and value.strip().lower() == "true" for name, value in headers.items()
        )

    def supports_box_questions(
        self, model: str, custom_llm_provider: str, headers: Mapping[str, str], deployment_flag: bool | None
    ) -> bool:
        return self.supports_assets(headers) and super().supports_box_questions(
            model, custom_llm_provider, headers, deployment_flag
        )

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        return super().transform_decisions_request(model, _with_images_as_assets(request), custom_llm_provider)
