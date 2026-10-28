from collections.abc import Mapping
from itertools import chain
from typing import Final, TypeAlias

from typing_extensions import assert_never

from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    decisions_text,
    image_media_type,
    input_images,
)
from litellm.types.decisions import (
    DecisionsBase64Image,
    DecisionsImage,
    DecisionsIRMessages,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsIRState,
    OpenAIDecisionInputImage,
    OpenAIDecisionInputMessage,
    OpenAIDecisionInputText,
    UnsupportedDecisionsRequest,
)

_StateItem: TypeAlias = str | Mapping[str, object]


def _data_url(image: DecisionsImage) -> str:
    if isinstance(image, DecisionsBase64Image):
        return f"data:{image_media_type(image)};base64,{image.base64}"
    return image


def _image_item(image: DecisionsImage) -> _StateItem:
    return {"type": "image_url", "image_url": {"url": _data_url(image)}}


def _part_item(part: OpenAIDecisionInputText | OpenAIDecisionInputImage) -> _StateItem:
    if isinstance(part, OpenAIDecisionInputText):
        return part.text
    return _image_item(part.image_url)


def _message_items(message: OpenAIDecisionInputMessage) -> tuple[_StateItem, ...]:
    if isinstance(message.content, str):
        return (message.content,)
    return tuple(map(_part_item, message.content))


def _multimodal_state(decision_input: DecisionsIRState | DecisionsIRMessages) -> tuple[_StateItem, ...]:
    match decision_input:
        case DecisionsIRState():
            return (decisions_text(decision_input.state), *map(_image_item, decision_input.images))
        case DecisionsIRMessages():
            return tuple(chain.from_iterable(map(_message_items, decision_input.messages)))
        case _:
            assert_never(decision_input)


class OpenRouterDecisionsConfig(BaseDecisionsConfig):
    path = "/alpha/decisions"
    api_key_env = ("OPENROUTER_API_KEY",)
    api_base_env = ("OPENROUTER_API_BASE",)
    supports_images = True
    image_media_types = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})
    inlines_remote_images = True

    def get_default_api_base(self) -> str | None:
        return "https://openrouter.ai/api"

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        cost: Final = response.usage.extra.get("cost") if response.usage is not None else None
        if not isinstance(cost, (int, float)):
            return None
        return float(cost)

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        body: Final = super().transform_decisions_request(model, request, custom_llm_provider)
        if isinstance(body, UnsupportedDecisionsRequest) or not input_images(request.input):
            return body
        return {
            **{key: value for key, value in body.items() if key != "images"},
            "state": _multimodal_state(request.input),
        }
