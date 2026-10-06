from collections.abc import Mapping

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


class OpenRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: dict


class OpenRouterImageCompletionTokensDetails(BaseModel):
    image_tokens: int | None = None
    reasoning_tokens: int | None = None


class OpenRouterImagePromptTokensDetails(BaseModel):
    image_tokens: int | None = None
    text_tokens: int | None = None


class OpenRouterImageUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None
    cost_details: Mapping[str, float | None] | None = None
    completion_tokens_details: OpenRouterImageCompletionTokensDetails | None = None
    prompt_tokens_details: OpenRouterImagePromptTokensDetails | None = None


class OpenRouterImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    media_type: str | None = None
    revised_prompt: str | None = None

    @model_validator(mode="after")
    def _require_image_payload(self) -> "OpenRouterImageData":
        if self.b64_json is None and self.url is None:
            raise ValueError("image entry has neither b64_json nor url")
        return self


class OpenRouterImagesResponse(BaseModel):
    data: tuple[OpenRouterImageData, ...] = Field(min_length=1)
    created: int | None = None
    model: str | None = None
    usage: OpenRouterImageUsage | None = None
