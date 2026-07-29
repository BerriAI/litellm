from collections.abc import Mapping

from pydantic import BaseModel
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


class OpenRouterImagesResponse(BaseModel):
    """Response body of ``POST https://openrouter.ai/api/v1/images``."""

    data: tuple[OpenRouterImageData, ...]
    created: int | None = None
    model: str | None = None
    usage: OpenRouterImageUsage | None = None
