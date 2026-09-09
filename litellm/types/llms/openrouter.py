from collections.abc import Mapping

from pydantic import BaseModel
from typing_extensions import TypedDict


class OpenRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: dict


class OpenRouterImageCompletionTokensDetails(BaseModel):
    """``usage.completion_tokens_details`` on ``POST /api/v1/images`` responses."""

    image_tokens: int | None = None
    reasoning_tokens: int | None = None


class OpenRouterImagePromptTokensDetails(BaseModel):
    """``usage.prompt_tokens_details`` on ``POST /api/v1/images`` responses (image edit only)."""

    image_tokens: int | None = None
    text_tokens: int | None = None


class OpenRouterImageUsage(BaseModel):
    """Usage block of the OpenRouter unified image API response, incl. real cost for spend-log settlement."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None
    cost_details: Mapping[str, float | None] | None = None
    completion_tokens_details: OpenRouterImageCompletionTokensDetails | None = None
    prompt_tokens_details: OpenRouterImagePromptTokensDetails | None = None


class OpenRouterImageData(BaseModel):
    """One image entry in ``data[]``; OpenRouter always returns base64, never a bare URL, but both fields are modeled defensively."""

    b64_json: str | None = None
    url: str | None = None
    media_type: str | None = None
    revised_prompt: str | None = None


class OpenRouterImagesResponse(BaseModel):
    """
    Response body of ``POST https://openrouter.ai/api/v1/images``.

    Used by litellm/llms/openrouter/image_api.py to validate and parse both image
    generation and image edit responses, since both request types answer through this
    same endpoint and payload shape.
    """

    data: tuple[OpenRouterImageData, ...]
    created: int | None = None
    model: str | None = None
    usage: OpenRouterImageUsage | None = None
