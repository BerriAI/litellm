import json
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class OpenRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: Dict


class OpenRouterImageCompletionTokensDetails(BaseModel):
    image_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


class OpenRouterImageUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None
    cost_details: Optional[Mapping[str, Optional[float]]] = None
    completion_tokens_details: Optional[OpenRouterImageCompletionTokensDetails] = None


class OpenRouterImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    media_type: Optional[str] = None
    revised_prompt: Optional[str] = None


class OpenRouterImagesResponse(BaseModel):
    """Response body of ``POST https://openrouter.ai/api/v1/images``."""

    data: Tuple[OpenRouterImageData, ...]
    created: Optional[int] = None
    model: Optional[str] = None
    usage: Optional[OpenRouterImageUsage] = None
