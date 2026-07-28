from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


ClassificationInput: TypeAlias = str | list[str] | list[int] | list[list[int]]


class ClassificationRequest(BaseModel):
    model: str
    input: ClassificationInput
    user: str | None = None
    truncate_prompt_tokens: int | None = Field(default=None, ge=-1)
    truncation_side: Literal["left", "right"] | None = None
    priority: int = 0
    add_special_tokens: bool = True
    request_id: str | None = None
    use_activation: bool | None = None


class ClassificationData(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    label: str | None = None
    probs: list[float]
    num_classes: int


class ClassificationUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "list"
    created: int
    model: str
    data: list[ClassificationData]
    usage: ClassificationUsage
