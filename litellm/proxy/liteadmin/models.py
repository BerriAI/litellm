from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1, max_length=200)
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=20)
    inference_base_url: str = Field(max_length=2048)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(max_length=64)
    approved: bool
