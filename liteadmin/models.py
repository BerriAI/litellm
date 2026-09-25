from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1, max_length=200)
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=20)
    inference_base_url: str = Field(max_length=2048)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    name: str
    title: str
    arguments: dict[str, JsonValue]
    destructive: bool


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(max_length=64)
    approved: bool


class Completed(BaseModel):
    status: Literal["completed"] = "completed"
    key: str | None = None


class Unknown(BaseModel):
    status: Literal["unknown"] = "unknown"
    message: str = "The change could not be verified. Check the resource before trying again."


ActionResult = Annotated[Completed | Unknown, Field(discriminator="status")]
