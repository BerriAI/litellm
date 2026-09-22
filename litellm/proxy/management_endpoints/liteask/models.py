from collections.abc import Mapping
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class LiteAskMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class LiteAskChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    conversation_id: UUID
    messages: tuple[LiteAskMessage, ...] = Field(min_length=1, max_length=40)


class LiteAskApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    conversation_id: UUID
    token: str = Field(min_length=1, max_length=50000)


class LiteAskConfig(BaseModel):
    enabled: bool
    model: str | None
    can_execute_mutations: bool


class LiteAskProposal(BaseModel):
    token: str
    tool: str
    title: str
    arguments: Mapping[str, JsonValue]
    expires_at: int


class LiteAskResponse(BaseModel):
    message: str
    proposal: LiteAskProposal | None = None
    result: JsonValue = None
    generated_key: str | None = None
