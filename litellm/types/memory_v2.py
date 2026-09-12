from datetime import datetime
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

MemoryTarget: TypeAlias = Literal["gateway", "organization", "team", "project", "user", "key"]
MemoryScope: TypeAlias = Literal["key", "user", "team", "project", "organization"]
MemoryActivation: TypeAlias = Literal["disabled", "opt_in", "automatic"]


class MemoryPolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_type: MemoryTarget
    target_id: str = Field(min_length=1, max_length=256)
    activation: MemoryActivation
    scope: MemoryScope = "key"

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.target_type == "gateway" and self.target_id != "*":
            raise ValueError("The gateway target_id must be '*'")
        if self.target_type == "key" and (
            len(self.target_id) != 64 or any(c not in "0123456789abcdef" for c in self.target_id)
        ):
            raise ValueError("Use the key's hash, never its secret value")
        return self


class MemoryPolicy(MemoryPolicyInput):
    policy_id: str
    updated_at: datetime
    updated_by: str


class MemoryPreference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool


class MemoryStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    active: bool
    activation: MemoryActivation
    scope: MemoryScope | None
    opted_in: bool
    policy_id: str | None


class MemoryCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=160, pattern=r"^[a-zA-Z0-9_.-]+$")
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=8000)
    evidence: str = Field(min_length=1, max_length=2000)
    expected_revision: datetime | None = None


class MemoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    key: str
    title: str
    content: str
    evidence: str
    updated_at: datetime


class MemorySearch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(default="", max_length=500)
    limit: int = Field(default=8, ge=1, le=20)
    offset: int = Field(default=0, ge=0)


class MemoryRead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str = Field(min_length=1, max_length=64)
