import re
from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

MemoryTarget: TypeAlias = Literal["gateway", "organization", "team", "project", "user", "key"]
MemoryScope: TypeAlias = Literal["key", "user", "team", "project", "organization"]
MemoryActivation: TypeAlias = Literal["disabled", "opt_in", "automatic"]
MemoryKind: TypeAlias = Literal["workflow", "decision", "correction", "learning", "context", "disagreement"]
MemoryCertainty: TypeAlias = Literal["user_stated", "observed", "inferred"]


def _validate_search_query(value: str) -> str:
    if len(frozenset(re.findall(r"[\w-]{2,}", value.casefold()))) > 16:
        raise ValueError("Memory search accepts at most 16 distinct search terms")
    return value


MemoryQuery: TypeAlias = Annotated[
    str, AfterValidator(_validate_search_query), Field(description="Use at most 16 distinct search terms")
]


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
    when_to_use: str = Field(default="", max_length=700)
    scope: str = Field(default="", max_length=200)
    kind: MemoryKind = "context"
    certainty: MemoryCertainty = "observed"
    source: str = Field(default="", max_length=1000)


class MemoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    key: str
    title: str
    content: str
    evidence: str
    updated_at: datetime
    created_at: datetime | None = None
    actor: str | None = None
    when_to_use: str = ""
    scope: str = ""
    kind: MemoryKind = "context"
    certainty: MemoryCertainty = "observed"
    source: str = ""


class MemorySearch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: MemoryQuery = Field(default="", max_length=500)
    limit: int = Field(default=8, ge=1, le=20)
    offset: int = Field(default=0, ge=0)


class MemoryRead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str = Field(min_length=1, max_length=64)


class MemoryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=3, max_length=180)
    when_to_use: str = Field(min_length=5, max_length=700)
    content: str = Field(min_length=10, max_length=6000)
    kind: MemoryKind
    scope: str = Field(min_length=1, max_length=200)
    certainty: MemoryCertainty
    evidence: str = Field(min_length=5, max_length=2000)
    source: str = Field(min_length=3, max_length=1000)


class MemoryObservationCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observations: tuple[MemoryObservation, ...] = Field(max_length=8)
    checkpoint: str | None = Field(default=None, max_length=200)


class MemoryCatalogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class MemoryRecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: MemoryQuery = Field(default="", max_length=2000)
    scope: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=8, ge=1, le=30)


class MemoryReadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
