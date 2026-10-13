from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from litellm.constants import VIRTUAL_MODEL_ORDERED_STRATEGY, VIRTUAL_MODEL_PREFIX


def _validate_virtual_model_name(name: str) -> str:
    if not name:
        raise ValueError("virtual_model_name must be non-empty")
    if "/" in name:
        raise ValueError(f"virtual_model_name must not contain '/', got: {name}")
    if name.startswith(VIRTUAL_MODEL_PREFIX):
        raise ValueError(
            f"virtual_model_name must not start with reserved prefix '{VIRTUAL_MODEL_PREFIX}', got: {name}"
        )
    return name


VirtualModelRoutingStrategy: TypeAlias = Literal[
    "ordered",
    "simple-shuffle",
    "least-busy",
    "usage-based-routing",
    "usage-based-routing-v2",
    "latency-based-routing",
    "cost-based-routing",
]


class VirtualModelMemberRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    priority: Annotated[int, Field(ge=1, le=1000)] = 1


class VirtualModelCreateRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    virtual_model_name: str
    description: str | None = None
    members: Annotated[tuple[VirtualModelMemberRequest, ...], Field(min_length=1, max_length=50)]
    routing_strategy: VirtualModelRoutingStrategy = VIRTUAL_MODEL_ORDERED_STRATEGY
    num_retries: Annotated[int | None, Field(ge=0, le=3)] = None
    timeout: int | None = None
    stream_timeout: int | None = None
    allowed_fails: int | None = None
    cooldown_time: int | None = None
    team_id: str | None = None

    @field_validator("virtual_model_name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return _validate_virtual_model_name(v)


class VirtualModelPatchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    virtual_model_name: str | None = None
    description: str | None = None
    add_members: tuple[VirtualModelMemberRequest, ...] | None = None
    remove_members: tuple[str, ...] | None = None
    routing_strategy: VirtualModelRoutingStrategy | None = None
    num_retries: Annotated[int | None, Field(ge=0, le=3)] = None
    timeout: int | None = None
    stream_timeout: int | None = None
    allowed_fails: int | None = None
    cooldown_time: int | None = None
    team_id: str | None = None

    @field_validator("virtual_model_name")
    @classmethod
    def _validate_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_virtual_model_name(v)


class VirtualModelMemberResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    model_name: str | None = None
    priority: int


class VirtualModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    virtual_model_id: str
    virtual_model_name: str
    description: str | None = None
    members: tuple[VirtualModelMemberResponse, ...]
    routing_strategy: str
    num_retries: int | None = None
    timeout: int | None = None
    stream_timeout: int | None = None
    allowed_fails: int | None = None
    cooldown_time: int | None = None
    team_id: str | None = None
    live_member_count: int
    created_at: str | None = None
    created_by: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None
