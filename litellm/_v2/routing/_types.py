from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Strategy: TypeAlias = Literal["weighted", "least_busy", "latency", "usage"]


class ProviderParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    model: str = Field(min_length=1)
    api_key: str | None = Field(default=None, repr=False)
    api_base: str | None = None
    custom_llm_provider: str | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(min_length=1)
    litellm_params: ProviderParams
    model_info: Mapping[str, JsonValue] = Field(default_factory=dict)
    blocked: bool | None = None


class CooldownPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_failures: int = Field(ge=0, le=2**32 - 1)
    duration: float = Field(gt=0, allow_inf_nan=False)


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    retries: int = Field(default=0, ge=0, le=2**32 - 1)
    fallbacks: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)


class RouterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_list: tuple[ModelConfig, ...]
    strategy: Strategy = "weighted"
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    cooldown: CooldownPolicy | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    deployment_id: str
    model_name: str
    model: str


@dataclass(frozen=True, slots=True)
class SelectionContext:
    model: str
    candidates: tuple[Candidate, ...]
    attempt: int
    session_id: str | None


Selector: TypeAlias = Callable[[SelectionContext], str]


@dataclass(frozen=True, slots=True)
class Snapshot:
    generation: int
    closed: bool
    deployments: tuple[Candidate, ...]
