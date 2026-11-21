from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Focus: TypeAlias = Literal["general", "math", "coding", "agentic"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Priorities(FrozenModel):
    intelligence: int = Field(default=2, ge=0, le=3)
    speed: int = Field(default=2, ge=0, le=3)
    cost: int = Field(default=2, ge=0, le=3)


class Requirements(FrozenModel):
    reasoning: bool = False
    vision: bool = False
    function_calling: bool = False
    min_context_tokens: int | None = Field(default=None, ge=0)


class RecommendationRequest(FrozenModel):
    priorities: Priorities = Field(default_factory=Priorities)
    focus: frozenset[Focus] = frozenset({"general"})
    requirements: Requirements = Field(default_factory=Requirements)
    limit: int = Field(default=3, ge=1, le=20)
    include_external: bool = False


class Candidate(FrozenModel):
    model_group: str
    on_gateway: bool = True
    providers: tuple[str, ...] = ()
    underlying_models: tuple[str, ...] = ()
    mode: str | None = None
    supports_reasoning: bool = False
    supports_vision: bool = False
    supports_function_calling: bool = False
    max_input_tokens: float | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None


class Speed(FrozenModel):
    model_group: str
    latency_seconds: float | None = None
    output_tokens_per_second: float | None = None


class Signal(FrozenModel):
    raw: float | None
    normalized: float
    unknown: bool


class Signals(FrozenModel):
    intelligence: Signal
    speed: Signal
    cost: Signal


class Recommendation(FrozenModel):
    model_group: str
    on_gateway: bool = True
    providers: tuple[str, ...]
    score: float
    signals: Signals
    latency_seconds: float | None
    output_tokens_per_second: float | None
    capabilities: Requirements
    context_window: float | None
    reasons: tuple[str, ...]


class DataSource(FrozenModel):
    name: str
    url: str | None = None
    as_of: str | None = None
    available: bool
    description: str


class RecommendationResponse(FrozenModel):
    recommendations: tuple[Recommendation, ...]
    candidates_considered: int
    candidates_filtered: int
    data_sources: tuple[DataSource, ...]
