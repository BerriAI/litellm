from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm.rust_bridge import _native

ModelBudgetEntity: TypeAlias = Literal["Key", "User", "Team", "TeamMember", "EndUser"]


@dataclass(frozen=True, slots=True)
class ModelBudgetEntry:
    name: str
    max_budget: float | None
    duration: str | None
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class ModelBudgetResolveFacts:
    model: str
    bedrock_base: str
    bedrock_provider: str
    entries: tuple[ModelBudgetEntry, ...]
    key_entries: tuple[ModelBudgetEntry, ...] = ()
    team: bool = False
    access_groups: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ModelBudgetResolution:
    read_groups: bool
    entry: ModelBudgetEntry | None
    named_model: bool


@dataclass(frozen=True, slots=True)
class ModelBudgetSnapshot:
    max_budget: float | None = None
    local_spend: float | None = None
    legacy_spend: float | None = None
    shared_spend: float | None = None
    shared_started_at: float | None = None
    shared_window_seconds: float | None = None
    now: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelBudgetQueryFacts:
    entity: ModelBudgetEntity
    snapshot: ModelBudgetSnapshot
    duration_seconds: float | None = None
    process_uptime: float = 0.0
    same_counter: bool = False
    redis_present: bool = False
    shared_read: bool = False


@dataclass(frozen=True, slots=True)
class ModelBudgetDecision:
    read: Literal["Local", "Legacy", "Shared"] | None
    denied: bool
    current_spend: float


_RESOLVE: Final = TypeAdapter(ModelBudgetResolveFacts)
_RESOLUTION: Final = TypeAdapter(ModelBudgetResolution)
_QUERY: Final = TypeAdapter(ModelBudgetQueryFacts)
_SNAPSHOT: Final = TypeAdapter(ModelBudgetSnapshot)
_DECISION: Final = TypeAdapter(ModelBudgetDecision)


def resolve(facts: ModelBudgetResolveFacts) -> ModelBudgetResolution:
    return _RESOLUTION.validate_python(_native.request_auth_model_budget_resolve(_RESOLVE.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary


def query(facts: ModelBudgetQueryFacts) -> ModelBudgetDecision:
    return _DECISION.validate_python(_native.request_auth_model_budget_query(_QUERY.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary


def check(facts: ModelBudgetSnapshot) -> ModelBudgetDecision:
    return _DECISION.validate_python(_native.request_auth_model_budget_check(_SNAPSHOT.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes the validated frozen facts for the native boundary
