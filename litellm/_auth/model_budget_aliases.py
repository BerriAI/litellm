from dataclasses import dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.model_budget_data import ModelBudgetSnapshot
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class Row:
    key: str
    names: tuple[str, ...]
    prefix_known: bool | None = None
    provider_read: bool = False
    provider: str | None = None


@dataclass(frozen=True, slots=True)
class Facts:
    model: str
    rows: tuple[Row, ...] = ()


@dataclass(frozen=True, slots=True)
class Decision:
    read: Literal["Entry", "Prefix", "Provider"] | None
    key: str
    models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScopeFacts:
    models: tuple[str, ...]
    snapshots: tuple[ModelBudgetSnapshot, ...] = ()
    budget_present: bool = True


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    read_index: int | None
    result_index: int | None


_FACTS: Final = TypeAdapter(Facts)
_DECISION: Final = TypeAdapter(Decision)
_SCOPE: Final = TypeAdapter(ScopeFacts)
_SCOPE_DECISION: Final = TypeAdapter(ScopeDecision)


@native_boundary
def decide(facts: Facts) -> Decision:
    return _DECISION.validate_python(_native.request_auth_model_budget_aliases(_FACTS.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes typed native facts


@native_boundary
def scope(facts: ScopeFacts) -> ScopeDecision:
    return _SCOPE_DECISION.validate_python(_native.request_auth_model_budget_scope(_SCOPE.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes typed native facts
