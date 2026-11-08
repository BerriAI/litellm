from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class Gate:
    token_present: bool
    llm_route: bool | None = None
    unbilled: bool | None = None
    model_present: bool | None = None
    counter_count: int | None = None
    cost_read: bool = False
    cost: float | None = None


@dataclass(frozen=True, slots=True)
class Sources:
    key_present: bool
    team_present: bool
    team_id: str | None
    user_present: bool
    user_id: str | None
    apply_user_budget_to_team_keys: bool


Scope: TypeAlias = Literal["Key", "Team", "User", "EndUser", "Tag", "AccessGroup", "Member", "Organization", "Project"]


@dataclass(frozen=True, slots=True)
class Counter:
    scope: Scope
    limit: float | None


@dataclass(frozen=True, slots=True)
class Window:
    duration_present: bool
    limit: float | None
    start_present: bool | None = None


@dataclass(frozen=True, slots=True)
class Action:
    kind: Literal[
        "ReadRoute",
        "ReadUnbilled",
        "ReadModel",
        "ReadCounters",
        "ReadEstimate",
        "Reserve",
        "Skip",
        "ReadStart",
        "WarnWindow",
        "AppendWindow",
    ]


@dataclass(frozen=True, slots=True)
class SourcePlan:
    key: bool
    team: bool
    user: bool
    kind: Literal["Sources"] = "Sources"


@dataclass(frozen=True, slots=True)
class CounterPlan:
    include: bool
    kind: Literal["Counter"] = "Counter"


_ACTION: Final = TypeAdapter(Action)
_SOURCES: Final = TypeAdapter(SourcePlan)
_COUNTER: Final = TypeAdapter(CounterPlan)


@native_boundary
def action(facts: Gate | Window) -> Action:
    return _ACTION.validate_python(_native.request_auth_reservation(encode_facts(facts)))


@native_boundary
def sources(facts: Sources) -> SourcePlan:
    return _SOURCES.validate_python(_native.request_auth_reservation(encode_facts(facts)))


@native_boundary
def counter(scope: Scope, limit: float | None) -> bool:
    return _COUNTER.validate_python(_native.request_auth_reservation(encode_facts(Counter(scope, limit)))).include
