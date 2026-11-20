from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from litellm.integrations.custom_logger_v2.enums import Hook

StateT: Final = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class CallIds:
    call_id: str
    trace_id: str | None = None
    session_id: str | None = None
    parent_call_id: str | None = None
    connection_id: str | None = None


@dataclass(frozen=True, slots=True)
class CallKind:
    call_type: str
    route: str | None = None
    stream: bool = False
    synchronous: bool = False
    endpoint_type: str | None = None
    internal_origin: str | None = None


@dataclass(frozen=True, slots=True)
class CallTarget:
    model: str
    model_group: str | None = None
    provider: str | None = None
    api_base: str | None = None
    deployment_id: str | None = None
    attempt: int | None = None


@dataclass(frozen=True, slots=True)
class CallTiming:
    start: float
    provider_start: float | None = None
    first_token: float | None = None
    end: float | None = None


@dataclass(frozen=True, slots=True)
class CallFacts:
    ids: CallIds
    kind: CallKind
    target: CallTarget
    timing: CallTiming
    cache_hit: bool | None = None


@dataclass(frozen=True, slots=True)
class CallContext(Generic[StateT]):
    hook: Hook
    call: CallFacts | None
    state: StateT
