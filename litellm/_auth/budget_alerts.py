from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class Failure:
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class ThresholdFacts:
    key: str
    digit: bool
    digits: str | None = None
    ratio: float | None = None
    integer_failure: Failure | None = None
    ratio_failure: Failure | None = None
    text: str | None = None
    items: tuple[str | None, ...] | None = None


@dataclass(frozen=True, slots=True)
class ConfigFacts:
    global_entries: tuple[ThresholdFacts, ...] = ()
    local_entries: tuple[ThresholdFacts, ...] = ()
    global_failure: Failure | None = None
    local_failure: Failure | None = None
    order: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Facts:
    member: bool
    spend: float | None
    limit: float | None
    default_fraction: float
    config: ConfigFacts | None = None


@dataclass(frozen=True, slots=True)
class Threshold:
    key: str
    recipients: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    read_config: bool = False
    read_order: bool = False
    global_keys: tuple[str, ...] = ()
    local_keys: tuple[str, ...] = ()
    emit: bool = False
    thresholds: tuple[Threshold, ...] | None = None
    failure: Failure | None = None


_FACTS: Final = TypeAdapter(Facts)
_DECISION: Final = TypeAdapter(Decision)


@native_boundary
def plan(facts: Facts) -> Decision:
    return _DECISION.validate_python(_native.request_auth_budget_alert_plan(_FACTS.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts
