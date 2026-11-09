from dataclasses import dataclass
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import Field, TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class Spend:
    current: float
    verified: bool
    fallback: float
    limit: float | None
    window: bool
    fallback_authoritative: bool
    strict_before: bool | None = None
    authoritative_read: bool = False
    authoritative: float | None = None
    strict_after: bool | None = None


@dataclass(frozen=True, slots=True)
class ThrottleKey:
    counter_key: str
    token_present: bool
    token: str | None


@dataclass(frozen=True, slots=True)
class Overflow:
    current: float
    limit: float


@dataclass(frozen=True, slots=True)
class OverBudget:
    throttled: bool
    current: float
    limit: float
    reservation: float
    strict: bool


@dataclass(frozen=True, slots=True)
class ReservationFailure:
    strict: bool


@dataclass(frozen=True, slots=True)
class ReservationGroups:
    currents: tuple[float, ...]
    limits: tuple[float, ...]
    reservation: float


@dataclass(frozen=True, slots=True)
class SpendValue:
    kind: Literal["Value", "Repair"]
    spend: float


@dataclass(frozen=True, slots=True)
class Action:
    kind: Literal[
        "ReadStrictBefore",
        "ReadStrictAfter",
        "ReadFloor",
        "Unverifiable",
        "ReadThrottle",
        "Skip",
        "Release",
        "CheckOverflow",
        "Unavailable",
        "Compatible",
        "Inconsistent",
        "RejectResize",
    ]


@dataclass(frozen=True, slots=True)
class Rejected:
    kind: Literal["Reject"]
    current: float
    estimate: float | None


@dataclass(frozen=True, slots=True)
class Resize:
    kind: Literal["Resize"]
    cost: float


@dataclass(frozen=True, slots=True)
class Groups:
    kind: Literal["ReservationGroups"]
    groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Adjustment:
    current: float | None
    adjustment: float


@dataclass(frozen=True, slots=True)
class ResizeConsistency:
    inconsistent: bool
    reseed: bool


Decision: TypeAlias = Annotated[SpendValue | Action | Rejected | Resize | Groups, Field(discriminator="kind")]
_DECISION: Final = TypeAdapter[Decision](Decision)


@native_boundary
def decide(
    facts: Spend
    | ThrottleKey
    | Overflow
    | OverBudget
    | ReservationFailure
    | ReservationGroups
    | Adjustment
    | ResizeConsistency,
) -> Decision:
    return _DECISION.validate_python(_native.request_auth_spend_counter_policy(encode_facts(facts)))
