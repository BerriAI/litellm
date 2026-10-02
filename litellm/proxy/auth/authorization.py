from dataclasses import dataclass
from typing import Generic, TypeVar

GrantT = TypeVar("GrantT", covariant=True)


@dataclass(frozen=True, slots=True)
class AnyOf(Generic[GrantT]):
    grants: tuple[GrantT, ...]


def any_of(*grants: GrantT) -> AnyOf[GrantT]:
    return AnyOf(grants)
