"""Bounded scatter/gather over dataplanes. A dataplane that fails is a Missed value in the result, never an exception."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError

from litellm.proxy.control_plane.dataplanes.client import CallResult, DataplaneReply, DataplaneUnreachable
from litellm.proxy.control_plane.dataplanes.credentials import CredentialUnavailable
from litellm.proxy.control_plane.dataplanes.registry import Dataplane

T: Final = TypeVar("T")
M: Final = TypeVar("M")


@dataclass(frozen=True, slots=True)
class Failure:
    reason: str


@dataclass(frozen=True, slots=True)
class Missed:
    dataplane: Dataplane
    reason: str


@dataclass(frozen=True, slots=True)
class Reached(Generic[T]):
    dataplane: Dataplane
    value: T


@dataclass(frozen=True, slots=True)
class Gathered(Generic[T]):
    reached: tuple[Reached[T], ...]
    missed: tuple[Missed, ...]

    @property
    def partial(self) -> bool:
        return bool(self.missed)

    @property
    def nothing_answered(self) -> bool:
        return bool(self.missed) and not self.reached


@dataclass(frozen=True, slots=True)
class Fanout:
    concurrency: int

    async def gather(
        self, dataplanes: tuple[Dataplane, ...], fetch: Callable[[Dataplane], Awaitable[T | Failure]]
    ) -> Gathered[T]:
        gate: Final = asyncio.Semaphore(self.concurrency)

        async def bounded(dataplane: Dataplane) -> Reached[T] | Missed:
            async with gate:
                outcome: Final = await fetch(dataplane)
            if isinstance(outcome, Failure):
                return Missed(dataplane, outcome.reason)
            return Reached(dataplane, outcome)

        outcomes: Final = await asyncio.gather(*(bounded(dataplane) for dataplane in dataplanes))
        return Gathered(
            reached=tuple(outcome for outcome in outcomes if isinstance(outcome, Reached)),
            missed=tuple(outcome for outcome in outcomes if isinstance(outcome, Missed)),
        )


def parse_reply(result: CallResult, shape: TypeAdapter[M]) -> M | Failure:
    match result:
        case DataplaneReply(status=status, body=body) if status < 400:
            try:
                return shape.validate_python(body)
            except ValidationError as error:
                return Failure(f"unexpected response shape: {error.error_count()} validation error(s)")
        case DataplaneReply(status=status, credential=credential):
            return Failure(f"HTTP {status} with the {credential} credential")
        case DataplaneUnreachable(detail=detail, attempts=attempts):
            return Failure(f"unreachable after {attempts} attempt(s): {detail}")
        case CredentialUnavailable(reason=reason):
            return Failure(f"no usable credential: {reason}")


def describe_missed(missed: tuple[Missed, ...]) -> str:
    return ", ".join(f"{miss.dataplane.id}: {miss.reason}" for miss in missed)
