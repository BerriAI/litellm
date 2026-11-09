from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Final, NoReturn

from litellm._auth import spend_counter_policy as core
from litellm._auth.failure import NativeAuthRuntimeError


@dataclass(frozen=True, slots=True)
class SpendIO:
    read_strict: Callable[[], bool]
    read_floor: Callable[[], Awaitable[float | None]]
    repair: Callable[[float], Awaitable[None]]
    unavailable: Callable[[], NoReturn]


async def resolve_spend(facts: core.Spend, io: SpendIO) -> float:
    decision: Final = core.decide(facts)
    match decision:
        case core.SpendValue(kind="Value", spend=spend):
            return spend
        case core.SpendValue(kind="Repair", spend=spend):
            await io.repair(spend)
            return spend
        case core.Action(kind="ReadStrictBefore"):
            return await resolve_spend(replace(facts, strict_before=io.read_strict()), io)
        case core.Action(kind="ReadStrictAfter"):
            return await resolve_spend(replace(facts, strict_after=io.read_strict()), io)
        case core.Action(kind="ReadFloor"):
            return await resolve_spend(replace(facts, authoritative_read=True, authoritative=await io.read_floor()), io)
        case core.Action(kind="Unverifiable"):
            io.unavailable()
        case _:
            raise NativeAuthRuntimeError("Unexpected spend-counter policy decision")
