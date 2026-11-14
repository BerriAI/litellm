"""How strong each reasoning_effort level is, and rounding a requested level onto a declared set.

Imports nothing from the SDK, so the `lite` CLI, which loads without initializing litellm, can use it.
"""

from collections.abc import Sequence
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from typing_extensions import assert_never

ReasoningEffortRounding: TypeAlias = Literal["up", "down"]

REASONING_EFFORT_STRENGTH_ORDER: Final = ("minimal", "low", "medium", "high", "xhigh", "max")
REASONING_EFFORT_STRENGTH_RANK: Final = MappingProxyType(
    {effort: rank for rank, effort in enumerate(REASONING_EFFORT_STRENGTH_ORDER)}
)


def nearest_declared_reasoning_effort(
    requested: str, declared: Sequence[str], rounding: ReasoningEffortRounding = "up"
) -> str:
    """Rounding up takes the weakest declared level at least as strong as the request, else the
    strongest declared, so the caller gets no less reasoning than it asked for instead of a rejected
    call. Rounding down mirrors it: the strongest declared level at most as strong, else the
    weakest. none is the off switch rather than a strength, so it is never rounded onto the ladder
    and no level is rounded to it: a caller who turned reasoning off must not be billed for it, and
    a model that cannot turn it off says so itself. A level outside the strength order is likewise
    returned as is for upstream to judge."""
    ranked: Final = sorted(
        (effort for effort in declared if effort in REASONING_EFFORT_STRENGTH_RANK),
        key=lambda effort: REASONING_EFFORT_STRENGTH_RANK[effort],
    )
    if requested in ranked or requested not in REASONING_EFFORT_STRENGTH_RANK or not ranked:
        return requested
    strength: Final = REASONING_EFFORT_STRENGTH_RANK[requested]
    match rounding:
        case "up":
            return next((effort for effort in ranked if REASONING_EFFORT_STRENGTH_RANK[effort] >= strength), ranked[-1])
        case "down":
            return next(
                (effort for effort in reversed(ranked) if REASONING_EFFORT_STRENGTH_RANK[effort] <= strength), ranked[0]
            )
        case _:
            assert_never(rounding)
