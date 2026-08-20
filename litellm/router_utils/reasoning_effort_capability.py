"""Resolve which reasoning_effort values a deployment, and by intersection a model group, accepts.

The model-map flags carry different polarity per level, mirroring the provider gates
(gpt_5_transformation.py restricts xhigh to explicit opt-in and treats minimal/low as opt-out;
anthropic/chat/transformation.py rejects only xhigh/max without an explicit flag): medium and high
are unconditional for any reasoning model, none/minimal/low are supported unless the map explicitly
says false, and xhigh/max require an explicit true. Shipping the resolved list keeps that polarity
in one place instead of re-encoding it in every consumer.
"""

from collections.abc import Mapping, Sequence
from typing import Final

REASONING_EFFORT_CAPABILITY_ORDER: Final = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

_OPT_OUT_FLAGS: Final = (
    ("none", "supports_none_reasoning_effort"),
    ("minimal", "supports_minimal_reasoning_effort"),
    ("low", "supports_low_reasoning_effort"),
)
_OPT_IN_FLAGS: Final = (
    ("xhigh", "supports_xhigh_reasoning_effort"),
    ("max", "supports_max_reasoning_effort"),
)
_UNCONDITIONAL_EFFORTS: Final = frozenset(("medium", "high"))


def resolve_supported_reasoning_efforts(model_info: Mapping[str, object]) -> tuple[str, ...] | None:
    """None = no capability metadata for this deployment (e.g. a model absent from the model map,
    whose stub info carries no supports_reasoning key at all); () = reasoning unsupported."""
    if "supports_reasoning" not in model_info:
        return None
    if model_info.get("supports_reasoning") is not True:
        return ()
    opt_out: Final = frozenset(effort for effort, flag in _OPT_OUT_FLAGS if model_info.get(flag) is not False)
    opt_in: Final = frozenset(effort for effort, flag in _OPT_IN_FLAGS if model_info.get(flag) is True)
    allowed: Final = opt_out | _UNCONDITIONAL_EFFORTS | opt_in
    return tuple(effort for effort in REASONING_EFFORT_CAPABILITY_ORDER if effort in allowed)


def intersect_supported_reasoning_efforts(
    current: Sequence[str] | None,
    resolved: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Deployments without metadata (None) never narrow the group; an effort survives only when
    every deployment with metadata accepts it, so the group offers nothing routing could reject."""
    if resolved is None:
        return tuple(current) if current is not None else None
    if current is None:
        return tuple(resolved)
    keep: Final = frozenset(current) & frozenset(resolved)
    return tuple(effort for effort in REASONING_EFFORT_CAPABILITY_ORDER if effort in keep)
