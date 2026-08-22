"""Resolve which reasoning_effort values a deployment, and by intersection a model group, accepts.

The model-map flags carry different polarity per level, mirroring the provider gates
(gpt_5_transformation.py restricts xhigh to explicit opt-in and treats minimal/low as opt-out;
anthropic/chat/transformation.py rejects only xhigh/max without an explicit flag): medium and high
are unconditional for any reasoning model, minimal/low are supported unless the map explicitly
says false, and xhigh/max require an explicit true. Shipping the resolved list keeps that polarity
in one place instead of re-encoding it in every consumer.

The none level is the one flag whose polarity is provider-dependent. OpenAI never refuses it on the
request path (azure/chat/gpt_5_transformation.py is the only caller that does, and it raises
UnsupportedParamsError unless supports_none_reasoning_effort is explicitly true), so none is opt-out
everywhere except azure, where it is opt-in. Resolving it the other way for azure would advertise a
level litellm itself rejects, which is the failure this module exists to prevent.
"""

from collections.abc import Mapping, Sequence
from typing import Final

REASONING_EFFORT_CAPABILITY_ORDER: Final = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

_OPT_OUT_FLAGS: Final = (
    ("minimal", "supports_minimal_reasoning_effort"),
    ("low", "supports_low_reasoning_effort"),
)
_OPT_IN_FLAGS: Final = (
    ("xhigh", "supports_xhigh_reasoning_effort"),
    ("max", "supports_max_reasoning_effort"),
    ("ultra", "supports_ultra_reasoning_effort"),
)
_UNCONDITIONAL_EFFORTS: Final = frozenset(("medium", "high"))
_NONE_FLAG: Final = "supports_none_reasoning_effort"
_NONE_OPT_IN_PROVIDERS: Final = frozenset(("azure",))


def _supports_none_reasoning_effort(model_info: Mapping[str, object]) -> bool:
    """Opt-in on azure, whose gpt-5 config raises on reasoning_effort='none' without an explicit
    true; opt-out elsewhere, where no request path refuses the level. A missing azure flag defers to
    _supports_factory, the same resolver the azure gate calls, so its bare-model-name fallback
    (azure/gpt-5.2 inheriting the flag from gpt-5.2) reaches both sides alike."""
    flag: Final = model_info.get(_NONE_FLAG)
    if model_info.get("litellm_provider") not in _NONE_OPT_IN_PROVIDERS:
        return flag is not False
    if flag is not None:
        return flag is True
    model_key: Final = model_info.get("key")
    if not isinstance(model_key, str):
        return False
    from litellm.utils import (
        _supports_factory,  # pyright: ignore[reportPrivateUsage]  # the resolver the azure gate itself calls; a public wrapper would fork the fallback
    )

    return _supports_factory(model=model_key, custom_llm_provider=None, key=_NONE_FLAG)


def resolve_supported_reasoning_efforts(model_info: Mapping[str, object]) -> tuple[str, ...] | None:
    """None = nothing is known about this deployment, so it must not narrow its group; () = this is a
    known model that accepts no effort level, which correctly empties the group. Keeping the two
    apart matters because the router registers a deployment absent from the model map under a
    synthesized entry, and get_model_info then answers with supports_reasoning None exactly as it
    does for a mapped non-reasoning model. That entry carries no mode, which every real map entry for
    a routable model does, so an unset flag with no mode is the unknown case and one custom model in
    a group no longer wipes the levels its mapped siblings agree on."""
    if "supports_reasoning" not in model_info:
        return None
    supports_reasoning: Final = model_info.get("supports_reasoning")
    if supports_reasoning is None and model_info.get("mode") is None:
        return None
    if supports_reasoning is not True:
        return ()
    opt_out: Final = frozenset(effort for effort, flag in _OPT_OUT_FLAGS if model_info.get(flag) is not False)
    opt_in: Final = frozenset(effort for effort, flag in _OPT_IN_FLAGS if model_info.get(flag) is True)
    none_level: Final = frozenset(("none",)) if _supports_none_reasoning_effort(model_info) else frozenset()
    allowed: Final = opt_out | _UNCONDITIONAL_EFFORTS | opt_in | none_level
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
