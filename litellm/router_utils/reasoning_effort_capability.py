"""Resolve which reasoning_effort values a deployment, and by intersection a model group, accepts.

The model map's supports_*_reasoning_effort flags are the only signal, and each level's polarity
mirrors how a request path reads that same flag. medium and high are unconditional for a reasoning
model. minimal and low are opt-out: openai/chat/gpt_5_transformation.py refuses them only when the
map says false. xhigh and max are opt-in. none is opt-out everywhere except the azure gpt-5 family,
whose config raises UnsupportedParamsError without an explicit true.

xhigh is gated on the request path by the openai and azure gpt-5 configs. max is not gated there at
all: every entry carrying supports_max_reasoning_effort is Claude-family, and
anthropic/chat/transformation.py gates max on the output_config path while its reasoning_effort
path maps any level to a thinking budget. Making max opt-in is a deliberate trade, then, since an
explicit flag is the only signal that the tier is a real one rather than litellm rounding the level
to a budget, and a missing flag costs advisory metadata rather than a rejected request.

A deployment the map describes with no effort flags at all resolves to None rather than to the
opt-out defaults. 689 of the map's 854 reasoning entries carry no flag, and the o-series, xai and
bedrock nova entries among them take neither none nor minimal, so composing a set out of the
defaults alone would advertise levels those providers reject.

The advertisement order is the REASONING_EFFORT declaration order, which is presentation only. It
is not a strength scale and does not reconcile with bedrock's output_config ceiling order in
llms/bedrock/common_utils.py, which ranks max below xhigh while the thinking-budget constants rank
it above.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, get_args

import litellm
from litellm.types.llms.openai import REASONING_EFFORT

REASONING_EFFORT_ADVERTISEMENT_ORDER: Final = get_args(REASONING_EFFORT)
_EMPTY_ENTRY: Final[Mapping[str, object]] = MappingProxyType({})

_EFFORT_FLAGS: Final = (
    ("none", "supports_none_reasoning_effort"),
    ("minimal", "supports_minimal_reasoning_effort"),
    ("low", "supports_low_reasoning_effort"),
    ("xhigh", "supports_xhigh_reasoning_effort"),
    ("max", "supports_max_reasoning_effort"),
)
_OPT_OUT_EFFORTS: Final = ("minimal", "low")
_OPT_IN_EFFORTS: Final = ("xhigh", "max")
_UNCONDITIONAL_EFFORTS: Final = frozenset(("medium", "high"))


def _bare_model_entry(model_info: Mapping[str, object]) -> Mapping[str, object]:
    """The unprefixed twin of a provider-prefixed map entry, which is where the flags often live:
    azure/gpt-5-mini carries none of them while gpt-5-mini carries all three. The request-path
    gates resolve through the same twin (_supports_factory, #20885), so reading it here is what
    keeps the advertisement and the gate on the same answer."""
    key: Final = model_info.get("key")
    provider: Final = model_info.get("litellm_provider")
    if not isinstance(key, str) or not isinstance(provider, str) or not key.startswith(f"{provider}/"):
        return _EMPTY_ENTRY
    entry: Final[Mapping[str, object] | None] = litellm.model_cost.get(key.removeprefix(f"{provider}/"))
    return entry if entry is not None else _EMPTY_ENTRY


def _declared_effort_flags(model_info: Mapping[str, object]) -> Mapping[str, object]:
    bare: Final = _bare_model_entry(model_info)
    return MappingProxyType(
        {
            effort: model_info.get(flag) if model_info.get(flag) is not None else bare.get(flag)
            for effort, flag in _EFFORT_FLAGS
        }
    )


def _supports_none_reasoning_effort(model_info: Mapping[str, object], flag: object) -> bool:
    """Opt-in only where a request path refuses the level. AzureOpenAIGPT5Config raises
    UnsupportedParamsError on reasoning_effort='none' without an explicit true, and it is selected
    only for the gpt-5 family, so every other azure deployment keeps the opt-out default."""
    if model_info.get("litellm_provider") != "azure":
        return flag is not False

    from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config

    key: Final = model_info.get("key")
    if not isinstance(key, str) or not AzureOpenAIGPT5Config.is_model_gpt_5_model(key):
        return flag is not False
    return flag is True


def deployment_is_catalog_mapped(
    resolved_model_info: Mapping[str, object] | None,
    operator_model_info: Mapping[str, object],
) -> bool:
    """Whether the model map described this deployment, as opposed to the operator describing it.

    Every deployment is registered in the cost map under its own id, so a mode the operator wrote
    on an off-map deployment reads back here exactly like one the catalog supplied. Excluding it is
    what stops such a deployment from claiming to be a known non-reasoning model and emptying the
    levels its mapped siblings agree on.
    """
    if resolved_model_info is None or resolved_model_info.get("mode") is None:
        return False
    return operator_model_info.get("mode") is None


def resolve_supported_reasoning_efforts(
    model_info: Mapping[str, object],
    *,
    deployment_is_mapped: bool,
) -> tuple[str, ...] | None:
    """None = nothing is known about this deployment, so it must not narrow its group; () = a known
    model that accepts no effort level, which correctly empties the group.

    Telling those apart needs provenance the flattened ModelInfo does not carry. A deployment the
    map does not describe arrives with supports_reasoning None, exactly like a mapped non-reasoning
    model: 2273 of the map's 3165 entries omit the key rather than setting it false, so reading an
    unset flag as () would let one custom deployment empty every level its mapped siblings agree
    on. deployment_is_mapped is that provenance, and an operator who wants either answer for an
    off-map deployment gets it by setting supports_reasoning explicitly.
    """
    supports_reasoning: Final = model_info.get("supports_reasoning")
    if supports_reasoning is not True:
        return () if supports_reasoning is False or deployment_is_mapped else None

    flags: Final = _declared_effort_flags(model_info)
    if all(value is None for value in flags.values()):
        return None

    opt_out: Final = frozenset(effort for effort in _OPT_OUT_EFFORTS if flags[effort] is not False)
    opt_in: Final = frozenset(effort for effort in _OPT_IN_EFFORTS if flags[effort] is True)
    none_level: Final = (
        frozenset(("none",)) if _supports_none_reasoning_effort(model_info, flags["none"]) else frozenset()
    )
    allowed: Final = opt_out | _UNCONDITIONAL_EFFORTS | opt_in | none_level
    return tuple(effort for effort in REASONING_EFFORT_ADVERTISEMENT_ORDER if effort in allowed)


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
    return tuple(effort for effort in REASONING_EFFORT_ADVERTISEMENT_ORDER if effort in keep)
