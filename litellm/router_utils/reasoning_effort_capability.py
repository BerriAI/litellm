"""Resolve which reasoning_effort values a deployment, and by intersection a model group, accepts.

An entry that states its levels outright in reasoning_effort_levels is read first and wins
whole, for a model whose set the per-level flags cannot express: Kimi K3 takes low, high and max,
and no flag can drop medium because medium has none. Every other entry answers through the
supports_*_reasoning_effort flags below, whose polarity mirrors how a request path reads that same
flag. medium and high are unconditional for a reasoning model. minimal and low are opt-out:
openai/chat/gpt_5_transformation.py refuses them only when the map says false. xhigh and max are
opt-in. none is opt-out everywhere except the azure gpt-5 family, whose config raises
UnsupportedParamsError without an explicit true.

xhigh is gated on the request path by the openai and azure gpt-5 configs. max is not gated there at
all: outside the gpt-6-astra rows every entry carrying supports_max_reasoning_effort is Claude-family,
and anthropic/chat/transformation.py gates max on the output_config path while its reasoning_effort
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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, get_args

from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params
from litellm.litellm_core_utils.reasoning_effort_utils import reasoning_effort_from_thinking
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.types.llms.openai import REASONING_EFFORT

ReasoningEffortRounding: TypeAlias = Literal["up", "down"]
RequestedReasoningEffort: TypeAlias = str | Mapping[str, object] | None

REASONING_EFFORT_ADVERTISEMENT_ORDER: Final[tuple[str, ...]] = get_args(REASONING_EFFORT)
_EMPTY_ENTRY: Final[Mapping[str, object]] = MappingProxyType({})

_EFFORT_FLAGS: Final = (
    ("none", "supports_none_reasoning_effort"),
    ("minimal", "supports_minimal_reasoning_effort"),
    ("low", "supports_low_reasoning_effort"),
    ("xhigh", "supports_xhigh_reasoning_effort"),
    ("max", "supports_max_reasoning_effort"),
)
_DECLARED_EFFORTS_KEY: Final = "reasoning_effort_levels"
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


def declared_reasoning_efforts(model_info: Mapping[str, object]) -> tuple[str, ...] | None:
    """The entry's own answer, read through the same bare twin as the flags so both spellings of one
    model agree. Present-and-a-list IS the answer, so a declared [] correctly empties the group and
    an unknown level is dropped rather than raised: the bundled map is enum-validated by
    validate-model-prices-json, but an operator can put this key on a config.yaml model_info block
    where that schema never runs, and one mistyped level must not fail every sibling on the proxy."""
    own: Final = model_info.get(_DECLARED_EFFORTS_KEY)
    raw: Final = own if own is not None else _bare_model_entry(model_info).get(_DECLARED_EFFORTS_KEY)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    declared: Final = frozenset(effort for effort in raw if isinstance(effort, str))
    return tuple(effort for effort in REASONING_EFFORT_ADVERTISEMENT_ORDER if effort in declared)


def declared_reasoning_efforts_for_model(model: str, custom_llm_provider: str) -> tuple[str, ...] | None:
    """The levels an entry declares, resolved from the model string a provider config holds rather
    than from a router deployment's model_info.

    None means the map has no opinion, either because the entry declares nothing or because it
    describes no such model, so a caller keeps whatever it did before the entry was described. The
    entry is read straight off the map rather than through get_model_info, which raises for a model
    it does not know: a provider config runs on the request path for every model it serves, most of
    which the map never named, and a lookup miss there must not fail the call.
    """
    entry: Final = litellm.model_cost.get(f"{custom_llm_provider}/{model}") or litellm.model_cost.get(model)
    if not isinstance(entry, dict):
        return None
    return declared_reasoning_efforts(entry)


REASONING_EFFORT_STRENGTH_ORDER: Final = ("minimal", "low", "medium", "high", "xhigh", "max")
_STRENGTH_RANK: Final = MappingProxyType({effort: rank for rank, effort in enumerate(REASONING_EFFORT_STRENGTH_ORDER)})


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
        (effort for effort in declared if effort in _STRENGTH_RANK), key=lambda effort: _STRENGTH_RANK[effort]
    )
    if requested in ranked or requested not in _STRENGTH_RANK or not ranked:
        return requested
    strength: Final = _STRENGTH_RANK[requested]
    match rounding:
        case "up":
            return next((effort for effort in ranked if _STRENGTH_RANK[effort] >= strength), ranked[-1])
        case "down":
            return next((effort for effort in reversed(ranked) if _STRENGTH_RANK[effort] <= strength), ranked[0])
        case _:
            assert_never(rounding)


_ROUNDING_KEY: Final = "reasoning_effort_rounding"
_DEFAULT_EFFORT: Final = "default"
_EFFORT_KEY: Final = "effort"
_LOGGABLE_REQUESTED_LEVELS: Final = frozenset((*REASONING_EFFORT_ADVERTISEMENT_ORDER, _DEFAULT_EFFORT))
_UNRECOGNIZED_LEVEL: Final = "unrecognized"
_PROVIDERS_SENDING_THINKING_AS_EFFORT: Final = frozenset(("hosted_vllm",))


def deployment_reasoning_effort_levels(deployment_model_info: object) -> tuple[str, ...] | None:
    """Only the deployment's own declaration counts. The router copies it onto the shared
    provider/model map key, and map entries describe a model rather than what a deployment's
    endpoint accepts, so reading the map here would round requests for siblings and for models
    nobody declared levels on."""
    own_info: Final = as_str_mapping(deployment_model_info)
    return declared_reasoning_efforts(own_info) if own_info is not None else None


def deployment_rounds_reasoning_effort(deployment_model_info: object) -> bool:
    declared: Final = deployment_reasoning_effort_levels(deployment_model_info)
    return declared is not None and any(level in _STRENGTH_RANK for level in declared)


def with_adaptive_thinking_as_effort(
    params: Mapping[str, object], *, model: str, custom_llm_provider: str
) -> Mapping[str, object]:
    translated: Final = dict(params)
    AnthropicModelInfo.translate_legacy_thinking_for_adaptive_model(
        model=model, optional_params=translated, custom_llm_provider=custom_llm_provider
    )
    return translated


def thinking_budget_effort(thinking: object, *, model: str, custom_llm_provider: str) -> str | None:
    adaptive_config: Final = as_str_mapping(
        with_adaptive_thinking_as_effort(
            {"thinking": thinking}, model=model, custom_llm_provider=custom_llm_provider
        ).get("output_config")
    )
    if adaptive_config is not None:
        adaptive_effort: Final = adaptive_config.get(_EFFORT_KEY)
        return adaptive_effort if isinstance(adaptive_effort, str) else None
    if custom_llm_provider not in _PROVIDERS_SENDING_THINKING_AS_EFFORT and "thinking" in (
        get_supported_openai_params(model=model, custom_llm_provider=custom_llm_provider) or ()
    ):
        return None
    return reasoning_effort_from_thinking(thinking)


def adaptive_thinking_beside_effort(
    thinking: AnthropicThinkingParam | None, effort: object, *, model: str, custom_llm_provider: str
) -> AnthropicThinkingParam | None:
    if effort is None or effort == "none":
        return None
    translated: Final = with_adaptive_thinking_as_effort(
        {"thinking": thinking}, model=model, custom_llm_provider=custom_llm_provider
    )
    return thinking if "output_config" in translated else None


def _rounding(value: object) -> ReasoningEffortRounding | None:
    match value.strip().lower() if isinstance(value, str) else value:
        case "up":
            return "up"
        case "down":
            return "down"
        case None:
            return None
        case _:
            verbose_logger.warning(
                "Ignoring reasoning_effort_rounding=%s, expected 'up' or 'down'", _single_line(value)
            )
            return None


def _deployment_rounding(deployment_model_info: object) -> ReasoningEffortRounding:
    own_info: Final = as_str_mapping(deployment_model_info)
    own: Final = _rounding(own_info.get(_ROUNDING_KEY)) if own_info is not None else None
    return own or _rounding(litellm.reasoning_effort_rounding) or "up"


def _requested_level(requested: RequestedReasoningEffort) -> str | None:
    if isinstance(requested, str):
        return requested
    fields: Final = as_str_mapping(requested)
    if fields is None:
        return None
    level: Final = fields.get(_EFFORT_KEY)
    return level if isinstance(level, str) else None


def _with_level(requested: RequestedReasoningEffort, level: str | None) -> RequestedReasoningEffort:
    fields: Final = as_str_mapping(requested)
    if fields is None:
        return level
    if level is not None:
        return {**fields, _EFFORT_KEY: level}
    rest: Final = {key: value for key, value in fields.items() if key != _EFFORT_KEY}
    return rest or None


def model_group_from_metadata(*metadata: object) -> str | None:
    mappings: Final = (as_str_mapping(entry) for entry in metadata)
    groups: Final = (mapping.get("model_group") for mapping in mappings if mapping is not None)
    return next((group for group in groups if isinstance(group, str)), None)


@dataclass(frozen=True, slots=True)
class DeploymentReasoningEffort:
    value: RequestedReasoningEffort
    deployment_accepts_effort: bool


def resolve_deployment_reasoning_effort(
    requested: RequestedReasoningEffort,
    *,
    model: str,
    custom_llm_provider: str | None,
    deployment_model_info: object,
    model_group: str | None,
    drop_params: bool,
) -> DeploymentReasoningEffort:
    requested_level: Final = _requested_level(requested)
    unchanged: Final = DeploymentReasoningEffort(value=requested, deployment_accepts_effort=False)
    if requested_level is None:
        return unchanged
    declared: Final = deployment_reasoning_effort_levels(deployment_model_info)
    if declared is None:
        return unchanged
    rounding: Final = _deployment_rounding(deployment_model_info)
    rounded_level: Final = (
        None
        if requested_level == _DEFAULT_EFFORT
        else nearest_declared_reasoning_effort(requested_level, declared, rounding)
    )
    sent_level: Final = None if drop_params and rounded_level not in declared else rounded_level
    deployment_accepts_effort: Final = sent_level is not None and len(declared) > 0
    if sent_level == requested_level:
        return DeploymentReasoningEffort(value=requested, deployment_accepts_effort=deployment_accepts_effort)
    _log_remap(
        requested_level=requested_level,
        sent_level=sent_level,
        rounding=rounding,
        deployment_model=_provider_model(model, custom_llm_provider),
        deployment_model_info=deployment_model_info,
        model_group=model_group,
    )
    return DeploymentReasoningEffort(
        value=_with_level(requested, sent_level), deployment_accepts_effort=deployment_accepts_effort
    )


def _provider_model(model: str, custom_llm_provider: str | None) -> str:
    if custom_llm_provider is None or model.startswith(f"{custom_llm_provider}/"):
        return model
    return f"{custom_llm_provider}/{model}"


def _log_remap(
    *,
    requested_level: str,
    sent_level: str | None,
    rounding: ReasoningEffortRounding,
    deployment_model: str,
    deployment_model_info: object,
    model_group: str | None,
) -> None:
    own_info: Final = as_str_mapping(deployment_model_info)
    deployment_id: Final = own_info.get("id") if own_info is not None else None
    logged_requested_level: Final = _single_line(
        requested_level if requested_level in _LOGGABLE_REQUESTED_LEVELS else _UNRECOGNIZED_LEVEL
    )
    logged_sent_level: Final = _single_line(sent_level)
    logged_deployment_id: Final = _single_line(deployment_id)
    logged_model_group: Final = _single_line(model_group)
    logged_deployment_model: Final = _single_line(deployment_model)
    verbose_logger.info(
        "reasoning_effort %s sent as %s (rounding %s) to deployment %s, model group %s, model %s",
        logged_requested_level,
        logged_sent_level,
        rounding,
        logged_deployment_id,
        logged_model_group,
        logged_deployment_model,
        extra={
            "reasoning_effort_deployment_id": logged_deployment_id,
            "reasoning_effort_model_group": logged_model_group,
            "reasoning_effort_deployment_model": logged_deployment_model,
            "reasoning_effort_requested": logged_requested_level,
            "reasoning_effort_sent": logged_sent_level,
            "reasoning_effort_rounding": rounding,
        },
    )


def _single_line(value: object) -> str | None:
    return None if value is None else str(value).replace("\r", "").replace("\n", "")


def reasoning_effort_needs_allowing(
    resolved: DeploymentReasoningEffort, *, model: str, custom_llm_provider: str, base_model: str | None
) -> bool:
    """A provider that lists reasoning_effort maps it itself, so allowing the raw key as well would
    send it twice (anthropic turns it into thinking and rejects the raw key). A level outside the
    declaration is still sent once the deployment declares any, for its backend to judge."""
    if not resolved.deployment_accepts_effort:
        return False
    return "reasoning_effort" not in (
        get_supported_openai_params(model=model, custom_llm_provider=custom_llm_provider, base_model=base_model) or ()
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

    If supports_reasoning is unset but at least one per-level flag (e.g.
    supports_minimal_reasoning_effort) is present, treat it as implicitly True, since the
    per-level flags are evidence the model supports reasoning. An explicit False always wins:
    it is the operator's escape hatch and must not be overridden by inherited per-level flags.
    """
    supports_reasoning: Final = model_info.get("supports_reasoning")
    if supports_reasoning is False:
        return ()

    flags: Final = _declared_effort_flags(model_info)
    has_per_level_flag: Final = any(value is not None for value in flags.values())
    if supports_reasoning is not True and not has_per_level_flag:
        return () if deployment_is_mapped else None

    declared: Final = declared_reasoning_efforts(model_info)
    if declared is not None:
        return declared

    if all(value is None for value in flags.values()):
        return None

    opt_out: Final = frozenset(effort for effort in _OPT_OUT_EFFORTS if flags[effort] is not False)
    opt_in: Final = frozenset(effort for effort in _OPT_IN_EFFORTS if flags[effort] is True)
    none_level: Final[frozenset[str]] = (
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
