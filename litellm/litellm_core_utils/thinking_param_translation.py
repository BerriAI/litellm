from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping, Sequence

_SEND_VIA_EXTRA_BODY: Final = "extra_body"
_SEND_VIA_PROVIDER_MAPPED: Final = "provider_mapped"

_EFFORT_FALLBACKS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "xhigh": ("max", "high"),
        "max": ("xhigh", "high"),
        "medium": ("high", "low"),
        "minimal": ("low", "none"),
        "none": ("low",),
    }
)


@dataclass(frozen=True, slots=True)
class ThinkingParamsState:
    thinking: object | None
    reasoning_effort: object | None
    extra_body: Mapping[str, object]


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _thinking_enabled(thinking: object) -> bool:
    if isinstance(thinking, bool):
        return thinking
    if isinstance(thinking, str):
        return thinking.lower() in {"enabled", "true", "1", "auto"}
    if isinstance(thinking, Mapping):
        typ: Final = thinking.get("type")
        if isinstance(typ, str):
            return typ.lower() in {"enabled", "auto", "true"}
        enabled: Final = thinking.get("enabled")
        if isinstance(enabled, bool):
            return enabled
    return False


def _thinking_type_value(thinking: object, allowed: Sequence[str]) -> str | None:
    if isinstance(thinking, str):
        candidate: Final = thinking
    elif isinstance(thinking, Mapping):
        raw: Final = thinking.get("type")
        candidate = raw if isinstance(raw, str) else None
    elif isinstance(thinking, bool):
        candidate = "enabled" if thinking else "disabled"
    else:
        candidate = None
    if candidate is None:
        return None
    if not allowed or candidate in allowed:
        return candidate
    if candidate == "auto" and "enabled" in allowed:
        return "enabled"
    return None


def _thinking_payload(thinking: object, typ: str) -> Mapping[str, object]:
    if not isinstance(thinking, Mapping):
        return MappingProxyType({"type": typ})
    budget: Final = thinking.get("budget_tokens")
    if isinstance(budget, int):
        return MappingProxyType({"type": typ, "budget_tokens": budget})
    return MappingProxyType({"type": typ})


def _clamp_effort(value: object, allowed: Sequence[str]) -> str | None:
    if not isinstance(value, str):
        return None
    if not allowed:
        return None
    if value in allowed:
        return value
    for fallback in _EFFORT_FALLBACKS.get(value, ()):
        if fallback in allowed:
            return fallback
    return None


def _map_thinking_to_extra_body(
    *,
    thinking_param: str | None,
    thinking: object,
    thinking_values: Sequence[str],
) -> Mapping[str, object]:
    match thinking_param:
        case "thinking.type":
            typ: Final = _thinking_type_value(thinking, thinking_values)
            if typ is None:
                return MappingProxyType({})
            return MappingProxyType({"thinking": dict(_thinking_payload(thinking, typ))})
        case "thinking":
            if isinstance(thinking, Mapping):
                return MappingProxyType({"thinking": dict(thinking)})
            typ_only: Final = _thinking_type_value(thinking, thinking_values or ("enabled", "disabled"))
            if typ_only is None:
                return MappingProxyType({})
            return MappingProxyType({"thinking": {"type": typ_only}})
        case "enable_thinking":
            return MappingProxyType({"enable_thinking": _thinking_enabled(thinking)})
        case "chat_template_kwargs":
            return MappingProxyType(
                {"chat_template_kwargs": {"enable_thinking": _thinking_enabled(thinking)}}
            )
        case None:
            return MappingProxyType({})
        case _:
            return MappingProxyType({})


def translate_thinking_params(
    *,
    model_info: Mapping[str, object] | None,
    state: ThinkingParamsState,
) -> ThinkingParamsState:
    if model_info is None:
        return state

    send_via: Final = model_info.get("thinking_send_via")
    if send_via not in {_SEND_VIA_EXTRA_BODY, _SEND_VIA_PROVIDER_MAPPED}:
        return state

    supports_reasoning: Final = model_info.get("supports_reasoning") is True
    thinking_param_raw: Final = model_info.get("thinking_param")
    thinking_param: Final = thinking_param_raw if isinstance(thinking_param_raw, str) else None
    thinking_values: Final = _as_str_tuple(model_info.get("thinking_values"))
    effort_values: Final = _as_str_tuple(model_info.get("reasoning_effort_values"))

    if not supports_reasoning and send_via != _SEND_VIA_PROVIDER_MAPPED:
        return state

    thinking: Final = state.thinking
    effort: Final = state.reasoning_effort
    if thinking is None and effort is None:
        return state

    existing_extra: Final = dict(state.extra_body)
    patch: dict[str, object] = {}
    keep_thinking: Final = send_via == _SEND_VIA_PROVIDER_MAPPED

    if thinking is not None and send_via == _SEND_VIA_EXTRA_BODY:
        patch.update(
            _map_thinking_to_extra_body(
                thinking_param=thinking_param,
                thinking=thinking,
                thinking_values=thinking_values,
            )
        )

    if effort is not None:
        clamped: Final = _clamp_effort(effort, effort_values)
        if clamped is not None:
            patch["reasoning_effort"] = clamped
        elif not effort_values and send_via == _SEND_VIA_EXTRA_BODY and isinstance(effort, str):
            patch["reasoning_effort"] = effort

    if not patch:
        return state

    thinking_mapped: Final = any(
        key in patch for key in ("thinking", "enable_thinking", "chat_template_kwargs")
    )
    next_thinking: Final = thinking if (keep_thinking or not thinking_mapped) else None
    next_effort: Final = None if "reasoning_effort" in patch else effort
    merged_extra: Final = MappingProxyType({**existing_extra, **patch})
    return ThinkingParamsState(
        thinking=next_thinking,
        reasoning_effort=next_effort,
        extra_body=merged_extra,
    )


def apply_thinking_param_translation(
    *,
    model_info: Mapping[str, object] | None,
    thinking: object | None,
    reasoning_effort: object | None,
    existing_extra_body: Mapping[str, object] | None,
) -> ThinkingParamsState:
    base_extra: Final = (
        MappingProxyType(dict(existing_extra_body))
        if isinstance(existing_extra_body, Mapping)
        else MappingProxyType({})
    )
    return translate_thinking_params(
        model_info=model_info,
        state=ThinkingParamsState(
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            extra_body=base_extra,
        ),
    )
