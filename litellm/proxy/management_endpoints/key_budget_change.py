"""Classifies a /key/update request's budget edits as tightening or loosening the key's stored limits."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.models.team import BudgetLimitEntry
from litellm.models.verification_token import LiteLLM_VerificationToken
from litellm.proxy._types import UpdateKeyRequest


@dataclass(frozen=True, slots=True)
class KeyBudgetUnchanged:
    kind: Literal["unchanged"] = "unchanged"


@dataclass(frozen=True, slots=True)
class KeyBudgetTightened:
    kind: Literal["tightened"] = "tightened"


@dataclass(frozen=True, slots=True)
class KeyBudgetLoosened:
    kind: Literal["loosened"] = "loosened"


@dataclass(frozen=True, slots=True)
class KeyBudgetAdminOnly:
    kind: Literal["admin_only"] = "admin_only"


KeyBudgetChange: TypeAlias = KeyBudgetUnchanged | KeyBudgetTightened | KeyBudgetLoosened | KeyBudgetAdminOnly

_FieldChange: TypeAlias = Literal["unchanged", "tightened", "loosened"]

SELF_SERVE_BUDGET_POLICY_SETTING: Final = "self_serve_budget_policy"
SelfServeBudgetPolicy: TypeAlias = Literal["disabled", "lower_only", "ceiling"]

_STORED_WINDOWS: Final = TypeAdapter(list[BudgetLimitEntry])
_POLICY: Final = TypeAdapter(SelfServeBudgetPolicy)


def resolve_self_serve_budget_policy(general_settings: Mapping[str, object]) -> SelfServeBudgetPolicy:
    configured: Final = general_settings.get(SELF_SERVE_BUDGET_POLICY_SETTING, "disabled")
    try:
        return _POLICY.validate_python(configured)
    except ValidationError:
        verbose_proxy_logger.warning(
            "general_settings.%s=%r is not one of disabled, lower_only, ceiling; using disabled",
            SELF_SERVE_BUDGET_POLICY_SETTING,
            configured,
        )
        return "disabled"


def _max_budget_change(data: UpdateKeyRequest, existing: LiteLLM_VerificationToken) -> _FieldChange:
    if "max_budget" not in data.model_fields_set or data.max_budget == existing.max_budget:
        return "unchanged"
    if data.max_budget is None:
        return "loosened"
    if existing.max_budget is None or data.max_budget < existing.max_budget:
        return "tightened"
    return "loosened"


def _duration(window: BudgetLimitEntry) -> str:
    return window.budget_duration


def _caps_by_duration(windows: Sequence[BudgetLimitEntry]) -> Mapping[str, float]:
    grouped: Final = groupby(sorted(windows, key=_duration), key=_duration)
    return MappingProxyType({duration: min(w.max_budget for w in group) for duration, group in grouped})


def _stored_caps(existing: LiteLLM_VerificationToken) -> Mapping[str, float] | None:
    try:
        return _caps_by_duration(_STORED_WINDOWS.validate_python(existing.budget_limits or []))
    except ValidationError:
        return None


def _budget_limits_change(data: UpdateKeyRequest, existing: LiteLLM_VerificationToken) -> _FieldChange:
    """
    Window spend counters are keyed by the exact `budget_duration` string, so a window only
    counts as kept when the same string survives with a cap no higher than before.
    """
    if "budget_limits" not in data.model_fields_set:
        return "unchanged"
    stored: Final = _stored_caps(existing)
    if stored is None:
        return "loosened"
    requested: Final = _caps_by_duration(data.budget_limits or ())
    kept_or_lowered: Final = all(d in requested and requested[d] <= cap for d, cap in stored.items())
    return "tightened" if kept_or_lowered else "loosened"


@dataclass(frozen=True, slots=True)
class EffectiveKeyBudget:
    max_budget: float | None
    budget_limits: tuple[BudgetLimitEntry, ...]


def stored_budget_windows(existing: LiteLLM_VerificationToken) -> tuple[BudgetLimitEntry, ...]:
    """The key's stored windows, or none when the column holds something the window model cannot read."""
    try:
        return tuple(_STORED_WINDOWS.validate_python(existing.budget_limits or []))
    except ValidationError:
        return ()


def effective_key_budget(data: UpdateKeyRequest, existing: LiteLLM_VerificationToken) -> EffectiveKeyBudget:
    """The budget the key would carry after the update, with omitted fields keeping their stored values."""
    max_budget: Final = data.max_budget if "max_budget" in data.model_fields_set else existing.max_budget
    if "budget_limits" in data.model_fields_set:
        return EffectiveKeyBudget(max_budget=max_budget, budget_limits=tuple(data.budget_limits or ()))
    return EffectiveKeyBudget(max_budget=max_budget, budget_limits=stored_budget_windows(existing))


def classify_key_budget_change(data: UpdateKeyRequest, existing: LiteLLM_VerificationToken) -> KeyBudgetChange:
    overrides_linked_budget: Final = (
        existing.budget_id is not None and _max_budget_change(data, existing) != "unchanged"
    )
    if data.spend is not None or "soft_budget" in data.model_fields_set or overrides_linked_budget:
        return KeyBudgetAdminOnly()
    changes: Final = frozenset((_max_budget_change(data, existing), _budget_limits_change(data, existing)))
    if "loosened" in changes:
        return KeyBudgetLoosened()
    if "tightened" in changes:
        return KeyBudgetTightened()
    return KeyBudgetUnchanged()
