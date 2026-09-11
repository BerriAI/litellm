"""Conduct Guard as a LiteLLM guardrail.

Thin adapter over the ``conduct-litellm-guard`` PyPI package. The
adapter, response-envelope parser, session-ID chain, fail-mode logic,
and the ``guard_check_prompt`` wire client all live in that package —
this file only wires the base runtime into LiteLLM's ``CustomGuardrail``
contract.

Install: ``pip install "conduct-litellm-guard>=0.2.2"``
Source:  https://github.com/sseshachala/conductai/tree/main/packages/conduct-litellm-guard
Docs:    https://conductai.ai/guard
"""

from __future__ import annotations

from typing import Any, ClassVar

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

_IMPORT_ERROR_MESSAGE = (
    "conduct-litellm-guard is required for the Conduct guardrail. "
    'Install it with: pip install "conduct-litellm-guard>=0.2.2"'
)


# ── Base plugin import — deferred to init-time ─────────────────────────
# Reason: the guardrail-hook auto-discovery loop treats a module-level
# ``raise ImportError`` as "hook unavailable" and silently drops the
# registration. A user who installed LiteLLM but forgot the
# ``conduct-litellm-guard`` dependency would see their config load with
# no guardrail active and no error message (cursor[bot] finding on
# BerriAI/litellm#38143). Import here without raising; surface the
# missing dep at ``__init__`` time when it is actionable.

try:
    from conduct_litellm_guard import ConductGuard as _BaseConductGuard
    from conduct_litellm_guard.guardrail import (
        ConductGuardBlocked as ConductGuardrailBlocked,
    )
    from conduct_litellm_guard.guardrail import GuardDecision  # noqa: F401 — re-exported

    _IMPORT_ERROR: ImportError | None = None
except ImportError as _e:
    _BaseConductGuard = None  # type: ignore[assignment,misc]
    ConductGuardrailBlocked = None  # type: ignore[assignment,misc]
    GuardDecision = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = _e


# When the base package isn't installed we still need a real class so
# LiteLLM's registry lookup succeeds; the friendly error surfaces on
# construction.
_ParentClass = _BaseConductGuard if _BaseConductGuard is not None else CustomGuardrail


class ConductGuardrail(_ParentClass):  # type: ignore[valid-type,misc]
    """LiteLLM adapter over ``conduct_litellm_guard.ConductGuard``.

    Subclass exists so we can:
    - Advertise supported event hooks honestly to LiteLLM (see
      ``get_supported_event_hooks``).
    - Raise a friendly error at construction time when the base package
      isn't installed (rather than at module import — see comment
      above).
    """

    # Advertised event hooks. The plugin currently runs at pre_call
    # (input rail) — a policy block short-circuits before the model
    # sees the prompt, which is the semantic LiteLLM users expect for
    # a "guardrail". ``during_call`` / ``post_call`` support lands with
    # 0.3.x once the underlying Conduct response gate is wired through
    # ``guard_check_response`` (tracked in the plugin repo). Advertising
    # only pre_call today prevents silent bypass of ``during_call``
    # configurations — see veria-ai finding on BerriAI/litellm#38143.
    SUPPORTED_EVENT_HOOKS: ClassVar[tuple[GuardrailEventHooks, ...]] = (GuardrailEventHooks.pre_call,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _IMPORT_ERROR is not None or _BaseConductGuard is None:
            raise ImportError(_IMPORT_ERROR_MESSAGE) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        """LiteLLM calls this during config validation to reject
        unsupported ``mode:`` values (e.g. ``during_call`` while only
        pre_call is implemented)."""
        return list(cls.SUPPORTED_EVENT_HOOKS)


__all__ = ["ConductGuardrail", "ConductGuardrailBlocked", "GuardDecision"]
