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

from typing import ClassVar

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

_IMPORT_ERROR_MESSAGE = (
    "conduct-litellm-guard is required for the Conduct guardrail. "
    'Install it with: pip install "conduct-litellm-guard>=0.2.2"'
)


# ── Base plugin import — deferred to init-time via raise_if_missing_package ──
# Raising ImportError at module load caused the guardrail-hook auto-loader to
# treat a missing ``conduct-litellm-guard`` as "hook unavailable" and silently
# drop the registration. Users saw configs load with no guardrail active and
# no error message. Instead we fall back to ``CustomGuardrail`` at module load
# so the class hierarchy stays intact; ``initialize_guardrail`` (in
# ``__init__.py``) calls :func:`raise_if_missing_package` before construction
# so the friendly error surfaces when actionable.
# (cursor[bot] finding on BerriAI/litellm#38143.)

try:
    from conduct_litellm_guard import ConductGuard as _BaseConductGuard
    from conduct_litellm_guard.guardrail import (
        ConductGuardBlocked as ConductGuardrailBlocked,
    )
    from conduct_litellm_guard.guardrail import GuardDecision

    _IMPORT_ERROR: ImportError | None = None
except ImportError as _import_err:
    _BaseConductGuard = CustomGuardrail
    ConductGuardrailBlocked = None
    GuardDecision = None
    _IMPORT_ERROR = _import_err


class ConductGuardrail(_BaseConductGuard):
    """LiteLLM adapter over ``conduct_litellm_guard.ConductGuard``.

    Inherits its ``__init__`` from the base runtime when the standalone
    package is installed; otherwise inherits from ``CustomGuardrail``
    and ``initialize_guardrail`` short-circuits with a friendly error
    before this class is ever constructed.

    Only two additions on this side:
    - ``SUPPORTED_EVENT_HOOKS`` / ``get_supported_event_hooks`` so
      LiteLLM validates configs against modes we actually implement.
    """

    # Advertised event hooks. The plugin currently runs at pre_call
    # (input rail) — a policy block short-circuits before the model
    # sees the prompt, which is the semantic LiteLLM users expect for
    # a "guardrail". ``during_call`` / ``post_call`` support lands
    # with plugin 0.3.x once the underlying Conduct response gate is
    # wired through ``guard_check_response``. Advertising only
    # pre_call today prevents silent bypass of ``during_call``
    # configurations — see veria-ai finding on BerriAI/litellm#38143.
    SUPPORTED_EVENT_HOOKS: ClassVar[tuple[GuardrailEventHooks, ...]] = (GuardrailEventHooks.pre_call,)

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        """LiteLLM calls this during config validation to reject
        unsupported ``mode:`` values (e.g. ``during_call`` while only
        pre_call is implemented)."""
        return list(cls.SUPPORTED_EVENT_HOOKS)


def raise_if_missing_package() -> None:
    """Called by ``initialize_guardrail`` before constructing the class.

    Surfaces the friendly ``pip install`` error at the actionable moment
    (config load) rather than silently dropping the hook at module load.
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(_IMPORT_ERROR_MESSAGE) from _IMPORT_ERROR


__all__ = [
    "ConductGuardrail",
    "ConductGuardrailBlocked",
    "GuardDecision",
    "raise_if_missing_package",
]
