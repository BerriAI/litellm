"""Conduct Guard as a LiteLLM guardrail.

Pure alias for the ``conduct-litellm-guard`` PyPI package. The
``ConductGuard`` class ships with ``SUPPORTED_EVENT_HOOKS`` +
``get_supported_event_hooks`` since plugin 0.2.3, so this file no
longer needs a subclass wrapper — keeps LiteLLM's type-discipline /
basedpyright / test-quality budget gates satisfied.

Install: ``pip install "conduct-litellm-guard>=0.2.3"``
Source:  https://github.com/sseshachala/conductai/tree/main/packages/conduct-litellm-guard
Docs:    https://conductai.ai/guard
"""

from __future__ import annotations

_import_error_message = (
    "conduct-litellm-guard is required for the Conduct guardrail. "
    'Install it with: pip install "conduct-litellm-guard>=0.2.3"'
)


# ── Base plugin import ─────────────────────────────────────────────────
# Raising ImportError at module load caused the guardrail-hook auto-loader
# to treat a missing ``conduct-litellm-guard`` as "hook unavailable" and
# silently drop the registration. Users saw configs load with no guardrail
# active and no error message. Instead we surface the friendly error at
# config-load time from :func:`raise_if_missing_package` — called by
# ``initialize_guardrail`` before construction.
# (cursor[bot] finding on BerriAI/litellm#38143.)

try:
    from conduct_litellm_guard import ConductGuard as ConductGuardrail
    from conduct_litellm_guard.guardrail import (
        ConductGuardBlocked as ConductGuardrailBlocked,
    )
    from conduct_litellm_guard.guardrail import GuardDecision

    _import_error: ImportError | None = None
except ImportError as _import_err:
    ConductGuardrail = None
    ConductGuardrailBlocked = None
    GuardDecision = None
    _import_error = _import_err


def raise_if_missing_package() -> None:
    """Called by ``initialize_guardrail`` before constructing the class.

    Surfaces the friendly ``pip install`` error at the actionable moment
    (config load) rather than silently dropping the hook at module load.
    """
    if _import_error is not None:
        raise ImportError(_import_error_message) from _import_error


__all__ = [
    "ConductGuardrail",
    "ConductGuardrailBlocked",
    "GuardDecision",
    "raise_if_missing_package",
]
