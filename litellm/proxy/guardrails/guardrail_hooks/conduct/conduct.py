"""Conduct Guard as a LiteLLM guardrail.

Thin alias for the ``conduct-litellm-guard`` PyPI package. The
``ConductGuard`` class ships with ``SUPPORTED_EVENT_HOOKS`` +
``get_supported_event_hooks`` since plugin 0.2.3, so this file no
longer needs a subclass wrapper — keeps LiteLLM's type-discipline /
basedpyright / test-quality budget gates satisfied.

When the standalone package isn't installed we still register a real
stub class so LiteLLM's guardrail registry can scan
``get_supported_event_hooks`` at load time without crashing (see
BerriAI/litellm#38143 CI regression: registry iteration expects every
registered class to expose the hooks classmethod). Instantiation of
the stub raises ``ImportError`` via ``raise_if_missing_package``.

Install: ``pip install "conduct-litellm-guard>=0.2.3"``
Source:  https://github.com/sseshachala/conductai/tree/main/packages/conduct-litellm-guard
Docs:    https://conductai.ai/guard
"""

from __future__ import annotations

from typing import ClassVar

from litellm.integrations.custom_guardrail import CustomGuardrail

_import_error_message = (
    "conduct-litellm-guard is required for the Conduct guardrail. "
    'Install it with: pip install "conduct-litellm-guard>=0.2.3"'
)


try:
    from conduct_litellm_guard import ConductGuard as ConductGuardrail
    from conduct_litellm_guard.guardrail import (
        ConductGuardBlocked as ConductGuardrailBlocked,
    )
    from conduct_litellm_guard.guardrail import GuardDecision

    _import_error: ImportError | None = None
except ImportError as _err:

    class ConductGuardrail(CustomGuardrail):
        """Stub used when ``conduct-litellm-guard`` isn't installed.

        Exposes the class-level surface LiteLLM's guardrail registry
        scans at load time — ``SUPPORTED_EVENT_HOOKS`` +
        ``get_supported_event_hooks`` — so the registry doesn't crash
        when this hook is discovered without the runtime dependency.
        ``initialize_guardrail`` calls ``raise_if_missing_package``
        before ever constructing this class, so users see a friendly
        ``pip install`` error rather than the stub silently activating.
        """

        SUPPORTED_EVENT_HOOKS: ClassVar[tuple[str, ...]] = ("pre_call",)

        @classmethod
        def get_supported_event_hooks(cls) -> list[str]:
            return list(cls.SUPPORTED_EVENT_HOOKS)

    ConductGuardrailBlocked = None
    GuardDecision = None
    _import_error = _err


def raise_if_missing_package() -> None:
    """Called by ``initialize_guardrail`` before constructing the class.

    Surfaces the friendly ``pip install`` error at the actionable moment
    (config load) rather than silently dropping the hook or letting the
    stub run.
    """
    if _import_error is not None:
        raise ImportError(_import_error_message) from _import_error


__all__ = [
    "ConductGuardrail",
    "ConductGuardrailBlocked",
    "GuardDecision",
    "raise_if_missing_package",
]
