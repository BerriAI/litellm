"""Conduct Guard as a LiteLLM guardrail.

Thin re-export. The adapter, response-envelope parser, session-ID chain,
and fail-mode logic all live in the `conduct-litellm-guard` PyPI package,
which is where issues, versioning, and standalone-user support live.

Install: `pip install conduct-litellm-guard`
Source:  https://github.com/sseshachala/conductai/tree/main/packages/conduct-litellm-guard
Docs:    https://conductai.ai/guard
"""

from __future__ import annotations

try:
    from conduct_litellm_guard import ConductGuard as ConductGuardrail
    from conduct_litellm_guard.guardrail import (
        ConductGuardBlocked as ConductGuardrailBlocked,
    )
    from conduct_litellm_guard.guardrail import GuardDecision
except ImportError as _e:
    raise ImportError(
        "conduct-litellm-guard is required for the Conduct guardrail. Install with: pip install conduct-litellm-guard"
    ) from _e


__all__ = ["ConductGuardrail", "ConductGuardrailBlocked", "GuardDecision"]
