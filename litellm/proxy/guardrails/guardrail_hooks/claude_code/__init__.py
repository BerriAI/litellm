"""
Claude Code guardrail integrations for LiteLLM.

Focused policy-enforcement guardrail for Claude Code deployments:

- claude_code_block_expensive_flags — block expensive API flags (fast mode, etc.)

Hosted tool blocking is handled by the provider-agnostic 'block_hosted_tools'
guardrail (see guardrail_hooks/block_hosted_tools/).
"""

from typing import TYPE_CHECKING

import litellm
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .block_expensive_flags import ClaudeCodeBlockExpensiveFlagsGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


# ------------------------------------------------------------------ #
# Per-guardrail initializer functions                                  #
# ------------------------------------------------------------------ #


def _init_block_expensive_flags(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> ClaudeCodeBlockExpensiveFlagsGuardrail:
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("ClaudeCodeBlockExpensiveFlagsGuardrail requires a guardrail_name")
    instance = ClaudeCodeBlockExpensiveFlagsGuardrail(
        guardrail_name=guardrail_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


# ------------------------------------------------------------------ #
# Registries consumed by the guardrail loader                          #
# ------------------------------------------------------------------ #

guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.CLAUDE_CODE_BLOCK_EXPENSIVE_FLAGS.value: _init_block_expensive_flags,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.CLAUDE_CODE_BLOCK_EXPENSIVE_FLAGS.value: ClaudeCodeBlockExpensiveFlagsGuardrail,
}

__all__ = [
    "ClaudeCodeBlockExpensiveFlagsGuardrail",
]
