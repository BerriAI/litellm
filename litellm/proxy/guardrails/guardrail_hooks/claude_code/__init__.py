"""
Claude Code guardrail integrations for LiteLLM.

Two focused policy-enforcement guardrails for Claude Code deployments:

1. claude_code_prompt_cache        — auto-inject Anthropic prompt-caching headers
2. claude_code_block_expensive_flags — block expensive API flags (fast mode, etc.)

Hosted tool blocking is handled by the provider-agnostic 'block_hosted_tools'
guardrail (see guardrail_hooks/block_hosted_tools/).
"""

from typing import TYPE_CHECKING

import litellm
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .block_expensive_flags import ClaudeCodeBlockExpensiveFlagsGuardrail
from .prompt_cache import ClaudeCodePromptCacheGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


# ------------------------------------------------------------------ #
# Per-guardrail initializer functions                                  #
# ------------------------------------------------------------------ #


def _init_prompt_cache(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> ClaudeCodePromptCacheGuardrail:
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("ClaudeCodePromptCacheGuardrail requires a guardrail_name")
    instance = ClaudeCodePromptCacheGuardrail(
        guardrail_name=guardrail_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


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
    SupportedGuardrailIntegrations.CLAUDE_CODE_PROMPT_CACHE.value: _init_prompt_cache,
    SupportedGuardrailIntegrations.CLAUDE_CODE_BLOCK_EXPENSIVE_FLAGS.value: _init_block_expensive_flags,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.CLAUDE_CODE_PROMPT_CACHE.value: ClaudeCodePromptCacheGuardrail,
    SupportedGuardrailIntegrations.CLAUDE_CODE_BLOCK_EXPENSIVE_FLAGS.value: ClaudeCodeBlockExpensiveFlagsGuardrail,
}

__all__ = [
    "ClaudeCodePromptCacheGuardrail",
    "ClaudeCodeBlockExpensiveFlagsGuardrail",
]
