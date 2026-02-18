"""
Block Hosted Tools guardrail for LiteLLM.

Blocks platform-executed server-side tools from Anthropic, OpenAI, and Gemini.
Provider tool lists are maintained in per-provider YAML files (anthropic.yaml,
openai.yaml, gemini.yaml) alongside this module.
"""

from typing import TYPE_CHECKING

import litellm
from litellm.types.guardrails import SupportedGuardrailIntegrations

from .guardrail import BlockHostedToolsGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams", guardrail: "Guardrail"
) -> BlockHostedToolsGuardrail:
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("BlockHostedToolsGuardrail requires a guardrail_name")
    instance = BlockHostedToolsGuardrail(
        guardrail_name=guardrail_name,
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.BLOCK_HOSTED_TOOLS.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.BLOCK_HOSTED_TOOLS.value: BlockHostedToolsGuardrail,
}

__all__ = ["BlockHostedToolsGuardrail", "initialize_guardrail"]
