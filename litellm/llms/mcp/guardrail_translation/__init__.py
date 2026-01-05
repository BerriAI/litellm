"""Guardrail translation handler wiring for MCP tool calls."""

from litellm.llms.mcp.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.types.utils import CallTypes

# These mappings live under litellm.llms so the unified guardrail discovery can
# treat MCP tooling like any other provider endpoint.
guardrail_translation_mappings = {
    CallTypes.call_mcp_tool: MCPGuardrailTranslationHandler,
}

__all__ = ["guardrail_translation_mappings", "MCPGuardrailTranslationHandler"]
