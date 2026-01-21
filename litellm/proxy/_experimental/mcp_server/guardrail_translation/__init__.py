"""Guardrail translation mapping for MCP tool calls."""

from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.types.utils import CallTypes

# This mapping lives alongside the MCP server implementation because MCP
# integrations are managed by the proxy subsystem, not litellm.llms providers.
# Unified guardrails import this module explicitly to register the handler.

guardrail_translation_mappings = {
    CallTypes.call_mcp_tool: MCPGuardrailTranslationHandler,
}

__all__ = ["guardrail_translation_mappings", "MCPGuardrailTranslationHandler"]
