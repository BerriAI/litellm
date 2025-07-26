from typing import Any
from .mcp_guardrail import MCPGuardrail

def initialize_mcp(guardrail_config: dict[str, Any], **kwargs):
    """
    Initialize MCP guardrail from configuration.
    
    Args:
        guardrail_config: Dictionary containing guardrail configuration
        **kwargs: Additional keyword arguments
        
    Returns:
        MCPGuardrail: Initialized MCP guardrail instance
    """
    return MCPGuardrail(**guardrail_config, **kwargs)

# Registry for guardrail initializers
guardrail_initializer_registry = {
    "mcp": initialize_mcp,
}

__all__ = ["MCPGuardrail", "initialize_mcp", "guardrail_initializer_registry"] 