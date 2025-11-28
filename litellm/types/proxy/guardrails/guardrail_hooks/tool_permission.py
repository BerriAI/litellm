# Tool Permission Guardrail Type Definitions
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class ToolPermissionRule(BaseModel):
    """
    A rule defining permission for a specific tool or tool pattern
    """

    id: str = Field(description="Unique identifier for the rule")
    tool_name: str = Field(
        description="Tool name or pattern (e.g., 'Bash', 'mcp__github_*', 'mcp__github_*_read', '*_read')"
    )
    decision: Literal["allow", "deny"] = Field(
        description="Whether to allow or deny this tool usage"
    )
    allowed_param_patterns: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional regex map enforcing nested parameter values using dot/[] paths",
    )


class ToolResult(BaseModel):
    """
    Represents a tool_result block to be added to the response
    """

    type: str = Field(default="tool_result", description="Should be 'tool_result'")
    tool_use_id: str = Field(
        description="ID of the tool use this result corresponds to"
    )
    content: str = Field(description="Result content")
    is_error: bool = Field(default=True, description="Whether this is an error result")


class PermissionError(BaseModel):
    """
    Error information for permission denial
    """

    tool_name: str = Field(description="Name of the denied tool")
    rule_id: Optional[str] = Field(description="ID of the rule that caused denial")
    message: str = Field(description="Error message")


class ToolPermissionGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters exposed to the UI for the Tool Permission guardrail."""

    rules: Optional[List[ToolPermissionRule]] = Field(
        default=None,
        description="Ordered allow/deny rules. Patterns support * wildcards and optional regex constraints on tool arguments.",
    )
    default_action: Literal["allow", "deny"] = Field(
        default="deny", description="Fallback decision when no rule matches"
    )
    on_disallowed_action: Literal["block", "rewrite"] = Field(
        default="block",
        description="Choose whether disallowed tools block the request or get rewritten out of the payload",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "LiteLLM Tool Permission Guardrail"
