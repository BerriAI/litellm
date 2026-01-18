# Tool Permission Guardrail Type Definitions
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .base import GuardrailConfigModel


class ToolPermissionRule(BaseModel):
    """
    A rule defining permission for a specific tool or tool pattern
    """

    id: str = Field(description="Unique identifier for the rule")
    tool_name: Optional[str] = Field(
        default=None,
        description="Regex pattern applied to the tool's function name",
    )
    tool_type: Optional[str] = Field(
        default=None,
        description="Regex pattern applied to the tool type (e.g., function)",
    )
    decision: Literal["allow", "deny"] = Field(
        description="Whether to allow or deny this tool usage"
    )
    allowed_param_patterns: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional regex map enforcing nested parameter values using dot/[] paths",
    )

    @field_validator("tool_name", "tool_type", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return stripped
        return value

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, v):
        """Normalize decision to lowercase to handle case-insensitive input."""
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def _ensure_target_present(self):
        if self.tool_name is None and self.tool_type is None:
            raise ValueError(
                "Each rule must specify at least a tool_name or tool_type regex"
            )
        return self


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
        description="Ordered allow/deny rules. Patterns use regex for tool names/types and optional regex constraints on tool arguments.",
    )
    default_action: Literal["allow", "deny"] = Field(
        default="deny", description="Fallback decision when no rule matches"
    )
    on_disallowed_action: Literal["block", "rewrite"] = Field(
        default="block",
        description="Choose whether disallowed tools block the request or get rewritten out of the payload",
    )

    @field_validator("default_action", mode="before")
    @classmethod
    def normalize_default_action(cls, v):
        """Normalize default_action to lowercase to handle case-insensitive input."""
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("on_disallowed_action", mode="before")
    @classmethod
    def normalize_on_disallowed_action(cls, v):
        """Normalize on_disallowed_action to lowercase to handle case-insensitive input."""
        if isinstance(v, str):
            return v.lower()
        return v

    @staticmethod
    def ui_friendly_name() -> str:
        return "LiteLLM Tool Permission Guardrail"
