"""
Policy validation type definitions.

These types are used for validating policy configurations and returning
validation results.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PolicyValidationErrorType(str, Enum):
    """Types of validation errors that can occur."""

    INVALID_GUARDRAIL = "invalid_guardrail"
    INVALID_TEAM = "invalid_team"
    INVALID_KEY = "invalid_key"
    INVALID_MODEL = "invalid_model"
    INVALID_INHERITANCE = "invalid_inheritance"
    CIRCULAR_INHERITANCE = "circular_inheritance"
    INVALID_SCOPE = "invalid_scope"
    INVALID_SYNTAX = "invalid_syntax"


class PolicyValidationError(BaseModel):
    """
    Represents a validation error or warning for a policy.
    """

    policy_name: str = Field(description="Name of the policy with the issue.")
    error_type: PolicyValidationErrorType = Field(
        description="Type of validation error."
    )
    message: str = Field(description="Human-readable error message.")
    field: Optional[str] = Field(
        default=None,
        description="Specific field that caused the error (e.g., 'guardrails.add', 'scope.teams').",
    )
    value: Optional[str] = Field(
        default=None,
        description="The invalid value that caused the error.",
    )

    model_config = ConfigDict(extra="forbid")


class PolicyValidationResponse(BaseModel):
    """
    Response from policy validation.

    - `valid`: True if no blocking errors were found
    - `errors`: List of blocking errors (prevent policy from being applied)
    - `warnings`: List of non-blocking warnings (policy can still be applied)
    """

    valid: bool = Field(description="True if the policy configuration is valid.")
    errors: List[PolicyValidationError] = Field(
        default_factory=list,
        description="List of blocking validation errors.",
    )
    warnings: List[PolicyValidationError] = Field(
        default_factory=list,
        description="List of non-blocking validation warnings.",
    )

    model_config = ConfigDict(extra="forbid")


class PolicyValidateRequest(BaseModel):
    """
    Request body for the /policy/validate endpoint.
    """

    policies: Dict[str, Any] = Field(
        description="Policy configuration to validate. Map of policy names to policy definitions."
    )

    model_config = ConfigDict(extra="forbid")
