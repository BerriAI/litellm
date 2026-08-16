"""
Type definitions for the LiteLLM Policy Engine.

This module re-exports types from litellm.types.proxy.policy_engine for backward compatibility.
The canonical location for these types is litellm/types/proxy/policy_engine/.
"""

# Re-export all types from the new location
from litellm.types.proxy.policy_engine import (  # Policy types; Validation types; Resolver types
    Policy,
    PolicyConfig,
    PolicyGuardrails,
    PolicyMatchContext,
    PolicyScope,
    PolicyValidateRequest,
    PolicyValidationError,
    PolicyValidationErrorType,
    PolicyValidationResponse,
    ResolvedPolicy,
)

__all__ = [
    # Policy types
    "Policy",
    "PolicyConfig",
    "PolicyGuardrails",
    "PolicyScope",
    # Validation types
    "PolicyValidateRequest",
    "PolicyValidationError",
    "PolicyValidationErrorType",
    "PolicyValidationResponse",
    # Resolver types
    "PolicyMatchContext",
    "ResolvedPolicy",
]
