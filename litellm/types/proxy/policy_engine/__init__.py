"""
Type definitions for the LiteLLM Policy Engine.

The Policy Engine allows administrators to define policies that combine guardrails
with scoping rules. Policies can target specific teams, API keys, and models using
wildcard patterns, and support inheritance from base policies.
"""

from litellm.types.proxy.policy_engine.policy_types import (
    Policy,
    PolicyConfig,
    PolicyGuardrails,
    PolicyScope,
)
from litellm.types.proxy.policy_engine.resolver_types import (
    PolicyGuardrailsResponse,
    PolicyInfoResponse,
    PolicyListResponse,
    PolicyMatchContext,
    PolicyScopeResponse,
    PolicySummaryItem,
    PolicyTestResponse,
    ResolvedPolicy,
)
from litellm.types.proxy.policy_engine.validation_types import (
    PolicyValidateRequest,
    PolicyValidationError,
    PolicyValidationErrorType,
    PolicyValidationResponse,
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
    # API Response types
    "PolicyGuardrailsResponse",
    "PolicyInfoResponse",
    "PolicyListResponse",
    "PolicyScopeResponse",
    "PolicySummaryItem",
    "PolicyTestResponse",
]
