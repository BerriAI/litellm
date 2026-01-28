"""
Type definitions for the LiteLLM Policy Engine.

The Policy Engine allows administrators to define policies that combine guardrails
with scoping rules. Policies can target specific teams, API keys, and models using
wildcard patterns, and support inheritance from base policies.

Configuration:
- `policies`: Define WHAT guardrails to apply (with inheritance and conditions)
- `policy_attachments`: Define WHERE policies apply (teams, keys, models)
"""

from litellm.types.proxy.policy_engine.policy_types import (
    Policy,
    PolicyAttachment,
    PolicyCondition,
    PolicyConfig,
    PolicyGuardrails,
    PolicyScope,
)
from litellm.types.proxy.policy_engine.resolver_types import (
    PolicyAttachmentCreateRequest,
    PolicyAttachmentDBResponse,
    PolicyAttachmentListResponse,
    PolicyConditionRequest,
    PolicyCreateRequest,
    PolicyDBResponse,
    PolicyGuardrailsResponse,
    PolicyInfoResponse,
    PolicyListDBResponse,
    PolicyListResponse,
    PolicyMatchContext,
    PolicyScopeResponse,
    PolicySummaryItem,
    PolicyTestResponse,
    PolicyUpdateRequest,
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
    "PolicyCondition",
    "PolicyAttachment",
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
    # CRUD Request/Response types
    "PolicyConditionRequest",
    "PolicyCreateRequest",
    "PolicyUpdateRequest",
    "PolicyDBResponse",
    "PolicyListDBResponse",
    "PolicyAttachmentCreateRequest",
    "PolicyAttachmentDBResponse",
    "PolicyAttachmentListResponse",
]
