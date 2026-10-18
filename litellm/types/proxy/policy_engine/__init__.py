"""
Type definitions for the LiteLLM Policy Engine.

The Policy Engine allows administrators to define policies that combine guardrails
with scoping rules. Policies can target specific teams, API keys, and models using
wildcard patterns, and support inheritance from base policies.

Configuration:
- `policies`: Define WHAT guardrails to apply (with inheritance and conditions)
- `policy_attachments`: Define WHERE policies apply (teams, keys, models)
"""

from litellm.types.proxy.policy_engine.pipeline_types import (
    GuardrailPipeline,
    PipelineExecutionResult,
    PipelineStep,
    PipelineStepResult,
)
from litellm.types.proxy.policy_engine.policy_types import (
    Policy,
    PolicyAttachment,
    PolicyCondition,
    PolicyConfig,
    PolicyGuardrails,
    PolicyScope,
)
from litellm.types.proxy.policy_engine.resolver_types import (
    AttachmentImpactResponse,
    PipelineTestRequest,
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
    PolicyMatchDetail,
    PolicyResolveRequest,
    PolicyResolveResponse,
    PolicyScopeResponse,
    PolicySummaryItem,
    PolicyTestResponse,
    PolicyUpdateRequest,
    PolicyVersionCompareResponse,
    PolicyVersionCreateRequest,
    PolicyVersionListResponse,
    PolicyVersionStatusUpdateRequest,
    ResolvedPolicy,
)
from litellm.types.proxy.policy_engine.validation_types import (
    PolicyValidateRequest,
    PolicyValidationError,
    PolicyValidationErrorType,
    PolicyValidationResponse,
)

__all__ = [
    "AttachmentImpactResponse",
    # Pipeline types
    "GuardrailPipeline",
    "PipelineExecutionResult",
    "PipelineStep",
    "PipelineStepResult",
    # Pipeline test types
    "PipelineTestRequest",
    # Policy types
    "Policy",
    "PolicyAttachment",
    "PolicyAttachmentCreateRequest",
    "PolicyAttachmentDBResponse",
    "PolicyAttachmentListResponse",
    "PolicyCondition",
    # CRUD Request/Response types
    "PolicyConditionRequest",
    "PolicyConfig",
    "PolicyCreateRequest",
    "PolicyDBResponse",
    "PolicyGuardrails",
    # API Response types
    "PolicyGuardrailsResponse",
    "PolicyInfoResponse",
    "PolicyListDBResponse",
    "PolicyListResponse",
    # Resolver types
    "PolicyMatchContext",
    "PolicyMatchDetail",
    # Resolve types
    "PolicyResolveRequest",
    "PolicyResolveResponse",
    "PolicyScope",
    "PolicyScopeResponse",
    "PolicySummaryItem",
    "PolicyTestResponse",
    "PolicyUpdateRequest",
    # Validation types
    "PolicyValidateRequest",
    "PolicyValidationError",
    "PolicyValidationErrorType",
    "PolicyValidationResponse",
    "PolicyVersionCompareResponse",
    # Policy versioning
    "PolicyVersionCreateRequest",
    "PolicyVersionListResponse",
    "PolicyVersionStatusUpdateRequest",
    "ResolvedPolicy",
]
