"""
Policy resolver type definitions.

These types are used for matching requests to policies and resolving
the final guardrails list.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PolicyMatchContext(BaseModel):
    """
    Context used to match a request against policies.

    Contains the team alias, key alias, and model from the incoming request.
    """

    team_alias: Optional[str] = Field(
        default=None,
        description="Team alias from the request.",
    )
    key_alias: Optional[str] = Field(
        default=None,
        description="API key alias from the request.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name from the request.",
    )

    model_config = ConfigDict(extra="forbid")


class ResolvedPolicy(BaseModel):
    """
    Result of resolving a policy with its inheritance chain.

    Contains the final list of guardrails after applying all add/remove operations.
    """

    policy_name: str = Field(description="Name of the resolved policy.")
    guardrails: List[str] = Field(
        default_factory=list,
        description="Final list of guardrail names to apply.",
    )
    inheritance_chain: List[str] = Field(
        default_factory=list,
        description="List of policy names in the inheritance chain (from root to this policy).",
    )

    model_config = ConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# API Response Types
# ─────────────────────────────────────────────────────────────────────────────


class PolicyScopeResponse(BaseModel):
    """Scope configuration for a policy."""

    teams: List[str] = Field(default_factory=list)
    keys: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)


class PolicyGuardrailsResponse(BaseModel):
    """Guardrails configuration for a policy."""

    add: List[str] = Field(default_factory=list)
    remove: List[str] = Field(default_factory=list)


class PolicyInfoResponse(BaseModel):
    """Response for /policy/info/{policy_name} endpoint."""

    policy_name: str
    inherit: Optional[str] = None
    scope: PolicyScopeResponse
    guardrails: PolicyGuardrailsResponse
    resolved_guardrails: List[str]
    inheritance_chain: List[str]


class PolicySummaryItem(BaseModel):
    """Summary of a single policy for list endpoint."""

    inherit: Optional[str] = None
    scope: PolicyScopeResponse
    guardrails: PolicyGuardrailsResponse
    resolved_guardrails: List[str]
    inheritance_chain: List[str]


class PolicyListResponse(BaseModel):
    """Response for /policy/list endpoint."""

    policies: Dict[str, PolicySummaryItem]
    total_count: int


class PolicyTestResponse(BaseModel):
    """Response for /policy/test endpoint."""

    context: PolicyMatchContext
    matching_policies: List[str]
    resolved_guardrails: List[str]
    message: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Request/Response Types for Policy Endpoints
# ─────────────────────────────────────────────────────────────────────────────


class PolicyConditionRequest(BaseModel):
    """Condition for when a policy applies."""

    model: Optional[str] = Field(
        default=None,
        description="Model name pattern (exact match or regex) for when policy applies.",
    )


class PolicyCreateRequest(BaseModel):
    """Request body for creating a new policy."""

    policy_name: str = Field(description="Unique name for the policy.")
    inherit: Optional[str] = Field(
        default=None,
        description="Name of parent policy to inherit from.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the policy.",
    )
    guardrails_add: Optional[List[str]] = Field(
        default=None,
        description="List of guardrail names to add.",
    )
    guardrails_remove: Optional[List[str]] = Field(
        default=None,
        description="List of guardrail names to remove (from inherited).",
    )
    condition: Optional[PolicyConditionRequest] = Field(
        default=None,
        description="Condition for when this policy applies.",
    )


class PolicyUpdateRequest(BaseModel):
    """Request body for updating a policy."""

    policy_name: Optional[str] = Field(
        default=None,
        description="New name for the policy.",
    )
    inherit: Optional[str] = Field(
        default=None,
        description="Name of parent policy to inherit from.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the policy.",
    )
    guardrails_add: Optional[List[str]] = Field(
        default=None,
        description="List of guardrail names to add.",
    )
    guardrails_remove: Optional[List[str]] = Field(
        default=None,
        description="List of guardrail names to remove (from inherited).",
    )
    condition: Optional[PolicyConditionRequest] = Field(
        default=None,
        description="Condition for when this policy applies.",
    )


class PolicyDBResponse(BaseModel):
    """Response for a policy from the database."""

    policy_id: str = Field(description="Unique ID of the policy.")
    policy_name: str = Field(description="Name of the policy.")
    inherit: Optional[str] = Field(default=None, description="Parent policy name.")
    description: Optional[str] = Field(default=None, description="Policy description.")
    guardrails_add: List[str] = Field(
        default_factory=list, description="Guardrails to add."
    )
    guardrails_remove: List[str] = Field(
        default_factory=list, description="Guardrails to remove."
    )
    condition: Optional[Dict[str, Any]] = Field(
        default=None, description="Policy condition."
    )
    created_at: Optional[datetime] = Field(
        default=None, description="When the policy was created."
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="When the policy was last updated."
    )
    created_by: Optional[str] = Field(default=None, description="Who created the policy.")
    updated_by: Optional[str] = Field(
        default=None, description="Who last updated the policy."
    )


class PolicyListDBResponse(BaseModel):
    """Response for listing policies from the database."""

    policies: List[PolicyDBResponse] = Field(
        default_factory=list, description="List of policies."
    )
    total_count: int = Field(default=0, description="Total number of policies.")


# ─────────────────────────────────────────────────────────────────────────────
# Policy Attachment CRUD Types
# ─────────────────────────────────────────────────────────────────────────────


class PolicyAttachmentCreateRequest(BaseModel):
    """Request body for creating a policy attachment."""

    policy_name: str = Field(description="Name of the policy to attach.")
    scope: Optional[str] = Field(
        default=None,
        description="Use '*' for global scope (applies to all requests).",
    )
    teams: Optional[List[str]] = Field(
        default=None,
        description="Team aliases or patterns this attachment applies to.",
    )
    keys: Optional[List[str]] = Field(
        default=None,
        description="Key aliases or patterns this attachment applies to.",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Model names or patterns this attachment applies to.",
    )


class PolicyAttachmentDBResponse(BaseModel):
    """Response for a policy attachment from the database."""

    attachment_id: str = Field(description="Unique ID of the attachment.")
    policy_name: str = Field(description="Name of the attached policy.")
    scope: Optional[str] = Field(default=None, description="Scope of the attachment.")
    teams: List[str] = Field(default_factory=list, description="Team patterns.")
    keys: List[str] = Field(default_factory=list, description="Key patterns.")
    models: List[str] = Field(default_factory=list, description="Model patterns.")
    created_at: Optional[datetime] = Field(
        default=None, description="When the attachment was created."
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="When the attachment was last updated."
    )
    created_by: Optional[str] = Field(
        default=None, description="Who created the attachment."
    )
    updated_by: Optional[str] = Field(
        default=None, description="Who last updated the attachment."
    )


class PolicyAttachmentListResponse(BaseModel):
    """Response for listing policy attachments."""

    attachments: List[PolicyAttachmentDBResponse] = Field(
        default_factory=list, description="List of policy attachments."
    )
    total_count: int = Field(default=0, description="Total number of attachments.")
