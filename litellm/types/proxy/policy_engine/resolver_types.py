"""
Policy resolver type definitions.

These types are used for matching requests to policies and resolving
the final guardrails list.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyMatchContext(BaseModel):
    """
    Context used to match a request against policies.

    Contains the team alias, key alias, and model from the incoming request.
    """

    team_alias: str | None = Field(
        default=None,
        description="Team alias from the request.",
    )
    key_alias: str | None = Field(
        default=None,
        description="API key alias from the request.",
    )
    model: str | None = Field(
        default=None,
        description="Model name from the request.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tags from key/team metadata.",
    )

    model_config = ConfigDict(extra="forbid")


class ResolvedPolicy(BaseModel):
    """
    Result of resolving a policy with its inheritance chain.

    Contains the final list of guardrails after applying all add/remove operations.
    """

    policy_name: str = Field(description="Name of the resolved policy.")
    guardrails: list[str] = Field(
        default_factory=list,
        description="Final list of guardrail names to apply.",
    )
    inheritance_chain: list[str] = Field(
        default_factory=list,
        description="List of policy names in the inheritance chain (from root to this policy).",
    )

    model_config = ConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# API Response Types
# ─────────────────────────────────────────────────────────────────────────────


class PolicyScopeResponse(BaseModel):
    """Scope configuration for a policy."""

    teams: list[str] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PolicyGuardrailsResponse(BaseModel):
    """Guardrails configuration for a policy."""

    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)


class PolicyInfoResponse(BaseModel):
    """Response for /policy/info/{policy_name} endpoint."""

    policy_name: str
    inherit: str | None = None
    scope: PolicyScopeResponse
    guardrails: PolicyGuardrailsResponse
    resolved_guardrails: list[str]
    inheritance_chain: list[str]


class PolicySummaryItem(BaseModel):
    """Summary of a single policy for list endpoint."""

    inherit: str | None = None
    scope: PolicyScopeResponse
    guardrails: PolicyGuardrailsResponse
    resolved_guardrails: list[str]
    inheritance_chain: list[str]


class PolicyListResponse(BaseModel):
    """Response for /policy/list endpoint."""

    policies: dict[str, PolicySummaryItem]
    total_count: int


class PolicyTestResponse(BaseModel):
    """Response for /policy/test endpoint."""

    context: PolicyMatchContext
    matching_policies: list[str]
    resolved_guardrails: list[str]
    message: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Request/Response Types for Policy Endpoints
# ─────────────────────────────────────────────────────────────────────────────


class PolicyConditionRequest(BaseModel):
    """Condition for when a policy applies."""

    model: str | None = Field(
        default=None,
        description="Model name pattern (exact match or regex) for when policy applies.",
    )


class PolicyCreateRequest(BaseModel):
    """Request body for creating a new policy."""

    policy_name: str = Field(description="Unique name for the policy.")
    inherit: str | None = Field(
        default=None,
        description="Name of parent policy to inherit from.",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the policy.",
    )
    guardrails_add: list[str] | None = Field(
        default=None,
        description="List of guardrail names to add.",
    )
    guardrails_remove: list[str] | None = Field(
        default=None,
        description="List of guardrail names to remove (from inherited).",
    )
    condition: PolicyConditionRequest | None = Field(
        default=None,
        description="Condition for when this policy applies.",
    )
    pipeline: dict[str, Any] | None = Field(
        default=None,
        description="Optional guardrail pipeline for ordered execution. Contains 'mode' and 'steps'.",
    )


class PolicyUpdateRequest(BaseModel):
    """Request body for updating a policy."""

    policy_name: str | None = Field(
        default=None,
        description="New name for the policy.",
    )
    inherit: str | None = Field(
        default=None,
        description="Name of parent policy to inherit from.",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the policy.",
    )
    guardrails_add: list[str] | None = Field(
        default=None,
        description="List of guardrail names to add.",
    )
    guardrails_remove: list[str] | None = Field(
        default=None,
        description="List of guardrail names to remove (from inherited).",
    )
    condition: PolicyConditionRequest | None = Field(
        default=None,
        description="Condition for when this policy applies.",
    )
    pipeline: dict[str, Any] | None = Field(
        default=None,
        description="Optional guardrail pipeline for ordered execution. Contains 'mode' and 'steps'.",
    )


class PolicyDBResponse(BaseModel):
    """Response for a policy from the database."""

    policy_id: str = Field(description="Unique ID of the policy.")
    policy_name: str = Field(description="Name of the policy.")
    version_number: int = Field(default=1, description="Version number of this policy.")
    version_status: str = Field(
        default="production",
        description="One of: draft, published, production.",
    )
    parent_version_id: str | None = Field(default=None, description="Policy ID this version was cloned from.")
    is_latest: bool = Field(
        default=True,
        description="True if this is the latest version by version_number.",
    )
    published_at: datetime | None = Field(default=None, description="When this version was published.")
    production_at: datetime | None = Field(default=None, description="When this version was promoted to production.")
    inherit: str | None = Field(default=None, description="Parent policy name.")
    description: str | None = Field(default=None, description="Policy description.")
    guardrails_add: list[str] = Field(default_factory=list, description="Guardrails to add.")
    guardrails_remove: list[str] = Field(default_factory=list, description="Guardrails to remove.")
    condition: dict[str, Any] | None = Field(default=None, description="Policy condition.")
    pipeline: dict[str, Any] | None = Field(default=None, description="Optional guardrail pipeline.")
    created_at: datetime | None = Field(default=None, description="When the policy was created.")
    updated_at: datetime | None = Field(default=None, description="When the policy was last updated.")
    created_by: str | None = Field(default=None, description="Who created the policy.")
    updated_by: str | None = Field(default=None, description="Who last updated the policy.")
    definition_location: Literal["db", "config"] = Field(
        default="db",
        description="Where this policy is defined: 'db' (database) or 'config' (config.yaml).",
    )


class PolicyListDBResponse(BaseModel):
    """Response for listing policies from the database."""

    policies: list[PolicyDBResponse] = Field(default_factory=list, description="List of policies.")
    total_count: int = Field(default=0, description="Total number of policies.")


# ─────────────────────────────────────────────────────────────────────────────
# Policy Versioning Types
# ─────────────────────────────────────────────────────────────────────────────


class PolicyVersionCreateRequest(BaseModel):
    """Request body for creating a new policy version (draft)."""

    source_policy_id: str | None = Field(
        default=None,
        description="Policy ID to clone from. If None, clone from current production version.",
    )


class PolicyVersionStatusUpdateRequest(BaseModel):
    """Request body for updating a policy version's status."""

    version_status: str = Field(
        description="New status: 'published' or 'production'.",
    )


class PolicyVersionListResponse(BaseModel):
    """Response for listing all versions of a policy."""

    policy_name: str = Field(description="Name of the policy.")
    versions: list[PolicyDBResponse] = Field(
        default_factory=list, description="All versions ordered by version_number desc."
    )
    total_count: int = Field(default=0, description="Total number of versions.")


class PolicyVersionCompareResponse(BaseModel):
    """Response for comparing two policy versions."""

    version_a: PolicyDBResponse = Field(description="First version.")
    version_b: PolicyDBResponse = Field(description="Second version.")
    field_diffs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Field name -> {version_a: val, version_b: val} for differing fields.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Policy Attachment CRUD Types
# ─────────────────────────────────────────────────────────────────────────────


class PolicyAttachmentCreateRequest(BaseModel):
    """Request body for creating a policy attachment."""

    policy_name: str = Field(description="Name of the policy to attach.")
    scope: str | None = Field(
        default=None,
        description="Use '*' for global scope (applies to all requests).",
    )
    teams: list[str] | None = Field(
        default=None,
        description="Team aliases or patterns this attachment applies to.",
    )
    keys: list[str] | None = Field(
        default=None,
        description="Key aliases or patterns this attachment applies to.",
    )
    models: list[str] | None = Field(
        default=None,
        description="Model names or patterns this attachment applies to.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tag patterns this attachment applies to. Supports wildcards (e.g., health-*).",
    )


class PolicyAttachmentDBResponse(BaseModel):
    """Response for a policy attachment from the database."""

    attachment_id: str = Field(description="Unique ID of the attachment.")
    policy_name: str = Field(description="Name of the attached policy.")
    scope: str | None = Field(default=None, description="Scope of the attachment.")
    teams: list[str] = Field(default_factory=list, description="Team patterns.")
    keys: list[str] = Field(default_factory=list, description="Key patterns.")
    models: list[str] = Field(default_factory=list, description="Model patterns.")
    tags: list[str] = Field(default_factory=list, description="Tag patterns.")
    created_at: datetime | None = Field(default=None, description="When the attachment was created.")
    updated_at: datetime | None = Field(default=None, description="When the attachment was last updated.")
    created_by: str | None = Field(default=None, description="Who created the attachment.")
    updated_by: str | None = Field(default=None, description="Who last updated the attachment.")
    definition_location: Literal["db", "config"] = Field(
        default="db",
        description="Where this attachment is defined: 'db' (database) or 'config' (config.yaml).",
    )


class PolicyAttachmentListResponse(BaseModel):
    """Response for listing policy attachments."""

    attachments: list[PolicyAttachmentDBResponse] = Field(
        default_factory=list, description="List of policy attachments."
    )
    total_count: int = Field(default=0, description="Total number of attachments.")


# ─────────────────────────────────────────────────────────────────────────────
# Policy Resolve Types
# ─────────────────────────────────────────────────────────────────────────────


class PipelineTestRequest(BaseModel):
    """Request body for testing a guardrail pipeline with sample messages."""

    pipeline: dict[str, Any] = Field(
        description="Pipeline definition with 'mode' and 'steps'.",
    )
    test_messages: list[dict[str, str]] = Field(
        description="Test messages to run through the pipeline, e.g. [{'role': 'user', 'content': '...'}].",
    )


class PolicyResolveRequest(BaseModel):
    """Request body for resolving effective policies/guardrails for a context."""

    team_alias: str | None = Field(default=None, description="Team alias to resolve for.")
    key_alias: str | None = Field(default=None, description="Key alias to resolve for.")
    model: str | None = Field(default=None, description="Model name to resolve for.")
    tags: list[str] | None = Field(default=None, description="Tags to resolve for.")


class PolicyMatchDetail(BaseModel):
    """Details about why a specific policy matched."""

    policy_name: str = Field(description="Name of the matched policy.")
    matched_via: str = Field(
        description="How the policy was matched (e.g., 'tag:healthcare', 'team:health-team', 'scope:*')."
    )
    guardrails_added: list[str] = Field(
        default_factory=list,
        description="Guardrails this policy contributes.",
    )


class PolicyResolveResponse(BaseModel):
    """Response for resolving effective policies/guardrails for a context."""

    effective_guardrails: list[str] = Field(
        default_factory=list,
        description="Final list of guardrails that would be applied.",
    )
    matched_policies: list[PolicyMatchDetail] = Field(
        default_factory=list,
        description="Details about each matched policy and why it matched.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Attachment Impact Estimation Types
# ─────────────────────────────────────────────────────────────────────────────


class AttachmentImpactResponse(BaseModel):
    """Response for estimating the impact of a policy attachment."""

    affected_keys_count: int = Field(
        default=0,
        description="Number of keys that would be affected (named + unnamed).",
    )
    affected_teams_count: int = Field(
        default=0,
        description="Number of teams that would be affected (named + unnamed).",
    )
    unnamed_keys_count: int = Field(default=0, description="Number of affected keys without an alias.")
    unnamed_teams_count: int = Field(default=0, description="Number of affected teams without an alias.")
    sample_keys: list[str] = Field(
        default_factory=list,
        description="Sample of affected key aliases (up to 10).",
    )
    sample_teams: list[str] = Field(
        default_factory=list,
        description="Sample of affected team aliases (up to 10).",
    )
