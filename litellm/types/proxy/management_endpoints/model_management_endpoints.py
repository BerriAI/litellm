from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...router import ModelGroupInfo


class ModelGroupInfoProxy(ModelGroupInfo):
    is_public_model_group: bool = Field(default=False)
    health_status: str | None = Field(default=None)
    health_response_time: float | None = Field(default=None)
    health_checked_at: str | None = Field(default=None)


class UpdateUsefulLinksRequest(BaseModel):
    # Supports both old format (Dict[str, str]) and new format (Dict[str, Dict[str, Any]])
    # New format: { "displayName": { "url": "...", "index": 0 } }
    # Old format: { "displayName": "url" } (for backward compatibility)
    useful_links: dict[str, str | dict[str, Any]]


class AutoRouterClassifierDefaultPromptResponse(BaseModel):
    """The built-in system prompt an auto-router's LLM classifier uses when none is configured.

    Served so the dashboard's prompt editor prefills the rubric the proxy actually sends, rather than
    a copy in the frontend that drifts the moment the rubric is edited.
    """

    system_prompt: str


class NewModelGroupRequest(BaseModel):
    access_group: str  # The access group name (e.g., "production-models")
    model_names: list[str] | None = None  # Existing model groups to include - tags ALL deployments for each name
    model_ids: list[str] | None = None  # Specific deployment IDs to tag (more precise than model_names)


class NewModelGroupResponse(BaseModel):
    access_group: str
    model_names: list[str] | None = None
    model_ids: list[str] | None = None
    models_updated: int  # Number of models updated


class UpdateModelGroupRequest(BaseModel):
    model_names: list[str] | None = None  # Updated list of model groups to include - tags ALL deployments for each name
    model_ids: list[str] | None = None  # Specific deployment IDs to tag (more precise than model_names)


class DeleteModelGroupResponse(BaseModel):
    access_group: str
    models_updated: int  # Number of deployments where the access group was removed
    message: str


class AccessGroupBudget(BaseModel):
    budget_id: str
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None


class AccessGroupBudgetRequest(BaseModel):
    budget_id: str | None = None  # Link an existing budget instead of creating one
    max_budget: float | None = Field(default=None, ge=0)
    soft_budget: float | None = Field(default=None, ge=0)
    budget_duration: str | None = None

    # rejects tpm_limit/rpm_limit/max_parallel_requests: those are not enforced per access group
    model_config = ConfigDict(extra="forbid")


class AccessGroupBudgetResponse(BaseModel):
    access_group: str
    spend: float  # Shared spend accrued by every key that can reach this access group
    budget: AccessGroupBudget | None = None


class DeleteAccessGroupBudgetResponse(BaseModel):
    access_group: str
    budget_deleted: bool  # False when the access group had no budget to begin with
    message: str


class AccessGroupInfo(BaseModel):
    access_group: str
    model_names: list[str]  # List of model names in this access group
    deployment_count: int  # Total number of deployments with this access group
    spend: float | None = None  # Spend drawn against the group's shared budget
    budget: AccessGroupBudget | None = None


class ListAccessGroupsResponse(BaseModel):
    access_groups: list[AccessGroupInfo]


class ProviderModelDiscoveryRequest(BaseModel):
    """Body for POST /provider/models/discover. Exactly one of ``litellm_credential_name`` or
    inline ``api_key``/``api_base`` names the credential to probe; extra fields are rejected
    outright rather than silently ignored, since this is a security boundary -- see
    ``reject_server_owned_wif_params`` in the handler for why."""

    custom_llm_provider: str
    litellm_credential_name: str | None = None
    api_key: str | None = None
    api_base: str | None = None

    model_config = ConfigDict(extra="forbid")


class ProviderModelDiscoveryResponse(BaseModel):
    models: Sequence[str]
