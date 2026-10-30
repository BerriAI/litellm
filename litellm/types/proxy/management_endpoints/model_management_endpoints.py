from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Final

from pydantic import ConfigDict, Field, model_validator

from litellm.types.llms.base import LiteLLMBaseModel

from ...router import ModelGroupInfo, updateDeployment


class PatchModelRequest(updateDeployment):
    litellm_params_to_remove: tuple[Annotated[str, Field(min_length=1)], ...] = Field(
        default=(),
        description=(
            "Top-level litellm_params keys to delete. Omitted parameters remain unchanged. "
            "The required model key and keys also supplied in litellm_params cannot be removed. "
            "Removing an absent key or repeating a key is a no-op."
        ),
    )

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    @model_validator(mode="after")
    def validate_param_removals(self) -> "PatchModelRequest":
        removals: Final = frozenset(self.litellm_params_to_remove)
        if "model" in removals:
            raise ValueError("The required litellm_params.model cannot be removed")
        conflicts: Final[frozenset[str]] = (
            removals.intersection(self.litellm_params.model_fields_set) if self.litellm_params else frozenset()
        )
        if conflicts:
            raise ValueError(f"Cannot set and remove the same litellm_params keys: {', '.join(sorted(conflicts))}")
        return self


class ModelGroupInfoProxy(ModelGroupInfo):
    is_public_model_group: bool = Field(default=False)
    health_status: str | None = Field(default=None)
    health_response_time: float | None = Field(default=None)
    health_checked_at: str | None = Field(default=None)
    custom_tags: tuple[str, ...] | None = None
    custom_metadata: Mapping[str, str | int | float | bool] | None = None


class UpdateUsefulLinksRequest(LiteLLMBaseModel):
    # Supports both old format (Dict[str, str]) and new format (Dict[str, Dict[str, Any]])
    # New format: { "displayName": { "url": "...", "index": 0 } }
    # Old format: { "displayName": "url" } (for backward compatibility)
    useful_links: dict[str, str | dict[str, object]]


class AutoRouterClassifierDefaultPromptResponse(LiteLLMBaseModel):
    """The built-in system prompt an auto-router's LLM classifier uses when none is configured.

    Served so the dashboard's prompt editor prefills the rubric the proxy actually sends, rather than
    a copy in the frontend that drifts the moment the rubric is edited.
    """

    system_prompt: str


class NewModelGroupRequest(LiteLLMBaseModel):
    access_group: str  # The access group name (e.g., "production-models")
    model_names: list[str] | None = None  # Existing model groups to include - tags ALL deployments for each name
    model_ids: list[str] | None = None  # Specific deployment IDs to tag (more precise than model_names)


class NewModelGroupResponse(LiteLLMBaseModel):
    access_group: str
    model_names: list[str] | None = None
    model_ids: list[str] | None = None
    models_updated: int  # Number of models updated


class UpdateModelGroupRequest(LiteLLMBaseModel):
    model_names: list[str] | None = None  # Updated list of model groups to include - tags ALL deployments for each name
    model_ids: list[str] | None = None  # Specific deployment IDs to tag (more precise than model_names)


class DeleteModelGroupResponse(LiteLLMBaseModel):
    access_group: str
    models_updated: int  # Number of deployments where the access group was removed
    message: str


class AccessGroupBudget(LiteLLMBaseModel):
    budget_id: str
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None


class AccessGroupBudgetRequest(LiteLLMBaseModel):
    budget_id: str | None = None  # Link an existing budget instead of creating one
    max_budget: float | None = Field(default=None, ge=0)
    soft_budget: float | None = Field(default=None, ge=0)
    budget_duration: str | None = None

    # rejects tpm_limit/rpm_limit/max_parallel_requests: those are not enforced per access group
    model_config = ConfigDict(extra="forbid")


class AccessGroupBudgetResponse(LiteLLMBaseModel):
    access_group: str
    spend: float  # Shared spend accrued by every key that can reach this access group
    budget: AccessGroupBudget | None = None


class DeleteAccessGroupBudgetResponse(LiteLLMBaseModel):
    access_group: str
    budget_deleted: bool  # False when the access group had no budget to begin with
    message: str


class AccessGroupInfo(LiteLLMBaseModel):
    access_group: str
    model_names: list[str]  # List of model names in this access group
    deployment_count: int  # Total number of deployments with this access group
    spend: float | None = None  # Spend drawn against the group's shared budget
    budget: AccessGroupBudget | None = None


class ListAccessGroupsResponse(LiteLLMBaseModel):
    access_groups: list[AccessGroupInfo]
