from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel

from ...router import ModelGroupInfo, ModelInfo
from .management_v1 import ResourceResponse


class ModelGroupInfoProxy(ModelGroupInfo):
    is_public_model_group: bool = Field(default=False)
    health_status: str | None = Field(default=None)
    health_response_time: float | None = Field(default=None)
    health_checked_at: str | None = Field(default=None)


class ModelGroupInfoResponse(ResourceResponse[tuple[ModelGroupInfoProxy, ...]]):
    """`{data: [...]}` of `GET /model_group/info`: one `ModelGroupInfoProxy` per model group the caller may see."""

    model_config = ConfigDict(frozen=True)


class ModelInfoV1LiteLLMParams(LiteLLMBaseModel):
    """A deployment's litellm_params as `GET /model/info` serves them.

    Every configured key passes through except credentials: api_key, client_secret, vertex credentials and AWS
    access keys are removed and every other secret-shaped value is masked, except litellm_credential_name. With the
    proxy started from the CLI with `--model`, every unset parameter arrives as the string "None", or as an empty
    string when its name is secret-shaped.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    model: str


class ModelInfoV1Deployment(LiteLLMBaseModel):
    """One deployment as `GET /model/info` and `GET /v1/model/info` report it.

    Any further deployment-level key passes through from the router row unchanged.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    model_name: str = Field(
        description=(
            "The name requests use for this deployment. A team deployment stored under the proxy's internal "
            "`model_name_<team_id>_<uuid>` name reports its team_public_model_name instead."
        )
    )
    litellm_params: ModelInfoV1LiteLLMParams
    model_info: ModelInfo = Field(
        description=(
            "The configured model_info merged with the cost map entry for the model (pricing, token limits, "
            "mode, supports_* capabilities, supported_openai_params, litellm_provider, key), the proxy's "
            "discovered model info and pricing_overrides. With the proxy database connected, direct_access is "
            "added, and access_via_team_ids for an admin or for a key that belongs to a user."
        )
    )


class ModelInfoV1Response(ResourceResponse[tuple[ModelInfoV1Deployment, ...] | ModelInfoV1Deployment]):
    """`{data: ...}` of `GET /model/info` and `GET /v1/model/info`.

    `data` is one `ModelInfoV1Deployment` per deployment the caller may see. With the proxy started from the CLI
    with `--model` and no config, `data` is that single deployment as one object instead of a list.
    """

    model_config = ConfigDict(frozen=True)


class UpdateUsefulLinksRequest(LiteLLMBaseModel):
    # Supports both old format (Dict[str, str]) and new format (Dict[str, Dict[str, Any]])
    # New format: { "displayName": { "url": "...", "index": 0 } }
    # Old format: { "displayName": "url" } (for backward compatibility)
    useful_links: dict[str, str | dict[str, Any]]


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
