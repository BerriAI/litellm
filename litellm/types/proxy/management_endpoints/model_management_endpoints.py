from typing import Any

from pydantic import BaseModel, Field

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


class AccessGroupInfo(BaseModel):
    access_group: str
    model_names: list[str]  # List of model names in this access group
    deployment_count: int  # Total number of deployments with this access group


class ListAccessGroupsResponse(BaseModel):
    access_groups: list[AccessGroupInfo]
