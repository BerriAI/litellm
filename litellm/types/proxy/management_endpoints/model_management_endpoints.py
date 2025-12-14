from typing import Dict, List, Union, Any

from pydantic import BaseModel, Field

from ...router import ModelGroupInfo


class ModelGroupInfoProxy(ModelGroupInfo):
    is_public_model_group: bool = Field(default=False)


class UpdateUsefulLinksRequest(BaseModel):
    # Supports both old format (Dict[str, str]) and new format (Dict[str, Dict[str, Any]])
    # New format: { "displayName": { "url": "...", "index": 0 } }
    # Old format: { "displayName": "url" } (for backward compatibility)
    useful_links: Dict[str, Union[str, Dict[str, Any]]]


class NewModelGroupRequest(BaseModel):
    access_group: str  # The access group name (e.g., "production-models")
    model_names: List[str]  # Existing model groups to include (e.g., ["gpt-4", "claude-3"])


class NewModelGroupResponse(BaseModel):
    access_group: str
    model_names: List[str]
    models_updated: int  # Number of models updated


class UpdateModelGroupRequest(BaseModel):
    model_names: List[str]  # Updated list of model groups to include


class DeleteModelGroupResponse(BaseModel):
    access_group: str
    models_updated: int  # Number of deployments where the access group was removed
    message: str


class AccessGroupInfo(BaseModel):
    access_group: str
    model_names: List[str]  # List of model names in this access group
    deployment_count: int  # Total number of deployments with this access group


class ListAccessGroupsResponse(BaseModel):
    access_groups: List[AccessGroupInfo]