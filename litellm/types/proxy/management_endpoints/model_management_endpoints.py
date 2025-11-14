from typing import Dict, List

from pydantic import BaseModel, Field

from ...router import ModelGroupInfo


class ModelGroupInfoProxy(ModelGroupInfo):
    is_public_model_group: bool = Field(default=False)


class UpdateUsefulLinksRequest(BaseModel):
    useful_links: Dict[str, str]


class NewModelGroupRequest(BaseModel):
    access_group: str  # The access group name (e.g., "production-models")
    model_names: List[str]  # Existing model groups to include (e.g., ["gpt-4", "claude-3"])

class NewModelGroupResponse(BaseModel):
    access_group: str
    model_names: List[str]
    models_updated: int  # Number of models updated